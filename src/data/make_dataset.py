# -*- coding: utf-8 -*-
import os
import pandas as pd
import click
import logging
from pathlib import Path
import numpy as np
from sklearn.linear_model import LinearRegression
import concurrent.futures

from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm
from scipy.signal import medfilt
import ruptures as rpt
from scipy.stats import mode


def cut_window(data_cycle, overlap=0.8, base_length=96):
    step = int((1 - overlap) * base_length)
    n_windows = np.ceil((data_cycle.shape[0] - base_length) / step)
    indexer = np.arange(base_length)[None, :] + step * np.arange(n_windows)[:, None]
    return indexer.astype('int'), int(n_windows)


def get_a_NN_object(df, n_wells_start, n_wells_end=2000, window_length=45):
    df['GR_medfilt'] = medfilt(df['GR'], 11)
    well_ids = df['well_id'].unique().tolist()[n_wells_start:(n_wells_start + n_wells_end)]
    df_wells = df[df['well_id'].isin(well_ids)]
    idxr, n_wins = cut_window(df_wells['GR_medfilt'], overlap=0.8, base_length=window_length)
    windows = df_wells['GR_medfilt'].values[idxr]
    labels = df_wells['label'].values[idxr]
    NN = NearestNeighbors(n_neighbors=10, metric='braycurtis')
    NN.fit(windows)
    return NN, labels


def augment_a_well_nearest_neighbors(df, NN, labels, bl=40):
    df_tmp = df.copy()
    df_tmp['GR_nn'] = medfilt(df['GR'], 11)
    idxr, n_win = cut_window(df_tmp['GR_nn'].values, overlap=0.0, base_length=bl)
    windows = df_tmp['GR_nn'].values[idxr]
    dist, idw = NN.kneighbors(windows, n_neighbors=5)
    dist_flip, idw_flip = NN.kneighbors(np.fliplr(windows), n_neighbors=5)

    pred = mode(labels[idw, :], axis=1)[0]
    pred = pred.reshape(pred.shape[0], pred.shape[2]).flatten()

    pred_lr = mode(labels[idw_flip, :], axis=1)[0]
    pred_lr = pred_lr.reshape(pred_lr.shape[0], pred_lr.shape[2])

    df_labels_windows = pd.DataFrame(
        {'row_id': idxr.flatten(), f'label_nn_{bl}': pred, f'flip_same_{bl}': np.equal(pred, pred_lr.flatten()),
         f'same_after_inv_{bl}': np.equal(pred, np.fliplr(pred_lr).flatten())})
    res = df_labels_windows.groupby('row_id').mean().reset_index()
    res = df_tmp.join(res, on='row_id', rsuffix='_r').drop(columns=['row_id_r', 'GR_nn'])
    return res


def divide_block(x):
    algo = rpt.Pelt(model="rbf", min_size=5, jump=3).fit(x)
    inds = np.array(algo.predict(pen=5))
    counts = np.zeros(x.shape[0])
    counts[:inds[0]] = np.arange(inds[0])
    res = np.zeros(x.shape[0])
    for i in range(len(inds) - 1):
        res[inds[i]:inds[i + 1]] = i
        counts[inds[i]:inds[i + 1]] = np.arange(inds[i + 1] - inds[i])

    return res, counts


def diff_sum(x):
    if x.shape[0] > 1:
        res = np.diff(x).sum()
    else:
        res=-99
    return res


def diff_max(x):
    if x.shape[0]>1:
        res = np.nanmax(np.diff(x))
    else:
        res=-99
    return res


def diff_min(x):
    if x.shape[0] > 1:
        res = np.nanmin(np.diff(x))
    else:
        res=-99
    return res


def abs_diff_sum(x):
    if x.shape[0] > 1:
        res = np.abs(np.diff(x)).sum()
    else:
        res=-99
    return res


def rolling_slope(x):
    model_ols = LinearRegression()
    idx = np.arange(x.shape[0])
    if np.isnan(x).sum() > 0 or (x.shape[0]<2) :
        res = [np.nan]
    else:
        model_ols.fit(np.array(idx).reshape(-1, 1), np.array(x).reshape(-1, 1))
        res = model_ols.coef_[0]
    return res


def rolling_r2(x):
    model_ols = LinearRegression()
    idx = np.arange(x.shape[0])
    if (np.isnan(x).sum() > 0) or (x.shape[0]<2) :
        res=np.nan
    else:
        model_ols.fit(idx.reshape(-1, 1), x.reshape(-1, 1))
        res = model_ols.score(idx.reshape(-1, 1), x.reshape(-1, 1))

    return res


def half_rolling_slope(x):
    model_ols = LinearRegression()
    idx = np.arange(max(x.shape[0] // 2, 1))
    if (np.isnan(x).sum() > 0) or (x.shape[0]<2) :
        res=[np.nan]
    else:
        model_ols.fit(np.array(idx).reshape(-1, 1), np.array(x)[idx].reshape(-1, 1))
        res = model_ols.coef_[0]
    return res


def make_parabolic(w, amp=50):
    x = [0, w / 2, w]
    y = [amp, 0, amp]
    p_coef = np.polyfit(x, y, 2)
    p = np.poly1d(p_coef)
    x_new = np.arange(w)
    res = p(x_new)
    return res


def corr_with_parabolic(x, y):
    if x.shape[0] == y.shape[0]:
        res = np.corrcoef(x, y)[0, 1]
    else:
        res = np.nan
    return res


def corr_with_parabolic_grp(x):
    if x.shape[0] > 3:

        y = make_parabolic(w=x.shape[0])
        res = np.corrcoef(x, y)[0, 1]
    else:
        res=np.nan
    return res


def value_middle_vs_ends(x):
    x = np.array(x)
    left = x[:2].mean()
    right = x[-2:].mean()
    mid = x[x.shape[0] // 2]
    res = (left + right) * 0.5 - mid
    return res


def calculate_lags(s, note='series', lags=None):
    if lags is None:
        lags = [1, 2, 3, 4]
    names = []
    series = []
    for l in lags:
        series.append(s.shift(l))
        names.append(f"{note}_lag_{l}")
    df = pd.DataFrame(dict(zip(names, series)))
    return df


def apply_rolling_functions(df, col='GR', window=10,
                            func=None, center=True):
    template = make_parabolic(w=window)
    if func is None:
        func = {'mean': np.mean, 'std': np.std, 'diff_sum': diff_sum, 'abs_diff_sum': abs_diff_sum,
                'corr_parabolic': lambda x: corr_with_parabolic(x, template),
                'slope': rolling_slope, 'mid_vs_end': value_middle_vs_ends, 'max_diff': diff_max, 'min_diff': diff_min}
    names = []
    for k, v in func.items():
        series = df.loc[:, col].rolling(window=window, center=center, min_periods=window // 2).apply(v, raw=True)
        colname = f'{k}_{window}_{col}'
        df.loc[:, colname] = series.values
        names.append(colname)
    # df[f'slope_diff_{window}'] = df[f'half_slope_{window}_{col}'] - df[f'slope_{window}_{col}']
    df.index = np.arange(df.shape[0])
    return df[names]


def apply_grouped_functions(df, col='GR', groups='grp',
                            func=None):
    template = make_parabolic(w=50)

    if func is None:
        func = {'mean': np.mean, 'std': np.std, 'diff_sum': diff_sum, 'abs_diff_sum': abs_diff_sum,
                'slope': lambda x: rolling_slope(x)[0], 'mid_vs_end': value_middle_vs_ends, 'max_diff': diff_max,
                'min_diff': diff_min,
                'half_slope': lambda x: half_rolling_slope(x)[0],
                'corr_parabolic': lambda x: corr_with_parabolic_grp(x),
                'size': lambda x: x.shape[0]}
    names = []
    for k, v in func.items():
        # series = df.groupby(groups)[col].transform(v)
        colname = f'{col}_{k}'
        # df.loc[:, colname] = series.values
        grouped_stats = df.groupby(groups)[col].apply(v)
        grouped_stats_prev = df.groupby(groups)[col].apply(v).shift(1)
        grouped_stats_next = df.groupby(groups)[col].apply(v).shift(-1)
        grouped_stats_diffp = grouped_stats - grouped_stats_prev
        grouped_stats_diffn = grouped_stats - grouped_stats_next

        df = df.join(grouped_stats, on=groups, rsuffix=f'_{k}')
        df = df.join(grouped_stats_diffp, on=groups, rsuffix=f'_{k}_dp')
        df = df.join(grouped_stats_diffn, on=groups, rsuffix=f'_{k}_dn')

        names.append(colname)
        names.append(f'{col}_{k}_dp')
        names.append(f'{col}_{k}_dn')

    df[f'{col}_slopediff'] = df[f'{col}_slope'] - df[f'{col}_half_slope']
    df.index = np.arange(df.shape[0])
    return df[names]


def preprocess_a_well(df_well, windows=[15, 30, 65]):
    df_well['GR_medfilt'] = medfilt(df_well['GR'], 11)
    df_well['GR_diff'] = df_well['GR_medfilt'].diff()
    df_well['GR_shifted'] = df_well['GR_medfilt'].shift(100)
    block, counts = divide_block(medfilt(df_well['GR'], 11))
    df_well['grp'] = (df_well[f'label_nn_{windows[0]}'] != df_well[f'label_nn_{windows[0]}'].shift(1)).cumsum().fillna('pad')
    df_well['block'] = block
    df_well['counts'] = counts
    # Add lag variables:
    df_lags = calculate_lags(s=df_well['GR_medfilt'], note='GR_medfilt', lags=np.arange(-50, 50, 5))

    label_lags = []
    for w in windows:
        df_label_lags = calculate_lags(s=df_well[f'label_nn_{w}'], note=f'label_nn_{w}', lags=np.arange(-50, 50, 5))
        label_lags.append(df_label_lags.copy())

    df_feats_ws = apply_rolling_functions(df_well.copy(), window=10, col='GR_medfilt')
    df_feats_wm = apply_rolling_functions(df_well.copy(), window=20, col='GR_medfilt')
    df_feats_wl = apply_rolling_functions(df_well.copy(), window=50, col='GR_medfilt')
    df_feats_wxl = apply_rolling_functions(df_well.copy(), window=120, col='GR_medfilt')
    df_feats_wl_right = apply_rolling_functions(df_well.copy(), window=100, col='GR_medfilt', center=False)
    # df_feats_wl_right_shift = apply_rolling_functions(df_well.copy(), window=100, col='GR_shifted', center=False)
    df_feats_grouped = apply_grouped_functions(df_well.copy(), col='GR_medfilt', groups='grp')
    # df_feats = pd.concat([df_well,df_lags,df_feats_ws, df_feats_wm, df_feats_wl,df_feats_wl_right,df_feats_wxl], axis=1)
    df_feats = pd.concat(
        [df_well, df_lags, df_feats_ws, df_feats_wm, df_feats_wl, df_feats_wl_right, df_feats_wxl,
         df_feats_grouped] + label_lags, axis=1)

    return df_feats


def preprocess_dataset(df, n_wells=50, NN_list=[], labels_list=[]):
    df_well_list = []
    wells = df['well_id'].unique().tolist()[:n_wells]
    list_df_wells = [df.loc[df['well_id'].isin([w]), :].copy() for w in wells]
    for df in list_df_wells:
        df.index = np.arange(df.shape[0])

    for NN, labels in zip(NN_list, labels_list):
        for i, df in enumerate(list_df_wells):
            tmp = augment_a_well_nearest_neighbors(df, NN, labels, bl=labels.shape[1])
            list_df_wells[i] = tmp
    for w in tqdm(list_df_wells):
        df_well = w
        df_new = preprocess_a_well(df_well.copy(),windows=[15])
        df_well_list.append(df_new.copy())

    df_preprocessed = pd.concat(df_well_list, axis=0)
    df_preprocessed.index = np.arange(df_preprocessed.shape[0])
    return df_preprocessed


def preprocess_dataset_parallel(df, n_wells=50, NN_list=[], labels_list=[]):
    wells = df['well_id'].unique().tolist()[:n_wells]
    list_df_wells = [df.loc[df['well_id'].isin([w]), :].copy() for w in wells]
    for df in list_df_wells:
        df.index = np.arange(df.shape[0])

    for NN, labels in zip(NN_list, labels_list):
        for i, df in enumerate(list_df_wells):
            tmp = augment_a_well_nearest_neighbors(df, NN, labels, bl=labels.shape[1])
            list_df_wells[i] = tmp

    with concurrent.futures.ProcessPoolExecutor(max_workers=4) as executor:
        results = list(tqdm(executor.map(preprocess_a_well, list_df_wells), total=len(list_df_wells)))

    df_preprocessed = pd.concat(results, axis=0)
    df_preprocessed.index = np.arange(df_preprocessed.shape[0])
    return df_preprocessed


def main(input_filepath, output_filepath):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')
    df_train = pd.read_csv(os.path.join(input_filepath, 'train_lofi_rowid_Nov13.csv'))
    df_test = pd.read_csv(os.path.join(input_filepath, 'test_lofi_rowid_Nov13.csv'))

    fname_final_train = os.path.join(output_filepath, 'train.pck')

    n_test = df_test['well_id'].unique().shape[0]
    n_wells = 200
    NN_l, labels_l = get_a_NN_object(df_train, n_wells,n_wells_end=400, window_length=65)
    NN_m, labels_m = get_a_NN_object(df_train, 400,800, window_length=30)
    NN_s, labels_s = get_a_NN_object(df_train, 800,1200, window_length=15)

    df_train_processed = preprocess_dataset_parallel(df_train, n_wells=n_wells, NN_list=[NN_l, NN_m, NN_s],
                                                     labels_list=[labels_l, labels_m, labels_s])
    #df_train_processed = preprocess_dataset(df_train, n_wells=n_wells, NN_list=[NN_s],
    #                                                  labels_list=[labels_s])

    df_train_processed.to_pickle(fname_final_train)

    fname_final_test = os.path.join(output_filepath, 'test.pck')

    # df_test_processed = preprocess_dataset_parallel(df_test, n_wells=n_test, NN=NN, labels=labels)
    # df_test_processed.to_pickle(fname_final_test)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    input_filepath = os.path.join(project_dir, 'data', 'raw')
    output_filepath = os.path.join(project_dir, 'data', 'processed')
    os.makedirs(output_filepath, exist_ok=True)
    main(input_filepath, output_filepath)
