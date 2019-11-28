# -*- coding: utf-8 -*-
import os
import pickle

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
from scipy.stats import mode, kurtosis
from statsmodels.tsa.stattools import acf

log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
logging.basicConfig(level=logging.INFO, format=log_fmt)


def calculate_derivatives(df, order=1, cols=['A', 'B']):
    for c in cols:
        colname = f'{c}_{order}'
        df[colname] = df[c].values
        for i in range(order):
            df[colname] = df[colname].diff()
    return df


def cut_window(data_cycle, overlap=0.8, base_length=96):
    step = int((1 - overlap) * base_length)
    n_windows = np.ceil((data_cycle.shape[0] - base_length) / step)
    indexer = np.arange(base_length)[None, :] + step * np.arange(n_windows)[:, None]
    return indexer.astype('int'), int(n_windows)


def get_a_NN_object(df, n_wells_start, n_wells_end=2000, window_length=45, kernel_size=11):
    df['GR_medfilt'] = medfilt(df['GR'], kernel_size)
    well_ids = df['well_id'].unique().tolist()[n_wells_start:(n_wells_start + n_wells_end)]
    df_wells = df[df['well_id'].isin(well_ids)]
    idxr, n_wins = cut_window(df_wells['GR_medfilt'], overlap=0.8, base_length=window_length)
    windows = df_wells['GR_medfilt'].values[idxr]
    labels = df_wells['label'].values[idxr]
    NN = NearestNeighbors(n_neighbors=10, metric='braycurtis')
    NN.fit(windows)
    return NN, labels


def augment_a_well_nearest_neighbors(df, NN, labels, bl=40, kernel_size=11):
    df_tmp = df.copy()
    df_tmp['GR_nn'] = medfilt(df['GR'], kernel_size)
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


def divide_block_steps(df, t=24):
    df['step'] = (df['GR'].diff().abs() >= t)
    df_steps = df[df['step']]
    inds = np.hstack([0, np.array(df_steps['row_id']), df['row_id'].max() + 1])
    counts = np.zeros(df.shape[0])
    counts[:inds[0]] = np.arange(inds[0])
    res = np.zeros(df.shape[0])
    for i in range(len(inds) - 1):
        res[inds[i]:inds[i + 1]] = i
        counts[inds[i]:inds[i + 1]] = np.arange(inds[i + 1] - inds[i])

    df['prev_val_step'] = np.nan
    df.loc[df['step'], 'prev_val_step'] = df['GR'].diff()
    df['prev_val_step'] = df['prev_val_step'].fillna(method='ffill').astype('float')

    df['next_val_step'] = np.nan
    df.loc[df['step'], 'next_val_step'] = df['GR'].diff()
    df['next_val_step'] = df['next_val_step'].fillna(method='bfill').astype('float')

    df['count_step'] = counts
    df['count_groups'] = res
    df['count_step_norm'] = df['count_step'] / df.groupby('count_groups')['count_step'].transform('max')
    df.drop(columns=['step'], inplace=True)
    return df


def diff_sum(x):
    if x.shape[0] > 1:
        res = np.diff(x).sum()
    else:
        res = -99
    return res


def diff_max(x):
    if x.shape[0] > 1:
        res = np.nanmax(np.diff(x))
    else:
        res = -99
    return res


def diff_min(x):
    if x.shape[0] > 1:
        res = np.nanmin(np.diff(x))
    else:
        res = -99
    return res


def abs_diff_sum(x):
    if x.shape[0] > 1:
        res = np.abs(np.diff(x)).sum()
    else:
        res = -99
    return res


def abs_diff_exceed_count(x, t=24):
    if x.shape[0] > 1:
        res = (np.abs(np.diff(x)) > t).sum()
    else:
        res = -99
    return res


def calculate_acf(x):
    if x.shape[0] > 2:
        res = np.mean(np.abs(acf(x, fft=False)[1:]))
    else:
        res = -99
    return res


def rolling_slope(x):
    model_ols = LinearRegression()
    idx = np.arange(x.shape[0])
    if np.isnan(x).sum() > 0 or (x.shape[0] < 2):
        res = [np.nan]
    else:
        model_ols.fit(np.array(idx).reshape(-1, 1), np.array(x).reshape(-1, 1))
        res = model_ols.coef_[0]
    return res


def acf_residuals(x):
    model_ols = LinearRegression()
    idx = np.arange(x.shape[0])
    if (np.isnan(x).sum() > 0) or (x.shape[0] < 2):
        res = np.nan
    else:
        model_ols.fit(idx.reshape(-1, 1), np.array(x).reshape(-1, 1))
        resid = model_ols.predict(idx.reshape(-1, 1)) - np.array(x).reshape(-1, 1)
        acf_resid = acf(resid.flatten(), fft=False)[1:]
        # thresholds =
        # num_thresh = np.abs(acf_resid) > (np.range(acf_resid.shape[0])+1)
        res = np.abs(acf_resid).mean()

    return res


def rolling_r2(x):
    model_ols = LinearRegression()
    idx = np.arange(x.shape[0])
    if (np.isnan(x).sum() > 0) or (x.shape[0] < 2):
        res = np.nan
    else:
        model_ols.fit(idx.reshape(-1, 1), x.reshape(-1, 1))
        res = model_ols.score(idx.reshape(-1, 1), x.reshape(-1, 1))

    return res


def half_rolling_slope(x):
    model_ols = LinearRegression()
    idx = np.arange(max(x.shape[0] // 2, 1))
    if (np.isnan(x).sum() > 0) or (x.shape[0] < 2):
        res = [np.nan]
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
        res = np.nan
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
        if l != 0:
            series.append(s.shift(l))
            names.append(f"{note}_lag_{l}")
    df = pd.DataFrame(dict(zip(names, series)))
    return df


def maxmindiff(x):
    x = np.array(x)
    res = np.max(x) - np.min(x)
    return res


def dist_between_maxmin(x):
    x = np.array(x)
    res = (np.argmax(x) - np.argmin(x))
    return res


def dist_between_maxmin_norm(x):
    x = np.array(x)
    if x.shape[0] > 2:
        res = (np.argmax(x) - np.argmin(x)) / x.shape[0]
    else:
        res = -99
    return res


def apply_rolling_functions(df, col='GR', window=10,
                            func=None, center=True):
    template = make_parabolic(w=window)
    if func is None:
        func = {'max': np.max, 'min': np.min, 'maxmindiff': maxmindiff, 'dist_between_maxmin': dist_between_maxmin,
                'mean': np.mean, 'std': np.std, 'diff_sum': diff_sum, 'abs_diff_sum': abs_diff_sum,
                'corr_parabolic': lambda x: corr_with_parabolic(x, template),
                'slope': rolling_slope, 'mid_vs_end': value_middle_vs_ends, 'max_diff': diff_max, 'min_diff': diff_min,
                'kurtosis': kurtosis, 'abs_diff_exceed_count': abs_diff_exceed_count, 'acf': calculate_acf,
                'acf_resid': acf_residuals}
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
    if func is None:
        func = {'mean': np.mean, 'std': np.std, 'diff_sum': diff_sum, 'abs_diff_sum': abs_diff_sum,
                'dist_between_maxmin': dist_between_maxmin, 'dist_between_maxmin_norm': dist_between_maxmin_norm,
                'slope': lambda x: rolling_slope(x)[0], 'mid_vs_end': value_middle_vs_ends, 'max_diff': diff_max,
                'min_diff': diff_min,
                'half_slope': lambda x: half_rolling_slope(x)[0],
                'corr_parabolic': lambda x: corr_with_parabolic_grp(x),
                'size': lambda x: x.shape[0], 'kurtosis': kurtosis, 'abs_diff_exceed_count': abs_diff_exceed_count,
                'acf': calculate_acf}
    names = []
    for k, v in func.items():
        # series = df.groupby(groups)[col].transform(v)
        colname = f'{col}_{k}_{groups}'
        # df.loc[:, colname] = series.values
        grouped_stats = df.groupby(groups)[col].apply(v)
        grouped_stats_prev = df.groupby(groups)[col].apply(v).shift(1)
        grouped_stats_next = df.groupby(groups)[col].apply(v).shift(-1)
        grouped_stats_diffp = grouped_stats - grouped_stats_prev
        grouped_stats_diffn = grouped_stats - grouped_stats_next

        df = df.join(grouped_stats, on=groups, rsuffix=f'_{k}_{groups}')
        df = df.join(grouped_stats_diffp, on=groups, rsuffix=f'_{k}_{groups}_dp')
        df = df.join(grouped_stats_diffn, on=groups, rsuffix=f'_{k}_{groups}_dn')

        names.append(colname)
        names.append(f'{col}_{k}_{groups}_dp')
        names.append(f'{col}_{k}_{groups}_dn')

    df[f'{col}_slopediff_{groups}'] = df[f'{col}_slope_{groups}'] - df[f'{col}_half_slope_{groups}']
    df.index = np.arange(df.shape[0])
    return df[names]


def preprocess_a_well(df_well, windows=[15, 30, 65]):
    df_well['GR_medfilt'] = medfilt(df_well['GR'], 5)
    df_well['GR_medfilt_s'] = medfilt(df_well['GR'], 3)
    df_well['GR_diff_medfilt'] = df_well['GR_medfilt'].diff()
    df_well['GR_diff_scale'] = df_well['GR_medfilt'] - df_well['GR_medfilt_s']
    df_well['GR_diff_scale_large'] = medfilt(df_well['GR'], 3) - medfilt(df_well['GR_medfilt'], 31)
    df_well['GR_shifted'] = df_well['GR_medfilt'].shift(100)
    df_well['resid'] = medfilt(df_well['GR'], 11) - df_well['GR_medfilt']
    df_well['resid_s'] = df_well['GR'] - df_well['GR_medfilt']

    # Calculate derivatives:
    df_well = calculate_derivatives(df_well, order=2, cols=['GR_medfilt', 'GR_medfilt_s', 'GR'])
    df_well = calculate_derivatives(df_well, order=3, cols=['GR_medfilt', 'GR_medfilt_s', 'GR'])

    block, counts = divide_block(medfilt(df_well['GR'], 11))
    df_well['grp'] = (df_well[f'label_nn_{windows[0]}'] != df_well[f'label_nn_{windows[0]}'].shift(1)).cumsum().fillna(
        'pad')
    df_well['block'] = block
    df_well['counts'] = counts
    # Divide by quick changes in GR:
    df_well = divide_block_steps(df_well, 20)

    # Add lag variables:
    df_lags = calculate_lags(s=df_well['GR_medfilt'], note='GR_medfilt', lags=np.arange(-50, 50, 5))

    label_lags = []
    for w in windows:
        df_label_lags = calculate_lags(s=df_well[f'label_nn_{w}'], note=f'label_nn_{w}', lags=np.arange(-50, 50, 5))
        label_lags.append(df_label_lags.copy())
    df_feats_wxs = apply_rolling_functions(df_well.copy(), window=5, col='GR_medfilt')
    df_feats_ws = apply_rolling_functions(df_well.copy(), window=10, col='GR_medfilt')
    df_feats_wm = apply_rolling_functions(df_well.copy(), window=20, col='GR_medfilt')
    df_feats_wl = apply_rolling_functions(df_well.copy(), window=50, col='GR_medfilt')

    df_feats_wxs_resid = apply_rolling_functions(df_well.copy(), window=5, col='resid')
    df_feats_ws_resid = apply_rolling_functions(df_well.copy(), window=10, col='resid')
    df_feats_wl_resid = apply_rolling_functions(df_well.copy(), window=50, col='resid')
    df_feats_grouped_small_scale_resid = apply_grouped_functions(df_well.copy(), col='resid', groups='grp')
    df_feats_grouped_GR_changes_resid = apply_grouped_functions(df_well.copy(), col='resid', groups='count_groups')

    resid_list = [df_feats_wxs_resid, df_feats_ws_resid, df_feats_wl_resid, df_feats_grouped_small_scale_resid,
                  df_feats_grouped_GR_changes_resid]

    df_feats_wxl = apply_rolling_functions(df_well.copy(), window=120, col='GR_medfilt')
    df_feats_wl_right = apply_rolling_functions(df_well.copy(), window=100, col='GR_medfilt', center=False)
    # df_feats_wl_right_shift = apply_rolling_functions(df_well.copy(), window=100, col='GR_shifted', center=False)
    df_feats_grouped = apply_grouped_functions(df_well.copy(), col='GR_medfilt', groups='grp')
    # Smaller scale features:
    df_feats_small_scale_wxs = apply_rolling_functions(df_well.copy(), window=5, col='GR_medfilt_s')
    df_feats_small_scale_ws = apply_rolling_functions(df_well.copy(), window=11, col='GR_medfilt_s')
    df_feats_small_scale_wm = apply_rolling_functions(df_well.copy(), window=21, col='GR_medfilt_s')
    df_feats_grouped_small_scale = apply_grouped_functions(df_well.copy(), col='GR_medfilt_s', groups='grp')
    df_feats_grouped_GR_changes = apply_grouped_functions(df_well.copy(), col='GR', groups='count_groups')

    # df_feats = pd.concat([df_well,df_lags,df_feats_ws, df_feats_wm, df_feats_wl,df_feats_wl_right,df_feats_wxl], axis=1)
    df_feats = pd.concat(
        [df_well, df_lags, df_feats_wxs, df_feats_ws, df_feats_wm, df_feats_wl, df_feats_wl_right, df_feats_wxl,
         df_feats_grouped, df_feats_small_scale_wm, df_feats_small_scale_ws, df_feats_grouped_small_scale,
         df_feats_small_scale_wxs, df_feats_grouped_GR_changes] + resid_list + label_lags, axis=1)

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
        df_new = preprocess_a_well(df_well.copy(), windows=[15])
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

    # Scale the data:
    # df_train['GR'] = (df_train['GR'] - df_train['GR'].mean()) / df_train['GR'].std()

    n_wells = 200
    nn_dict = {}
    if not (os.path.exists(os.path.join(output_filepath, 'NN.pck'))):
        NN_l, labels_l = get_a_NN_object(df_train, n_wells, n_wells_end=400, window_length=65)
        NN_m, labels_m = get_a_NN_object(df_train, 400, 800, window_length=30, kernel_size=11)
        NN_s, labels_s = get_a_NN_object(df_train, 800, 1200, window_length=15, kernel_size=3)
        nn_dict['NN_l'] = NN_l
        nn_dict['NN_m'] = NN_m
        nn_dict['NN_s'] = NN_s
        nn_dict['label_l'] = labels_l
        nn_dict['label_m'] = labels_m
        nn_dict['label_s'] = labels_s
        logger.info('saving pickle NN')
        with open(os.path.join(output_filepath, 'NN.pck'), 'wb') as f:
            pickle.dump(nn_dict, f)
    else:
        logger.info('reading NN pickle...')
        with open(os.path.exists(os.path.join(output_filepath, 'NN.pck')), 'rb') as f:
            nn_dict = pickle.load(f)
        NN_l = nn_dict['NN_l']
        NN_m = nn_dict['NN_m']
        NN_s = nn_dict['NN_s']
        labels_l = nn_dict['label_l']
        labels_m = nn_dict['label_m']
        labels_s = nn_dict['label_s']

    df_train_processed = preprocess_dataset_parallel(df_train, n_wells=n_wells, NN_list=[NN_l, NN_m, NN_s],
                                                     labels_list=[labels_l, labels_m, labels_s])
    # df_train_processed = preprocess_dataset(df_train, n_wells=n_wells, NN_list=[NN_s],
    #                                                  labels_list=[labels_s])

    df_train_processed.to_pickle(fname_final_train)

    fname_final_test = os.path.join(output_filepath, 'test.pck')

    # df_test_processed = preprocess_dataset_parallel(df_test, n_wells=n_test, NN=NN, labels=labels)
    # df_test_processed.to_pickle(fname_final_test)


if __name__ == '__main__':
    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    input_filepath = os.path.join(project_dir, 'data', 'raw')
    output_filepath = os.path.join(project_dir, 'data', 'processed')
    os.makedirs(output_filepath, exist_ok=True)
    main(input_filepath, output_filepath)
