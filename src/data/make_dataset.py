# -*- coding: utf-8 -*-
import os
import pandas as pd
import click
import logging
from pathlib import Path
import numpy as np
from sklearn.linear_model import LinearRegression
import concurrent.futures
from tqdm import tqdm
from scipy.signal import medfilt


def diff_sum(x):
    res = np.diff(x).sum()
    return res


def abs_diff_sum(x):
    res = np.abs(np.diff(x)).sum()
    return res


def rolling_slope(x):
    model_ols = LinearRegression()
    idx = np.arange(x.shape[0])
    model_ols.fit(idx.reshape(-1, 1), x.reshape(-1, 1))
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
    res = np.correlate(x, y).max()
    return res


def value_middle_vs_ends(x):
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
                            func=None):
    template = make_parabolic(w=window)
    if func is None:
        func = {'mean': np.mean, 'std': np.std, 'diff_sum': diff_sum, 'abs_diff_sum': abs_diff_sum,
                'slope': rolling_slope, 'mid_vs_end': value_middle_vs_ends}
    names = []
    for k, v in func.items():
        series = df.loc[:, col].rolling(window=window, center=True, min_periods=1).apply(v, raw=True)
        colname = f'{k}_{window}_{col}'
        df.loc[:, colname] = series.values
        names.append(colname)
    df.index = np.arange(df.shape[0])
    return df[names]


def preprocess_a_well(df_well):
    df_well['GR_medfilt'] = medfilt(df_well['GR'], 21)
    # Add lag variables:
    df_lags = calculate_lags(s=df_well['GR_medfilt'], note='GR_medfilt', lags=np.arange(-50, 50, 5))

    df_feats_ws = apply_rolling_functions(df_well.copy(), window=10, col='GR_medfilt')
    df_feats_wm = apply_rolling_functions(df_well.copy(), window=20, col='GR_medfilt')
    df_feats_wl = apply_rolling_functions(df_well.copy(), window=50, col='GR_medfilt')
    df_feats_wxl = apply_rolling_functions(df_well.copy(), window=120, col='GR_medfilt')
    df_feats = pd.concat([df_well,df_lags,df_feats_ws, df_feats_wm, df_feats_wl, df_feats_wxl], axis=1)
    return df_feats


def preprocess_dataset(df, n_wells=50):
    df_well_list = []
    wells = df['well_id'].unique().tolist()[:n_wells]
    for w in tqdm(wells):
        df_well = df[df['well_id'] == w]
        df_new = preprocess_a_well(df_well.copy())
        df_well_list.append(df_new.copy())

    df_preprocessed = pd.concat(df_well_list, axis=0)
    df_preprocessed.index = np.arange(df_preprocessed.shape[0])
    return df_preprocessed


def preprocess_dataset_parallel(df, n_wells=50):
    wells = df['well_id'].unique().tolist()[:n_wells]
    list_df_wells = [df.loc[df['well_id'].isin([w]), :].copy() for w in wells]
    for df in list_df_wells:
        df.index=np.arange(df.shape[0])

    with concurrent.futures.ProcessPoolExecutor(max_workers=12) as executor:
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

    df_train_processed = preprocess_dataset_parallel(df_train, n_wells=5)
    df_train_processed.to_pickle(fname_final_train)

    #  fname_final_test = os.path.join(output_filepath, 'test.pck')

    # df_test_processed = preprocess_dataset_parallel(df_test, n_wells=n_test)
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
