# -*- coding: utf-8 -*-
import os
import pandas as pd
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
import numpy as np
from sklearn.linear_model import LinearRegression
import tqdm


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


def apply_rolling_functions(df, col='GR', window=10,
                            func=None):
    if func is None:
        func = {'mean': np.mean, 'std': np.std, 'diff_sum': diff_sum, 'abs_diff_sum': abs_diff_sum,
                'slope': rolling_slope}
    names = []
    for k, v in func.items():
        series = df[col].rolling(window=window, center=True, min_periods=1).apply(v, raw=True)
        colname = f'{k}_{window}_{col}'
        df.loc[:, colname] = series.values
        names.append(colname)
    df.index = np.arange(df.shape[0])
    return df[names]


def preprocess_a_well(df_well):
    df_feats_w20 = apply_rolling_functions(df_well, window=20)
    df_feats_w60 = apply_rolling_functions(df_well, window=60)
    df_feats_w150 = apply_rolling_functions(df_well, window=150)
    df_feats = pd.concat([df_well, df_feats_w20, df_feats_w60, df_feats_w150], axis=1)
    return df_feats


def preprocess_dataset(df, n_wells=50):
    df_well_list = []
    wells = df['well_id'].unique().tolist()[:n_wells]
    for w in tqdm.tqdm(wells):
        df_well = df[df['well_id'] == w]
        df_new = preprocess_a_well(df_well.copy())
        df_well_list.append(df_new.copy())

    df_preprocessed = pd.concat(df_well_list, axis=0)
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

    fname_final_train = os.path.join(output_filepath, 'train.csv')
    fname_final_test = os.path.join(output_filepath, 'test.csv')

    n_test = df_test['well_id'].unique().shape[0]

    df_train_processed = preprocess_dataset(df_train, n_wells=50)
    df_test_processed = preprocess_dataset(df_test, n_wells=n_test)

    df_train_processed.to_csv(fname_final_train)
    df_test_processed.to_csv(fname_final_test)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())
    input_filepath = os.path.join(project_dir, 'data','raw')
    output_filepath = os.path.join(project_dir, 'data','processed')
    os.makedirs(output_filepath,exist_ok=True)
    main(input_filepath, output_filepath)
