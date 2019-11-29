# -*- coding: utf-8 -*-
import os
import pickle

import pandas as pd
# import click
import logging
from pathlib import Path
import numpy as np
from sklearn.linear_model import LinearRegression
import concurrent.futures
from tqdm import tqdm
from scipy.signal import medfilt


def fliplabel(x):
    if x == 3:
        return 4
    if x == 4:
        return 3
    return x


def preprocess_a_well(df_well):
    df_well['GR_medfilt'] = medfilt(df_well['GR'], 1)
    x = df_well['GR_medfilt'].values
    y = df_well['label'].values

    return x, y


def preprocess_a_well_test(df_well):
    df_well['GR_medfilt'] = medfilt(df_well['GR'], 1)
    x = df_well['GR_medfilt'].values
    return x


def preprocess_a_well_flip(df_well):
    df_well['gr_flipped'] = df_well['GR'].values[::-1]
    df_well['label_flipped'] = df_well['label'].values[::-1]
    df_well['label_flipped'] = df_well['label_flipped'].apply(lambda x: fliplabel(x))

    x = df_well['gr_flipped'].values
    y = df_well['label_flipped'].values

    return x, y


def to_ohe(x):
    n_values = np.max(x) + 1
    y = np.eye(n_values)[x]
    return y


def preprocess_dataset_parallel(df, n_wells=50):
    wells = df['well_id'].unique().tolist()[:n_wells]
    list_df_wells = [df.loc[df['well_id'].isin([w]), :].copy() for w in wells]
    for df in list_df_wells:
        df.index = np.arange(df.shape[0])

    with concurrent.futures.ProcessPoolExecutor(max_workers=10) as executor:
        results = list(tqdm(executor.map(preprocess_a_well, list_df_wells), total=len(list_df_wells)))

    with concurrent.futures.ProcessPoolExecutor(max_workers=10) as executor:
        results_flipped = list(tqdm(executor.map(preprocess_a_well_flip, list_df_wells), total=len(list_df_wells)))

    X = np.array([r[0] for r in results])
    y = np.array([r[1] for r in results])
    y = to_ohe(y)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))

    X_flipped = np.array([r[0] for r in results_flipped])
    y_flipped = np.array([r[1] for r in results_flipped])
    y_flipped = to_ohe(y_flipped)
    X_flipped = np.reshape(X_flipped, (X_flipped.shape[0], X_flipped.shape[1], 1))

    X_all = np.concatenate((X, X_flipped))
    y_all = np.concatenate((y, y_flipped))
    np.random.seed(123)
    inds = np.arange(X_all.shape[0])
    np.random.shuffle(inds)

    X_all = X_all[inds, :, :]
    y_all = y_all[inds, :, :]

    print(X_all.shape)
    print(y_all.shape)
    data_dict = {'X': X_all, 'y': y_all, 'X_small': X, 'y_small': y}

    return data_dict


def preprocess_dataset_test(df_test):
    wells = df_test['well_id'].sort_values().unique().tolist()
    list_df_wells = [df_test.loc[df_test['well_id'].isin([w]), :].copy() for w in wells]
    for df in list_df_wells:
        df.index = np.arange(df.shape[0])

    with concurrent.futures.ProcessPoolExecutor(max_workers=10) as executor:
        results = list(tqdm(executor.map(preprocess_a_well_test, list_df_wells), total=len(list_df_wells)))

    X = np.array([r for r in results])
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))

    data_dict = {'X': X, 'df_test': df_test}

    return data_dict


def main(input_filepath, output_filepath):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')
    df_train = pd.read_csv(os.path.join(input_filepath, 'train_lofi_rowid_Nov13.csv'))
    df_test = pd.read_csv(os.path.join(input_filepath, 'test_lofi_rowid_Nov13.csv'))

    fname_final_train = os.path.join(output_filepath, 'train_nn.pck')
    fname_final_test = os.path.join(output_filepath, 'test_nn.pck')

    n_test = df_test['well_id'].unique().shape[0]
    n_train = df_train['well_id'].unique().shape[0]
    print(n_train)
    data_dict = preprocess_dataset_parallel(df_train, n_wells=4000)
    with open(fname_final_train, 'wb') as f:
        pickle.dump(data_dict, f)

    data_dict_test = preprocess_dataset_test(df_test)
    with open(fname_final_test, 'wb') as f:
        pickle.dump(data_dict_test, f)


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
