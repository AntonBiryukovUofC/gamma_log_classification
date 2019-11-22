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

    df_train_processed = preprocess_dataset_parallel(df_train, n_wells=75)
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
