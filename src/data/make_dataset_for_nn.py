# -*- coding: utf-8 -*-
import concurrent.futures
import logging
import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import medfilt
from tqdm import tqdm


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

def rescale_full_well_chain_to_new_df(df,n_wells=20,n_points=1100):
    df_train=df.copy()
    y0 = df_train[df_train['label'] == 0].groupby('well_id')['GR'].mean().to_frame()
    y2 = df_train[df_train['label'] == 2].groupby('well_id')['GR'].mean().to_frame()

    df_train_new = df_train.copy().join(y0.reset_index(), rsuffix='_0', on='well_id').drop(columns='well_id_0')
    df_train_new = df_train_new.join(y2.reset_index(), rsuffix='_2', on='well_id').drop(columns='well_id_2')
    z0, z2 = 110, 50
    scale = (z0 - z2) / (df_train_new['GR_0'] - df_train_new['GR_2'])
    df_train_new['GR'] = (df_train_new['GR'] - df_train_new['GR_2']) * scale + z2
    id_start = np.random.randint(0,df_train.shape[0]-n_points-1,size=n_wells)
    df_slice_list = []
    for i in range(n_wells):
        gr = df_train_new['GR'].values[id_start[i]:(id_start[i]+n_points)]
        label = df_train_new['label'].values[id_start[i]:(id_start[i]+n_points)]
        row_id = np.arange(n_points)
        well_id= i*np.ones_like(row_id)
        df_slice = pd.DataFrame({'GR':gr,'row_id':row_id,'well_id':well_id,'label':label})
        df_slice_list.append(df_slice)
    df_slices = pd.concat(df_slice_list,axis=0)
    df_slices.index = np.arange(df_slices.shape[0])
    return df_slices



def preprocess_a_well_fake(df_well):
    y0 = np.mean(df_well[df_well['label'] == 0]['GR'])
    y2 = np.mean(df_well[df_well['label'] == 2]['GR'])
    z0, z2 = np.random.uniform(low=103, high=140, size=1), np.random.uniform(low=40, high=57, size=1)
    scale = (z0 - z2) / (y0 - y2)
    df_well['GR_leveled'] = (df_well['GR'] - y2) * scale + z2
    x = df_well['GR_leveled'].values
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


def preprocess_dataset_parallel(df, n_wells=50,n_wells_sliced = 5000):
    df_sliced = rescale_full_well_chain_to_new_df(df,n_wells=n_wells_sliced)

    wells = df['well_id'].unique().tolist()[:n_wells]
    wells_sliced = df_sliced['well_id'].unique().tolist()

    # originals
    list_df_wells = [df.loc[df['well_id'].isin([w]), :].copy() for w in wells]
    for df in list_df_wells:
        df.index = np.arange(df.shape[0])
    # Fakes
    list_df_wells_fakes = [df_sliced.loc[df_sliced['well_id'].isin([w]), :].copy() for w in wells_sliced]
    print(len(list_df_wells_fakes))
    for df in list_df_wells_fakes:
        df.index = np.arange(df.shape[0])

    with concurrent.futures.ProcessPoolExecutor(max_workers=10) as executor:
        results = list(tqdm(executor.map(preprocess_a_well, list_df_wells), total=len(list_df_wells)))

    with concurrent.futures.ProcessPoolExecutor(max_workers=10) as executor:
        results_flipped = list(tqdm(executor.map(preprocess_a_well_flip, list_df_wells), total=len(list_df_wells)))

    with concurrent.futures.ProcessPoolExecutor(max_workers=10) as executor:
        results_fake = list(
            tqdm(executor.map(preprocess_a_well_fake, list_df_wells_fakes), total=len(list_df_wells_fakes)))

    X = np.array([r[0] for r in results])
    y = np.array([r[1] for r in results])
    y = to_ohe(y)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))

    X_flipped = np.array([r[0] for r in results_flipped])
    y_flipped = np.array([r[1] for r in results_flipped])
    y_flipped = to_ohe(y_flipped)
    X_flipped = np.reshape(X_flipped, (X_flipped.shape[0], X_flipped.shape[1], 1))

    X_fake = np.array([r[0] for r in results_fake])
    y_fake = np.array([r[1] for r in results_fake])
    y_fake = to_ohe(y_fake)
    X_fake = np.reshape(X_fake, (X_fake.shape[0], X_fake.shape[1], 1))

    X_all = np.concatenate((X, X_flipped))
    y_all = np.concatenate((y, y_flipped))

    X_with_fake = np.concatenate((X, X_flipped, X_fake))
    y_with_fake = np.concatenate((y, y_flipped, y_fake))

    np.random.seed(123)
    inds = np.arange(X_all.shape[0])
    np.random.shuffle(inds)
    X_all = X_all[inds, :, :]
    y_all = y_all[inds, :, :]

    inds_with_fake = np.arange(X_with_fake.shape[0])
    np.random.shuffle(inds_with_fake)
    X_with_fake = X_with_fake[inds_with_fake, :, :]
    y_with_fake = y_with_fake[inds_with_fake, :, :]

    print(X_all.shape)
    print(y_all.shape)
    print(f'Shape with Fakes: {X_with_fake.shape}')

    data_dict = {'X': X_all, 'y': y_all, 'X_small': X, 'y_small': y, 'X_with_fake': X_with_fake,
                 'y_with_fake': y_with_fake}

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
    data_dict = preprocess_dataset_parallel(df_train, n_wells=4000,n_wells_sliced=8000)
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
