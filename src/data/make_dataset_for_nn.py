# -*- coding: utf-8 -*-
import concurrent.futures
import logging
import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import medfilt, resample
from sklearn.model_selection import GroupKFold
from tqdm import tqdm
from sklearn.neighbors import KNeighborsClassifier
from scipy.signal import resample
from sklearn.neighbors import KNeighborsClassifier


def squeeze_stretch(s, y, scale=1.1):
    n_old = s.shape[0]
    knn = KNeighborsClassifier(n_neighbors=3, weights='uniform')
    if scale >= 1:
        n_new = scale * s.shape[0]
        s_new = resample(s, int(n_new))
        y_new = resample(y, int(n_new))
        mid_point = int(n_new) // 2
        confident_samples = np.ceil(y_new) == np.round(y_new)
        # Get KNN on confident samples
        x_axis = np.arange(s_new.shape[0])
        X = x_axis[confident_samples].reshape(-1, 1)
        y = np.abs(np.ceil(y_new[confident_samples]))
        print(y.shape)
        knn.fit(X, y)
        y_new = knn.predict(x_axis.reshape(-1, 1))
        result_x = s_new[(mid_point - 550):(mid_point + 550)]
        result_y = y_new[(mid_point - 550):(mid_point + 550)]
        result_y[result_y > 4] = 4
    else:
        n_new = scale * s.shape[0]
        s_new = resample(s, int(n_new))
        y_new = resample(y, int(n_new))
        x_axis = np.arange(s_new.shape[0])

        confident_samples = np.ceil(y_new) == np.round(y_new)
        X_knn = x_axis[confident_samples].reshape(-1, 1)
        y_knn = np.abs(np.ceil(y_new[confident_samples]))
        knn.fit(X_knn, y_knn)
        y_new = knn.predict(x_axis.reshape(-1, 1))

        pad_width = int(n_old - n_new)
        if pad_width % 2 == 0:
            lp = rp = pad_width // 2
        else:
            lp = pad_width // 2
            rp = lp + 1
        s_new = np.pad(s_new, (lp, rp), mode='constant')
        y_new = np.pad(y_new, (lp, rp), mode='constant')
        low = np.quantile(s[y < 1], 0.15)
        high = np.quantile(s[y < 1], 0.85)

        rand_num = np.random.uniform(low, high, lp + rp)
        s_new[:lp] = rand_num[:lp]
        s_new[-rp:] = rand_num[lp:]
        y_new[:lp] = 0
        y_new[-rp:] = 0
        result_x = s_new
        result_y = np.round(np.abs(y_new))
        result_y[result_y > 4] = 4
    return result_x, result_y


def fliplabel(x):
    if x == 3:
        return 4
    if x == 4:
        return 3
    return x


def preprocess_a_well_test(df_well):
    df_well['GR_medfilt'] = medfilt(df_well['GR'], 1)
    x = df_well['GR_medfilt'].values
    return x


def preprocess_a_well_flip(df_well):
    df_well['gr_flipped'] = df_well['GR'].values[::-1]
    df_well['label_flipped'] = df_well['label'].values[::-1]
    # df_well['label_flipped'] = df_well['label_flipped'].apply(lambda x: fliplabel(x))

    x = df_well['gr_flipped'].values
    y = df_well['label_flipped'].values

    return x, y


def to_ohe(x):
    n_values = np.max(x) + 1
    y = np.eye(n_values)[x]
    return y


def preprocess_a_well(df_well):
    df_well['GR_medfilt'] = medfilt(df_well['GR'], 1)
    x = df_well['GR_medfilt'].values
    y = df_well['label'].values

    return x, y


def preprocess_a_well_fake(df_well):
    x = df_well['GR'].values
    y = df_well['label'].values

    return x, y


def create_normalized_gr(df_train):
    x0 = df_train.groupby('well_id')['GR'].quantile(0.715).to_frame()
    x2 = x0.copy()
    x2['GR'] = 0.33 + 0.4 * x2['GR']

    df_train_new = df_train.copy()
    # df_train_new = df_train.copy().join(x0.reset_index(), rsuffix='_0', on='well_id').drop(columns='well_id_0')
    # df_train_new = df_train_new.join(x2.reset_index(), rsuffix='_2', on='well_id').drop(columns='well_id_2')
    df_train_new = pd.merge(df_train_new, x0.reset_index(), on='well_id',
                            suffixes=('', '_0'), how='left')
    df_train_new = pd.merge(df_train_new, x2.reset_index(), on='well_id',
                            suffixes=('', '_2'), how='left')

    #        new_row = (X[i, :] - bottom) / (top - bottom) - 0.5
    df_train_new['GR_raw'] = df_train_new['GR']
    df_train_new['GR_leveled'] = (df_train_new['GR'] - df_train_new['GR_2']) / (
                df_train_new['GR_0'] - df_train_new['GR_2']) - 0.5
    df_train_new['GR'] =df_train_new['GR_leveled']
    #df_train_new.to_pickle('/home/geoanton/Repos/gamma_log_classification/data/processed/df_normalized_script.pck')
    #logging.info('saved the normalized DF')
    return df_train_new


def create_new_wells_from_normalized(df, n_wells=20, n_points=1100):
    df_train_new = df.copy()
    id_start = np.random.randint(0, df_train_new.shape[0] - n_points - 1, size=n_wells)
    df_slice_list = []
    dropped_count = 0
    for i in range(n_wells):
        gr = df_train_new['GR_leveled'].values[id_start[i]:(id_start[i] + n_points)]
        label = df_train_new['label'].values[id_start[i]:(id_start[i] + n_points)]
        row_id = np.arange(n_points)
        well_id = i * np.ones_like(row_id)
        df_slice = pd.DataFrame({'GR': gr, 'row_id': row_id, 'well_id': well_id, 'label': label})
        if df_slice['GR'].isna().sum() == 0:
            df_slice_list.append(df_slice)
        else:
            dropped_count += 1
    df_slices = pd.concat(df_slice_list, axis=0)
    df_slices.index = np.arange(df_slices.shape[0])
    print(f'Dropped {dropped_count} wells due to NaN in GR')
    return df_slices


def rescale_X_to_maxmin(X, note='note'):
    logging.info(f'Performing a rescale on {note}..')
    for i in range(X.shape[0]):
        top = np.quantile(X[i, :], 0.715)
        bottom = 0.33 + 0.4 * top
        new_row = (X[i, :] - bottom) / (top - bottom) - 0.5
        X[i, :] = new_row
    return X


def preprocess_dataset_parallel(df, n_wells=50, n_wells_sliced=5000, df_normalized=None):
    df_sliced = create_new_wells_from_normalized(df, n_wells=n_wells_sliced)

    # df_sliced.to_pickle(os.path.join(project_dir, 'data', 'processed','sliced_well.pck'))
    # print('Saved sliced!')
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

    with concurrent.futures.ProcessPoolExecutor(max_workers=10) as executor:
        results_fake_flipped = list(
            tqdm(executor.map(preprocess_a_well_flip, list_df_wells_fakes), total=len(list_df_wells_fakes)))

    X = np.array([r[0] for r in results])
    y = np.array([r[1] for r in results])
    y = to_ohe(y)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))

    X_flipped = np.array([r[0] for r in results_flipped])
    y_flipped = np.array([r[1] for r in results_flipped])
    y_flipped = to_ohe(y_flipped)
    X_flipped = np.reshape(X_flipped, (X_flipped.shape[0], X_flipped.shape[1], 1))

    # Preprocess Fake Wells:

    X_fake = np.array([r[0] for r in results_fake])
    y_fake = np.array([r[1] for r in results_fake])
    y_fake = to_ohe(y_fake)
    X_fake = np.reshape(X_fake, (X_fake.shape[0], X_fake.shape[1], 1))
    idx_fake_nonna = np.logical_not(np.isnan(X_fake).any(axis=1))[:, 0]
    X_fake = X_fake[idx_fake_nonna, :]
    y_fake = y_fake[idx_fake_nonna, :]

    X_fake_flipped = np.array([r[0] for r in results_fake_flipped])
    y_fake_flipped = np.array([r[1] for r in results_fake_flipped])
    y_fake_flipped = to_ohe(y_fake_flipped)
    X_fake_flipped = np.reshape(X_fake_flipped, (X_fake_flipped.shape[0], X_fake_flipped.shape[1], 1))
    idx_fake_nonna_flipped = np.logical_not(np.isnan(X_fake_flipped).any(axis=1))[:, 0]
    X_fake_flipped = X_fake_flipped[idx_fake_nonna_flipped, :]
    y_fake_flipped = y_fake_flipped[idx_fake_nonna_flipped, :]

    # Tie regular data & fake data:
    X_with_fake = np.concatenate((X, X_fake))
    y_with_fake = np.concatenate((y, y_fake))
    np.random.seed(123)
    inds_with_fake = np.arange(X_with_fake.shape[0])
    np.random.shuffle(inds_with_fake)
    X_with_fake = X_with_fake[inds_with_fake, :, :]
    y_with_fake = y_with_fake[inds_with_fake, :, :]

    # Tie regular and flipped fake data:
    X_with_fake_flipped = np.concatenate((X_flipped, X_fake_flipped))
    y_with_fake_flipped = np.concatenate((y_flipped, y_fake_flipped))
    np.random.seed(123)
    inds_with_fake_flipped = np.arange(X_with_fake_flipped.shape[0])
    np.random.shuffle(inds_with_fake_flipped)
    X_with_fake_flipped = X_with_fake_flipped[inds_with_fake_flipped, :, :]
    y_with_fake_flipped = y_with_fake_flipped[inds_with_fake_flipped, :, :]

    print(f'Shape with Fakes: {X_with_fake.shape}')

    # TODO Added rescaling
    # Rescaling
    #X_with_fake = rescale_X_to_maxmin(X_with_fake, note='with_Fake')
    #X_fake = rescale_X_to_maxmin(X_fake, note='fake_Only')
    #X_with_fake_flipped = rescale_X_to_maxmin(X_with_fake_flipped, note='with_Fake flipped')
    #X_fake_flipped = rescale_X_to_maxmin(X_fake_flipped, note='fake_Only flipped')

    # Regular data
    data_dict = {'X_with_fake': X_with_fake,
                 'y_with_fake': y_with_fake,
                 'X_fake_only': X_fake,
                 'y_fake_only': y_fake}
    # Fliplr
    data_dict_fliplr = {'X_with_fake': X_with_fake_flipped,
                        'y_with_fake': y_with_fake_flipped,
                        'X_fake_only': X_fake_flipped,
                        'y_fake_only': y_fake_flipped}
    # Flipud
    data_dict_flipud = {'X_with_fake': X_with_fake * (-1),
                        'y_with_fake': y_with_fake,
                        'X_fake': X_fake * (-1),
                        'y_fake': y_fake}

    return data_dict, data_dict_fliplr, data_dict_flipud


def preprocess_dataset_test(df_test):
    wells = df_test['well_id'].sort_values().unique().tolist()
    list_df_wells = [df_test.loc[df_test['well_id'].isin([w]), :].copy() for w in wells]
    for df in list_df_wells:
        df.index = np.arange(df.shape[0])

    with concurrent.futures.ProcessPoolExecutor(max_workers=10) as executor:
        results = list(tqdm(executor.map(preprocess_a_well_test, list_df_wells), total=len(list_df_wells)))

    X = np.array([r for r in results])
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))

    # TODO Added rescaling
    X = rescale_X_to_maxmin(X, note='TEST')
    data_dict = {'X': X, 'df_test': df_test}

    return data_dict


# TODO Make sure this spits out a rescaled dataset
def preprocess_dataset_validation(df):
    wells = df['well_id'].unique().tolist()

    # originals
    list_df_wells = [df.loc[df['well_id'].isin([w]), :].copy() for w in wells]
    for df in list_df_wells:
        df.index = np.arange(df.shape[0])

    with concurrent.futures.ProcessPoolExecutor(max_workers=10) as executor:
        results = list(tqdm(executor.map(preprocess_a_well, list_df_wells), total=len(list_df_wells)))

    X = np.array([r[0] for r in results])
    y = np.array([r[1] for r in results])
    y = to_ohe(y)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))

    #X = rescale_X_to_maxmin(X, note='VALIDATION')
    data_dict = {'X': X, 'y': y}
    return data_dict


def main(input_filepath, output_filepath):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')
    df_train = pd.read_csv(os.path.join(input_filepath, 'train_lofi_rowid_Nov13.csv'))
    df_test = pd.read_csv(os.path.join(input_filepath, 'test_lofi_rowid_Nov13.csv'))

    fname_final_test = os.path.join(output_filepath, 'test_nn.pck')

    n_test = df_test['well_id'].unique().shape[0]
    n_train = df_train['well_id'].unique().shape[0]
    logger.info(f'Wells in train total = {n_train}')
    cv = GroupKFold(n_splits=5)
    # make the normalized DF once:
    df_normalized = create_normalized_gr(df_train)
    # bring the normalized track into the original
    df_train['GR_leveled'] = df_normalized['GR_leveled']
    df_train['GR_raw'] = df_train['GR']
    df_train['GR'] = df_train['GR_leveled']

    for k, (train_index, test_index) in enumerate(cv.split(df_train, df_train['label'], df_train['well_id'])):
        logger.info(f'Prepping fold {k}')

        df_train_fold = df_train.iloc[train_index, :].copy()
        df_train_fold.index = np.arange(df_train_fold.shape[0])

        df_val_fold = df_train.iloc[test_index, :].copy()
        df_val_fold.index = np.arange(df_val_fold.shape[0])

        n_wells_train = df_train_fold['well_id'].unique().shape[0]
        data_dict_train, data_dict_train_lr, data_dict_train_ud = preprocess_dataset_parallel(df_train_fold,
                                                                                              n_wells=n_wells_train,
                                                                                              n_wells_sliced=4500)
        data_dict_val = preprocess_dataset_validation(df_val_fold)

        for data_type, data_dict in zip(['regular', 'lr', 'ud'],
                                        [data_dict_train, data_dict_train_lr, data_dict_train_ud]):
            fname_final_train = os.path.join(output_filepath, f'{data_type}_train_nn_{k}.pck')
            fold_dict = {}

            logger.info(f'Saving {data_type} in {fname_final_train}')
            fold_dict[f'data_dict_train_{k}'] = data_dict
            fold_dict[f'data_dict_test_{k}'] = data_dict_val
            with open(fname_final_train, 'wb') as f:
                pickle.dump(fold_dict, f)

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
