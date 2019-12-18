import concurrent

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm

from config import TRAIN_NAME, TEST_NAME, DATA_PATH


def rescale_one_row(s):
    new_row = MinMaxScaler(feature_range=(-0.5, 0.5)).fit_transform(s)

    return new_row


def rescale_X_to_maxmin( X, note='note'):
    print(f'Performing a rescale on {note}..')
    list_wells = [X[i, :] for i in range(X.shape[0])]
    X_new = X.copy()

    with concurrent.futures.ProcessPoolExecutor(max_workers=1) as executor:
        results = list(executor.map(rescale_one_row, list_wells))

    for i in range(X.shape[0]):
        X_new[i, :, 0] = results[i].reshape(1, -1)

    return X_new


def add_linear_drift_shift(x, slope, labels=None, h=0.33, l=-0.33):
    delta = np.arange(x.shape[0]) * slope
    y = x + delta
    if labels is not None:
        y_series = pd.Series(labels)
        groups = (y_series != y_series.shift(1)).cumsum().fillna('pad')
        groups.index = y_series.index
        groups[y_series == 0] = -20
        rand_grp_shift = np.random.uniform(l, h, groups.unique().shape[0])
        map_grp = dict(zip(groups.unique(), rand_grp_shift))
        shifts = np.array([map_grp[x] for x in groups])
        y = y + shifts

    return y


def preprocessing_initial(df, note='MinMax', add_trend=False):
    train_wells = df['well_id'].unique().tolist()

    # data
    X = np.zeros((
        len(train_wells),
        1104,
        1
    ))

    # labels
    y = np.zeros((
        len(train_wells),
        1104,
        5
    ))

    GR = df['GR'].values
    label = df['label'].values
    np.random.seed(42)
    slope = np.random.uniform(low=5e-3, high=5e-2, size=len(train_wells))
    if add_trend:
        print(f'Adding trends to {note}')
    for i in range(len(train_wells)):

        temp = label[i * 1100:(i + 1) * 1100]

        GR_temp = GR[i * 1100:(i + 1) * 1100]
        if add_trend:
            GR_temp = add_linear_drift_shift(GR_temp, slope[i], labels=temp)
        GR_temp = np.reshape(GR_temp, [GR_temp.shape[0], 1])

        X[i, :1100, 0] = GR_temp[:, 0]
        X[i, 1100:, 0] = X[i, 1096:1100, 0]

        for j in range(1100):
            if temp[j] == 0:
                y[i, j, 0] = 1
            if temp[j] == 1:
                y[i, j, 1] = 1
            if temp[j] == 2:
                y[i, j, 2] = 1
            if temp[j] == 3:
                y[i, j, 3] = 1
            if temp[j] == 4:
                y[i, j, 4] = 1

            y[i, 1100:, :] = y[i, 1096:1100, :]


    return X, y


def augment_xstarter_data( X,seed=42,std=0.06):
    np.random.seed(seed)
    for i in range(X.shape[0]):
        X[i, :, 0] = X[i, :, 0] + std * np.random.normal(size=(X.shape[1]))
    return X


def load_data( data_path=DATA_PATH,
                 test_name=TEST_NAME,
                 train_name=TRAIN_NAME):
    # load test and train
    df_test = pd.read_csv(data_path + test_name, index_col=None, header=0)
    df_train_xstart = pd.read_csv(data_path + 'train.csv', index_col=None, header=0)
    df_train_xstart['well_id'] += 4000
    df_test['label'] = np.nan
    df_train = pd.read_csv(data_path + train_name, index_col=None, header=0)
    df_train, y_train = preprocessing_initial(df_train.drop('row_id', axis=1), note='Train')
    df_train_xstart, y_train_xstart = preprocessing_initial(df_train_xstart.drop('row_id', axis=1),
                                                                 note='Train_xstart', add_trend=False)
    df_train_xstart = augment_xstarter_data(df_train_xstart)
    df_test, y_test = preprocessing_initial(df_test.drop('row_id', axis=1), note='Test')
    return df_train, y_train, df_test, df_train_xstart, y_train_xstart