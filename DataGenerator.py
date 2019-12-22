# import
import concurrent
import gc

from scipy.signal import resample, hilbert
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
import pandas as pd
from Decompose.SBD import *
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold

def rescale_one_row(s):
    new_row = MinMaxScaler(feature_range=(-0.5, 0.5)).fit_transform(s.reshape(-1, 1))

    return new_row


def to_ohe(x, n_values=5):
    y = np.eye(n_values)[x]
    return y


def add_linear_drift_shift(x, slope, labels=None, h=0.25, l=-0.25):
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


def add_diff_leaky_feature(X, orders):
    print('adding leaky features !')
    X_leak = np.reshape(X[:, :, 1], (X.shape[0], X.shape[1], 1))
    diffs = []
    for ord in orders:
        print(f'Processing {ord} diff of leaky feature')
        tmp = X_leak.copy()
        for i in range(ord):
            tmp = np.diff(tmp, axis=1)
        tmp = np.pad(tmp, pad_width=((0, 0), (ord, 0), (0, 0)))
        diffs.append(tmp)
    diffs_array = np.concatenate(diffs, axis=2)
    res = np.concatenate((X, diffs_array), axis=2)
    return res


class DataGenerator:

    def __init__(self,
                 data_path=DATA_PATH,
                 test_name=TEST_NAME,
                 train_name=TRAIN_NAME,
                 input_size=INPUT_SIZE,
                 target=TARGET,
                 dataset_mode='normal',
                 add_trend=False,
                 orders=None,
                 use_diffs_leaky=False,
                 use_neighbor = False
                 ):

        if orders is None:
            orders = [2,3,4,5]
        self.add_trend = add_trend
        self.use_neighbor = use_neighbor
        self.input_size = input_size
        self.target = target
        self.dataset_mode = dataset_mode
        self.X_train, self.y_train, self.X_test = self.load_data(data_path, test_name,
                                                                 train_name)

        # self.X_train,self.y_train = self.squeeze_stretch(self.X_train,self.y_train)

        # self.X_test, dummy = self.squeeze_stretch(self.X_test, self.y_train[self.X_test.shape[0],:,:])

        # apply subband decomposition
        shape_train = (self.X_train.shape[0], self.X_train.shape[1], 1)
        SBD_arr = SBD(self.X_train[:, :, 0].reshape(shape_train))
        self.X_train = np.concatenate((self.X_train, SBD_arr), axis=2)

        diff_sig = np.diff(self.X_train, axis=1)
        diff_sig1 = np.zeros((diff_sig.shape[0], diff_sig.shape[1] + 1, diff_sig.shape[2]))
        diff_sig1[:, :-1, :] = diff_sig
        self.X_train = np.concatenate((self.X_train, diff_sig1), axis=2)
        if use_diffs_leaky:
            print('Applying diffs for leaky features in Train')
            self.X_train = add_diff_leaky_feature(self.X_train, orders=orders)

        # apply subband decomposition
        shape_test = (self.X_test.shape[0], self.X_test.shape[1], 1)
        SBD_arr = SBD(self.X_test[:, :, 0].reshape(shape_test))
        self.X_test = np.concatenate((self.X_test, SBD_arr), axis=2)

        diff_sig = np.diff(self.X_test, axis=1)
        diff_sig1 = np.zeros((diff_sig.shape[0], diff_sig.shape[1] + 1, diff_sig.shape[2]))
        diff_sig1[:, :-1, :] = diff_sig
        self.X_test = np.concatenate((self.X_test, diff_sig1), axis=2)
        if use_diffs_leaky:
            print('Applying diffs for leaky features in Test')
            self.X_test = add_diff_leaky_feature(self.X_test, orders=orders)

        del SBD_arr, diff_sig1
        gc.collect()


        if self.use_neighbor:
            # add nearest neighbours
            dummy = np.zeros((self.X_test.shape[0], self.X_test.shape[1], self.y_train.shape[2]))
            test_pairs, y_test_pairs = self.nearest_test(self.X_train[:, :, 0:1], self.y_train, self.X_test[:, :, 0:1],
                                                         dummy)

            self.X_test = np.concatenate((self.X_test, test_pairs - self.X_test[:, :, 0:1]), axis=2)
            self.X_test = np.concatenate((self.X_test, test_pairs), axis=2)
            self.X_test = np.concatenate((self.X_test, y_test_pairs), axis=2)

            del dummy, test_pairs, y_test_pairs
            gc.collect()



    def nearest_train(self, X, y):

        pairs = np.zeros((X.shape[0], X.shape[1], 1))
        y_pairs = np.zeros((y.shape[0], y.shape[1], 5))

        # kfold cross-validation
        kf = KFold(4, shuffle=True, random_state=42)
        for fold, (train_ind, val_ind) in enumerate(kf.split(X)):
            print('Processing of the fold number: ', fold)
            # get train and validations sessions
            X_train = X[train_ind, :, :]
            y_train = y[train_ind, :, :]

            X_val = X[val_ind, :, :]
            y_val = y[val_ind, :, :]

            pairs[val_ind, :, :], y_pairs[val_ind, :, :] = self.find_neighbours(X_train, y_train, X_val, y_val)

        # return pairs,y_pairs
        return pairs, y_pairs

    def nearest_test(self, X_train, y_train, X_test, y_test):

        print('Processing of the test')
        # return pairs,y_pairs
        return self.find_neighbours(X_train, y_train, X_test, y_test)

    def find_neighbours(self, X_train, y_train, X_val, y_val):

        pairs = np.zeros((X_val.shape[0], X_val.shape[1], 1))
        y_pairs = np.zeros((y_val.shape[0], y_val.shape[1], 5))

        for i in tqdm(range(X_val.shape[0])):
            corr_vector = np.zeros((X_train.shape[0]))
            for j in range(X_train.shape[0]):
                corr_vector[j] = self.find_corr(X_val[i, :, 0], X_train[j, :, 0])

            pairs[i, :, :] = X_train[np.argmax(corr_vector), :, :]
            y_pairs[i, :, :] = y_train[np.argmax(corr_vector), :, :]

        return pairs, y_pairs

    def find_corr(self, X1, X2):

        # corr = np.convolve(X1, X2, mode='same')
        corr = np.correlate(X1, X2)

        return np.max(corr)




    def load_data(self, data_path, test_name, train_name):

        # load test and train
        df_test = pd.read_csv(data_path + test_name, index_col=None, header=0)
        self.df_test = df_test

        df_test['label'] = np.nan

        df_train = pd.read_csv(data_path + train_name, index_col=None, header=0)

        df_train, y_train = self.preprocessing_initial(df_train.drop('row_id', axis=1), note='Train')
        df_test, y_test = self.preprocessing_initial(df_test.drop('row_id', axis=1), note='Test')
        return df_train, y_train, df_test

    # return df_train, y_train, df_test, df_train_xstart, y_train_xstart

    def augment_xstarter_data(self, X):

        for i in range(X.shape[0]):
            X[i, :, 0] = X[i, :, 0] + 0.06 * np.random.normal(size=(X.shape[1]))

        return X

    def get_train_val(self, train_ind, val_ind):

        # get trian samples
        X_train = self.X_train[train_ind, :, :]
        y_train = self.y_train[train_ind, :, :]

        # get validation samples
        X_val = self.X_train[val_ind, :, :]
        y_val = self.y_train[val_ind, :, :]


        if self.use_neighbor:
            pairs, y_pairs = self.nearest_train(X_train[:, :, 0:1], y_train)
            X_train = np.concatenate((X_train, pairs - X_train[:, :, 0:1]), axis=2)
            X_train = np.concatenate((X_train, pairs), axis=2)
            X_train = np.concatenate((X_train, y_pairs), axis=2)

            pairs, y_pairs = self.nearest_test(X_train[:, :, 0:1], y_train, X_val[:, :, 0:1], y_val)
            X_val = np.concatenate((X_val, pairs - X_val[:, :, 0:1]), axis=2)
            X_val = np.concatenate((X_val, pairs), axis=2)
            X_val = np.concatenate((X_val, y_pairs), axis=2)

        return X_train, y_train, X_val, y_val

    def rescale_X_to_maxmin(self, X, note='note'):
        print(f'Performing a rescale on {note}..')
        list_wells = [X[i, :, 0] for i in range(X.shape[0])]
        X_new = X.copy()

        with concurrent.futures.ProcessPoolExecutor(max_workers=1) as executor:
            results = list(tqdm(executor.map(rescale_one_row, list_wells), total=len(list_wells)))

        for i in range(X.shape[0]):
            X_new[i, :, 0] = results[i].reshape(1, -1)

        return X_new

    def preprocessing_initial(self, df, note='MinMax', add_trend=False):

        train_wells = df['well_id'].unique().tolist()

        # data
        X = np.zeros((
            len(train_wells),
            1104,
            2
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
            GR_leak = GR_temp * 100
            GR_leak = GR_leak - np.floor(GR_leak)
            GR_leak = np.round(GR_leak, 11)

            if add_trend:
                GR_temp = add_linear_drift_shift(GR_temp, slope[i], labels=temp, h=35, l=-35)
            GR_temp = np.reshape(GR_temp, [GR_temp.shape[0], 1])
            GR_leak = np.reshape(GR_leak, [GR_leak.shape[0], 1])

            X[i, :1100, 0] = GR_temp[:, 0]
            X[i, :1100, 1] = GR_leak[:, 0]

            X[i, 1100:, 0] = X[i, 1096:1100, 0]
            X[i, 1100:, 1] = X[i, 1096:1100, 1]

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

        X = self.rescale_X_to_maxmin(X, note=note)
        if self.dataset_mode == 'lr':
            X = np.fliplr(X)
            y = np.fliplr(y)
        if self.dataset_mode == 'ud':
            X = X * (-1)

        # X = envelope_scaling(X)

        # for i in range(X.shape[0]):
        #    X[0,:,:] = medfilt(X[0,:,:],3)

        return X, y
