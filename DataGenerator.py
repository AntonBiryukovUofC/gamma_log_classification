# import
from numpy import random
from scipy.signal import resample
from sklearn.neighbors import KNeighborsClassifier
from tqdm import trange

from config import *
from Decompose.SBD import *


def to_ohe(x):
    n_values = np.max(x) + 1
    y = np.eye(n_values)[x]
    return y


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
        print(confident_samples.sum())
        X_knn = x_axis[confident_samples].reshape(-1, 1)
        y_knn = np.abs(np.ceil(y_new[confident_samples]))
        print(y.shape)
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


def get_stretch_scaled(X_train, y_train, seed=123, low=0.75, high=1.1):
    n_wells = X_train.shape[0]
    y_new = np.zeros_like(y_train)
    X_new = np.zeros_like(X_train)
    for i in trange(n_wells):
        rng = random.Random(seed + i)
        scale = rng.uniform(low, high)
        for j in X_train.shape[2]:
            s = X_train[i, :, j]
            labels_well = np.argmax(y_train[i, :, :], axis=1)
            s_new, label = squeeze_stretch(s, labels_well, scale=scale)
            X_new[i, :, j] = s_new
            y_new[i, :, :] = to_ohe(label)

    X_all = np.concatenate(X_train,X_new,axis=0)
    y_all = np.concatenate(y_train, y_new, axis=0)
    return X_all,y_all


class DataGenerator:

    def __init__(self,
                 data_path=DATA_PATH,
                 test_name=TEST_NAME,
                 train_name=TRAIN_NAME,
                 input_size=INPUT_SIZE,
                 target=TARGET,
                 ):

        self.input_size = input_size
        self.target = target

        self.X_train, self.y_train, self.X_test = self.load_data(data_path, test_name, train_name)

        print(data_path)
        self.X_train, self.y_train = get_stretch_scaled(self.X_train,self.y_train)

        # apply subband decomposition
        self.X_train = SBD(self.X_train)
        self.X_test = SBD(self.X_test)

    def load_data(self, data_path, test_name, train_name):

        # load test and train
        df_test = pd.read_csv(data_path + test_name, index_col=0, header=0)
        self.df_test = df_test

        df_test['label'] = np.nan

        df_train = pd.read_csv(data_path + train_name, index_col=0, header=0)

        df_train, y_train = self.preprocessing_initial(df_train)
        df_test, y_test = self.preprocessing_initial(df_test)

        return df_train, y_train, df_test

    def get_train_val(self, train_ind, val_ind):

        # get trian samples
        X_train = self.X_train[train_ind, :, :]
        y_train = self.y_train[train_ind, :, :]
        # Augmentation
        X_train = get_stretch_scaled(X_train, y_train)

        # get validation samples
        X_val = self.X_train[val_ind, :, :]
        y_val = self.y_train[val_ind, :, :]

        return X_train, y_train, X_val, y_val

    def preprocessing_initial(self, df):

        scaler = MinMaxScaler()

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

        for i in range(len(train_wells)):

            GR_temp = GR[i * 1100:(i + 1) * 1100]
            GR_temp = np.reshape(GR_temp, [GR_temp.shape[0], 1])
            # GR_temp = scaler.fit_transform(GR_temp)

            X[i, :1100, 0] = GR_temp[:, 0]
            X[i, 1100:, 0] = X[i, 1096:1100, 0]

            temp = label[i * 1100:(i + 1) * 1100]

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

        X = self.rescale_X_to_maxmin(X)

        return X, y

    def rescale_X_to_maxmin(self, X, note='note'):
        for i in range(X.shape[0]):
            top = np.quantile(X[i, :], 0.715)
            bottom = 0.33 + 0.4 * top
            new_row = (X[i, :] - bottom) / (top - bottom) - 0.5
            X[i, :] = new_row
        return X
