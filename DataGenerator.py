# import
import concurrent
import random
from scipy.signal import resample
from sklearn.neighbors import KNeighborsClassifier
from tqdm import trange, tqdm

from config import *
from Decompose.SBD import *


def to_ohe(x, n_values=5):
    y = np.eye(n_values)[x]
    return y


def stretch_well(s, y, scale=1.1):
    knn = KNeighborsClassifier(n_neighbors=3, weights='uniform')

    if scale >= 1:
        n_new = scale * s.shape[0]
        s_new = resample(s, int(n_new))
        y_new = resample(y, int(n_new))
        confident_samples = np.ceil(y_new) == np.round(y_new)
        # Get KNN on confident samples
        x_axis = np.arange(s_new.shape[0])
        X = x_axis[confident_samples].reshape(-1, 1)
        y = np.abs(np.ceil(y_new[confident_samples]))
        knn.fit(X, y)
        y_new = knn.predict(x_axis.reshape(-1, 1))
        result_x = s_new
        result_y = y_new
        result_y[result_y > 4] = 4
        result_y[result_y < 0] = 0
    else:
        raise ValueError
    return result_x, result_y


def get_stretch_well(x, scale=2):
    X_train, y_train = x[0],x[1]

    if len(X_train.shape) == 2:
        X_train = np.expand_dims(X_train, axis=0)
        y_train = np.expand_dims(y_train,axis=0)
    n_wells = X_train.shape[0]
    new_length = scale * X_train.shape[1]
    y_new = np.zeros((y_train.shape[0], new_length, y_train.shape[2]))
    X_new = np.zeros((X_train.shape[0], new_length, X_train.shape[2]))

    for i in range(n_wells):
        for j in range(X_train.shape[2]):
            s = X_train[i, :, j]
            labels_well = np.argmax(y_train[i, :, :], axis=1)
            s_new, label = stretch_well(s, labels_well, scale=scale)

            X_new[i, :, j] = s_new
            y_new[i, :, :] = to_ohe(label.astype(int))

    return X_new, y_new


def get_stretch_well_parallel(X_train, y_train, scale=2):
    n_wells = X_train.shape[0]
    print(f'Dealing with {n_wells}')
    new_length = scale * X_train.shape[1]
    print(f'New length {new_length}')

    y_new = np.zeros((y_train.shape[0], new_length, y_train.shape[2]))
    X_new = np.zeros((X_train.shape[0], new_length, X_train.shape[2]))

    list_wells = [(X_train[i, :, :],y_train[i,:,:]) for i in range(X_train.shape[0])]

    with concurrent.futures.ProcessPoolExecutor(max_workers=10) as executor:
        results = list(tqdm(executor.map(get_stretch_well, list_wells), total=len(list_wells)))
    X_new = np.concatenate([r[0] for r in results],axis=0)
    y_new = np.concatenate([r[1] for r in results], axis=0)

    return X_new, y_new


class DataGenerator:

    def __init__(self,
                 data_path=DATA_PATH,
                 test_name=TEST_NAME,
                 train_name=TRAIN_NAME,
                 input_size=INPUT_SIZE,
                 target=TARGET,
                 scale=SCALE,
                 ):

        self.input_size = input_size
        self.target = target


        self.X_train, self.y_train, self.X_test = self.load_data(data_path,test_name,train_name)

        #self.X_train,self.y_train = self.squeeze_stretch(self.X_train,self.y_train)

        #self.X_test, dummy = self.squeeze_stretch(self.X_test, self.y_train[self.X_test.shape[0],:,:])

        """ 
        #self.X_train, self.y_train = get_stretch_well_parallel(X_train=self.X_train, y_train=self.y_train, scale=scale)
        #self.X_test, dummy = get_stretch_well_parallel(X_train=self.X_test,
                                                       y_train=self.y_train[0:self.X_test.shape[0], :, :],
                                                       scale=scale)
        """


        # apply subband decomposition
        SBD_arr = SBD(self.X_train)
        self.X_train = np.concatenate((self.X_train,SBD_arr),axis=2)

        diff_sig = np.diff(self.X_train, axis=1)
        diff_sig1 = np.zeros((diff_sig.shape[0], diff_sig.shape[1] + 1, diff_sig.shape[2]))
        diff_sig1[:, :-1, :] = diff_sig
        self.X_train = np.concatenate((self.X_train, diff_sig1), axis=2)


        # apply subband decomposition
        SBD_arr = SBD(self.X_test)
        self.X_test = np.concatenate((self.X_test, SBD_arr), axis=2)

        diff_sig = np.diff(self.X_test, axis=1)
        diff_sig1 = np.zeros((diff_sig.shape[0], diff_sig.shape[1] + 1, diff_sig.shape[2]))
        diff_sig1[:, :-1, :] = diff_sig
        self.X_test = np.concatenate((self.X_test, diff_sig1), axis=2)

        del SBD_arr,diff_sig1
        gc.collect()

    def load_data(self, data_path, test_name, train_name):

        # load test and train
        df_test = pd.read_csv(data_path + test_name, index_col=None, header=0)
        self.df_test = df_test

        df_test['label'] = np.nan

        df_train = pd.read_csv(data_path + train_name, index_col=None, header=0)

        df_train, y_train = self.preprocessing_initial(df_train.drop('row_id', axis=1))
        df_test, y_test = self.preprocessing_initial(df_test.drop('row_id', axis=1))

        return df_train, y_train, df_test


    def get_train_val(self, train_ind, val_ind):

        # get trian samples
        X_train = self.X_train[train_ind, :, :]
        y_train = self.y_train[train_ind, :, :]



        # get validation samples
        X_val = self.X_train[val_ind, :, :]
        y_val = self.y_train[val_ind, :, :]

        return X_train, y_train, X_val, y_val


    def preprocessing_initial(self, df):

        scaler = StandardScaler()

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

        #for i in range(X.shape[0]):
        #    X[0,:,:] = medfilt(X[0,:,:],3)

        return X, y

    def rescale_X_to_maxmin(self, X, note='note'):
        for i in range(X.shape[0]):
            top = np.quantile(X[i, :], 0.715)
            bottom = 0.33 + 0.4 * top
            new_row = (X[i, :] - bottom) / (top - bottom) - 0.5
            X[i, :] = new_row
        return X



