#import
from config import *
from Decompose.SBD import *


class DataGenerator:

    def __init__(self,
                 data_path=DATA_PATH,
                 test_name=TEST_NAME,
                 train_name=TRAIN_NAME,
                 input_size = INPUT_SIZE,
                 target = TARGET,
                 ):


        self.input_size = input_size
        self.target = target

        self.X_train, self.y_train, self.X_test = self.load_data(data_path,test_name,train_name)

        print(data_path)



        # apply subband decomposition

        SBD_arr = SBD(self.X_train)
        self.X_train = np.concatenate((self.X_train,SBD_arr),axis=2)
        SBD_arr = SBD(self.X_test)
        self.X_test = np.concatenate((self.X_test, SBD_arr), axis=2)

        del SBD_arr
        gc.collect()



    def load_data(self,data_path,test_name,train_name):


        # load test and train
        df_test = pd.read_csv(data_path + test_name,index_col=None, header=0)
        self.df_test = df_test

        df_test['label'] = np.nan

        df_train = pd.read_csv(data_path + train_name,index_col=None, header=0)

        df_train,y_train = self.preprocessing_initial(df_train.drop('row_id',axis=1))
        df_test,y_test = self.preprocessing_initial(df_test.drop('row_id',axis=1))

        return df_train, y_train, df_test


    def get_train_val(self,train_ind,val_ind):


        #get trian samples
        X_train = self.X_train[train_ind,:,:]
        y_train = self.y_train[train_ind,:,:]



        # get validation samples
        X_val = self.X_train[val_ind,:,:]
        y_val = self.y_train[val_ind,:,:]

        return X_train, y_train, X_val, y_val


    def preprocessing_initial(self,df):

        scaler = MinMaxScaler()

        train_wells = df['well_id'].unique().tolist()



        #data
        X = np.zeros((
            len(train_wells),
            1104,
            1
        ))




        #labels
        y = np.zeros((
            len(train_wells) ,
            1104,
            5
        ))



        GR = df['GR'].values
        label = df['label'].values

        for i in range(len(train_wells)):

            GR_temp = GR[i*1100:(i+1)*1100]
            GR_temp = np.reshape(GR_temp,[GR_temp.shape[0],1])
            #GR_temp = scaler.fit_transform(GR_temp)

            X[i,:1100,0] = GR_temp[:,0]
            X[i, 1100:, 0] = X[i, 1096:1100, 0]

            temp = label[i*1100:(i+1)*1100]



            for j in range(1100):
                if temp[j] == 0:
                    y[i,j,0] = 1
                if temp[j] == 1:
                    y[i,j,1] = 1
                if temp[j] == 2:
                    y[i,j,2] = 1
                if temp[j] == 3:
                    y[i,j,3] = 1
                if temp[j] == 4:
                    y[i,j,4] = 1

                y[i, 1100:, :] = y[i, 1096:1100, :]

        X = self.rescale_X_to_maxmin(X)

        return X, y

    def rescale_X_to_maxmin(self,X, note='note'):
        for i in range(X.shape[0]):
            top = np.quantile(X[i, :], 0.715)
            bottom = 0.33 + 0.4 * top
            new_row = (X[i, :] - bottom) / (top - bottom) - 0.5
            X[i, :] = new_row
        return X





