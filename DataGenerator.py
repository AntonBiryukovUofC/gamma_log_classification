#import
from config import *



class DataGenerator:

    def __init__(self,
                 data_path=DATA_PATH,
                 test_name=TEST_NAME,
                 train_name=TRAIN_NAME,
                 input_size = INPUT_SIZE,
                 target = TARGET,
                 droplist = DROPLIST
                 ):


        self.input_size = input_size
        self.target = target

        self.X_train, self.y_train, self.well_id_train, self.X_test, self.df_test, self.well_id_test = self.load_data(data_path,test_name,train_name)

        print(data_path)


        """ 
        # apply subband decomposition
        self.X_train = SBD(self.X_train)
        self.X_test = SBD(self.X_test)
        """


    def load_data(self,data_path,test_name,train_name):


        # load test and train
        df_test = pd.read_csv(data_path + test_name, index_col=0, header=0)
        df_train = pd.read_csv(data_path + train_name, index_col=0, header=0)


        return self.preprocessing_initial(df_train,df_test)


    def get_train_val(self,train_ind,val_ind):


        #get trian samples
        X_train = self.X_train[train_ind,:,:]
        y_train = self.y_train[train_ind]

        # get validation samples
        X_val = self.X_train[val_ind, :,:]
        y_val = self.y_train[val_ind]




        return X_train, y_train, X_val, y_val


    def preprocessing_initial(self,df_train,df_test):


        test_wells = df_test['well_id'].unique().tolist()
        train_wells = df_train['well_id'].unique().tolist()


        one_well_size = df_train[df_train['well_id']==train_wells[0]].shape[0]-self.input_size+1


        #data
        X_train = np.zeros((
            len(train_wells) * one_well_size,
            self.input_size,
            1
        ))

        X_test = np.zeros((
            len(test_wells) * one_well_size,
            self.input_size,
            1
        ))


        #labels
        y_train = np.zeros((
            len(train_wells) * one_well_size,
            self.input_size,
            1
        ))


        #insex of wells
        well_id_train = np.zeros((X_train.shape[0]))
        well_id_test = np.zeros((X_test.shape[0]))


        #get target column
        target = df_train[self.target].values
        df_train = df_train.drop(self.target,axis=1)

        #get data columns
        df_train = df_train.values
        df_test = df_test.values


        for ind,i in enumerate(train_wells):

            temp = df_train[df_train[:,0] == i]
            temp_target = target[df_train[:,0] == i]

            for j in range(one_well_size):

                X_train[j + ind*one_well_size,:,0] = temp[j:j + self.input_size,1]
                y_train[j + ind*one_well_size,:,0] = temp_target[j:j + self.input_size]
                well_id_train[j + ind*one_well_size] = i


        for ind,i in enumerate(test_wells):

            temp = df_test[df_test[:,0] == i]

            for j in range(one_well_size):

                X_test[j + ind*one_well_size,:,0] = temp[j:j + self.input_size,1]
                well_id_test[j + ind*one_well_size] = i

        return X_train, y_train, well_id_train, X_test, df_test, well_id_test





