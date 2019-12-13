# import
from config import *
from DataGenerator import *
import pickle

""" 
class Batch_processing(keras.utils.Sequence):
    'Generates data for Keras'

    def __init__(self, index_list,
                 GetData,
                 train_ind,
                 val_ind,
                 train = True,
                 batch_size=BATCH_SIZE
                 ):

        self.train = train
        self.index_list = list(index_list)
        self.batch_size = batch_size
        self.GetData = GetData

        self.train_ind = train_ind
        self.val_ind = val_ind


    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.index_list) / self.batch_size))



    def __getitem__(self, index):

        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.index_list[index * self.batch_size:(index + 1) * self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.index_list[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def __data_generation(self, list_IDs_temp):

        X_train, y_train, X_val, y_val = self.GetData(train_ind,val_ind)

        if train:
            return X_train, y_train
        else:
            return X_val, y_val
"""


class Pipeline():

    def __init__(self,
                 model_func,
                 start_fold,

                 n_fold=N_FOLD,
                 epochs=N_EPOCH,
                 batch_size=BATCH_SIZE,
                 lr=LR,
                 patience=PATIENCE,
                 min_delta=MIN_DELTA,
                 model_name=MODEL_NAME,
                 GetData=DataGenerator(),
                 pic_folder=PIC_FOLDER,

                 stacking_folder=STACKING_FOLDER,
                 submit_fopder=SUBMIT_FOLDER

                 ):
        # load the model
        self.model_func = model_func
        self.start_fold = start_fold

        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = lr
        self.n_fold = n_fold
        self.model_name = model_name

        self.GetData = GetData

        self.pic_folder = pic_folder
        self.stacking_folder = stacking_folder
        self.submit_fopder = submit_fopder

        # early stopping
        self.earlystopper = EarlyStopping(monitor='val_loss', patience=patience, verbose=1, mode='min',
                                          min_delta=min_delta)

        self.reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1,
                                           patience=int(patience / 3), min_lr=self.lr / 1000, verbose=1)

    def validation(self, weights_location_list):
        print('Validation started')
        assert len(weights_location_list) == 5
        # kfold cross-validation
        kf = KFold(self.n_fold, shuffle=True, random_state=42)

        predictions = np.zeros((self.GetData.X_test.shape[0], self.GetData.X_test.shape[1], 5))
        score = 0
        pred_val_dict = {}
        for fold, (train_ind, val_ind) in enumerate(kf.split(self.GetData.X_train)):
            weights_loc = weights_location_list[fold]

            X_train, y_train, X_val, y_val = self.GetData.get_train_val(train_ind, val_ind)
            self.model = self.model_func(input_size=INPUT_SIZE, hyperparams=HYPERPARAM)
            self.model = load_model(weights_loc)

            pred_val = self.model.predict(X_val)
            pred_val_dict[fold] = pred_val.copy()

        return pred_val_dict


start_fold = 0
weights_location_list = ["/home/anton/Repos/gamma_log_classification_unet/data/weights/UNET_model_0_.h5",
                         "/home/anton/Repos/gamma_log_classification_unet/data/weights/UNET_model_1_.h5",
                         "/home/anton/Repos/gamma_log_classification_unet/data/weights/UNET_model_2_.h5",
                         "/home/anton/Repos/gamma_log_classification_unet/data/weights/UNET_model_3_.h5",
                         "/home/anton/Repos/gamma_log_classification_unet/data/weights/UNET_model_4_.h5"]
CV = Pipeline(DL_model, start_fold)
pred = CV.validation(weights_location_list)
file_pickle = '/home/anton/tmp_unets/OOF/UNET_OOF.pcl'
with open(file_pickle, 'wb') as f:
    pickle.dump(pred, f)
