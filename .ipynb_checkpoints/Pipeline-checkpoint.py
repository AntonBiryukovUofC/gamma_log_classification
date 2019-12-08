#import
from config import *
from DataGenerator import *

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
                 model,

                 n_fold = N_FOLD,
                 epochs = N_EPOCH,
                 batch_size = BATCH_SIZE,
                 lr = LR,
                 patience = PATIENCE,
                 min_delta = MIN_DELTA,
                 model_name = MODEL_NAME,
                 GetData=DataGenerator()

                 ):

        # load the model
        self.model = model

        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = lr
        self.n_fold = n_fold
        self.model_name = model_name

        self.GetData = GetData



        # early stopping
        self.earlystopper = EarlyStopping(monitor='val_loss', patience=patience, verbose=1, mode='min', min_delta=min_delta)


    def train(self):

        checkpointer = ModelCheckpoint(self.model_name, monitor='val_loss',
                                            mode='min', verbose=1, save_best_only=True)

        # kfold cross-validation
        kf = KFold(self.n_fold, shuffle=True, random_state=42)

        for fold, (train_ind, val_ind) in enumerate(kf.split(self.GetData.X_train)):

            X_train, y_train, X_val, y_val = self.GetData.get_train_val(train_ind, val_ind)


            self.model.compile(optimizer=Adam(self.lr), loss='categorical_crossentropy', metrics=['accuracy'])

            # train model
            results = self.model.fit(X_train,y_train, batch_size=self.batch_size, epochs=self.epochs,
                                callbacks=[self.earlystopper, checkpointer],
                                validation_data=(X_val,y_val))

            print(results)

            """ 
            # plot training curves
            plt.figure(figsize=(10, 5))
            plt.plot(train_loss)
            plt.plot(val_loss)
            plt.legend(['Train loss', 'Test loss'])
            plt.title('Training curves')
            """

        return 0

#def target_metric(y_pred,y_true)