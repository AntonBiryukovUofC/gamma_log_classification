# import
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold

from config import *
from DataGenerator import *
from utils import augment_xstarter_data, rescale_X_to_maxmin, load_data


def batch_generator(X, Xs, y, ys, batch_size=16):
    '''
    Return a random image from X, y
    '''

    while True:
        # choose batch_size random images / labels from the data
        idx_cax = np.random.randint(0, X.shape[0], batch_size // 2)
        idx_xs = np.random.randint(0, Xs.shape[0], batch_size // 2)
        X_cax_batch = X[idx_cax, :, :]
        X_xs_batch = Xs[idx_xs, :, :]
        y_cax_batch = y[idx_cax, :, :]
        y_xs_batch = ys[idx_xs, :, :]
        # Do SBD, diffs, normalizing
        # CAX Data
        X_cax_batch = rescale_X_to_maxmin(X_cax_batch, note='batch_train_CAX')
        SBD_arr_cax = SBD(X_cax_batch)
        X_cax_batch = np.concatenate((X_cax_batch, SBD_arr_cax), axis=2)
        diff_sig = np.diff(X_cax_batch, axis=1)
        diff_sig1 = np.zeros((diff_sig.shape[0], diff_sig.shape[1] + 1, diff_sig.shape[2]))
        diff_sig1[:, :-1, :] = diff_sig
        X_cax_batch = np.concatenate((X_cax_batch, diff_sig1), axis=2)
        # XStart Data
        slope = np.random.uniform(low=5e-3, high=5e-2, size=X_xs_batch.shape[0])
        for i in range(X_xs_batch.shape[0]):
            # Assign a random trend and shift up/down
            X_xs_batch[i, :, 0] = add_linear_drift_shift(X_xs_batch[i, :, 0], slope=slope[i], l=-25, h=25)
        X_xs_batch = rescale_X_to_maxmin(X_xs_batch, note='batch_train_xstarter')
        X_xs_batch = augment_xstarter_data(X_xs_batch, seed=idx_cax[0])
        SBD_arr_xs = SBD(X_xs_batch)
        X_xs_batch = np.concatenate((X_xs_batch, SBD_arr_xs), axis=2)
        diff_sig = np.diff(X_xs_batch, axis=1)
        diff_sig1 = np.zeros((diff_sig.shape[0], diff_sig.shape[1] + 1, diff_sig.shape[2]))
        diff_sig1[:, :-1, :] = diff_sig
        X_xs_batch = np.concatenate((X_xs_batch, diff_sig1), axis=2)

        X_batch = np.concatenate((X_cax_batch, X_xs_batch), axis=0)
        y_batch = np.concatenate((y_cax_batch, y_xs_batch), axis=0)

        yield X_batch, y_batch


class Pipeline():

    def __init__(self,

                 GetData,
                 model_func,
                 start_fold,
                 gpu,

                 n_fold=N_FOLD,
                 epochs=N_EPOCH,
                 batch_size=BATCH_SIZE,
                 lr=LR,
                 patience=PATIENCE,
                 min_delta=MIN_DELTA,
                 model_name=MODEL_NAME,
                 pic_folder=PIC_FOLDER,

                 stacking_folder=STACKING_FOLDER,
                 debug_folder=DEBUG_FOLDER,
                 submit_folder=SUBMIT_FOLDER

                 ):

        # load the model
        self.model_func = model_func
        self.start_fold = start_fold
        self.gpu = gpu

        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = lr
        self.n_fold = n_fold
        self.model_name = model_name

        self.GetData = GetData

        self.pic_folder = pic_folder
        self.stacking_folder = stacking_folder
        self.debug_folder = debug_folder
        self.submit_fopder = submit_folder

        # early stopping
        self.earlystopper = EarlyStopping(monitor='val_accuracy', patience=patience, verbose=1, mode='max',
                                          min_delta=min_delta)

        self.reduce_lr = ReduceLROnPlateau(monitor='val_accuracy', factor=0.1,
                                           patience=int(patience / 3), min_lr=self.lr / 1000, verbose=1, mode='max', )

    def train(self, optimizer=None):
        if optimizer is None:
            optimizer = Adam(self.lr)

        X_train, y_train, X_test, X_train_xstart, y_train_xstart = load_data()

        # kfold cross-validation
        kf = KFold(self.n_fold, shuffle=True, random_state=42)



        predictions = np.zeros((self.GetData.X_test.shape[0], self.GetData.X_test.shape[1], 5))
        pred_val = np.zeros((self.GetData.X_train.shape[0], self.GetData.X_train.shape[1], 5))

        score = 0
        for fold, (train_ind, val_ind) in enumerate(kf.split(self.GetData.X_train)):

            if fold != self.start_fold:
                continue

           # _, _, X_val, y_val = self.GetData.get_train_val(train_ind, val_ind)

            gen_function = batch_generator(X_train,X_train_xstart,y_train,y_train_xstart,batch_size=32)
            checkpointer = ModelCheckpoint(self.model_name + '_' + str(fold) + f'_{self.gpu}.h5',
                                           monitor='val_accuracy',
                                           mode='max', verbose=1, save_best_only=True)

            self.model = self.model_func(input_size=(self.GetData.X_train.shape[1], self.GetData.X_train.shape[2]),
                                         hyperparams=HYPERPARAM)

            self.model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

            # train model
            history = self.model.fit_generator(generator=gen_function,  epochs=self.epochs,steps_per_epoch=7000 // 64,
                                     callbacks=[self.earlystopper, checkpointer,self.reduce_lr])

            pred_val[val_ind, :, :] = self.model.predict(X_val)

            pred_val_processed = predictions_postprocess(pred_val[val_ind, :, :])
            y_val = predictions_postprocess(y_val)

            predictions += self.model.predict(self.GetData.X_test) / self.n_fold

            np.save(self.debug_folder + str(fold) + '_', pred_val)

            score += target_metric(pred_val_processed, y_val) / self.n_fold

            fig = plt.figure()
            accuracy = history.history['accuracy']
            plt.plot(accuracy)
            plt.plot(history.history['val_accuracy'])
            plt.legend(['train_accuracy', 'val_accuracy'])
            plt.savefig(self.pic_folder + 'accuracy_' + str(fold) + '.png')

            fig = plt.figure()
            plt.plot(history.history['loss'])
            plt.plot(history.history['val_loss'])
            plt.legend(['train_loss', 'val_loss'])
            plt.savefig(self.pic_folder + 'loss_' + str(fold) + '.png')

        predictions = predictions[:, :1100:, :]
        submit = prepare_test(predictions, self.GetData.df_test)
        submit[['row_id', 'well_id', 'label']].to_csv(self.stacking_folder + "_" + str(score) + '_LSTM_.csv',
                                                      index=False)

        predictions = predictions[:, :1100, :]
        np.save(self.stacking_folder + str(score) + '_.csv', predictions)
        np.save(self.debug_folder + str(score) + '_.csv', pred_val)

        return score


def target_metric(y_pred, y_true):
    y_pred = np.reshape(y_pred, (y_pred.shape[0] * y_pred.shape[1]))
    y_true = np.reshape(y_true, (y_true.shape[0] * y_true.shape[1]))

    return accuracy_score(y_true, y_pred)


def predictions_postprocess(pred):
    final_predicitons = np.zeros((pred.shape[0], pred.shape[1]))

    for i in range(pred.shape[0]):
        for j in range(pred.shape[1]):
            m_val = np.max(pred[i, j, :])
            final_predicitons[i, j] = np.where(pred[i, j, :] == m_val)[0][0]

    return final_predicitons[:, :1100]


def prepare_test(pred_test, df_test):
    wells = df_test['well_id'].sort_values().unique().tolist()
    list_df_wells = [df_test.loc[df_test['well_id'].isin([w]), :].copy() for w in wells]

    for df in list_df_wells:
        df.index = np.arange(df.shape[0])

    for i, df_well in enumerate(list_df_wells):
        df_well['label'] = np.argmax(pred_test[i, :], axis=1)

    result = pd.concat(list_df_wells, axis=0)
    return result
