# import
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from category_encoders import CountEncoder
from config import *
from DataGenerator import *


def build_encoder(X_train):
    all_vals = pd.Series(X_train[:, :, 1].flatten())
    ce = dict(all_vals.value_counts() / (X_train.shape[0] * X_train.shape[1]))
    return ce


def encode(X, encoder):
    def func(x):
        res = encoder.get(x, 0)
        return res

    f = np.vectorize(func)
    freq_encoded = f(X[:, :, 1])
    # rescale it:
    freq_values_unique = list(encoder.values())
    freq_encoded = (freq_encoded - np.min(freq_values_unique)) / (
            np.max(freq_values_unique) - np.min(freq_values_unique)) - 0.5
    freq_encoded = np.reshape(freq_encoded, (freq_encoded.shape[0], freq_encoded.shape[1], 1))
    freq_encoded_diff = np.pad(np.diff(freq_encoded, axis=1), pad_width=((0, 0), (1, 0), (0, 0)))
    X = np.concatenate((X, freq_encoded, freq_encoded_diff), axis=2)

    return X


class Pipeline():

    def __init__(self,

                 GetData,
                 model_func,
                 start_fold,
                 gpu,
                 batch_size,

                 n_fold=N_FOLD,
                 epochs=N_EPOCH,
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

        self.reduce_lr = ReduceLROnPlateau(monitor='val_accuracy', factor=0.3,
                                           patience=int(patience / 5), min_lr=self.lr / 1000, verbose=1, mode='max', )

    def train(self, optimizer=None):
        if optimizer is None:
            optimizer = Adam(self.lr, clipnorm=1.0, clipvalue=0.5)

        # kfold cross-validation
        kf = KFold(self.n_fold, shuffle=True, random_state=42)

        predictions = np.zeros((self.GetData.X_test.shape[0], self.GetData.X_test.shape[1], 5))
        pred_val = np.zeros((self.GetData.X_train.shape[0], self.GetData.X_train.shape[1], 5))

        score = 0
        for fold, (train_ind, val_ind) in enumerate(kf.split(self.GetData.X_train)):

            if fold != self.start_fold:
                continue

            X_train, y_train, X_val, y_val = self.GetData.get_train_val(train_ind, val_ind)
            # Add encoding:
            #print('Encoding...')
            #encoder = build_encoder(X_train)
            #X_val = encode(X_val, encoder)
            #X_train = encode(X_train, encoder)
            #X_test = encode(self.GetData.X_test,encoder)
            #print('Done encoding!')
            checkpointer = ModelCheckpoint(
                f'{self.model_name}_{fold}_{self.gpu}_{self.batch_size}' + '_{epoch:02d}_{val_accuracy:.5f}.h5',
                monitor='val_accuracy',
                mode='max', verbose=1, save_best_only=True)

            self.model = self.model_func(input_size=(X_train.shape[1], X_train.shape[2]),
                                         hyperparams=HYPERPARAM)

            self.model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

            # train model
            history = self.model.fit(X_train, y_train, batch_size=self.batch_size, epochs=self.epochs,
                                     callbacks=[self.earlystopper, checkpointer, self.reduce_lr],
                                     validation_data=(X_val, y_val))

            pred_val[val_ind, :, :] = self.model.predict(X_val)

            pred_val_processed = predictions_postprocess(pred_val[val_ind, :, :])
            y_val = predictions_postprocess(y_val)

            predictions += self.model.predict(X_test) / self.n_fold

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
