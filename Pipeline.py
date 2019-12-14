#import
from config import *
from DataGenerator import *

class Pipeline():

    def __init__(self,

                 GetData,
                 model_func,
                 start_fold,
                 gpu,

                 n_fold = N_FOLD,
                 epochs = N_EPOCH,
                 batch_size = BATCH_SIZE,
                 lr = LR,
                 patience = PATIENCE,
                 min_delta = MIN_DELTA,
                 model_name = MODEL_NAME,
                 pic_folder = PIC_FOLDER,

                 stacking_folder = STACKING_FOLDER,
                 debug_folder = DEBUG_FOLDER,
                 submit_fopder = SUBMIT_FOLDER

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
        self.submit_fopder = submit_fopder

        # early stopping
        self.earlystopper = EarlyStopping(monitor='val_accuracy', patience=patience, verbose=1, mode='max', min_delta=min_delta)

        self.reduce_lr = ReduceLROnPlateau(monitor='val_accuracy', factor=0.1,
                                      patience=int(patience/3), min_lr=self.lr/1000, verbose = 1,mode='max',)

    def train(self):

        # kfold cross-validation
        kf = KFold(self.n_fold, shuffle=True, random_state=42)

        predictions = np.zeros((self.GetData.X_test.shape[0],self.GetData.X_test.shape[1],5))
        pred_val = np.zeros((self.GetData.X_train.shape[0],self.GetData.X_train.shape[1],5))

        score = 0
        for fold, (train_ind, val_ind) in enumerate(kf.split(self.GetData.X_train)):


            if fold != self.start_fold:
                continue

            X_train, y_train, X_val, y_val = self.GetData.get_train_val(train_ind, val_ind)

            checkpointer = ModelCheckpoint(self.model_name +'_'+str(fold)+f'_{self.gpu}.h5', monitor='val_accuracy',
                                           mode='max', verbose=1, save_best_only=True)

            self.model = self.model_func(input_size=(self.GetData.X_train.shape[1],self.GetData.X_train.shape[2]) ,hyperparams=HYPERPARAM)

            self.model.compile(optimizer=Adam(self.lr), loss='categorical_crossentropy', metrics=['accuracy'])

            # train model
            history = self.model.fit(X_train,y_train, batch_size=self.batch_size, epochs=self.epochs,
                                callbacks=[self.earlystopper, checkpointer, self.reduce_lr],
                                validation_data=(X_val,y_val))

            pred_val[val_ind,:,:] = self.model.predict(X_val)

            pred_val_processed = predictions_postprocess(pred_val[val_ind,:,:])
            y_val = predictions_postprocess(y_val)

            predictions += self.model.predict(self.GetData.X_test)/self.n_fold


            np.save(self.debug_folder + str(fold) + '_', pred_val)


            score += target_metric(pred_val_processed,y_val)/self.n_fold

            fig = plt.figure()
            accuracy = history.history['accuracy']
            plt.plot(accuracy)
            plt.plot(history.history['val_accuracy'])
            plt.legend(['train_accuracy','val_accuracy'])
            plt.savefig(self.pic_folder + 'accuracy_' + str(fold) + '.png')

            fig = plt.figure()
            plt.plot(history.history['loss'])
            plt.plot(history.history['val_loss'])
            plt.legend(['train_loss', 'val_loss'])
            plt.savefig(self.pic_folder + 'loss_' + str(fold) + '.png')




        predictions = predictions[:, :1100:, :]
        submit = prepare_test(predictions, self.GetData.df_test)
        submit[['row_id', 'well_id', 'label']].to_csv(self.stacking_folder +"_" +str(score)+'_LSTM_.csv', index=False)


        predictions = predictions[:,:1100,:]
        np.save(self.stacking_folder+str(score)+'_.csv',predictions)
        np.save( self.debug_folder+str(score)+'_.csv',pred_val)

        return score

def target_metric(y_pred,y_true):

    y_pred = np.reshape(y_pred,(y_pred.shape[0]*y_pred.shape[1]))
    y_true = np.reshape(y_true, (y_true.shape[0] * y_true.shape[1]))

    return accuracy_score(y_true, y_pred)

def predictions_postprocess(pred):

    final_predicitons = np.zeros((pred.shape[0],pred.shape[1]))

    for i in range(pred.shape[0]):
        for j in range(pred.shape[1]):
            m_val = np.max(pred[i,j, :])
            final_predicitons[i,j] = np.where(pred[i,j, :] == m_val)[0][0]

    return final_predicitons[:,:1100]


def prepare_test(pred_test, df_test):
    wells = df_test['well_id'].sort_values().unique().tolist()
    list_df_wells = [df_test.loc[df_test['well_id'].isin([w]), :].copy() for w in wells]

    for df in list_df_wells:
        df.index = np.arange(df.shape[0])

    for i, df_well in enumerate(list_df_wells):
        df_well['label'] = np.argmax(pred_test[i, :], axis=1)

    result = pd.concat(list_df_wells, axis=0)
    return result