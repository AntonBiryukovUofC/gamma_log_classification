# import
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import KFold

from config import *
from DataGenerator import *
import pickle


def prepare_test(pred_test, df_test):
    wells = df_test['well_id'].sort_values().unique().tolist()
    list_df_wells = [df_test.loc[df_test['well_id'].isin([w]), :].copy() for w in wells]

    for df in list_df_wells:
        df.index = np.arange(df.shape[0])

    for i, df_well in enumerate(list_df_wells):
        df_well['label'] = np.argmax(pred_test[i, :], axis=1)

    result = pd.concat(list_df_wells, axis=0)
    return result


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
        predictions_test = np.zeros((self.GetData.X_test.shape[0], 1104, 5))

        for fold, (train_ind, val_ind) in enumerate(kf.split(self.GetData.X_train)):
            print(f'Doing fold {fold}')
            weights_loc = weights_location_list[fold]

            X_train, y_train, X_val, y_val = self.GetData.get_train_val(train_ind, val_ind)
            # self.model = self.model_func(input_size=INPUT_SIZE, hyperparams=HYPERPARAM)
            self.model = load_model(weights_loc)

            pred_val = self.model.predict(X_val)
            pred_val_dict[fold] = pred_val.copy()
            pred_val_dict[f'{fold}_y_val'] = y_val.copy()

            predictions_test += self.model.predict(self.GetData.X_test) / 5

        predictions_test = predictions_test[:, :1100:, :]

        return pred_val_dict, predictions_test


start_fold = 0
test = pd.read_csv('./data/raw/test_cax.csv')
path = "./data/weights/"
weights_location_list = [path+"LSTM_model_0_97159.h5",
                         path+"LSTM_model_1_97207.h5",
                         path+"LSTM_model_2_97138.h5",
                         path+"LSTM_model_3_97309.h5",
                         path+"LSTM_model_4_97075.h5"]
CV = Pipeline(DL_model, start_fold)
pred, pred_test = CV.validation(weights_location_list)

location_test_ = "./data/test_lstm_preds.pck"
with open(location_test_, 'wb') as f:
    pickle.dump(pred_test, f)

file_pickle = './data/LSTM_OOF.pcl'
with open(file_pickle, 'wb') as f:
    pickle.dump(pred, f)

submit = prepare_test(pred_test, test)
submit[['unique_id', 'label']].to_csv('./data/result/LSTM_submit.csv', index=False)
