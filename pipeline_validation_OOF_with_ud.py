# import
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import KFold

from Pipeline import build_encoder, encode
from config import *
from DataGenerator import *
import pickle

BATCH_SIZE = 48


def prepare_df(pred, df, col="label"):
    wells = df['well_id'].sort_values().unique().tolist()
    list_df_wells = [df.loc[df['well_id'].isin([w]), :].copy() for w in wells]
    for df in list_df_wells:
        df.index = np.arange(df.shape[0])
    for i, df_well in enumerate(list_df_wells):
        df_well[col] = np.argmax(pred[i, :], axis=1)
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

    def validation(self, weights_location_list, freq_encoder=True):
        print('Validation started')
        assert len(weights_location_list) == 15
        # kfold cross-validation
        kf = KFold(self.n_fold, shuffle=True, random_state=42)

        predictions_test = np.zeros((self.GetData.X_test.shape[0], 1104, 5))
        predictions_train = np.zeros((self.GetData.X_train.shape[0], 1104, 5))

        for fold, (train_ind, val_ind) in enumerate(kf.split(self.GetData.X_train)):
            print(f'Doing fold {fold}')

            X_train, y_train, X_val, y_val = self.GetData.get_train_val(train_ind, val_ind)
            if freq_encoder:
                encoder = build_encoder(X_train)
                X_val = encode(X_val, encoder)
                # X_train = encode(X_train, encoder)
                X_test = encode(self.GetData.X_test, encoder)
            else:
                X_test = self.GetData.X_test
            
            k = 3
            for j in range(k):
                weights_loc = weights_location_list[k*fold + j]
                self.model = load_model(weights_loc)

                pred_val = self.model.predict(X_val)
                predictions_train[val_ind] += pred_val.copy() / k
                predictions_test += self.model.predict(X_test) / (5*k)

        predictions_test = predictions_test[:, :1100:, :]
        predictions_train = predictions_train[:, :1100:, :]

        return predictions_train, predictions_test


start_fold = 0
path = "./data/weights/"
add_trend = False
freq_enc = True
use_diffs_leaky = False

weights_location_dict = {'regular': [path + "LSTM_model_0_08_17_0.99564.h5",
                                     path + "LSTM_model_0_12_21_0.99541.h5",
                                     path + "LSTM_model_0_16_27_0.99550.h5",
                                     
                                     path + "LSTM_model_1_12_17_0.99587.h5",
                                     path + "LSTM_model_1_02_28_0.99583.h5",
                                     path + "LSTM_model_1_16_17_0.99562.h5",
                                   
                                     path + "LSTM_model_2_02_16_0.99537.h5",
                                     path + "LSTM_model_2_12_24_0.99536.h5",
                                     path + "LSTM_model_2_16_27_0.99525.h5",
                                     
                                     path + "LSTM_model_3_02_29_0.99527.h5",
                                     path + "LSTM_model_3_12_15_0.99520.h5",
                                     path + "LSTM_model_3_16_21_0.99517.h5",
                                     
                                     path + "LSTM_model_4_08_14_0.99543.h5",
                                     path + "LSTM_model_4_02_18_0.99563.h5",
                                     path + "LSTM_model_4_12_19_0.99547.h5",
                                    ],
                         'ud': [path + "LSTM_model_ud_0_8_18_0.99574.h5",
                                path + "LSTM_model_ud_0_2_21_0.99571.h5",
                                path + "LSTM_model_ud_0_4_16_0.99574.h5",

                                path + "LSTM_model_ud_1_4_23_0.99585.h5",
                                path + "LSTM_model_ud_1_2_17_0.99570.h5",
                                path + "LSTM_model_ud_1_8_19_0.99575.h5",

                                path + "LSTM_model_ud_2_8_25_0.99558.h5",
                                path + "LSTM_model_ud_2_4_26_0.99550.h5",
                                path + "LSTM_model_ud_2_2_18_0.99518.h5",

                                path + "LSTM_model_ud_3_8_25_0.99553.h5",
                                path + "LSTM_model_ud_3_2_16_0.99543.h5",
                                path + "LSTM_model_ud_3_4_30_0.99528.h5",

                                path + "LSTM_model_ud_4_4_22_0.99578.h5",
                                path + "LSTM_model_ud_4_8_23_0.99544.h5",
                                path + "LSTM_model_ud_4_2_20_0.99543.h5",
                               ]}

GetData_ud = DataGenerator(add_trend=add_trend, use_diffs_leaky=use_diffs_leaky, dataset_mode='ud')
GetData_regular = DataGenerator(add_trend=add_trend, use_diffs_leaky=use_diffs_leaky, dataset_mode='regular')

CV = Pipeline(DL_model, start_fold, GetData=GetData_regular)
CV_ud = Pipeline(DL_model, start_fold, GetData=GetData_ud)

pred_train, pred_test = CV.validation(weights_location_dict['regular'], freq_encoder=freq_enc)
pred_train_ud, pred_test_ud = CV_ud.validation(weights_location_dict['ud'], freq_encoder=freq_enc)

pred_test += pred_test_ud
pred_train += pred_train_ud

test = pd.read_csv('./data/raw/test_cax.csv')
submit = prepare_df(pred_test, test)
submit[['unique_id', 'label']].to_csv('./data/result/LSTM_submit.csv', index=False)

train = pd.read_csv('./data/raw/train_cax.csv')
oof = prepare_df(pred_train, train, "pred_label")
oof.to_csv('./data/result/oof.csv', index=False)
