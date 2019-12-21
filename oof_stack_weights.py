import numpy as np
import pandas as pd
import pickle

from sklearn.metrics import accuracy_score

unet_oof = '/home/anton/tmp_unets/OOF/UNET_OOF.pcl'
lstm_oof = '/home/anton/tmp_unets/OOF/LSTM_OOF.pcl'
with open(unet_oof, 'rb') as f:
    unet_oof_dict = pickle.load(f)

with open(lstm_oof, 'rb') as f:
    lstm_oof_dict = pickle.load(f)


def get_score(unet_oof_dict,lstm_oof_dict,w_unet = 0.5,w_lstm = 0.5):

    #print('Scores over folds:')
    oof_unet_list =[]
    oof_lstm_list =[]
    oof_val_list =[]

    for fold in range(5):
        oof_unet_fold = unet_oof_dict[fold]
        oof_lstm_fold = lstm_oof_dict[fold]
        y_val = lstm_oof_dict[f'{fold}_y_val']
        oof_mix = oof_lstm_fold * w_lstm + oof_unet_fold * w_unet
        labels_lstm = np.argmax(oof_lstm_fold, axis=2)
        labels_unet = np.argmax(oof_unet_fold, axis=2)
        labels_yval = np.argmax(y_val, axis=2)
        # print(labels_lstm.flatten().shape)
        # print(labels_unet.flatten().shape)

        score_lstm = accuracy_score(labels_yval.flatten(), labels_lstm.flatten())
        score_unet = accuracy_score(labels_yval.flatten(), labels_unet.flatten())
        oof_unet_list.append(oof_unet_fold.copy())
        oof_lstm_list.append(oof_lstm_fold.copy())
        oof_val_list.append(y_val.copy())

        #print(f'LSTM :{score_lstm} UNET:{score_unet}')

    oof_unet_all = np.concatenate(oof_unet_list, axis=0)
    oof_lstm_all = np.concatenate(oof_lstm_list, axis=0)
    oof_yval_all = np.concatenate(oof_val_list, axis=0)
    oof_mix_all = oof_lstm_all * w_lstm + oof_unet_all * w_unet
    labels_lstm = np.argmax(oof_lstm_all, axis=2)
    labels_unet = np.argmax(oof_unet_all, axis=2)
    labels_yval = np.argmax(oof_yval_all, axis=2)
    labels_mix = np.argmax(oof_mix_all, axis=2)


    score_unet_overall = accuracy_score(labels_yval.flatten(), labels_unet.flatten())
    score_lstm_overall = accuracy_score(labels_yval.flatten(), labels_lstm.flatten())
    score_mix_overall = accuracy_score(labels_yval.flatten(), labels_mix.flatten())

    print(
        f'Overall score LSTM :{score_lstm_overall:.5f} UNET:{score_unet_overall:.5f}, Mix with weights unet*{w_unet:.3f} + lstm*{w_lstm:.3f}: {score_mix_overall:.8f} ')
    return score_mix_overall

for i in np.arange(0,1.05,0.01):
    get_score(unet_oof_dict,lstm_oof_dict,w_unet=i,w_lstm=1-i)
