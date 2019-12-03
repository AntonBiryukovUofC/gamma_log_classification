import pandas as pd
import logging
import os
import pickle
from pathlib import Path

from keras.callbacks import Callback, ModelCheckpoint
from keras.engine.saving import load_model
from keras.optimizers import *
from keras_contrib.callbacks import CyclicLR
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import KFold
import tensorflow as tf
import altair as alt
# gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9)
#
# tf.Session(config=tf.ConfigProto(gpu_options=gpu_options,allow_soft_placement=True)
from src.models.models import create_unet, create_fcn_like, create_fcn__multiple_heads

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)

project_dir = Path(__file__).resolve().parents[2]

log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
logging.basicConfig(level=logging.INFO, format=log_fmt)

exclude_cols = ['row_id']
tgt = 'label'


def fliplabel(pred):
    tmp = pred.copy()
    pred[:, :, 3] = tmp[:, :, 4]
    pred[:, :, 4] = tmp[:, :, 3]
    return pred


def tta_predict(model, X_holdout):
    pred = model.predict(np.fliplr(X_holdout))
    pred = np.fliplr(pred)
    pred = fliplabel(pred)

    return pred

def predict_with_mode(model, X_holdout, mode):
    if mode == 'regular':
        pred = model.predict(X_holdout)
    if mode == 'lr':
        pred = model.predict(np.fliplr(X_holdout))
        pred = np.fliplr(pred)
    if mode == 'ud':
        pred = model.predict(X_holdout*(-1))
        tmp_3 =pred[:,:,3].copy()
        tmp_0 = pred[:, :, 0].copy()

        pred[:,:,3] = pred[:,:,4]
        pred[:, :, 4] = tmp_3
        pred[:,:,0] = pred[:,:,2]
        pred[:, :, 2] = tmp_0


    return pred


def prepare_test(pred_test, df_test):
    wells = df_test['well_id'].sort_values().unique().tolist()
    list_df_wells = [df_test.loc[df_test['well_id'].isin([w]), :].copy() for w in wells]
    for df in list_df_wells:
        df.index = np.arange(df.shape[0])

    for i, df_well in enumerate(list_df_wells):
        df_well['label'] = np.argmax(pred_test[i, :], axis=1)[2:(pred_test.shape[1] - 2)]

    result = pd.concat(list_df_wells, axis=0)
    return result


def main(input_file_path, output_file_path, n_splits=5):
    input_file_name_test = os.path.join(input_file_path, "test_nn.pck")

    with open(input_file_name_test, 'rb') as f:
        results_test = pickle.load(f)

    X_test = results_test['X']
    print(results_test['df_test'].shape)
    X_test = np.pad(X_test, pad_width=((0, 0), (2, 2), (0, 0)), mode='edge')

    fold_models = {0:{'lr':f'/home/anton/tmp_unets/fold0/unet-lr.hdf5',
                   'regular':f'/home/anton/tmp_unets/fold0/unet-regular.hdf5'},
                   1:{'lr':f'/home/anton/tmp_unets/fold1/unet-lr.hdf5',
                   'regular':f'/home/anton/tmp_unets/fold1/unet-regular.hdf5'},
                   2:{'lr':f'/home/anton/tmp_unets/fold2/unet-lr.hdf5',
                   'regular':f'/home/anton/tmp_unets/fold2/unet-regular.hdf5'}}
    preds_test_all = np.zeros((X_test.shape[0],X_test.shape[1],5))
    for k in fold_models.keys():
        #model = create_unet((X_test.shape[1], 1), init_power=5, kernel_size=5, dropout=0.1)
        #model.compile(loss='categorical_crossentropy', optimizer=SGD(lr=0.04),
                      #metrics=['acc', 'categorical_crossentropy'])
        #model.load_weights(fold_models[k])
        for mode in fold_models[k].keys():
            logging.info(f'Loading a model {fold_models[k][mode]}')

            model = load_model(fold_models[k][mode])
            pred_test = predict_with_mode(model,X_test,mode)
            preds_test_all+=pred_test
            del model


    df_test = prepare_test(preds_test_all, results_test['df_test'])
    return df_test, X_test, pred_test


if __name__ == "__main__":
    # not used in this stub but often useful for finding various files

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    input_file_path = os.path.join(project_dir, "data", "processed")
    output_file_path = os.path.join(project_dir, "models")
    os.makedirs(input_file_path, exist_ok=True)
    os.makedirs(output_file_path, exist_ok=True)

    df_test, X_test, pred_test = main(input_file_path, output_file_path)

    df_test[['row_id', 'well_id', 'label']].to_csv(os.path.join(input_file_path, 'submit_nn.csv'), index=False)
    # df_test.to_csv(os.path.join(input_file_path, 'submit_nn.csv'), index=False)
    with open(os.path.join(project_dir, "data", "interim", "test_dict"), 'wb') as f:
        pickle.dump((X_test, pred_test), f)
    # df = preds_wsc
