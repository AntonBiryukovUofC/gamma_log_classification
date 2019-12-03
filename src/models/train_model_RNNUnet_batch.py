import logging
import os
import pickle
import sys

sys.path.insert(0, './')

import click

from keras.callbacks import Callback, ModelCheckpoint
from keras.optimizers import SGD
from keras_contrib.callbacks import CyclicLR

from keras.backend.tensorflow_backend import set_session
import tensorflow as tf

from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import KFold

from src.models.models import create_fcn__multiple_heads, create_unet, create_unet_bidirectRNN
from pathlib import Path
import numpy as np




project_dir = Path(__file__).resolve().parents[2]

log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
logging.basicConfig(level=logging.INFO, format=log_fmt)

exclude_cols = ['row_id']
tgt = 'label'
input_file_path = os.path.join(project_dir, "data", "processed")
output_file_path = os.path.join(project_dir, "models")
os.makedirs(input_file_path, exist_ok=True)
os.makedirs(output_file_path, exist_ok=True)


def predict_with_mode(model, X_holdout, mode):
    if mode == 'regular':
        pred = model.predict(X_holdout)
    if mode == 'lr':
        pred = model.predict(np.fliplr(X_holdout))
        pred = np.fliplr(pred)
    if mode == 'ud':
        pred = model.predict(X_holdout*(-1))
        pred = pred

    return pred


def transform_holdout(X_holdout, y_holdout, mode):
    if mode == 'regular':
        X = X_holdout
        y = y_holdout
    if mode == 'lr':
        X = np.fliplr(X_holdout)
        y = np.fliplr(y_holdout)
    if mode == 'ud':
        X = X_holdout*(-1)
        y = y_holdout

    return X,y


@click.command()
@click.option('--epochs', default=20, help='number of epochs')
@click.option('--input_file_path', default=input_file_path, help='input file location (of train_nn.pck)')
@click.option('--output_file_path', default=output_file_path, help='output folder (for model weights)')
@click.option('--fold', default=0, help='fold to train')
@click.option('--gpu', default=0, help='gpu to train')
@click.option('--weights', default='', help='weights location')
@click.option('--dropout', default=0.1, help='dropout rate')
@click.option('--batch_size', default=8, help='batch size')
@click.option('--epochs_per_cycle', default=4, help='cycles per epoch')
@click.option('--mode', default='regular', help='mode of training [regular,lr,ud]')
@click.option('--kernel_size', default=5, help='Kernel size')
@click.option('--init_power', default=5, help='Num filters (power of 2) at the first Conv Layer')
@click.option('--lr_base', default=3e-3, help='Num filters (power of 2) at the first Conv Layer')
@click.option('--lr_top', default=4e-2, help='Num filters (power of 2) at the first Conv Layer')
def main(input_file_path, output_file_path, fold, dropout, weights, epochs, batch_size, gpu, epochs_per_cycle,mode,kernel_size,
         init_power,
         lr_base,
         lr_top
         ):
    # For multi gpu support
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)    

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

    n_splits = 5
    k = fold
    assert k<n_splits
    input_file_name = os.path.join(input_file_path, f"{mode}_train_nn_{k}.pck")

    # input_file_name_test = os.path.join(input_file_path, "Test_final.pck")
    # output_file_name = os.path.join(output_file_path, f"models_lgbm.pck")

    with open(input_file_name, 'rb') as f:
        results = pickle.load(f)

    X, y = results[f'data_dict_train_{k}']['X_with_fake'], results[f'data_dict_train_{k}']['y_with_fake']
    X = np.pad(X, pad_width=((0, 0), (2, 2), (0, 0)), mode='edge')
    y = np.pad(y, pad_width=((0, 0), (2, 2), (0, 0)), mode='edge')

    X_holdout, y_holdout = results[f'data_dict_test_{k}']['X'], results[f'data_dict_test_{k}']['y']
    X_holdout = np.pad(X_holdout, pad_width=((0, 0), (2, 2), (0, 0)), mode='edge')
    y_holdout = np.pad(y_holdout, pad_width=((0, 0), (2, 2), (0, 0)), mode='edge')


    logging.info(f'Shape of train: {X.shape}')
    logging.info(f'Shape of val: {X_holdout.shape}')

    f1_scores = []
    scores = []
    #X = (X - X.mean()) / X.std()
    #X_holdout = (X_holdout - X.mean()) / X.std()

    # clr = SGDRScheduler(min_lr=1e-2,max_lr=5e-1,steps_per_epoch=np.ceil(3200/32))
        # Skip other than k-th fold

    model_output_folder = os.path.join(output_file_path, f'RNNUnet-fold_{k}_mode_{mode}')
    os.makedirs(model_output_folder, exist_ok=True)
    model_output_file = os.path.join(model_output_folder, "weights.{epoch:02d}-{val_acc:.4f}.hdf5")

    model_checkpoint = ModelCheckpoint(model_output_file,
                                       monitor='val_acc', verbose=0,
                                       save_best_only=True, save_weights_only=False,
                                       mode='auto', period=1)

    clr = CyclicLR(base_lr=lr_base, max_lr=lr_top, step_size=epochs_per_cycle * X.shape[0] / batch_size,
                   mode='triangular')

    print(X.shape)

    model = create_unet_bidirectRNN((X.shape[1], 1), init_power=init_power, kernel_size=kernel_size, dropout=dropout)
    # model = load_model('/home/anton/Repos/gamma_log_classification/models/weights.18-0.17.hdf5')
    X_holdout_adjusted,y_holdout_adjusted = transform_holdout(X_holdout,y_holdout,mode = mode)

    model.compile(loss='categorical_crossentropy', optimizer=SGD(lr=0.04),
                  metrics=['acc', 'categorical_crossentropy'])
    if weights != '':
        model.load_weights(weights)
    model.fit(
        X,
        y,
        verbose=1,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[model_checkpoint, clr],
        validation_data=(X_holdout_adjusted, y_holdout_adjusted),
    )

    pred = predict_with_mode(model,X_holdout,mode)

    score = accuracy_score(np.argmax(y_holdout, axis=2).flatten(), np.argmax(pred, axis=2).flatten())
    f1_sc = f1_score(np.argmax(y_holdout, axis=2).flatten(), np.argmax(pred, axis=2).flatten(), labels=[1, 2, 3, 4],
                     average='weighted')
    f1_scores.append(f1_sc)
    scores.append(score)

    logging.info(f"{k} - Holdout score = {score}, f1 = {f1_sc}")

    logging.info(f" Holdout score = {np.mean(scores)} , std = {np.std(scores)}")
    logging.info(f" Holdout F1 score = {np.mean(f1_scores)} , std = {np.std(f1_scores)}")


if __name__ == "__main__":
    main()
