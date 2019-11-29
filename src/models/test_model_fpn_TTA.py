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


class SGDRScheduler(Callback):
    '''Cosine annealing learning rate scheduler with periodic restarts.
    # Usage
        ```python
            schedule = SGDRScheduler(min_lr=1e-5,
                                     max_lr=1e-2,
                                     steps_per_epoch=np.ceil(epoch_size/batch_size),
                                     lr_decay=0.9,
                                     cycle_length=5,
                                     mult_factor=1.5)
            model.fit(X_train, Y_train, epochs=100, callbacks=[schedule])
        ```
    # Arguments
        min_lr: The lower bound of the learning rate range for the experiment.
        max_lr: The upper bound of the learning rate range for the experiment.
        steps_per_epoch: Number of mini-batches in the dataset. Calculated as `np.ceil(epoch_size/batch_size)`.
        lr_decay: Reduce the max_lr after the completion of each cycle.
                  Ex. To reduce the max_lr by 20% after each cycle, set this value to 0.8.
        cycle_length: Initial number of epochs in a cycle.
        mult_factor: Scale epochs_to_restart after each full cycle completion.
    # References
        Blog post: jeremyjordan.me/nn-learning-rate
        Original paper: http://arxiv.org/abs/1608.03983
    '''

    def __init__(self,
                 min_lr,
                 max_lr,
                 steps_per_epoch,
                 lr_decay=1,
                 cycle_length=10,
                 mult_factor=2):

        self.min_lr = min_lr
        self.max_lr = max_lr
        self.lr_decay = lr_decay

        self.batch_since_restart = 0
        self.next_restart = cycle_length

        self.steps_per_epoch = steps_per_epoch

        self.cycle_length = cycle_length
        self.mult_factor = mult_factor

        self.history = {}

    def clr(self):
        '''Calculate the learning rate.'''
        fraction_to_restart = self.batch_since_restart / (self.steps_per_epoch * self.cycle_length)
        lr = self.min_lr + 0.5 * (self.max_lr - self.min_lr) * (1 + np.cos(fraction_to_restart * np.pi))
        return lr

    def on_train_begin(self, logs={}):
        '''Initialize the learning rate to the minimum value at the start of training.'''
        logs = logs or {}
        K.set_value(self.model.optimizer.lr, self.max_lr)

    def on_batch_end(self, batch, logs={}):
        '''Record previous batch statistics and update the learning rate.'''
        logs = logs or {}
        self.history.setdefault('lr', []).append(K.get_value(self.model.optimizer.lr))
        for k, v in logs.items():
            self.history.setdefault(k, []).append(v)

        self.batch_since_restart += 1
        K.set_value(self.model.optimizer.lr, self.clr())

    def on_epoch_end(self, epoch, logs={}):
        '''Check for end of current cycle, apply restarts when necessary.'''
        if epoch + 1 == self.next_restart:
            self.batch_since_restart = 0
            self.cycle_length = np.ceil(self.cycle_length * self.mult_factor)
            self.next_restart += self.cycle_length
            self.max_lr *= self.lr_decay
            self.best_weights = self.model.get_weights()

    def on_train_end(self, logs={}):
        '''Set weights to the values from the end of the most recent cycle for best performance.'''
        self.model.set_weights(self.best_weights)


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


project_dir = Path(__file__).resolve().parents[2]

log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
logging.basicConfig(level=logging.INFO, format=log_fmt)

exclude_cols = ['row_id']
tgt = 'label'


def prepare_test(pred_test, df_test):

    wells = df_test['well_id'].sort_values().unique().tolist()
    list_df_wells = [df_test.loc[df_test['well_id'].isin([w]), :].copy() for w in wells]
    for df in list_df_wells:
        df.index = np.arange(df.shape[0])

    for i, df_well in enumerate(list_df_wells):
        df_well['label'] = np.argmax(pred_test[i,:], axis=1)[2:(pred_test.shape[1]-2)]

    result = pd.concat(list_df_wells,axis=0)
    return result


def main(input_file_path, output_file_path, n_splits=5):
    input_file_name = os.path.join(input_file_path, "train_nn.pck")
    input_file_name_test = os.path.join(input_file_path, "test_nn.pck")

    with open(input_file_name, 'rb') as f:
        results = pickle.load(f)
    with open(input_file_name_test, 'rb') as f:
        results_test = pickle.load(f)

    X, y = results['X_small'], results['y_small']
    X_test = results_test['X']
    print(results_test['df_test'].shape)
    X = np.pad(X, pad_width=((0, 0), (2, 2), (0, 0)), mode='edge')
    y = np.pad(y, pad_width=((0, 0), (2, 2), (0, 0)), mode='edge')
    X_test = np.pad(X_test, pad_width=((0, 0), (2, 2), (0, 0)), mode='edge')

    cv = KFold(n_splits)
    f1_scores = []
    scores = []
    # preds_holdout = np.ones((X.shape[0], 5)) * (-50)
    mu = X.mean()
    std = X.std()

    X_test = (X_test - mu) / std
    X = (X - mu) / std

    for k, (train_index, test_index) in enumerate(cv.split(X, y)):
        X_train, X_holdout = X[train_index, :], X[test_index, :]
        y_train, y_holdout = y[train_index], y[test_index]

        print(X_train.shape)

        model = create_fcn__multiple_heads((X.shape[1], 1), init_power=6, kernel_size=(3, 7, 11))
        model.compile(loss='categorical_crossentropy', optimizer=SGD(lr=0.04),
                      metrics=['acc', 'categorical_crossentropy'])
        model.load_weights('/home/anton/Repos/gamma_log_classification/models/weights.16-0.9796.hdf5')

        pred = model.predict(X_holdout)
        pred_tta = tta_predict(model, X_holdout)

        pred_test = model.predict(X_test)
        pred_test_tta = tta_predict(model, X_test)
        pred_test_blend = (pred_test + pred_test_tta) * 0.5

        blend = pred * 0.5 + pred_tta * 0.5

        score = accuracy_score(np.argmax(y_holdout, axis=2).flatten(), np.argmax(pred, axis=2).flatten())
        f1_sc = f1_score(np.argmax(y_holdout, axis=2).flatten(), np.argmax(pred, axis=2).flatten(), labels=[1, 2, 3, 4],
                         average='weighted')
        score_tta = accuracy_score(np.argmax(y_holdout, axis=2).flatten(), np.argmax(pred_tta, axis=2).flatten())
        f1_sc_tta = f1_score(np.argmax(y_holdout, axis=2).flatten(), np.argmax(pred_tta, axis=2).flatten(),
                             labels=[1, 2, 3, 4],
                             average='weighted')
        score_blend = accuracy_score(np.argmax(y_holdout, axis=2).flatten(), np.argmax(blend, axis=2).flatten())
        f1_scores.append(f1_sc)
        scores.append(score)

        logging.info(
            f"{k} - Holdout score = {score}, f1 = {f1_sc} TTA= {score_tta}, f1 = {f1_sc_tta} | blend accuracy = {score_blend}")
        break

    df_test = prepare_test(pred_test_blend, results_test['df_test'])
    logging.info(f" Holdout score = {np.mean(scores)} , std = {np.std(scores)}")
    logging.info(f" Holdout F1 score = {np.mean(f1_scores)} , std = {np.std(f1_scores)}")
    return df_test,X_test,pred_test


if __name__ == "__main__":
    # not used in this stub but often useful for finding various files

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    input_file_path = os.path.join(project_dir, "data", "processed")
    output_file_path = os.path.join(project_dir, "models")
    os.makedirs(input_file_path, exist_ok=True)
    os.makedirs(output_file_path, exist_ok=True)

    df_test,X_test,pred_test = main(input_file_path, output_file_path)

    df_test[['row_id', 'well_id', 'label']].to_csv(os.path.join(input_file_path, 'submit_nn.csv'), index=False)
    #df_test.to_csv(os.path.join(input_file_path, 'submit_nn.csv'), index=False)
    with open(os.path.join(project_dir, "data", "interim","test_dict"),'wb') as f:
        pickle.dump((X_test,pred_test),f)
    # df = preds_wsc
