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

from src.models.models import create_fcn__multiple_heads
from pathlib import Path
import numpy as np

# gpus = tf.config.experimental.list_physical_devices('GPU')
# if gpus:
#     try:
#         # Currently, memory growth needs to be the same across GPUs
#         for gpu in gpus:
#             tf.config.experimental.set_memory_growth(gpu, True)
#         logical_gpus = tf.config.experimental.list_logical_devices('GPU')
#         print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
#     except RuntimeError as e:
#         # Memory growth must be set before GPUs have been initialized
#         print(e)


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


project_dir = Path(__file__).resolve().parents[2]

log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
logging.basicConfig(level=logging.INFO, format=log_fmt)

exclude_cols = ['row_id']
tgt = 'label'
input_file_path = os.path.join(project_dir, "data", "processed")
output_file_path = os.path.join(project_dir, "models")
os.makedirs(input_file_path, exist_ok=True)
os.makedirs(output_file_path, exist_ok=True)




@click.command()
@click.option('--epochs', default=20, help='number of epochs')
@click.option('--input_file_path', default=input_file_path, help='input file location (of train_nn.pck)')
@click.option('--output_file_path', default=output_file_path, help='output folder (for model weights)')
@click.option('--fold', default=0, help='fold to train')
@click.option('--gpu', default=0, help='gpu to train')
@click.option('--weights', default='', help='weights location')
@click.option('--dropout', default=0.1, help='dropout rate')
@click.option('--batch_size', default=8, help='batch size')
@click.option('--cycles_per_epoch', default=4, help='cycles per epoch')
def main(input_file_path, output_file_path,fold,dropout,weights,epochs,batch_size,gpu,cycles_per_epoch):
    n_splits = 5
    input_file_name = os.path.join(input_file_path, "train_nn.pck")

    # input_file_name_test = os.path.join(input_file_path, "Test_final.pck")
    # output_file_name = os.path.join(output_file_path, f"models_lgbm.pck")

    with open(input_file_name, 'rb') as f:
        results = pickle.load(f)
    X, y = results['X'], results['y']
    X = np.pad(X, pad_width=((0, 0), (2, 2), (0, 0)), mode='edge')
    y = np.pad(y, pad_width=((0, 0), (2, 2), (0, 0)), mode='edge')
    cv = KFold(n_splits)
    f1_scores = []
    scores = []
    X = (X - X.mean()) / X.std()

    # clr = SGDRScheduler(min_lr=1e-2,max_lr=5e-1,steps_per_epoch=np.ceil(3200/32))
    for k, (train_index, test_index) in enumerate(cv.split(X, y)):
        # Skip other than k-th fold
        if k!= fold:
            continue
        X_train, X_holdout = X[train_index, :], X[test_index, :]
        y_train, y_holdout = y[train_index], y[test_index]

        model_output_folder = os.path.join(output_file_path, f'fold_{k}')
        os.makedirs(model_output_folder,exist_ok=True)
        model_output_file = os.path.join(model_output_folder, "weights.{epoch:02d}-{val_acc:.4f}.hdf5")

        model_checkpoint = ModelCheckpoint(model_output_file,
            monitor='val_acc', verbose=0,
            save_best_only=True, save_weights_only=False,
            mode='auto', period=1)

        clr = CyclicLR(base_lr=2e-3, max_lr=4e-2, step_size=cycles_per_epoch * X_train.shape[0] / batch_size)

        print(X_train.shape)

        model = create_fcn__multiple_heads((X.shape[1], 1), init_power=6, kernel_size=(3, 7, 11), dropout=dropout)
        # model = load_model('/home/anton/Repos/gamma_log_classification/models/weights.18-0.17.hdf5')
        model.compile(loss='categorical_crossentropy', optimizer=SGD(lr=0.04),
                      metrics=['acc', 'categorical_crossentropy'])
        if weights != '':
            model.load_weights(weights)
        model.fit(
            X_train,
            y_train,
            verbose=1,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[model_checkpoint, clr],
            validation_data=(X_holdout, y_holdout),
        )


        pred = model.predict(X_holdout)

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

