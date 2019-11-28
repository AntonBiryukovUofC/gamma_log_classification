import logging
import os
import pickle
from pathlib import Path

import numpy as np
import tensorflow as tf
from keras.callbacks import Callback
from keras.layers import Dense, concatenate
from keras.layers import Input, UpSampling1D, Flatten, Average, Reshape
from keras.layers.convolutional import Conv1D, Conv2D, UpSampling2D
from keras.layers.core import Dropout
from keras.layers.pooling import MaxPooling1D, MaxPooling2D
from keras.models import Model
from keras.optimizers import *
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import KFold


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




def create_unet(input_size, init_power=4, kernel_size=3, dropout=0.3):
    # Build U-Net model
    inputs = Input(input_size)
    c1 = Conv1D(2 ** init_power, kernel_size, activation='relu', padding='same')(inputs)
    c1 = Dropout(dropout)(c1)
    c1 = Conv1D(2 ** init_power, kernel_size, activation='relu', padding='same')(c1)
    p1 = MaxPooling1D(2)(c1)
    c2 = Conv1D(2 ** (init_power + 1), kernel_size, activation='relu', padding='same')(p1)
    c2 = Dropout(dropout)(c2)
    c2 = Conv1D(2 ** (init_power + 1), kernel_size, activation='relu', padding='same')(c2)
    p2 = MaxPooling1D(2)(c2)

    c3 = Conv1D(2 ** (init_power + 2), kernel_size, activation='relu', padding='same')(p2)
    c3 = Dropout(dropout)(c3)
    c3 = Conv1D(2 ** (init_power + 2), kernel_size, activation='relu', padding='same')(c3)
    p3 = MaxPooling1D(2)(c3)

    c4 = Conv1D(2 ** (init_power + 3), kernel_size, activation='relu', padding='same')(p3)
    c4 = Dropout(dropout)(c4)
    c4 = Conv1D(2 ** (init_power + 3), kernel_size, activation='relu', padding='same')(c4)
    p4 = MaxPooling1D(2)(c4)

    c5 = Conv1D(2 ** (init_power + 4), kernel_size, activation='relu', padding='same')(p4)
    c5 = Dropout(dropout)(c5)
    c5 = Conv1D(2 ** (init_power + 4), kernel_size, activation='relu', padding='same')(c5)

    u6 = UpSampling1D(2)(c5)
    u6 = concatenate([u6, c4])
    c6 = Conv1D(2 ** (init_power + 3), kernel_size, activation='relu', padding='same')(u6)
    c6 = Dropout(dropout)(c6)
    c6 = Conv1D(2 ** (init_power + 3), kernel_size, activation='relu', padding='same')(c6)

    u7 = UpSampling1D(2)(c6)
    u7 = concatenate([u7, c3])
    c7 = Conv1D(2 ** (init_power + 2), kernel_size, activation='relu', padding='same')(u7)
    c7 = Dropout(dropout)(c7)
    c7 = Conv1D(2 ** (init_power + 2), kernel_size, activation='relu', padding='same')(c7)

    u8 = UpSampling1D(2)(c7)
    u8 = concatenate([u8, c2])
    c8 = Conv1D(2 ** (init_power + 1), kernel_size, activation='relu', padding='same')(u8)
    c8 = Dropout(dropout)(c8)
    c8 = Conv1D(2 ** (init_power + 1), kernel_size, activation='relu', padding='same')(c8)

    u9 = UpSampling1D(2)(c8)
    u9 = concatenate([u9, c1])
    c9 = Conv1D(2 ** init_power, kernel_size, activation='relu', padding='same')(u9)
    c9 = Dropout(dropout)(c9)
    c9 = Conv1D(2 ** init_power, kernel_size, activation='relu', padding='same')(c9)
    outputs = Conv1D(5, 1, activation='softmax')(c9)
    # outputs = Lambda(func, output_shape = [1]) (outputs)
    model = Model(inputs=[inputs], outputs=[outputs])
    print(model.summary())
    model.compile(loss='categorical_crossentropy', optimizer=SGD(lr=0.05), metrics=['acc', 'categorical_crossentropy'])

    return model

project_dir = Path(__file__).resolve().parents[2]

log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
logging.basicConfig(level=logging.INFO, format=log_fmt)

exclude_cols = ['row_id']
tgt = 'label'


def main(input_file_path, output_file_path, n_splits=5):
    input_file_name = os.path.join(input_file_path, "train_nn.pck")

    # input_file_name_test = os.path.join(input_file_path, "Test_final.pck")
    # output_file_name = os.path.join(output_file_path, f"models_lgbm.pck")

    with open(input_file_name, 'rb') as f:
        results = pickle.load(f)
    X, y = results['X'], results['y']
    X = np.pad(X,pad_width=((0,0),(2,2),(0,0)),mode='edge')
    y = np.pad(y, pad_width=((0,0),(2,2),(0,0)), mode='edge')
    cv = KFold(n_splits)
    f1_scores = []
    scores = []
    preds_holdout = np.ones((X.shape[0], 5)) * (-50)
    X = (X-X.mean())/X.std()


    #clr = SGDRScheduler(min_lr=1e-2,max_lr=5e-1,steps_per_epoch=np.ceil(3200/32))

    for k, (train_index, test_index) in enumerate(cv.split(X, y)):
        X_train, X_holdout = X[train_index, :], X[test_index, :]
        y_train, y_holdout = y[train_index], y[test_index]
        
        
        print(X_train.shape)
        
        model = create_unet((X.shape[1], 1),init_power=5,kernel_size=5)
        model.fit(
            X_train,
            y_train,
            verbose=1,
            epochs=20,
            batch_size=4,
        #    callbacks=[clr],
            validation_data = (X_holdout,y_holdout),
        )

        pred = model.predict(X_holdout)

        score = accuracy_score(np.argmax(y_holdout, axis=2).flatten(), np.argmax(pred, axis=2).flatten())
        f1_sc = f1_score(np.argmax(y_holdout, axis=2).flatten(), np.argmax(pred, axis=2).flatten(), labels=[1, 2, 3, 4], average='weighted')
        f1_scores.append(f1_sc)
        scores.append(score)
        
        logging.info(f"{k} - Holdout score = {score}, f1 = {f1_sc}")
        break
    logging.info(f" Holdout score = {np.mean(scores)} , std = {np.std(scores)}")
    logging.info(f" Holdout F1 score = {np.mean(f1_scores)} , std = {np.std(f1_scores)}")



if __name__ == "__main__":
    # not used in this stub but often useful for finding various files

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    input_file_path = os.path.join(project_dir, "data", "processed")
    output_file_path = os.path.join(project_dir, "models")
    os.makedirs(input_file_path, exist_ok=True)
    os.makedirs(output_file_path, exist_ok=True)

    preds_wsc = main(input_file_path, output_file_path)

    df = preds_wsc
