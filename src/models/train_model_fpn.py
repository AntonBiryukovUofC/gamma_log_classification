import logging
import os
import pickle
from pathlib import Path

import numpy as np
import tensorflow as tf
from keras.layers import Dense
from keras.layers import Input, UpSampling1D, Flatten, Average, Reshape
from keras.layers.convolutional import Conv1D
from keras.layers.core import Dropout
from keras.layers.pooling import MaxPooling1D
from keras.models import Model
from keras.optimizers import *
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import KFold



def create_cnn_fpn_like(input_size, dropout=0.5, n_dense=16, n_conv=64):
    inputs = Input(input_size)
    conv1_1 = Conv1D(n_conv, 4, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
    pool1_1 = MaxPooling1D(pool_size=2)(conv1_1)
    flat_1 = Flatten()(pool1_1)
    dense1_1 = Dense(n_dense, activation='relu')(flat_1)
    dense1_1 = Dropout(dropout)(dense1_1)
    dense1_1 = Reshape((n_dense, 1), input_shape=(n_dense,))(dense1_1)
    conv2_1 = Conv1D(n_conv, 4, activation='tanh', padding='same', kernel_initializer='he_normal')(dense1_1)
    upsamp1_1 = UpSampling1D(size=2)(conv2_1)
    final_1 = Conv1D(1, 4, activation='linear', padding='same', kernel_initializer='he_normal')(upsamp1_1)
    final_1_d = Dense(5, activation='softmax')(final_1)

    conv1_2 = Conv1D(n_conv, 8, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
    pool1_2 = MaxPooling1D(pool_size=2)(conv1_2)
    flat_2 = Flatten()(pool1_2)
    dense1_2 = Dense(n_dense, activation='relu')(flat_2)
    dense1_2 = Dropout(dropout)(dense1_2)
    dense1_2 = Reshape((n_dense, 1), input_shape=(n_dense,))(dense1_2)
    conv2_2 = Conv1D(n_conv, 8, activation='tanh', padding='same', kernel_initializer='he_normal')(dense1_2)
    upsamp1_2 = UpSampling1D(size=2)(conv2_2)


    final_2 = Conv1D(1, 8, activation='linear', padding='same', kernel_initializer='he_normal')(upsamp1_2)
    final_2_d = Dense(5, activation='softmax')(final_2)

    conv1_3 = Conv1D(n_conv, 16, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
    pool1_3 = MaxPooling1D(pool_size=2)(conv1_3)
    flat_3 = Flatten()(pool1_3)
    dense1_3 = Dense(n_dense, activation='relu')(flat_3)
    dense1_3 = Dropout(dropout)(dense1_3)
    dense1_3 = Reshape((n_dense, 1), input_shape=(n_dense,))(dense1_3)
    # conv2_3 = Conv1D(n_conv, 16, activation='relu', padding='same', kernel_initializer='he_normal')(dense1_3)
    conv2_3 = Conv1D(n_conv, 16, activation='tanh', padding='same', kernel_initializer='he_normal')(dense1_3)
    upsamp1_3 = UpSampling1D(size=2)(conv2_3)
    final_3 = Conv1D(1, 16, activation='linear', padding='same', kernel_initializer='he_normal')(upsamp1_3)
    final_3_d = Dense(5, activation='softmax')(final_3)

    final = Average()([final_1_d, final_2_d, final_3_d])
    model = Model(inputs=inputs, outputs=final)
    print(model.summary())
    model.compile(loss='categorical_crossentropy', optimizer=SGD(lr=0.01))

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
    cv = KFold(n_splits)
    f1_scores = []
    scores = []
    preds_holdout = np.ones((X.shape[0], 5)) * (-50)
    X = (X-X.mean())/X.std()

    for k, (train_index, test_index) in enumerate(cv.split(X, y)):
        X_train, X_holdout = X[train_index, :], X[test_index, :]
        y_train, y_holdout = y[train_index], y[test_index]

        model = create_cnn_fpn_like((X.shape[1], 1), n_dense=550)
        model.fit(
            X_train,
            y_train,
            verbose=1,
            epochs=5
        )

        pred = model.predict(X_holdout)

        score = accuracy_score(np.argmax(y_holdout, axis=2).flatten(), np.argmax(pred, axis=2).flatten())
        f1_sc = f1_score(np.argmax(y_holdout, axis=2).flatten(), np.argmax(pred, axis=2).flatten(), labels=[1, 2, 3, 4], average='weighted')
        f1_scores.append(f1_sc)
        scores.append(score)

        logging.info(f"{k} - Holdout score = {score}, f1 = {f1_sc}")

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
