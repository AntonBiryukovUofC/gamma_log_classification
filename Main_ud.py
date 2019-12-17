#import modules
import logging as log
import os

import click
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session

from Pipeline import *
from config import *


@click.command()
@click.option('--start_fold', default=0, help='fold to train')
@click.option('--gpu', default=0, help='gpu to train on')
def main(start_fold,gpu):
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
    sess = tf.Session(config=config)
    set_session(sess)  # set this TensorFlow session as the default session for Keras

    GetData = DataGenerator(dataset_mode='ud')
    CV = Pipeline(GetData, DL_model, start_fold, gpu,model_name=MODEL_PATH + 'LSTM_model_ud')
    score = CV.train()
    log.info(f'Model accuracy = {score}')

if __name__ == "__main__":
    main()
