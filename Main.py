#import modules
from DataGenerator import *
from Pipeline import *
from config import *
import sys
import logging as log
import click
import os
from adabound import AdaBound
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session

@click.command()
@click.option('--start_fold', default=0, help='fold to train')
@click.option('--gpu', default=0, help='gpu to train on')
@click.option('--add_trend', help='add trend to xstarter ?',is_flag=True)
def main(start_fold,gpu,add_trend):
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
    sess = tf.Session(config=config)
    set_session(sess)  # set this TensorFlow session as the default session for Keras
    if add_trend:
        log.info('Will add trend to XEEK Train data')
    GetData = DataGenerator(add_trend=add_trend)
    CV = Pipeline(GetData, DL_model, start_fold, gpu)
    score = CV.train()
    log.info(f'Model accuracy = {score}')

if __name__ == "__main__":
    main()

