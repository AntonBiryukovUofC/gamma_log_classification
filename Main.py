#import modules
from DataGenerator import *
from Pipeline import *
from config import *
import sys
import logging as log
import click
import os

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session

@click.command()
@click.option('--start_fold', default=0, help='fold to train')
@click.option('--gpu', default=0, help='gpu to train on')
@click.option('--batch', default=64, help='batch size')
@click.option('--add_trend', help='add trend to xstarter ?',is_flag=True)
@click.option('--freq_enc', help='use freq encoder?',is_flag=True)
@click.option('--use_diffs_leaky', help='calculate diffs for the leaky feature?',is_flag=True)
@click.option('--use_neighbor', help='Add nearest neighbors as additiona channels?',is_flag=True)

def main(start_fold,gpu,batch,add_trend,freq_enc,use_diffs_leaky,use_neighbor):
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
    sess = tf.Session(config=config)
    set_session(sess)  # set this TensorFlow session as the default session for Keras
    if add_trend:
        log.info('Will add trend to XEEK Train data')
    GetData = DataGenerator(add_trend=add_trend,use_diffs_leaky=use_diffs_leaky,use_neighbor=use_neighbor)
    CV = Pipeline(GetData, DL_model, start_fold, gpu, batch)
    score = CV.train(freq_encoder=freq_enc)
    log.info(f'Model accuracy = {score}')

if __name__ == "__main__":
    main()

