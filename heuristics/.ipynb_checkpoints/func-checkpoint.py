#imports

#data processing
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler

#deep learning
#keras
import keras
import tensorflow as tf
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as keras
from keras.callbacks import EarlyStopping, ModelCheckpoint
np.random.seed(42)

# condiguration of GPU
from keras import backend as K

tf.debugging.set_log_device_placement(True)



#visualization
import matplotlib.pyplot as plt


#show avaliable devices for CUDA to use GPU
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

#system libs
import os
import gc
#fix random seed
np.random.seed(42)


#names:
DATA_PATH = './data/raw/'
TRAIN_NAME = 'train.csv'
TEST_NAME = 'test.csv'

TARGET = 'label'
DROPLIST = []

# imodel settings
from model.config_DNN import *