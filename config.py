#imports

#data processing
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import accuracy_score
from scipy.signal import medfilt
from scipy.signal import hilbert
from obspy.signal.filter import envelope

#deep learning
#keras
import keras
import tensorflow as tf
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, ReduceLROnPlateau
from keras import backend as keras
from keras.callbacks import EarlyStopping, ModelCheckpoint

from sklearn.neighbors import KNeighborsClassifier
from scipy.signal import resample

np.random.seed(42)

from tensorflow import set_random_seed
tf.set_random_seed(42)

# condiguration of GPU
from keras import backend as K


#K.set_floatx('float64')

#visualization
import matplotlib.pyplot as plt



#system libs
import os
import gc



#names:
DATA_PATH = './data/raw/'
TRAIN_NAME = 'train_cax.csv'
TEST_NAME = 'test_cax.csv'

TARGET = 'label'
DROPLIST = []

# imodel settings
from model.config_LSTM import *



PIC_FOLDER = './data/pictures/'
STACKING_FOLDER = './data/stacking/'
SUBMIT_FOLDER = './data/result/'
DEBUG_FOLDER = './data/debug/'

for f in [PIC_FOLDER,STACKING_FOLDER,SUBMIT_FOLDER,DEBUG_FOLDER]:
    os.makedirs(f,exist_ok=True)


import tensorflow as tf
from tensorflow.compat.v1.keras.backend import set_session
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
config.log_device_placement = True  # to log device placement (on which device the operation ran)
sess = tf.compat.v1.Session(config=config)
set_session(sess)