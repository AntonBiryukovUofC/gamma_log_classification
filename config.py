#imports

#data processing
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import accuracy_score
from scipy.signal import medfilt

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
DEBUG_FOLDER = '/data/debug/'

for f in [PIC_FOLDER,STACKING_FOLDER,SUBMIT_FOLDER,DEBUG_FOLDER]:
    os.makedirs(f,exist_ok=True)