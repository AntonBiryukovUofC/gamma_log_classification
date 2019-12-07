#imports

#data processing
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score

#deep learning
#keras
import keras
import tensorflow as tf
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, ReduceLROnPlateau
from keras import backend as keras
from keras.callbacks import EarlyStopping, ModelCheckpoint
np.random.seed(42)

# condiguration of GPU
from keras import backend as K




#visualization
import matplotlib.pyplot as plt



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
from model.config_UNET import *



PIC_FOLDER = './data/pictures/'
STACKING_FOLDER = './data/stacking/'
SUBMIT_FOLDER = './data/result/'

