#imports

#data processing
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

#deep learning
#keras
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as keras
from keras.callbacks import EarlyStopping, ModelCheckpoint


# imodel settings
from model.config_VGG import *

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

