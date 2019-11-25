
# import keras
import numpy as np
import os
import numpy as np
from keras.models import *
from keras.layers import *





# define architecture for EEG recognizer
def VGG(input_size ,hyperparams):

    input_size = (input_size,1)

    inputs = Input(input_size)
    conv1 = Conv1D(hyperparams['n_filt_1'], hyperparams['kern_size_1'], activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
    pool1 = MaxPooling1D(pool_size=2)(conv1)

    conv2 = Conv1D(hyperparams['n_filt_2'], hyperparams['kern_size_2'], activation='relu', padding='same',
                   kernel_initializer='he_normal')(pool1)
    pool2 = MaxPooling1D(pool_size=2)(conv2)


    conv3 = Conv1D(hyperparams['n_filt_3'], hyperparams['kern_size_3'], activation='relu', padding='same',
                   kernel_initializer='he_normal')(pool2)
    pool3 = MaxPooling1D(pool_size=2)(conv3)


    conv4 = Conv1D(hyperparams['n_filt_2'], hyperparams['kern_size_2'], activation='relu', padding='same',
                   kernel_initializer='he_normal')(pool3)
    pool4 = UpSampling1D(size=2)(conv4)

    conv5 = Conv1D(hyperparams['n_filt_1'], hyperparams['kern_size_1'], activation='relu', padding='same',
                   kernel_initializer='he_normal')(pool4)
    pool5 = UpSampling1D(size=2)(conv5)


    final = Conv1D(1, hyperparams['kern_size_1'], activation='sigmoid', padding='same',
                   kernel_initializer='he_normal')(pool5)
    pool5 = UpSampling1D(size=2)(conv5)



    model = Model(input = inputs, output = final)

    model.summary()

    return model