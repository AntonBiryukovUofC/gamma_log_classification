
# import keras
import numpy as np
import os
import numpy as np
from keras.models import *
from keras.layers import *





# define architecture for EEG recognizer
def DL_model(input_size ,hyperparams):

    input_size = (input_size,1)

    inputs = Input(input_size)
    dense1 = Dense(hyperparams['N_1'],activation='relu')(inputs)
    dense1 = Dropout(0.5)(dense1)

    dense2 = Dense(hyperparams['N_1'], activation='relu')(dense1)
    dense2 = Dropout(0.5)(dense2)

    dense3 = Dense(hyperparams['N_1'], activation='relu')(dense2)
    dense3 = Dropout(0.5)(dense3)

    final = Conv1D(5, hyperparams['kern_size'], activation='sigmoid', padding='same',
                   kernel_initializer='he_normal')(dense3)

    model = Model(input = inputs, output = final)

    model.summary()

    return model