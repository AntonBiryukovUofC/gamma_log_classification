# import keras
import numpy as np
import os
import numpy as np
from keras.models import *
from keras.layers import *



def DL_model(input_size, hyperparams ):

    init_power = hyperparams['init_power']
    kernel_size = hyperparams['kernel_size']
    dropout = hyperparams['dropout']

    input_size = (input_size, 34)

    # Build U-Net model
    inputs = Input(input_size)
    c1 = Conv1D(2 ** init_power, kernel_size, activation='relu', padding='same')(inputs)
    c1 = Dropout(dropout)(c1)
    c1 = Conv1D(2 ** init_power, kernel_size, activation='relu', padding='same')(c1)
    p1 = MaxPooling1D(2)(c1)
    c2 = Conv1D(2 ** (init_power + 1), kernel_size, activation='relu', padding='same')(p1)
    c2 = Dropout(dropout)(c2)
    c2 = Conv1D(2 ** (init_power + 1), kernel_size, activation='relu', padding='same')(c2)
    p2 = MaxPooling1D(2)(c2)

    c3 = Conv1D(2 ** (init_power + 2), kernel_size, activation='relu', padding='same')(p2)
    c3 = Dropout(dropout)(c3)
    c3 = Conv1D(2 ** (init_power + 2), kernel_size, activation='relu', padding='same')(c3)
    p3 = MaxPooling1D(2)(c3)

    c4 = Conv1D(2 ** (init_power + 3), kernel_size, activation='relu', padding='same')(p3)
    c4 = Dropout(dropout)(c4)
    c4 = Conv1D(2 ** (init_power + 3), kernel_size, activation='relu', padding='same')(c4)
    #p4 = MaxPooling1D(2)(c4)

    """
    c5 = Conv1D(2 ** (init_power + 4), kernel_size, activation='relu', padding='same')(p4)
    c5 = Dropout(dropout)(c5)
    c5 = Conv1D(2 ** (init_power + 4), kernel_size, activation='relu', padding='same')(c5)

    u6 = UpSampling1D(2)(c5)
    u6 = concatenate([u6, c4])
    c6 = Conv1D(2 ** (init_power + 3), kernel_size, activation='relu', padding='same')(u6)
    c6 = Dropout(dropout)(c6)
    c6 = Conv1D(2 ** (init_power + 3), kernel_size, activation='relu', padding='same')(c6)
    """

    u7 = UpSampling1D(2)(c4)
    #u7 = UpSampling1D(2)(c6)
    u7 = concatenate([u7, c3])
    c7 = Conv1D(2 ** (init_power + 2), kernel_size, activation='relu', padding='same')(u7)
    c7 = Dropout(dropout)(c7)
    c7 = Conv1D(2 ** (init_power + 2), kernel_size, activation='relu', padding='same')(c7)

    u8 = UpSampling1D(2)(c7)
    u8 = concatenate([u8, c2])
    c8 = Conv1D(2 ** (init_power + 1), kernel_size, activation='relu', padding='same')(u8)
    c8 = Dropout(dropout)(c8)
    c8 = Conv1D(2 ** (init_power + 1), kernel_size, activation='relu', padding='same')(c8)

    u9 = UpSampling1D(2)(c8)
    u9 = concatenate([u9, c1])
    c9 = Conv1D(2 ** init_power, kernel_size, activation='relu', padding='same')(u9)
    c9 = Dropout(dropout)(c9)
    c9 = Conv1D(2 ** init_power, kernel_size, activation='relu', padding='same')(c9)
    outputs = Conv1D(5, 1, activation='softmax')(c9)
    # outputs = Lambda(func, output_shape = [1]) (outputs)
    model = Model(inputs=[inputs], outputs=[outputs])
    print(model.summary())
    #model.compile(loss='categorical_crossentropy', optimizer=SGD(lr=0.04), metrics=['acc', 'categorical_crossentropy'])

    return model