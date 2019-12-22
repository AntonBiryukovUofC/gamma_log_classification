# import keras
import numpy as np
import os
import numpy as np
from keras.models import *
from keras.layers import *


def DL_model(input_size, hyperparams):
    from keras.layers import TimeDistributed

    dropout = hyperparams['dropout']

    input = Input(input_size)
    bd1 = Bidirectional(CuDNNLSTM(256, return_sequences=True))(input)
    a1 = Activation('elu')(bd1)
    bd2 = Bidirectional(CuDNNLSTM(256, return_sequences=True))(a1)
    a2 = Activation('elu')(bd2)
    bd3 = Bidirectional(CuDNNLSTM(256, return_sequences=True))(a2)
    a3 = Activation('elu')(bd3)

    # output = TimeDistributed(Dense(5, activation='softmax'))(a3)

    output = Conv1D(64, 5, activation='elu', padding='same')(a3)
    output = Dropout(dropout)(output)
    output = Conv1D(5, 1, activation='softmax', padding='same')(output)

    model = Model(inputs=[input], outputs=[output])
    print(model.summary())
    return model
