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
    bd1 = Bidirectional(CuDNNGRU(256, return_sequences=True))(input)
    a1 = Activation('relu')(bd1)
    bd2 = Bidirectional(CuDNNGRU(128, return_sequences=True))(a1)
    a2 = Activation('relu')(bd2)
    bd3 = Bidirectional(CuDNNGRU(128, return_sequences=True))(a2)
    a3 = Activation('relu')(bd3)
    #bd4 = Bidirectional(CuDNNGRU(128, return_sequences=True))(a3)
    #a4 = Activation('relu')(bd4)

    output = TimeDistributed(Dense(5,activation='softmax'))(a3)
    model = Model(inputs=[input], outputs=[output])
    print(model.summary())
    return model