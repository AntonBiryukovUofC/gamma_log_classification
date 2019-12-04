from keras import Input, Model
from keras.layers import Conv1D, Dropout, MaxPooling1D, UpSampling1D, concatenate, Average, Bidirectional, LSTM, \
    CuDNNLSTM, CuDNNGRU, ReLU, BatchNormalization,add,Add
from keras.optimizers import SGD
import tensorflow.keras.backend as K
import numpy as np
def weightedLoss(originalLossFunc, weightsList):

    def lossFunc(true, pred):

        axis = -1 #if channels last
        #axis=  1 #if channels first


        #argmax returns the index of the element with the greatest value
        #done in the class axis, it returns the class index
        classSelectors = K.argmax(true, axis=axis)

        #considering weights are ordered by class, for each class
        #true(1) if the class index is equal to the weight index
        one64 = np.ones(1, dtype=np.int64)
        classSelectors = [K.equal(one64[0] * i, classSelectors) for i in range(len(weightsList))]
        #casting boolean to float for calculations
        #each tensor in the list contains 1 where ground true class is equal to its index
        #if you sum all these, you will get a tensor full of ones.
        classSelectors = [K.cast(x, K.floatx()) for x in classSelectors]

        #for each of the selections above, multiply their respective weight
        weights = [sel * w for sel,w in zip(classSelectors, weightsList)]

        #sums all the selections
        #result is a tensor with the respective weight for each element in predictions
        weightMultiplier = weights[0]
        for i in range(1, len(weights)):
            weightMultiplier = weightMultiplier + weights[i]


        #make sure your originalLossFunc only collapses the class axis
        #you need the other axes intact to multiply the weights tensor
        loss = originalLossFunc(true,pred)
        loss = loss * weightMultiplier

        return loss
    return lossFunc


def get_sample_weights(y_true, y_pred, cost_m):
    num_classes = len(cost_m)

    y_pred.shape.assert_has_rank(2)
    y_pred.shape[1].assert_is_compatible_with(num_classes)
    y_pred.shape.assert_is_compatible_with(y_true.shape)

    y_pred = K.one_hot(K.argmax(y_pred), num_classes)

    y_true_nk1 = K.expand_dims(y_true, 2)
    y_pred_n1k = K.expand_dims(y_pred, 1)
    cost_m_1kk = K.expand_dims(cost_m, 0)

    sample_weights_nkk = cost_m_1kk * y_true_nk1 * y_pred_n1k
    sample_weights_n = K.sum(sample_weights_nkk, axis=[1, 2])

    return sample_weights_n

def create_unet(input_size, init_power=4, kernel_size=3, dropout=0.3):
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
    p4 = MaxPooling1D(2)(c4)

    c5 = Conv1D(2 ** (init_power + 4), kernel_size, activation='relu', padding='same')(p4)
    c5 = Dropout(dropout)(c5)
    c5 = Conv1D(2 ** (init_power + 4), kernel_size, activation='relu', padding='same')(c5)

    u6 = UpSampling1D(2)(c5)
    u6 = concatenate([u6, c4])
    c6 = Conv1D(2 ** (init_power + 3), kernel_size, activation='relu', padding='same')(u6)
    c6 = Dropout(dropout)(c6)
    c6 = Conv1D(2 ** (init_power + 3), kernel_size, activation='relu', padding='same')(c6)

    u7 = UpSampling1D(2)(c6)
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
    model.compile(loss='categorical_crossentropy', optimizer=SGD(lr=0.04), metrics=['acc', 'categorical_crossentropy'])

    return model



def create_unet_bidirectRNN(input_size, init_power=4, kernel_size=3, dropout=0.3):
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
    p4 = MaxPooling1D(2)(c4)

    c5 = Conv1D(2 ** (init_power + 4), kernel_size, activation='relu', padding='same')(p4)
    c5 = Dropout(dropout)(c5)
    c5 = Conv1D(2 ** (init_power + 4), kernel_size, activation='relu', padding='same')(c5)

    u6 = UpSampling1D(2)(c5)
    u6 = concatenate([u6, c4])
    c6 = Conv1D(2 ** (init_power + 3), kernel_size, activation='relu', padding='same')(u6)
    c6 = Dropout(dropout)(c6)
    c6 = Conv1D(2 ** (init_power + 3), kernel_size, activation='relu', padding='same')(c6)

    u7 = UpSampling1D(2)(c6)
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
    outputs = Bidirectional(CuDNNLSTM(64,return_sequences=True))(c9)
    outputs = ReLU()(outputs)
    outputs = Conv1D(5, 1, activation='softmax')(outputs)
    model = Model(inputs=[inputs], outputs=[outputs])
    print(model.summary())
    model.compile(loss='categorical_crossentropy', optimizer=SGD(lr=0.04), metrics=['acc', 'categorical_crossentropy'])

    return model




def create_fcn_like(input_size, init_power=4, kernel_size=3, dropout=0.3):
    # Build U-Net model
    inputs = Input(input_size)
    c1 = Conv1D(2 ** init_power, kernel_size, activation='relu', padding='same')(inputs)
    c1 = Dropout(dropout)(c1)
    c1 = Conv1D(2 ** init_power, kernel_size, activation='relu', padding='same')(c1)
    p1 = MaxPooling1D(2)(c1)

    c4 = Conv1D(2 ** (init_power + 1), kernel_size, activation='relu', padding='same')(p1)
    c4 = Dropout(dropout)(c4)
    c4 = Conv1D(2 ** (init_power + 1), kernel_size, activation='relu', padding='same')(c4)
    p4 = MaxPooling1D(2)(c4)

    c5 = Conv1D(2 ** (init_power + 2), kernel_size, activation='relu', padding='same')(p4)
    c5 = Dropout(dropout)(c5)
    c5 = Conv1D(2 ** (init_power + 2), kernel_size, activation='relu', padding='same')(c5)

    u6 = UpSampling1D(2)(c5)
    c6 = Conv1D(2 ** (init_power + 1), kernel_size, activation='relu', padding='same')(u6)
    c6 = Dropout(dropout)(c6)
    c6 = Conv1D(2 ** (init_power + 1), kernel_size, activation='relu', padding='same')(c6)

    u9 = UpSampling1D(2)(c6)
    c9 = Conv1D(2 ** init_power, kernel_size, activation='relu', padding='same')(u9)
    c9 = Dropout(dropout)(c9)
    c9 = Conv1D(2 ** init_power, kernel_size, activation='relu', padding='same')(c9)
    outputs = Conv1D(5, 1, activation='softmax')(c9)
    # outputs = Lambda(func, output_shape = [1]) (outputs)
    model = Model(inputs=[inputs], outputs=[outputs])
    print(model.summary())
    model.compile(loss='categorical_crossentropy', optimizer=SGD(lr=0.04), metrics=['acc', 'categorical_crossentropy'])

    return model

def create_fcn__multiple_heads(input_size, init_power=4, kernel_size=(3,3,3), dropout=0.3):
    # Build U-Net model
    inputs = Input(input_size)
    c1_a = Conv1D(2 ** init_power, kernel_size[0], activation='relu', padding='same')(inputs)
    c1_a = Dropout(dropout)(c1_a)
    c1_a = Conv1D(2 ** init_power,  kernel_size[0], activation='relu', padding='same')(c1_a)
    p1_a = MaxPooling1D(2)(c1_a)
    c4_a = Conv1D(2 ** (init_power + 1),  kernel_size[0], activation='relu', padding='same')(p1_a)
    c4_a = Dropout(dropout)(c4_a)
    c4_a = Conv1D(2 ** (init_power + 1),  kernel_size[0], activation='relu', padding='same')(c4_a)
    p4_a = MaxPooling1D(2)(c4_a)
    c5_a = Conv1D(2 ** (init_power + 2),  kernel_size[0], activation='relu', padding='same')(p4_a)
    c5_a = Dropout(dropout)(c5_a)
    c5_a = Conv1D(2 ** (init_power + 2),  kernel_size[0], activation='relu', padding='same')(c5_a)
    u6_a = UpSampling1D(2)(c5_a)
    c6_a = Conv1D(2 ** (init_power + 1),  kernel_size[0], activation='relu', padding='same')(u6_a)
    c6_a = Dropout(dropout)(c6_a)
    c6_a = Conv1D(2 ** (init_power + 1),  kernel_size[0], activation='relu', padding='same')(c6_a)
    u9_a = UpSampling1D(2)(c6_a)
    c9_a = Conv1D(2 ** init_power,  kernel_size[0], activation='relu', padding='same')(u9_a)
    c9_a = Dropout(dropout)(c9_a)
    c9_a = Conv1D(2 ** init_power,  kernel_size[0], activation='relu', padding='same')(c9_a)
    outputs_a = Conv1D(5, 1, activation='softmax')(c9_a)

    c1_b = Conv1D(2 ** init_power,  kernel_size[1], activation='relu', padding='same')(inputs)
    c1_b = Dropout(dropout)(c1_b)
    c1_b = Conv1D(2 ** init_power, kernel_size[1], activation='relu', padding='same')(c1_b)
    p1_b = MaxPooling1D(2)(c1_b)
    c4_b = Conv1D(2 ** (init_power + 1), kernel_size[1], activation='relu', padding='same')(p1_b)
    c4_b = Dropout(dropout)(c4_b)
    c4_b = Conv1D(2 ** (init_power + 1), kernel_size[1], activation='relu', padding='same')(c4_b)
    p4_b = MaxPooling1D(2)(c4_b)
    c5_b = Conv1D(2 ** (init_power + 2), kernel_size[1], activation='relu', padding='same')(p4_b)
    c5_b = Dropout(dropout)(c5_b)
    c5_b = Conv1D(2 ** (init_power + 2), kernel_size[1], activation='relu', padding='same')(c5_b)
    u6_b = UpSampling1D(2)(c5_b)
    c6_b = Conv1D(2 ** (init_power + 1), kernel_size[1], activation='relu', padding='same')(u6_b)
    c6_b = Dropout(dropout)(c6_b)
    c6_b = Conv1D(2 ** (init_power + 1), kernel_size[1], activation='relu', padding='same')(c6_b)
    u9_b = UpSampling1D(2)(c6_b)
    c9_b = Conv1D(2 ** init_power, kernel_size[1], activation='relu', padding='same')(u9_b)
    c9_b = Dropout(dropout)(c9_b)
    c9_b = Conv1D(2 ** init_power, kernel_size[1], activation='relu', padding='same')(c9_b)
    outputs_b = Conv1D(5, 1, activation='softmax')(c9_b)

    c1_c = Conv1D(2 ** init_power, kernel_size[2], activation='relu', padding='same')(inputs)
    c1_c = Dropout(dropout)(c1_c)
    c1_c = Conv1D(2 ** init_power, kernel_size[2], activation='relu', padding='same')(c1_c)
    p1_c = MaxPooling1D(2)(c1_c)
    c4_c = Conv1D(2 ** (init_power + 1), kernel_size[2], activation='relu', padding='same')(p1_c)
    c4_c = Dropout(dropout)(c4_c)
    c4_c = Conv1D(2 ** (init_power + 1), kernel_size[2], activation='relu', padding='same')(c4_c)
    p4_c = MaxPooling1D(2)(c4_c)
    c5_c = Conv1D(2 ** (init_power + 2), kernel_size[2], activation='relu', padding='same')(p4_c)
    c5_c = Dropout(dropout)(c5_c)
    c5_c = Conv1D(2 ** (init_power + 2), kernel_size[2], activation='relu', padding='same')(c5_c)
    u6_c = UpSampling1D(2)(c5_c)
    c6_c = Conv1D(2 ** (init_power + 1), kernel_size[2], activation='relu', padding='same')(u6_c)
    c6_c = Dropout(dropout)(c6_c)
    c6_c = Conv1D(2 ** (init_power + 1), kernel_size[2], activation='relu', padding='same')(c6_c)
    u9_c = UpSampling1D(2)(c6_c)
    c9_c = Conv1D(2 ** init_power, kernel_size[2], activation='relu', padding='same')(u9_c)
    c9_c = Dropout(dropout)(c9_c)
    c9_c = Conv1D(2 ** init_power, kernel_size[2], activation='relu', padding='same')(c9_c)
    outputs_c = Conv1D(5, 1, activation='softmax')(c9_c)
    final = Average()([outputs_a, outputs_b, outputs_c])

    # outputs = Lambda(func, output_shape = [1]) (outputs)
    model = Model(inputs=[inputs], outputs=[final])
    print(model.summary())

    return model



def create_unet_bidirectRNN_extra_step(input_size, init_power=4, kernel_size=3, dropout=0.3):
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
    p4 = MaxPooling1D(2)(c4)
    # Bottleneck
    c5 = Conv1D(2 ** (init_power + 4), kernel_size, activation='relu', padding='same')(p4)
    c5 = Dropout(dropout)(c5)
    c5 = Conv1D(2 ** (init_power + 4), kernel_size, activation='relu', padding='same')(c5)

    u6 = UpSampling1D(2)(c5)
    u6 = concatenate([u6, c4])
    c6 = Conv1D(2 ** (init_power + 3), kernel_size, activation='relu', padding='same')(u6)
    c6 = Dropout(dropout)(c6)
    c6 = Conv1D(2 ** (init_power + 3), kernel_size, activation='relu', padding='same')(c6)

    u7 = UpSampling1D(2)(c6)
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
    outputs = Bidirectional(CuDNNLSTM(64,return_sequences=True))(c9)
    outputs = ReLU()(outputs)
    outputs = Conv1D(5, 1, activation='softmax')(outputs)
    model = Model(inputs=[inputs], outputs=[outputs])
    print(model.summary())
    model.compile(loss='categorical_crossentropy', optimizer=SGD(lr=0.04), metrics=['acc', 'categorical_crossentropy'])

    return model


def encoder(x, filters=44, n_block=4, kernel_size=5, activation='relu',dropout=0.1):
    skip = []
    for i in range(n_block):
        x = Conv1D(filters * 2**i, kernel_size, activation=activation, padding='same')(x)
        x = Dropout(dropout)(x)
        x = Conv1D(filters * 2**i, kernel_size, activation=activation, padding='same')(x)
        skip.append(x)
        x = MaxPooling1D(2)(x)
    return x, skip


def bottleneck(x, filters_bottleneck, mode='cascade', depth=6,
               kernel_size=5, activation='relu'):
    dilated_layers = []
    if mode == 'cascade':  # used in the competition
        for i in range(depth):
            x = Conv1D(filters_bottleneck, kernel_size,
                       activation=activation, padding='same', dilation_rate=2**i)(x)
            dilated_layers.append(x)
        return add(dilated_layers)
    elif mode == 'parallel':  # Like "Atrous Spatial Pyramid Pooling"
        for i in range(depth):
            dilated_layers.append(
                Conv1D(filters_bottleneck, kernel_size,
                       activation=activation, padding='same', dilation_rate=2**i)(x)
            )
        return add(dilated_layers)


def decoder(x, skip, filters, n_block=3, kernel_size=5, activation='relu',dropout=0.1):
    for i in reversed(range(n_block)):
        x = UpSampling1D(2)(x)
        x = Conv1D(filters * 2**i, kernel_size, activation=activation, padding='same')(x)
        x = concatenate([skip[i], x])
        x = Conv1D(filters * 2**i, kernel_size, activation=activation, padding='same')(x)
        x = Dropout(dropout)(x)
        x = Conv1D(filters * 2**i, kernel_size, activation=activation, padding='same')(x)
    return x


def get_dilated_unet(
        input_shape=(1104,3),
        mode='cascade',
        filters=32,
        n_block=3,
        kernel_size =5,
        dropout=0.1,
        depth=6

):
    inputs = Input(input_shape)

    enc, skip = encoder(inputs, filters, n_block,kernel_size=kernel_size,dropout=dropout)
    bottle = bottleneck(enc, filters_bottleneck=filters * 2 ** n_block, mode=mode,kernel_size=kernel_size,depth=depth)
    dec = decoder(bottle, skip, filters, n_block,kernel_size=kernel_size,dropout=dropout)
    classify = Conv1D(5, 1, activation='softmax')(dec)

    model = Model(inputs=inputs, outputs=classify)
    print(model.summary())
    model.compile(loss='categorical_crossentropy', optimizer=SGD(lr=0.04), metrics=['acc', 'categorical_crossentropy'])

    return model