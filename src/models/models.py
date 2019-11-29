from keras import Input, Model
from keras.layers import Conv1D, Dropout, MaxPooling1D, UpSampling1D, concatenate, Average
from keras.optimizers import SGD


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