from __future__ import print_function

import datetime
import os

from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras.engine.training import Model
from keras.initializers import RandomUniform
from keras.layers import Convolution1D, MaxPooling1D
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Input
from keras.layers import SpatialDropout1D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.embeddings import Embedding
from keras.metrics import binary_accuracy
from keras.optimizers import Adam
from keras.regularizers import l1, l2

import modelParameters

basename = "CNN"
suffix = datetime.datetime.now().strftime("%m%d_%I%M")
filename = "_".join([basename, suffix])

filename_config = "_".join([basename, "config", suffix])

batch_size = 35
nb_epoch = 2
region = 'same'

leaky_relu = LeakyReLU(alpha=0.13)

embedding_dims = 50
embedding_init = RandomUniform()
embedding_reg = l1(0.0)

convL2 = 0.001

num_filters1 = 250  # 1000
filter_length1 = 5
pool_len1 = 2
conv_init1 = "glorot_uniform"
conv_activation1 = 'relu'
conv_reg1 = l2(convL2)

num_filters2 = 200  # 800
filter_length2 = 3
pool_len2 = 2
conv_init2 = "glorot_uniform"
conv_activation2 = 'relu'
conv_reg2 = l2(convL2)

num_filters3 = 100  # 300
filter_length3 = 3
pool_len3 = 2
conv_init3 = "glorot_uniform"
conv_activation3 = 'relu'
conv_reg3 = l2(convL2)

num_filters4 = 500  # 300
filter_length4 = 4
pool_len4 = 2
conv_init4 = "glorot_uniform"
conv_activation4 = 'relu'
conv_reg4 = l2(convL2)

denseL2 = 0.001

dense_dims0 = 250  # 600
dense_activation0 = 'relu'
dense_init0 = "glorot_normal"
dense_reg0 = l2(denseL2)

dense_dims1 = 100  # 500
dense_activation1 = 'relu'
dense_init1 = "glorot_normal"
dense_reg1 = l2(denseL2)

dense_dims2 = 100  # 400
dense_activation2 = 'relu'
dense_init2 = "glorot_normal"
dense_reg2 = l2(denseL2)

dense_dims3 = 100
dense_activation3 = 'relu'
dense_init3 = "glorot_normal"
dense_reg3 = l2(denseL2)

dense_dims_final = 1
dense_activation_final = 'sigmoid'
dense_init_final = "glorot_normal"


# dense_l2_reg_final='N/A'


def build_CNN_model(inputType, do_training=False, model_inputs=None, loss_func='binary_crossentropy',
                    optimize_proc='adam', is_IntermediateModel=False, load_weight_path=None, **kwargs):
    """

    :param inputType:
    :param do_training:
    :param model_inputs:
    :param loss_func:
    :param optimize_proc:
    :param is_IntermediateModel:
    :param load_weight_path:
    :param kwargs:
    :return:
    """

    # assert not do_training and model_inputs, "if do_training then must pass in model_inputs dictionary"

    EMBEDDING_TYPE = 'embeddingMatrix'
    ONEHOT_TYPE = '1hotVector'

    defined_input_types = {EMBEDDING_TYPE, ONEHOT_TYPE}

    assert inputType in defined_input_types, "unknown input type {0}".format(inputType)

    if inputType is ONEHOT_TYPE:

        review_input = Input(shape=(modelParameters.MaxLen_w,), dtype='float32',
                             name="ONEHOT_INPUT")

        layer = Embedding(modelParameters.VocabSize_w + modelParameters.INDEX_FROM, embedding_dims,
                          embeddings_initializer=embedding_init, embeddings_regularizer=embedding_reg,
                          input_length=modelParameters.MaxLen_w, name='1hot_embeddingLayer')(review_input)

        layer = SpatialDropout1D(0.50)(layer)

    elif inputType is EMBEDDING_TYPE:
        review_input = Input(shape=(modelParameters.MaxLen_w, embedding_dims), dtype="float32", name="EMBEDDING_INPUT")
        layer = review_input

    else:
        raise ValueError("Bad inputType arg to build_CNN_model")

    layer = Convolution1D(filters=num_filters1,
                          kernel_size=filter_length1,
                          padding=region,
                          strides=1,
                          activation=conv_activation1,
                          kernel_initializer='glorot_uniform',
                          bias_initializer='zeros',
                          kernel_regularizer=conv_reg1,
                          dilation_rate=1,
                          name='ConvLayer1')(layer)

    layer = SpatialDropout1D(0.50)(layer)

    layer = MaxPooling1D(pool_size=pool_len1)(layer)

    # layer = Convolution1D(filters=num_filters2,
    #                       kernel_size=filter_length2,
    #                       padding=region,
    #                       strides=1,
    #                       activation=conv_activation2,
    #                       kernel_initializer=conv_init2,
    #                       kernel_regularizer=conv_reg2,
    #                       dilation_rate=1,
    #                       name='ConvLayer2')(layer)
    #
    # layer = SpatialDropout1D(0.50)(layer)
    #
    # layer = MaxPooling1D(pool_size=pool_len2)(layer)

    # layer = Convolution1D(filters=num_filters3,
    #                       kernel_size=filter_length3,
    #                       padding=region,
    #                       activation=conv_activation3,
    #                       kernel_initializer=conv_init3,
    #                       kernel_regularizer=conv_reg3,
    #                       dilation_rate=1,
    #                       name='ConvLayer3')(layer)
    #
    # layer = SpatialDropout1D(0.50)(layer)
    #
    # layer = MaxPooling1D(pool_size=pool_len3)(layer)



    # #layer = GlobalMaxPool1D()(layer)
    #
    # layer = Convolution1D(filters=num_filters4,
    #                       kernel_size=filter_length4,
    #                       padding=region,
    #                       activation=conv_activation4,
    #                       kernel_initializer=conv_init4,
    #                       kernel_regularizer=conv_reg4,
    #                       dilation_rate=1,
    #                       name='ConvLayer4')(layer)
    #
    # #layer = leaky_relu(layer)
    #
    # layer = SpatialDropout1D(0.50)(layer)
    #
    # layer = MaxPooling1D(pool_size=pool_len4)(layer)
    # #layer = GlobalMaxPool1D()(layer)
    #
    # # layer = BatchNormalization()(layer)

    layer = Flatten()(layer)

    layer = Dense(dense_dims0, activation=dense_activation0, kernel_regularizer=dense_reg0,
                  kernel_initializer='glorot_normal', bias_initializer='zeros',
                  name='dense0')(layer)

    layer = Dropout(0.50)(layer)

    layer = Dense(dense_dims1, activation=dense_activation1, kernel_regularizer=dense_reg1,
                  kernel_initializer='glorot_normal', bias_initializer='zeros',
                  name='dense1')(layer)

    layer = Dropout(0.50)(layer)

    # layer = Dense(dense_dims2, activation=dense_activation2, kernel_regularizer=dense_reg2,
    #               kernel_initializer=dense_init2,
    #               name='dense2')(layer)
    #
    #
    # layer = Dropout(0.50)(layer)
    #
    # layer = Dense(dense_dims3, activation=dense_activation3, kernel_regularizer=dense_reg3,
    #               kernel_initializer=dense_init3,
    #               name='dense3_outA')(layer)
    # #layer = leaky_relu(layer)
    #
    if is_IntermediateModel:
        return Model(inputs=[review_input], outputs=[layer], name="CNN_model")

    #
    # layer = Dropout(0.5)(layer)

    layer = Dense(dense_dims_final, activation=dense_activation_final, kernel_initializer=dense_init_final,
                  kernel_regularizer=dense_reg0,
                  name='output_Full')(layer)

    CNN_model = Model(inputs=[review_input], outputs=[layer], name="CNN_model")

    CNN_model.compile(optimizer=Adam(lr=0.001, decay=0.0), loss=loss_func, metrics=[binary_accuracy])

    if load_weight_path is not None:
        CNN_model.load_weights(load_weight_path)

    hist = ""
    if do_training:
        weightPath = os.path.join(modelParameters.WEIGHT_PATH, filename)
        configPath = os.path.join(modelParameters.WEIGHT_PATH, filename_config)

        with open(configPath + ".json", 'wb') as f:
            f.write(CNN_model.to_json())

        checkpoint = ModelCheckpoint(weightPath + '_W.{epoch:02d}-{val_loss:.4f}.hdf5',
                                     verbose=1, save_best_only=True, save_weights_only=False, monitor='val_loss')

        earlyStop = EarlyStopping(patience=3, verbose=1, monitor='val_loss')

        LRadjuster = ReduceLROnPlateau(monitor='val_loss', factor=0.30, patience=0, verbose=1, cooldown=1,
                                       min_lr=0.00001, epsilon=1e-2)

        call_backs = [checkpoint, earlyStop, LRadjuster]

        CNN_model.summary()

        hist = CNN_model.fit(*model_inputs['training'],
                             batch_size=batch_size,
                             epochs=nb_epoch, verbose=1,
                             validation_data=model_inputs['dev'],
                             callbacks=call_backs)

    return {"model": CNN_model, "hist": hist}


def save_CNNmodel_specs(model, hist, **kwargs):
    with open(os.path.join(modelParameters.SPECS_PATH, filename) + '.config', 'w') as f:
        f.write(str(model.get_config()))

    with open(os.path.join(modelParameters.SPECS_PATH, filename + '.json'), 'w') as f:
        f.write(model.to_json())

    with open(os.path.join(modelParameters.SPECS_PATH, filename) + '.hist', 'w') as f:
        f.write(str(hist.history))

    with open(os.path.join(modelParameters.SPECS_PATH, filename) + '.specs', 'w') as f:
        specs = """model: {}\nborder_mode: {}\nbatch_size: {}\nembedding_dims: {}\n
                num_filters1: {}\nfilter_length1: {}\npool_len1: {}\n
                num_filters2: {}\nfilter_length2: {}\npool_len2: {}\n
                num_filters3: {}\nfilter_length3: {}\npool_len3: {}\n
                num_filters4: {}\nfilter_length4: {}\npool_len4: {}\n
                dense_dims1: {}\ndense_dims2: {}\ndense_dims3: {}\n
                {moreArgs}\n""".format(basename,
                                       region,
                                       batch_size,
                                       embedding_dims,
                                       num_filters1,
                                       filter_length1,
                                       pool_len1,
                                       num_filters2,
                                       filter_length2,
                                       pool_len2,
                                       num_filters3,
                                       filter_length3,
                                       pool_len3,
                                       "NA",  # num_filters4,
                                       "NA",  # filter_length4,
                                       "NA",  # pool_len4,
                                       dense_dims1,
                                       dense_dims2,
                                       "NA",  # dense_dims3,
                                       moreArgs=kwargs
                                       )
        f.write(specs)


def cnn_test():
    from model_input_builders import build_CNN_input

    modelinputs = build_CNN_input(truncate='pre', padding='post', DEBUG=False)
    rv = build_CNN_model('1hotVector', True, modelinputs)
    return modelinputs, rv


if __name__ is "__main__":
    # modelinputs = build_CNN_input(truncate='pre', padding='post', DEBUG=False)
    # build_CNN_model('1hotVector', True,modelinputs)
    print("IN MAIN")
