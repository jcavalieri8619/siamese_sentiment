from __future__ import print_function

import datetime
import os

from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.engine.training import Model
from keras.layers import Convolution1D, MaxPooling1D
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Input
from keras.layers.embeddings import Embedding

import modelParameters

basename = "CNN_"
suffix = datetime.datetime.now().strftime("%m%d_%I%M")
filename = "_".join([basename, suffix])

batch_size = 60
nb_epoch = 8
region = 'same'

embedding_dims = 200
embedding_init = 'uniform'
embedding_l1_reg = 0.00000001

num_filters1 = 1000
filter_length1 = 2
pool_len1 = 2
conv_init1 = 'glorot_normal'
conv_activation1 = 'relu'
conv_l2_reg1 = 0.000001

num_filters2 = 800
filter_length2 = 3
pool_len2 = 2
conv_init2 = 'uniform'
conv_activation2 = 'relu'
conv_l2_reg2 = 0.000001

num_filters3 = 400
filter_length3 = 3
pool_len3 = 2
conv_init3 = 'glorot_normal'
conv_activation3 = 'relu'
conv_l2_reg3 = 0.000001

dense_dims1 = 800
dense_activation1 = 'relu'
dense_init1 = 'zero'
dense_l2_reg1 = 0.000001

dense_dims2 = 200
dense_activation2 = 'relu'
dense_init2 = 'glorot_uniform'
dense_l2_reg2 = 0.000001

dense_dims_final = 1
dense_activation_final = 'sigmoid'
dense_init_final = 'glorot_uniform'
# dense_l2_reg_final='N/A'

from model_input_builders import build_CNN_input


def build_CNN_model(inputType, do_training=False, model_inputs=None, loss_func='binary_crossentropy',
                    optimize_proc='adam', is_IntermediateModel=False, weight_path=None, **kwargs):
    """

    :param inputType:
    :param do_training:
    :param model_inputs:
    :param loss_func:
    :param optimize_proc:
    :param is_IntermediateModel:
    :param weight_path:
    :param kwargs:
    :return:
    """

    # assert not do_training and model_inputs, "if do_training then must pass in model_inputs dictionary"

    EMBEDDING_TYPE = 'embeddingMatrix'
    ONEHOT_TYPE = '1hotVector'

    defined_input_types = {EMBEDDING_TYPE, ONEHOT_TYPE}

    assert inputType in defined_input_types, "unknown input type {0}".format(inputType)

    if inputType is ONEHOT_TYPE:

        review_input = Input(shape=(modelParameters.MaxLen_w,), dtype='int32', name="ONEHOT_INPUT")

        layer = Embedding(modelParameters.VocabSize_w + modelParameters.INDEX_FROM, embedding_dims,
                          init=embedding_init, dropout=0.3,
                          input_length=modelParameters.MaxLen_w, name='1hot_embeddingLayer')(review_input)

    elif inputType is EMBEDDING_TYPE:
        review_input = Input(shape=(modelParameters.MaxLen_w, embedding_dims), dtype="float32", name="EMBEDDING_INPUT")
        layer = review_input

    else:
        raise ValueError("Bad inputType arg to build_CNN_model")

    layer = Convolution1D(nb_filter=num_filters1,
                          filter_length=filter_length1,
                          border_mode=region,
                          activation=conv_activation1,
                          init=conv_init1,
                          subsample_length=1,
                          name='ConvLayer1')(layer)

    layer = Dropout(0.25, )(layer)

    layer = MaxPooling1D(pool_length=pool_len1)(layer)

    layer = Convolution1D(nb_filter=num_filters2,
                          filter_length=filter_length2,
                          border_mode=region,
                          activation=conv_activation2,
                          init=conv_init2,
                          subsample_length=1,
                          name='ConvLayer2')(layer)

    layer = Dropout(0.30)(layer)

    layer = MaxPooling1D(pool_length=pool_len2)(layer)

    layer = Convolution1D(nb_filter=num_filters3,
                          filter_length=filter_length3,
                          border_mode=region,
                          activation=conv_activation3,
                          init=conv_init3,
                          subsample_length=1,
                          name='ConvLayer3')(layer)

    layer = Dropout(0.35)(layer)

    layer = MaxPooling1D(pool_length=pool_len3)(layer)

    layer = Flatten()(layer)

    layer = Dense(dense_dims1, activation=dense_activation1, init=dense_init1,
                  name='dense1')(layer)

    layer = Dropout(0.35)(layer)

    out_A = Dense(dense_dims2, activation=dense_activation2, init=dense_init2,
                  name='dense2_outA')(layer)

    if is_IntermediateModel:
        CNN_model = Model(input=[review_input], output=[out_A], name="CNN_model")
        return CNN_model

    out_A = Dropout(0.35)(out_A)

    out_B = Dense(dense_dims_final, activation=dense_activation_final, init=dense_init_final,
                  name='output_Full')(out_A)

    CNN_model = Model(input=[review_input], output=[out_B], name="CNN_model")

    CNN_model.compile(optimizer=optimize_proc, loss=loss_func)

    if weight_path is not None:
        CNN_model.load_weights(weight_path)

    if do_training:
        weightPath = os.path.join(modelParameters.WEIGHT_PATH, filename)
        checkpoint = ModelCheckpoint(weightPath + '_W.{epoch:02d}-{val_loss:.4f}.hdf5',
                                     verbose=1, save_best_only=True, monitor='val_loss')
        earlyStop = EarlyStopping(patience=1, verbose=1, monitor='val_loss')

        call_backs = [checkpoint, earlyStop]

        hist = CNN_model.fit(*model_inputs['training'],
                             batch_size=batch_size,
                             nb_epoch=nb_epoch, verbose=1,
                             validation_data=model_inputs['dev'],
                             callbacks=call_backs)

        return hist

    return CNN_model


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


##TESTING
modelinputs = build_CNN_input(truncate='pre', padding='post', DEBUG=False)
build_CNN_model('1hotVector', True, modelinputs)

if __name__ is "__main__":
    # modelinputs = build_CNN_input(truncate='pre', padding='post', DEBUG=False)
    # build_CNN_model('1hotVector', True,modelinputs)
    print("IN MAIN")
