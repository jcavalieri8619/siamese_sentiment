from __future__ import print_function

import datetime
import os

from keras.activations import sigmoid
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers import Input
from keras.layers.convolutional import Convolution1D, MaxPooling1D
from keras.layers.core import (Dense, Dropout, Flatten)
from keras.layers.embeddings import Embedding
from keras.models import Model
from keras.objectives import binary_crossentropy

import modelParameters
from convert_review import build_design_matrix

basename = "CNN_"
suffix = datetime.datetime.now().strftime("%m%d_%I%M")
filename = "_".join([basename, suffix])

disconnected = 'disconnected_layer'

batch_size = 80
region = 'same'

num_filters1 = 1300
filter_length1 = 2
stride_len1 = 1
pool_len1 = 2

num_filters2 = 800
filter_length2 = 3
stride_len2 = 1
pool_len2 = 2

num_filters3 = 500
filter_length3 = 4
stride_len3 = 1
pool_len3 = 2

num_filters4 = 300
filter_length4 = 5
stride_len4 = 1
pool_len4 = 2

embedding_dims = 200

dense_dims1 = 1250
dense_dims2 = 800
dense_dims3 = 250
num_epochs = 4


def build_CNN_input(vocab_size=modelParameters.VocabSize_w, usewords=True,
                    skiptop=modelParameters.skip_top,
                    devsplit=modelParameters.devset_split, verbose=True, **kwargs):
    """

    :param vocab_size:
    :param usewords:
    :param skiptop:
    :param devsplit:
    :param verbose:
    :param kwargs:
    :return:
    """

    if verbose:
        print('Building CNN Inputs')

    createValSet = kwargs.get('createValSet', True)
    DEBUG = kwargs.get("DEBUG", None)

    ((X_train, y_train), (X_dev, y_dev), (X_val, y_val)) = build_design_matrix(vocab_size=vocab_size,
                                                                               use_words=usewords,
                                                                               skip_top=skiptop,
                                                                               dev_split=devsplit,
                                                                               createValidationSet=createValSet,
                                                                               DEBUG=DEBUG
                                                                               )
    if verbose:
        print('X_train shape: {}'.format(X_train.shape))
        print('X_dev shape: {}'.format(X_dev.shape))
        print('X_test shape: {}'.format(X_val.shape))
        print('y_train shape: {}'.format(y_train.shape))
        print('y_dev shape: {}'.format(y_dev.shape))
        print('y_test shape: {}'.format(y_val.shape))

    return X_train, y_train, X_dev, y_dev, X_val, y_val


def build_CNN_model(inputType, loss_func=None, optimize_proc=None,
                    is_IntermediateModel=False, weight_path=None, **kwargs):
    """

    :param inputType:
    :param loss_func:
    :param optimize_proc:
    :param is_IntermediateModel:
    :param weight_path:
    :param kwargs:
    :return:
    """

    EMBEDDING_TYPE = 'embeddingMatrix'
    ONEHOT_TYPE = '1hotVector'

    defined_input_types = {EMBEDDING_TYPE, ONEHOT_TYPE}

    assert inputType in defined_input_types, "unknown input type"

    if inputType is ONEHOT_TYPE:

        review_input = Input(shape=(modelParameters.MaxLen_w,), dtype='int32', name="1hot_review")

        sharedEmbedding = Embedding(modelParameters.VocabSize_w + 3, embedding_dims,
                                    input_length=modelParameters.MaxLen_w, name='embeddingLayer')

        layer = sharedEmbedding(review_input)

    else:
        review_input = Input(shape=(modelParameters.MaxLen_w, embedding_dims), dtype="float32", name="embedding_review")
        layer = review_input

    sharedConv1 = Convolution1D(nb_filter=num_filters1,
                                filter_length=filter_length1,
                                border_mode=region,
                                activation='relu',
                                init='uniform',
                                subsample_length=stride_len1,
                                input_length=modelParameters.MaxLen_w,
                                input_shape=(modelParameters.MaxLen_w, embedding_dims),
                                name='ConvLayer1')

    layer = sharedConv1(layer, )

    layer = Dropout(0.25, )(layer)

    layer = MaxPooling1D()(layer)

    sharedConv2 = Convolution1D(nb_filter=num_filters2,
                                filter_length=filter_length2,
                                border_mode=region,
                                activation='relu',
                                init='uniform',
                                subsample_length=stride_len2,
                                name='ConvLayer2'
                                )

    layer = sharedConv2(layer, )

    layer = Dropout(0.30)(layer)

    layer = MaxPooling1D()(layer)

    sharedConv3 = Convolution1D(nb_filter=num_filters3,
                                filter_length=num_filters3,
                                border_mode=region,
                                activation='relu',
                                init='uniform',
                                subsample_length=stride_len3,
                                name='ConvLayer3'
                                )

    layer = sharedConv3(layer, )

    layer = Dropout(0.35)(layer)

    layer = MaxPooling1D()(layer)

    sharedConv4 = Convolution1D(nb_filter=num_filters4,
                                filter_length=filter_length4,
                                border_mode=region,
                                activation='relu',
                                init='uniform',
                                subsample_length=stride_len4,
                                name='ConvLayer4',

                                )

    layer = sharedConv4(layer, )

    layer = Dropout(0.35)(layer)

    layer = MaxPooling1D()(layer)

    layer = Flatten()(layer)

    sharedDense1 = Dense(dense_dims1, activation='relu',
                         name='denseLayer1', )

    layer = sharedDense1(layer)

    layer = Dropout(0.35)(layer)

    sharedDense2 = Dense(dense_dims2, activation='relu',
                         name='denseLayer2')
    layer = sharedDense2(layer)

    layer = Dropout(0.35)(layer)

    sharedDense3 = Dense(dense_dims3, activation='relu',
                         name='dense2_outputA')

    out_A = sharedDense3(layer, )

    if is_IntermediateModel:
        CNN_model = Model(input=[review_input], output=out_A, name="CNN_model")
        return CNN_model

    out_A = Dropout(0.20)(out_A)

    sidmoidLayer = Dense(1, activation=sigmoid,
                         name='output_B')

    out_B = sidmoidLayer(out_A, )

    CNN_model = Model(input=[review_input], output=[out_B], name="CNN_model")

    loss = loss_func if loss_func else binary_crossentropy
    optimizer = optimize_proc if optimize_proc else 'adam'

    CNN_model.compile(optimizer=optimizer, loss=loss)

    if weight_path is not None:
        CNN_model.load_weights(weight_path)

    return CNN_model


def train_CNN_model(model, X_train, y_train, X_dev, y_dev):
    """

    :param model:
    :param X_train:
    :param y_train:
    :param X_dev:
    :param y_dev:
    :return:
    """
    weightPath = os.path.join(modelParameters.WEIGHT_PATH, filename)
    checkpoint = ModelCheckpoint(weightPath + '_W.{epoch:02d}-{val_loss:.3f}.hdf5',
                                 verbose=1, save_best_only=True, monitor='val_loss')
    earlyStop = EarlyStopping(patience=1, verbose=1, monitor='val_loss')

    call_backs = [checkpoint, earlyStop]

    hist = model.fit(X_train, y_train, batch_size=batch_size,
                     nb_epoch=num_epochs, verbose=1,
                     validation_data=(X_dev, y_dev),
                     callbacks=call_backs)

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
                dense_dims1: {}\ndense_dims2: {}\ndense_dims3: {}\n""".format(basename,
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
                                                                              num_filters4,
                                                                              filter_length4,
                                                                              pool_len4,
                                                                              dense_dims1,
                                                                              dense_dims2,
                                                                              dense_dims3
                                                                              )
        f.write(specs)
