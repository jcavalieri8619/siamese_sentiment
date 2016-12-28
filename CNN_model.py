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
from convert_review import build_design_matrix

basename = "CNN_"
suffix = datetime.datetime.now().strftime("%m%d_%I%M")
filename = "_".join([basename, suffix])

USE_WORDS = True

if USE_WORDS:
    skipTop = modelParameters.skip_top
    vocabSize = modelParameters.VocabSize_w
    maxlen = modelParameters.MaxLen_w
else:
    skipTop = 0
    vocabSize = modelParameters.VocabSize_c
    maxlen = modelParameters.MaxLen_c

devSplit = modelParameters.devset_split

batch_size = 20
nb_epoch = 5
region = 'same'

embedding_dims = 200

num_filters1 = 1300
filter_length1 = 2
pool_len1 = 2

num_filters2 = 800
filter_length2 = 3
pool_len2 = 2

num_filters3 = 500
filter_length3 = 4
pool_len3 = 2

num_filters4 = 300
filter_length4 = 5
pool_len4 = 2

dense_dims1 = 1250
dense_dims2 = 800
dense_dims3 = 250


def build_CNN_input(vocab_size=vocabSize, usewords=USE_WORDS,
                    skiptop=skipTop, devsplit=devSplit, verbose=True, **kwargs):
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

    ((X_train, y_train), (X_dev, y_dev), (X_val, y_val)) = build_design_matrix(vocab_size=vocab_size,
                                                                               use_words=usewords,
                                                                               skip_top=skiptop,
                                                                               dev_split=devsplit,
                                                                               **kwargs
                                                                               )

    if verbose:
        print('X_train shape: {}'.format(X_train.shape))
        print('X_dev shape: {}'.format(X_dev.shape))
        print('X_test shape: {}'.format(X_val.shape))
        print('y_train shape: {}'.format(y_train.shape))
        print('y_dev shape: {}'.format(y_dev.shape))
        print('y_test shape: {}'.format(y_val.shape))

    return {'training': (X_train, y_train), "dev": (X_dev, y_dev), "val": (X_val, y_val)}


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

    assert inputType in defined_input_types, "unknown input type"

    if inputType is ONEHOT_TYPE:

        review_input = Input(shape=(modelParameters.MaxLen_w,), dtype='int32', name="1hot_review")

        layer = Embedding(modelParameters.VocabSize_w + modelParameters.INDEX_FROM, embedding_dims,
                          input_length=modelParameters.MaxLen_w, name='embeddingLayer')(review_input)

    else:
        review_input = Input(shape=(modelParameters.MaxLen_w, embedding_dims), dtype="float32", name="embedding_review")
        layer = review_input

    layer = Convolution1D(nb_filter=num_filters1,
                          filter_length=filter_length1,
                          border_mode=region,
                          activation='relu',
                          subsample_length=1,
                          name='ConvLayer1')(layer, )

    layer = Dropout(0.25, )(layer)

    layer = MaxPooling1D(pool_length=pool_len1)(layer)

    layer = Convolution1D(nb_filter=num_filters2,
                          filter_length=filter_length2,
                          border_mode=region,
                          activation='relu',
                          subsample_length=1,
                          name='ConvLayer2'
                          )(layer, )

    layer = Dropout(0.30)(layer)

    layer = MaxPooling1D(pool_length=pool_len2)(layer)

    layer = Convolution1D(nb_filter=num_filters3,
                          filter_length=num_filters3,
                          border_mode=region,
                          activation='relu',
                          subsample_length=1,
                          name='ConvLayer3'
                          )(layer, )

    layer = Dropout(0.35)(layer)

    layer = MaxPooling1D(pool_length=pool_len3)(layer)

    layer = Convolution1D(nb_filter=num_filters4,
                          filter_length=filter_length4,
                          border_mode=region,
                          activation='relu',
                          subsample_length=1,
                          name='ConvLayer4',
                          )(layer, )

    layer = Dropout(0.35)(layer)

    layer = MaxPooling1D(pool_length=pool_len4)(layer)

    layer = Flatten()(layer)

    layer = Dense(dense_dims1, activation='relu',
                  name='dense1')(layer)

    layer = Dropout(0.35)(layer)

    layer = Dense(dense_dims2, activation='relu',
                  name='dense2')(layer)

    layer = Dropout(0.35)(layer)

    out_A = Dense(dense_dims3, activation='relu',
                  name='dense3_outA')(layer)

    if is_IntermediateModel:
        CNN_model = Model(input=[review_input], output=[out_A], name="CNN_model")
        return CNN_model

    out_B = Dense(1, activation='sigmoid',
                  name='output_Full')(out_A, )

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

        return CNN_model, hist

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
                                       num_filters4,
                                       filter_length4,
                                       pool_len4,
                                       dense_dims1,
                                       dense_dims2,
                                       dense_dims3,
                                       moreArgs=kwargs
                                       )
        f.write(specs)
