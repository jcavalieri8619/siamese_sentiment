"""
Created by John P Cavalieri

"""
from __future__ import print_function

import datetime
import os

from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.engine.topology import merge
from keras.layers import Dense
from keras.layers import Input
from keras.models import Model

import modelParameters
from CNN_model import save_CNNmodel_specs, build_CNN_model
from loss_functions import contrastiveLoss
from siamese_activations import vectorDifference, squaredl2

USEWORDS = True

if USEWORDS:
    VocabSize = modelParameters.VocabSize_w
    maxReviewLen = modelParameters.MaxLen_w
    skipTop = modelParameters.skip_top
else:
    VocabSize = modelParameters.VocabSize_c
    maxReviewLen = modelParameters.MaxLen_c
    skipTop = 0

DEVSPLIT = modelParameters.devset_split

basename = "siamese"
suffix = datetime.datetime.now().strftime("%m%d_%I%M")
filename = "_".join([basename, suffix])

batch_size = 20
num_epochs = 4

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


def build_siamese_model(inputType, do_training=False, model_inputs=None, weight_path=None, verbose=True, **kwargs):
    """

    :param inputType:
    :param do_training:
    :param model_inputs:
    :param weight_path:
    :param verbose:
    :param kwargs:
    :return:
    """

    def merged_outshape(inputShapes):
        """
        dynamically computes output shape of merge layer
        :param inputShapes: list of input shapes
        :return:
        """
        shape = list(inputShapes)
        return shape[0]

    if verbose:
        print('Building siamese model')

    CNN_model = build_CNN_model(inputType=inputType, is_IntermediateModel=True)

    Lreview = Input(shape=(maxReviewLen,), dtype='int32', name="Lreview")
    Rreview = Input(shape=(maxReviewLen,), dtype='int32', name="Rreview")

    rightbranch = CNN_model([Lreview])
    leftbranch = CNN_model([Rreview])

    # first take the difference of the final feature representations from the CNN_model
    # represented by leftbranch and rightbranch
    merged_vector = merge([leftbranch, rightbranch], mode=vectorDifference, output_shape=merged_outshape,
                          name='merged_vector')

    # then that difference vector is fed into the final fully connected layer that
    # outputs the energy i.e. squared euclidian distance ||leftbranch-rightbranch||
    energy = Dense(1, activation=squaredl2,
                   name='energy_output')(merged_vector)

    siamese_model = Model(input=[Lreview, Rreview], output=[energy],
                          name="siamese_model")

    siamese_model.compile(optimizer='adam', loss=contrastiveLoss, )

    models = {'siamese': siamese_model, 'CNN': CNN_model}

    if weight_path:
        siamese_model.load_weights(weight_path)

    if do_training:
        X_left, y_left, X_right, y_right, similarity = model_inputs['training']

        Xdev_left, ydev_left, Xdev_right, ydev_right, dev_similarity = model_inputs['dev']

        weightPath = os.path.join(modelParameters.WEIGHT_PATH, filename)

        checkpoint = ModelCheckpoint(weightPath + '_W.{epoch:02d}-{val_loss:.4f}.hdf5',
                                     verbose=1, monitor='val_loss', save_best_only=True)

        earlyStop = EarlyStopping(patience=1, verbose=1, monitor='val_loss')

        call_backs = [checkpoint, earlyStop]

        hist = siamese_model.fit({'Lreview': X_left, 'Rreview': X_right, },
                                 {'energy_output': similarity},
                                 batch_size=batch_size,
                                 nb_epoch=num_epochs,
                                 verbose=1,
                                 validation_data=
                                 ({'Lreview': Xdev_left, 'Rreview': Xdev_right, },
                                  {'energy_output': dev_similarity}),
                                 callbacks=call_backs)

        return models, hist

    return models


def save_siameseModel_specs(models, hist, verbose=True, **kwargs):
    """

    :param models:
    :param hist:
    :param verbose:
    :param kwargs:
    :return:
    """
    if verbose:
        print("saving siamese model specs")
    save_CNNmodel_specs(models['CNN'], hist, kwargs)

    with open(os.path.join(modelParameters.SPECS_PATH, filename + '.json'), 'w') as f:
        f.write(models['siamese'].to_json())

    with open(os.path.join(modelParameters.SPECS_PATH, filename) + '.hist', 'w') as f:
        f.write(str(hist.history))

    with open(os.path.join(modelParameters.SPECS_PATH, filename) + '.config', 'w') as f:
        f.write(str(models['siamese'].get_config()))
