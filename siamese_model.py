"""
Created by John P Cavalieri

"""
from __future__ import print_function

import datetime
import os

from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras.layers import Dense
from keras.layers import Input
from keras.layers.merge import _Merge
from keras.models import Model

import modelParameters
from CNN_model import save_CNNmodel_specs, build_CNN_model
from loss_functions import contrastiveLoss
from siamese_activations import squaredl2

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
num_epochs = 10


class Diff(_Merge):
    """Layer that takes difference of a list of inputs.
    It takes as input a list of tensors,
    all of the same shape, and returns
    a single tensor (also of the same shape).
    """

    def _merge_function(self, inputs):
        output = inputs[0]
        for i in range(1, len(inputs)):
            output -= inputs[i]
        return output


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

    Lreview = Input(shape=(maxReviewLen,), dtype='float32', name="Lreview")
    Rreview = Input(shape=(maxReviewLen,), dtype='float32', name="Rreview")

    rightbranch = CNN_model([Lreview])
    leftbranch = CNN_model([Rreview])

    # first take the difference of the final feature representations from the CNN_model
    # represented by leftbranch and rightbranch
    merged_vector = Diff()([leftbranch, rightbranch], name='merged_vector')

    # then that difference vector is fed into the final fully connected layer that
    # outputs the energy i.e. squared euclidian distance ||leftbranch-rightbranch||
    energy = Dense(1, activation=squaredl2, name='energy_output')(merged_vector)

    siamese_model = Model(inputs=[Lreview, Rreview], outputs=[energy],
                          name="siamese_model")

    siamese_model.compile(optimizer='adam', loss=contrastiveLoss, metrics=['acc'])

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

        LRadjuster = ReduceLROnPlateau(monitor='val_loss', factor=0.30, patience=0, verbose=1, cooldown=1,
                                       min_lr=0.00001, epsilon=1e-2)

        call_backs = [checkpoint, earlyStop, LRadjuster]

        siamese_model.summary()

        hist = siamese_model.fit({'Lreview': X_left, 'Rreview': X_right, },
                                 {'energy_output': similarity},
                                 batch_size=batch_size,
                                 epochs=num_epochs,
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
