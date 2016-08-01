"""
Created by John P Cavalieri

"""
from __future__ import print_function

import datetime
import os

import modelParameters
from CNN_model import build_CNN_model
from convert_review import construct_designmatrix_pairs
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.engine.topology import merge
from keras.layers import Input
from keras.layers.core import (Dense)
from keras.models import Model
from loss_functions import contrastiveLoss
from siamese_activations import vectorDifference, squaredl2

DEVSPLIT = modelParameters.devset_split
USEWORDS = True

if USEWORDS:
    VocabSize = modelParameters.VocabSize_w
    maxReviewLen = modelParameters.MaxLen_w
    skipTop = modelParameters.skip_top
else:
    VocabSize = modelParameters.VocabSize_c
    maxReviewLen = modelParameters.MaxLen_c
    skipTop = 0

basename = "siamese"
suffix = datetime.datetime.now().strftime("%m%d_%I%M")
filename = "_".join([basename, suffix])

batch_size = 20

num_epochs = 4


def build_siamese_input(verbose=True):
    """

    :param verbose:
    :return:
    """

    if verbose:
        print('building pairs of reviews for siamese model input')

    ((trainingSets), (devSets), (devKNNsets), (testSets)) = construct_designmatrix_pairs(VocabSize,
                                                                                         useWords=USEWORDS,
                                                                                         skipTop=skipTop,
                                                                                         devSplit=DEVSPLIT)

    if verbose:
        print(len(trainingSets[0]), 'train sequences length')
        print(len(devSets[0]), 'dev sequences length')

        print(len(devKNNsets[0]), 'devKNN sequences length')
        print(len(testSets[0]), 'test sequences length')

        print('train shape:', trainingSets[0].shape)
        print('dev shape:', devSets[0].shape)
        print('devKNN shape:', devKNNsets[0].shape)
        print('test shape:', testSets[0].shape)

    return {'training': trainingSets, 'dev': devSets, 'KNN': devKNNsets, 'test': testSets}


def build_siamese_model(weight_path=None, verbose=True, **kwargs):
    """

    :param weight_path: path to saved weights for loading entire siamese model
    :param verbose:
    :param kwargs: if you want to load CNN_model individially with saved weights then pass keyword arg
                CNN_weight_path=path/to/weights
    :return: dict {'siamese':siamese_model, 'CNN':CNN_model}
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

    CNN_model = build_CNN_model(inputType='1hotVector',
                                is_IntermediateModel=True,
                                weight_path=kwargs.get('CNN_weight_path', None))

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

    siamese_model = Model(input=[Lreview, Rreview], output=energy,
                          name="siamese_model")

    siamese_model.compile(optimizer='adam', loss=contrastiveLoss, )

    return {'siamese': siamese_model, 'CNN': CNN_model}


def train_siamese_model(model, trainingSets, devSets):
    """

    :param model:
    :param trainingSets:
    :param devSets:
    :return:
    """

    # X_left and X_right are matrices with trainingSet rows and reviewLen columns
    # y_left and y_right are the corresponding sentiment labels i.e 0:negative 1:positive
    # similarity is 0 if X_left and X_right have same sentiment labels and 1 otherwise
    X_left, y_left, X_right, y_right, similarity = trainingSets

    Xdev_left, ydev_left, Xdev_right, ydev_right, dev_similarity = devSets

    weightPath = os.path.join(modelParameters.WEIGHT_PATH, filename)

    checkpoint = ModelCheckpoint(weightPath + '_W.{epoch:02d}-{val_loss_fn:.2f}.hdf5',
                                 verbose=1, monitor='val_loss_fn', save_best_only=True)

    earlyStop = EarlyStopping(patience=1, verbose=1, monitor='val_loss_fn')

    call_backs = [checkpoint, earlyStop]

    # todo pass actual siamese model in as arg not the dictionary containing both models

    hist = model['siamese'].fit({'Lreview': X_left, 'Rreview': X_right,},
                                {'energy_output': similarity},
                                batch_size=batch_size,
                                nb_epoch=num_epochs,
                                verbose=1,
                                validation_data=
                                ({'Lreview': Xdev_left, 'Rreview': Xdev_right,},
                                 {'energy_output': dev_similarity}),
                                callbacks=call_backs)

    try:
        with open(os.path.join(modelParameters.SPECS_PATH, filename + '.json'), 'w') as f:
            f.write(model['siamese'].to_json())
    except:
        print("error writing model json")
        pass

    try:
        with open(os.path.join(modelParameters.SPECS_PATH, filename) + '.hist', 'w') as f:
            f.write(str(hist.history))
    except:
        print("error writing training history")
        pass

    try:
        with open(os.path.join(modelParameters.SPECS_PATH, filename) + '.config', 'w') as f:
            f.write(str(model['siamese'].get_config()))
    except:
        print("error writing config")
        pass

    return hist
