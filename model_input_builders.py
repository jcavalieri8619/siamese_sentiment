import modelParameters
from convert_review import build_design_matrix, construct_designmatrix_pairs

USEWORDS = True

if USEWORDS:
    skipTop = modelParameters.skip_top
    VocabSize = modelParameters.VocabSize_w
    maxlen = modelParameters.MaxLen_w

else:
    skipTop = 0
    VocabSize = modelParameters.VocabSize_c
    maxlen = modelParameters.MaxLen_c

DEVSPLIT = modelParameters.devset_split


def build_CNN_input(vocab_size=VocabSize, usewords=USEWORDS,
                    skiptop=skipTop, devsplit=DEVSPLIT, verbose=True, **kwargs):
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


def build_siamese_input(verbose=True, **kwargs):
    """

    :param verbose:
    :return:
    """

    if verbose:
        print('building pairs of reviews for siamese model input')

    trainCutoff = kwargs.get('trainingSet_cutoff', 50000)

    ((trainingSets), (devSets), (devKNNsets), (valSets), (testSet)) = construct_designmatrix_pairs(VocabSize,
                                                                                                   useWords=USEWORDS,
                                                                                                   skipTop=skipTop,
                                                                                                   trainingSet_cutoff=trainCutoff,
                                                                                                   devSplit=DEVSPLIT)

    if verbose:
        print(len(trainingSets[0]), 'train sequences length')
        print(len(devSets[0]), 'dev sequences length')

        print(len(devKNNsets[0]), 'devKNN sequences length')
        print(len(valSets[0]), 'test sequences length')

        print('train shape:', trainingSets[0].shape)
        print('dev shape:', devSets[0].shape)
        print('devKNN shape:', devKNNsets[0].shape)
        print('test shape:', valSets[0].shape)

    return {'training': trainingSets, 'dev': devSets, 'KNNdev': devKNNsets, 'val': valSets, 'KNNtest': testSet}
