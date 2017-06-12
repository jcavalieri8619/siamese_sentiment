"""
Created by John P Cavalieri

"""
from __future__ import print_function

import cPickle
import os
from functools import partial
from itertools import combinations

import numpy

import modelParameters
from preprocess import (generate_word_list, generate_char_list,
                        generate_one_hot_maps, sentiment2reviews_map)

# if your computer cannot generate all the pairs of input data
# then set this to reduce the size of the original training data
# from 21,000 to some smaller number i.e 10,000.  The original size
# of dev set size can be set in modelParameters but at the default 14% its
# size is 3500 so you will probably want to reduce this as well.
TRAIN_LOW_RAM_CUTOFF = None
DEV_LOW_RAM_CUTOFF = None

RECONSTRUCTED_REVIEWS_PATH = "./model_data/reconstructed_reviews.txt"
PICKLED_CNN_INPUTS = "./model_data/CNN_inputs.pkl"
PICKLED_SIAMESE_INPUTS = "./model_data/SIAMESE_inputs.pkl"


def to_onehot_vector(reviewObject, one_hot_maps, use_words,
                     truncate, padding, maxlen=None, **kwargs):
    """

    :param reviewObject: tuple containing rating and movie review
    :param one_hot_maps: mapping from words to indices
    :param use_words: False to use chars
    :param maxlen: upper limit on words per review; default value set in modelParameters module
    :param truncate: 'pre' truncates at the beginning of review & 'post' truncates at the end of review
    :param padding: 'pre' pads beginning of review & 'post' pads at the end of review if review length < maxlen
    :param kwargs:
    :return: tuple containing a vector of 1hot indices representing the movie review and the movie rating
    """

    valid_truncate_padding_args = {'post', 'pre'}

    assert truncate in valid_truncate_padding_args and padding in valid_truncate_padding_args, \
        'to_onehot_vector truncate and padding args must be either \'pre\' or \'post\' '

    rating, review = reviewObject

    MAXlen = (maxlen if maxlen is not None else
              modelParameters.MaxLen_w if use_words else
              modelParameters.MaxLen_c)

    generate_symbol_list = generate_word_list if use_words else generate_char_list

    sym_list = generate_symbol_list(review)
    start_index = 0
    if truncate is 'pre':
        if len(sym_list) > MAXlen:
            sym_list = sym_list[-(MAXlen + 1):]
        elif padding is 'pre':
            start_index = MAXlen - len(sym_list) - 1

    elif truncate is 'post' and padding is 'pre':
        if len(sym_list) < MAXlen:
            start_index = MAXlen - len(sym_list) - 1

    vector_of_onehots = numpy.zeros((1, MAXlen), dtype='float32')
    vector_of_onehots[0, start_index] = modelParameters.START_SYMBOL
    for indx, word in enumerate(sym_list[:MAXlen - 1]):
        vector_of_onehots[0, (start_index + indx + 1)] = one_hot_maps.get(word, modelParameters.UNK_INDEX)

    return vector_of_onehots, rating


def reconstruct_reviews_from_designMatrix(designMatrix, targets, onehot_maps, ):
    """

    :param designMatrix:
    :param targets:
    :param onehot_maps:
    :return:
    """

    inverted_onehots = dict([(indx, wrd) for wrd, indx in onehot_maps.iteritems()])
    inverted_onehots[0] = '_PAD_'
    inverted_onehots[1] = '_START_'
    inverted_onehots[2] = '_UNK_'

    with open(RECONSTRUCTED_REVIEWS_PATH, 'w') as f:
        for (onehot_vector, label) in zip(designMatrix, targets):
            f.write("POSITIVE_REVIEW:\n" if label else "NEGATIVE_REVIEW:\n")
            for count, onehot_index in enumerate(onehot_vector):
                f.write(inverted_onehots[onehot_index])
                if not (count + 1) % 60:
                    f.write("\n")
                else:
                    f.write(" ")
            f.write('\n' * 2)


def build_design_matrix(vocab_size, use_words,
                        skip_top=0, maxlen=None, dev_split=None,
                        createValidationSet=True, truncate='post', padding='post',
                        pickle=False, verbose=True, **kwargs):
    """

    :param vocab_size:
    :param use_words:
    :param skip_top:
    :param maxlen:
    :param dev_split:
    :param createValidationSet:
    :param truncate:
    :param padding:
    :param pickle:
    :param verbose:
    :param kwargs:
    :return:
    """

    if pickle and os.path.exists(PICKLED_CNN_INPUTS):
        rv = cPickle.load(PICKLED_CNN_INPUTS)
        return rv

    review_iterator = list()

    MAXlen = (maxlen if maxlen is not None else
              modelParameters.MaxLen_w if use_words else
              modelParameters.MaxLen_c)

    testing_phase = kwargs.get('test_data', None)

    if testing_phase:
        # this test data is from kaggle competition and it consists of movie reviews and movie IDs
        # so that predicted sentiment can be paired wih a movie ID
        TESTDIR = './testing_data/test'
        # cannot use dev or validation sets with kaggle test data
        dev_split = createValidationSet = None

        if verbose:
            print("building test data objects")
            print("test data has no targets;\n"
                  "so the targets vector will contain ID of review at that index")
            if use_words:
                print("building TEST WORD design matrix")

            else:
                print("building TEST CHAR design matrix")

        for review_file in os.listdir(TESTDIR):
            with open(os.path.join(TESTDIR, review_file)) as f:
                # review id and review text in tuple
                review_iterator.append((review_file[:-4], f.read()))

        designMatrix = numpy.zeros((modelParameters.testingCount, MAXlen), dtype='float32')

        # for test data targets vector will hold review IDs; not ratings
        targets = numpy.zeros((modelParameters.testingCount, 1))

    else:
        # building TRAINING data

        if verbose:
            if use_words:
                print("building TRAINING WORD design matrix")

            else:
                print("building TRAINING CHAR design matrix")

        sentiment_reviews = sentiment2reviews_map()

        designMatrix = numpy.zeros((modelParameters.trainingCount, MAXlen), dtype='float32')

        targets = numpy.zeros((modelParameters.trainingCount, 1), dtype='float32')

        for label, file_map in sentiment_reviews.iteritems():
            for stars, reviewList in file_map.iteritems():
                for review in reviewList:
                    review_iterator.append((stars, review))

        numpy.random.shuffle(review_iterator)

    # now in common area where both test and training phase will execute
    word2index_mapping = generate_one_hot_maps(vocab_size, skip_top, use_words, kwargs.get("DEBUG"))

    func_to_one_hot = partial(to_onehot_vector,
                              one_hot_maps=word2index_mapping,
                              use_words=use_words,
                              truncate=truncate,
                              padding=padding,
                              maxlen=MAXlen,
                              )

    # # design matrix built in parallel because why not
    # workers = multiprocessing.Pool(multiprocessing.cpu_count())
    # results = workers.map(func_to_one_hot,
    #                       review_iterator)
    # workers.close()
    # workers.join()

    results = []
    for elem in review_iterator:
        results.append(func_to_one_hot(elem))

    if dev_split:

        assert isinstance(dev_split, int), "dev_split must be integer e.g. 14 implies 14%"

        if verbose:
            print("creating development set")

        numpy.random.shuffle(results)

        split = int((dev_split / 100.0) * modelParameters.trainingCount)
        dev_set = results[:split]
        results = results[split:]

        dev_designMatrix = numpy.zeros((len(dev_set), MAXlen), dtype='float32')
        dev_targets = numpy.zeros((len(dev_set), 1), dtype='float32')

        designMatrix = numpy.resize(designMatrix, (len(results), MAXlen))
        targets = numpy.resize(targets, (len(results), 1))

        for idx, (vector, rating) in enumerate(dev_set):
            dev_designMatrix[idx, :] = vector

            dev_targets[idx, 0] = rating >= 7

    for idx, (vector, rating) in enumerate(results):

        designMatrix[idx, :] = vector

        if testing_phase:
            # rating==review ID for test data in Kaggle format
            targets[idx, 0] = rating

        else:
            targets[idx, 0] = rating >= 7

    if verbose:
        print("finished generating design matrix and target vector")

    if kwargs.get("DEBUG"):
        print("reconstructing reviews from design matrix")
        reconstruct_reviews_from_designMatrix(designMatrix, targets, word2index_mapping)

    if not testing_phase and dev_split:

        if not createValidationSet:

            # using dev set but no validation set
            rv = ((designMatrix, targets), (dev_designMatrix, dev_targets),
                  (dev_designMatrix[:0, :0], dev_targets[:0, :0]))

            if pickle:
                with open(PICKLED_CNN_INPUTS, 'wb') as f:
                    cPickle.dump(rv, f, )

            return rv

        else:

            valsize = len(dev_designMatrix) / 4
            # return format is (trainingData tuple),(devData tuple),(valData tuple)
            rv = ((designMatrix, targets), (dev_designMatrix[valsize:], dev_targets[valsize:]),
                  (dev_designMatrix[:valsize], dev_targets[:valsize]))

            if pickle:
                with open(PICKLED_CNN_INPUTS, 'wb') as f:
                    cPickle.dump(rv, f, )

            return rv

    else:

        rv = designMatrix, targets

        if pickle:
            with open(PICKLED_CNN_INPUTS, 'wb') as f:
                cPickle.dump(rv, f, )

        return rv


def construct_kaggle_test_data(vocab_size, use_words):
    """
    generates testing data for kaggle competition. This data set does not include
    targets but instead has IDs so that kaggle can score the results
    :param vocab_size:
    :param use_words:
    :return:
    """
    return build_design_matrix(vocab_size=vocab_size,
                               use_words=use_words,
                               test_data=True)


def construct_designmatrix_pairs(VocabSize, useWords, skipTop=0, devSplit=None,
                                 trainingSet_cutoff=25000, devSet_cutoff=3000, verbose=True, **kwargs):
    """
    first generates standard design matrix and target vector then builds pairs
     for siamese input. Effectively we take all positive reviews choose 2, all
     negative reviews choose 2 and all reviews choose 2. Lecunn uses permutations,
     but that seems redundant. Ultimately the entire space of combinations will
     require an on-disk batch portion because its massive; for now I am just using a
     subset controlled by trainingSet_cutoff and devSet_cutoff in addition to limiting the
     number of mixed pairs created. If just creating the pairs requires too much
     RAM then set TRAIN_LOW_RAM_CUTOFF at the top of the module to
     some x << 20,000

    :param VocabSize: number of unique words in the vocabulary
    :param useWords: True is using words and false is using characters
    :param skipTop: stop word removal method--removes top K most frequent
            words in training data
    :param devSplit: percent of training data to split into dev set.
    :param kwargs:
    :return:
    """

    SIMILAR = 0
    DISIMILAR = 1

    TRAINCOMBO_SIZE = modelParameters.trainingComboSize
    DEVCOMBO_SIZE = modelParameters.devcomboSize

    ((X_train, y_train), (X_dev, y_dev), _) = build_design_matrix(VocabSize,
                                                                  use_words=useWords,
                                                                  skip_top=skipTop,
                                                                  dev_split=devSplit,
                                                                  createValidationSet=False)

    X_KNNtest = X_train[:2000]
    y_KNNtest = y_train[:2000]

    X_train = X_train[2000:]
    y_train = y_train[2000:]

    if TRAIN_LOW_RAM_CUTOFF is not None:
        X_train = X_train[:TRAIN_LOW_RAM_CUTOFF]
        y_train = y_train[:TRAIN_LOW_RAM_CUTOFF]

        X_dev = X_dev[:DEV_LOW_RAM_CUTOFF]
        y_dev = y_dev[:DEV_LOW_RAM_CUTOFF]

    trainPairs = [(x, y) for (x, y) in zip(X_train, y_train)]
    devPairs = [(x, y) for (x, y) in zip(X_dev, y_dev)]

    del X_train, y_train, X_dev, y_dev

    numpy.random.shuffle(trainPairs)
    numpy.random.shuffle(devPairs)

    posTrain = [pair for pair in trainPairs if pair[1]]

    negTrain = [pair for pair in trainPairs if not pair[1]]

    posDev = [pair for pair in devPairs if pair[1]]
    negDev = [pair for pair in devPairs if not pair[1]]

    Traincombo = list()
    Devcombo = list()

    if verbose:
        print("building pairs of design matrices and target vectors for positive movie reviews")

    pcombo = list(combinations(posTrain, 2))
    count = 0
    numpy.random.shuffle(pcombo)

    for item in pcombo:
        count += 1
        Traincombo.append((item[0], item[1], SIMILAR))
        if count >= TRAINCOMBO_SIZE:
            del pcombo
            break

    if verbose:
        print("building pairs of design matrices and target vectors for negative movie reviews")

    ncombo = list(combinations(negTrain, 2))
    count = 0
    numpy.random.shuffle(ncombo)

    for item in ncombo:
        count += 1
        Traincombo.append((item[0], item[1], SIMILAR))
        if count >= TRAINCOMBO_SIZE:
            del ncombo
            break

    if verbose:
        print("building pairs of dev design matrices and target vectors for positive movie reviews")

    pdcombo = list(combinations(posDev, 2))
    count = 0
    numpy.random.shuffle(pdcombo)
    for item in pdcombo:
        count += 1
        Devcombo.append((item[0], item[1], SIMILAR))
        if count >= DEVCOMBO_SIZE:
            del pdcombo
            break

    if verbose:
        print("building pairs of dev design matrices and target vectors for negative movie reviews")

    ndcombo = list(combinations(negDev, 2))
    count = 0
    numpy.random.shuffle(ndcombo)
    for item in ndcombo:

        count += 1
        Devcombo.append((item[0], item[1], SIMILAR))
        if count >= DEVCOMBO_SIZE:
            del ndcombo
            break

    if verbose:
        print("building pairs of design matrices and target vectors for both pos and neg movie reviews")

    allTrainCombo = list(combinations(trainPairs[:16000], 2))
    count = 0
    numpy.random.shuffle(allTrainCombo)

    for item in allTrainCombo:
        if item[0][1] == item[1][1]:
            continue
        assert item[0][1] != item[1][1], "mixed train combo logic error"
        count += 1
        Traincombo.append((item[0], item[1], DISIMILAR))
        if count >= 2 * TRAINCOMBO_SIZE:
            del allTrainCombo
            break

    if verbose:
        print("building pairs of dev design matrices and target vectors for both pos and neg movie reviews")

    allDevCombo = list(combinations(devPairs[:2000], 2))
    count = 0
    numpy.random.shuffle(allDevCombo)

    for item in allDevCombo:
        if item[0][1] != item[1][1]:
            continue
        assert item[0][1] == item[1][1], "mixed dev combo logic error"
        count += 1
        Devcombo.append((item[0], item[1], DISIMILAR))
        if count >= 2 * DEVCOMBO_SIZE:
            del allDevCombo
            break

    # form of combinations
    # [( (Xl,yl), (Xr,yr), SIM ), ...]

    X_left = numpy.zeros((len(Traincombo[:trainingSet_cutoff]), modelParameters.MaxLen_w), dtype='float32')
    X_right = numpy.zeros((len(Traincombo[:trainingSet_cutoff]), modelParameters.MaxLen_w), dtype='float32')

    y_left = numpy.zeros((len(Traincombo[:trainingSet_cutoff]), 1), dtype='float32')
    y_right = numpy.zeros((len(Traincombo[:trainingSet_cutoff]), 1), dtype='float32')

    similarity_labels = numpy.zeros((len(Traincombo[:trainingSet_cutoff]), 1), dtype='float32')

    Xval_left = numpy.zeros((len(Traincombo[trainingSet_cutoff:]), modelParameters.MaxLen_w), dtype='float32')
    Xval_right = numpy.zeros((len(Traincombo[trainingSet_cutoff:]), modelParameters.MaxLen_w), dtype='float32')

    yval_left = numpy.zeros((len(Traincombo[trainingSet_cutoff:]), 1), dtype='float32')
    yval_right = numpy.zeros((len(Traincombo[trainingSet_cutoff:]), 1), dtype='float32')

    similarity_vallabels = numpy.zeros((len(Traincombo[trainingSet_cutoff:]), 1), dtype='float32')

    numpy.random.shuffle(Traincombo)
    for idx, (left, right, similar_label) in enumerate(Traincombo[:trainingSet_cutoff]):
        X_left[idx, :], y_left[idx, 0] = left

        X_right[idx, :], y_right[idx, 0] = right

        similarity_labels[idx, 0] = similar_label

    for idx, (left, right, similar_label) in enumerate(Traincombo[trainingSet_cutoff:]):
        Xval_left[idx, :], yval_left[idx, 0] = left

        Xval_right[idx, :], yval_right[idx, 0] = right

        similarity_vallabels[idx, 0] = similar_label

    if devSplit is None:
        return ((X_left, y_left, X_right, y_right, similarity_labels), (None,),
                (Xval_left, yval_left, Xval_right, yval_right, similarity_vallabels),
                )

    Xdev_left = numpy.zeros((len(Devcombo[:devSet_cutoff]), modelParameters.MaxLen_w), dtype='float32')
    Xdev_right = numpy.zeros((len(Devcombo[:devSet_cutoff]), modelParameters.MaxLen_w), dtype='float32')

    ydev_left = numpy.zeros((len(Devcombo[:devSet_cutoff]), 1), dtype='float32')
    ydev_right = numpy.zeros((len(Devcombo[:devSet_cutoff]), 1), dtype='float32')
    similarity_devlabels = numpy.zeros((len(Devcombo[:devSet_cutoff]), 1), dtype='float32')

    XdevKnn_left = numpy.zeros((len(Devcombo[devSet_cutoff:]), modelParameters.MaxLen_w), dtype='float32')
    XdevKnn_right = numpy.zeros((len(Devcombo[devSet_cutoff:]), modelParameters.MaxLen_w), dtype='float32')

    ydevKnn_left = numpy.zeros((len(Devcombo[devSet_cutoff:]), 1), dtype='float32')
    ydevKnn_right = numpy.zeros((len(Devcombo[devSet_cutoff:]), 1), dtype='float32')
    similarity_devKnnlabels = numpy.zeros((len(Devcombo[devSet_cutoff:]), 1), dtype='float32')

    numpy.random.shuffle(Devcombo)
    for idx, (left, right, similar_label) in enumerate(Devcombo[:devSet_cutoff]):
        Xdev_left[idx, :], ydev_left[idx, 0] = left

        Xdev_right[idx, :], ydev_right[idx, 0] = right

        similarity_devlabels[idx, 0] = similar_label

    for idx, (left, right, similar_label) in enumerate(Devcombo[devSet_cutoff:]):
        XdevKnn_left[idx, :], ydevKnn_left[idx, 0] = left

        XdevKnn_right[idx, :], ydevKnn_right[idx, 0] = right

        similarity_devKnnlabels[idx, 0] = similar_label

    if verbose:
        print("finished building pairs of movie review design matrices")

    return ((X_left, y_left, X_right, y_right, similarity_labels),
            (Xdev_left, ydev_left, Xdev_right, ydev_right, similarity_devlabels),
            (XdevKnn_left, ydevKnn_left, XdevKnn_right, ydevKnn_right, similarity_devKnnlabels),
            (Xval_left, yval_left, Xval_right, yval_right, similarity_vallabels),
            (X_KNNtest, y_KNNtest))
