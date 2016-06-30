"""
Created by John P Cavalieri

"""
from __future__ import print_function

import cPickle
import multiprocessing
import os
import random
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

DESIGN_MATRIX_PATH_WORD = './model_data/designMatrix_w.pickle'
TARGET_VECTOR_PATH_WORD = './model_data/targetVect_w.pickle'

DESIGN_MATRIX_PATH_CHAR = './model_data/designMatrix_c.pickle'
TARGET_VECTOR_PATH_CHAR = './model_data/targetVect_c.pickle'

DEV_DESIGN_MATRIX_PATH_WORD = './model_data/dev_designMatrix_w.pickle'
DEV_TARGET_VECTOR_PATH_WORD = './model_data/dev_targetVect_w.pickle'

DEV_DESIGN_MATRIX_PATH_CHAR = './model_data/dev_designMatrix_c.pickle'
DEV_TARGET_VECTOR_PATH_CHAR = './model_data/dev_targetVect_c.pickle'

TEST_SET_DATA_PATH_WORD = './model_data/test_set_data_w.pickle'
TEST_SET_DATA_PATH_CHAR = './model_data/test_set_data_c.pickle'
TEST_SET_ID_VECTOR = './model_data/test_set_ID_vect.pickle'


# seed for consistency across calls
# random.seed( 1515 )


def to_onehot_vector(reviewObject, one_hot_maps, use_words, skip_top=0, maxlen=None, **kwargs):
	rating, review = reviewObject

	if use_words:
		MAXlen = maxlen if maxlen is not None else modelParameters.MaxLen_w
		vector_of_onehots = numpy.zeros((1, MAXlen), dtype='int32')
		vector_of_onehots += modelParameters.UNK_INDEX
		for indx, word in enumerate(generate_word_list(review)[:MAXlen]):
			vector_of_onehots[0, indx] = one_hot_maps[word]

	else:
		MAXlen = maxlen if maxlen is not None else modelParameters.MaxLen_c
		vector_of_onehots = numpy.zeros((1, maxlen), dtype='int32')
		vector_of_onehots += modelParameters.UNK_INDEX
		for indx, char in enumerate(generate_char_list(review)[:MAXlen]):
			vector_of_onehots[0, indx] = one_hot_maps[char]

	return (vector_of_onehots, rating)


def build_design_matrix(vocab_size, use_words,
                        skip_top=0, maxlen=None, dev_split=None,
                        verbose=True, **kwargs):
	"""

	:param vocab_size:
	:param use_words:
	:param skip_top:
	:param maxlen:
	:param dev_split:
	:param verbose:
	:param kwargs:
	:return:
	"""
	testing_phase = kwargs.get('test_data', None)

	if testing_phase:

		# in testing phase with no targets
		# attempting to find pickled data
		assert dev_split is not None, "cannot generate dev set for test data set"

		if use_words and os.path.isfile(TEST_SET_DATA_PATH_WORD):
			print("word TEST design matrix found, loading pickle")
			with open(TEST_SET_DATA_PATH_WORD, 'rb') as f:
				designMatrix = cPickle.load(f)
			with open(TEST_SET_ID_VECTOR, 'rb') as f:
				targets = cPickle.load(f)

			return (designMatrix, targets)

		elif not use_words and os.path.isfile(TEST_SET_DATA_PATH_CHAR):
			print("char TEST design matrix found, loading pickle")
			with open(TEST_SET_DATA_PATH_CHAR, 'rb') as f:
				designMatrix = cPickle.load(f)
			with open(TEST_SET_ID_VECTOR, 'rb') as f:
				targets = cPickle.load(f)

			return (designMatrix, targets)




	else:
		# in training phase with data and targets
		# attempting to find pickled data

		if use_words and os.path.isfile(DESIGN_MATRIX_PATH_WORD):
			print("word TRAINING design matrix found, loading pickle")
			with open(DESIGN_MATRIX_PATH_WORD, 'rb') as f:
				designMatrix = cPickle.load(f)
			with open(TARGET_VECTOR_PATH_WORD, 'rb') as f:
				targets = cPickle.load(f)

			if dev_split is not None:
				with open(DEV_DESIGN_MATRIX_PATH_WORD, 'rb') as f:
					dev_designMatrix = cPickle.load(f)
				with open(DEV_TARGET_VECTOR_PATH_WORD, 'rb') as f:
					dev_targets = cPickle.load(f)

				if not kwargs.get("noTest", False):
					# return format is (trainingData),(devData),(testData)
					return ((designMatrix, targets), (dev_designMatrix[500:], dev_targets[500:]),
					        (dev_designMatrix[:500], dev_targets[:500]))
				else:

					return ((designMatrix, targets), (dev_designMatrix, dev_targets))

			else:

				return (designMatrix, targets)

	# couldn't find any pickled data so generating it here

	if verbose:
		print("pickled data not found, building it...")

	one_hots = generate_one_hot_maps(vocab_size, skip_top, use_words)

	review_iterator = list()

	if testing_phase:
		# building testing data (test set is not equal to dev set)
		print("building test data objects")
		print("test data has no targets;\n"
		      "so the targets vector will contain ID of review at that index")

		review_iterator = list()
		for review_file in os.listdir('./testing_data/test/'):
			with open('./testing_data/test/' + review_file) as f:
				# review id and review text in tuple
				review_iterator.append((review_file[:-4], f.read()))

		if use_words:
			print("building TEST word design matrix")
			designMatrix = numpy.zeros((modelParameters.testingCount,
			                            (maxlen if maxlen is not None
			                             else modelParameters.MaxLen_w)), dtype='int32')

		else:
			print("building TEST char design matrix")
			designMatrix = numpy.zeros((modelParameters.testingCount,
			                            (maxlen if maxlen is not None
			                             else modelParameters.MaxLen_c)), dtype='int32')

		##for test data targets vector will hold review IDs; not ratings
		targets = numpy.zeros((modelParameters.testingCount, 1))



	else:
		# building training data

		sentiment_reviews = sentiment2reviews_map()

		if use_words:
			print("building TRAINING word design matrix")
			designMatrix = numpy.zeros((modelParameters.trainingCount,
			                            (maxlen if maxlen is not None
			                             else modelParameters.MaxLen_w)), dtype='int32')
		else:
			print("building TRAINING char design matrix")
			designMatrix = numpy.zeros((modelParameters.trainingCount,
			                            (maxlen if maxlen is not None
			                             else modelParameters.MaxLen_c)), dtype='int32')

		targets = numpy.zeros((modelParameters.trainingCount, 1), dtype='int32')

		for label, file_map in sentiment_reviews.iteritems():
			for stars, reviewList in file_map.iteritems():
				for review in reviewList:
					review_iterator.append((stars, review))

		random.shuffle(review_iterator)

	##now in common area where both test and training phase will execute
	MAXlen = (maxlen if maxlen is not None else
	          modelParameters.MaxLen_w if use_words else
	          modelParameters.MaxLen_c)

	func_to_one_hot = partial(to_onehot_vector,
	                          one_hot_maps=one_hots,
	                          use_words=use_words,
	                          skip_top=skip_top,
	                          maxlen=MAXlen,
	                          )

	# design matrix built in parallel because why not
	workers = multiprocessing.Pool(processes=8)
	results = workers.map(func_to_one_hot,
	                      review_iterator)
	workers.close()
	workers.join()

	if dev_split is not None:
		print("creating dev set")
		random.shuffle(results)

		split = int((float(dev_split) / 100) * modelParameters.trainingCount)
		dev_set = results[:split]
		results = results[split:]

		dev_designMatrix = numpy.zeros((len(dev_set), MAXlen), dtype='int32')
		dev_targets = numpy.zeros((len(dev_set), 1), dtype='int32')

		designMatrix = numpy.resize(designMatrix, (len(results), MAXlen))
		targets = numpy.resize(targets, (len(results), 1))

		for idx, (vector, rating) in enumerate(dev_set):
			dev_designMatrix[idx, :] = vector

			dev_targets[idx, 0] = rating >= 7

	for idx, (vector, rating) in enumerate(results):
		designMatrix[idx, :] = vector
		if testing_phase:
			# rating==review ID for test data (test data != dev set)
			targets[idx, 0] = rating
		else:
			targets[idx, 0] = rating >= 7

	print("finished building data design matrix, now pickling")

	if testing_phase is not None:
		##test ID vector is same for both char and word
		with open(TEST_SET_ID_VECTOR, 'wb') as f:
			cPickle.dump(targets, f)

		if use_words:
			with open(TEST_SET_DATA_PATH_WORD, 'wb') as f:
				cPickle.dump(designMatrix, f)

		else:
			with open(TEST_SET_DATA_PATH_CHAR, 'wb') as f:
				cPickle.dump(designMatrix, f)


	else:
		if use_words:
			with open(DESIGN_MATRIX_PATH_WORD, 'wb') as f:
				cPickle.dump(designMatrix, f)
			with open(TARGET_VECTOR_PATH_WORD, 'wb') as f:
				cPickle.dump(targets, f)

			if dev_split is not None:
				with open(DEV_DESIGN_MATRIX_PATH_WORD, 'wb') as f:
					cPickle.dump(dev_designMatrix, f)
				with open(DEV_TARGET_VECTOR_PATH_WORD, 'wb') as f:
					cPickle.dump(dev_targets, f)

	if dev_split is not None:

		if not kwargs.get("noTest", False):
			# return format is (trainingData),(devData),(testData)
			return ((designMatrix, targets), (dev_designMatrix[500:], dev_targets[500:]),
			        (dev_designMatrix[:500], dev_targets[:500]))
		else:
			# using dev set but no test set
			return ((designMatrix, targets), (dev_designMatrix, dev_targets), (dev_designMatrix[:0], dev_targets[:0]))

	else:

		return (designMatrix, targets)


def get_testing_data(vocab_size, use_words):
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


def construct_review_pairs(VocabSize, useWords, skipTop=0, devSplit=None, **kwargs):
	"""
	first generates standard design matrix and target vector then builds pairs
	 for siamese input. Effectively we take all positive reviews choose 2, all
	 negative reviews choose 2 and all reviews choose 2. Lecunn uses permutations,
	 but that seems redundant. Ultimately the entire space of combinations will
	 require an on-dist batch portion because its massive; for now I am just using a
	 subset controlled by TRAIN_CUTOFF and DEV_CUTOFF in addition to limiting the
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

	# these cutoffs reduce the final training and dev set sizes
	# after they have been created to speed up training.
	# these paramters will not help if you are low on RAM. see top of module
	TRAIN_CUTOFF = 21500
	DEV_CUTOFF = 3000

	((X_train, y_train), (X_dev, y_dev)) = build_design_matrix(VocabSize,
	                                                           use_words=useWords,
	                                                           skip_top=skipTop,
	                                                           dev_split=devSplit,
	                                                           noTest=True)

	if TRAIN_LOW_RAM_CUTOFF is not None or \
					DEV_LOW_RAM_CUTOFF is not None:
		X_train = X_train[:TRAIN_LOW_RAM_CUTOFF]
		y_train = y_train[:TRAIN_LOW_RAM_CUTOFF]

		X_dev = X_dev[:DEV_LOW_RAM_CUTOFF]
		y_dev = y_dev[:DEV_LOW_RAM_CUTOFF]

	assert len(X_train) == len(y_train), "training data and targets different length"
	trainPairs = [(x, y) for (x, y) in zip(X_train, y_train)]
	devPairs = [(x, y) for (x, y) in zip(X_dev, y_dev)]

	del X_train, y_train, X_dev, y_dev

	random.shuffle(trainPairs)
	random.shuffle(devPairs)

	posTrain = [pair for pair in trainPairs if pair[1]]

	negTrain = [pair for pair in trainPairs if not pair[1]]

	posDev = [pair for pair in devPairs if pair[1]]
	negDev = [pair for pair in devPairs if not pair[1]]

	Traincombo = list()
	Devcombo = list()
	print("building positive combos")

	pcombo = list(combinations(posTrain, 2))

	count = 0
	random.shuffle(pcombo)
	for item in pcombo:

		assert item[0][1] == item[1][1], "pos train combo logic error"

		count += 1
		# for of postraincombo list is [((Xleft,yleft),(Xright,yright),similarity),...]
		Traincombo.append((item[0], item[1], SIMILAR))
		if count >= TRAINCOMBO_SIZE:
			del pcombo
			break

	print("building negative combos")
	ncombo = list(combinations(negTrain, 2))
	count = 0
	random.shuffle(ncombo)
	for item in ncombo:
		assert item[0][1] == item[1][1], "neg train combo logic error"

		count += 1
		Traincombo.append((item[0], item[1], SIMILAR))
		if count >= TRAINCOMBO_SIZE:
			del ncombo
			break

	print("building positive dev combos")
	pdcombo = list(combinations(posDev, 2))
	count = 0
	random.shuffle(pdcombo)
	for item in pdcombo:
		assert item[0][1] == item[1][1], "pos dev combo logic error"

		count += 1
		Devcombo.append((item[0], item[1], SIMILAR))
		if count >= DEVCOMBO_SIZE:
			del pdcombo
			break

	print("building negative dev combos")
	ndcombo = list(combinations(negDev, 2))
	count = 0
	random.shuffle(ndcombo)
	for item in ndcombo:
		assert item[0][1] == item[1][1], "neg dev combo logic error"

		count += 1
		Devcombo.append((item[0], item[1], SIMILAR))
		if count >= DEVCOMBO_SIZE:
			del ndcombo
			break

	print("building mixed combos")
	allTrainCombo = list(combinations(trainPairs[:16000], 2))

	count = 0
	random.shuffle(allTrainCombo)
	for item in allTrainCombo:
		if item[0][1] == item[1][1]:
			continue
		assert item[0][1] != item[1][1], "mixed train combo logic error"
		count += 1
		Traincombo.append((item[0], item[1], DISIMILAR))
		if count >= 2 * TRAINCOMBO_SIZE:
			del allTrainCombo
			break

	print("building mixed dev combos")
	allDevCombo = list(combinations(devPairs[:2000], 2))
	count = 0
	random.shuffle(allDevCombo)
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


	X_left = numpy.zeros((len(Traincombo[:TRAIN_CUTOFF]), modelParameters.MaxLen_w), dtype='int32')
	X_right = numpy.zeros((len(Traincombo[:TRAIN_CUTOFF]), modelParameters.MaxLen_w), dtype='int32')

	y_left = numpy.zeros((len(Traincombo[:TRAIN_CUTOFF]), 1), dtype='int32')
	y_right = numpy.zeros((len(Traincombo[:TRAIN_CUTOFF]), 1), dtype='int32')

	similarity_labels = numpy.zeros((len(Traincombo[:TRAIN_CUTOFF]), 1), dtype='int32')

	Xtest_left = numpy.zeros((len(Traincombo[TRAIN_CUTOFF:]), modelParameters.MaxLen_w), dtype='int32')
	Xtest_right = numpy.zeros((len(Traincombo[TRAIN_CUTOFF:]), modelParameters.MaxLen_w), dtype='int32')

	ytest_left = numpy.zeros((len(Traincombo[TRAIN_CUTOFF:]), 1), dtype='int32')
	ytest_right = numpy.zeros((len(Traincombo[TRAIN_CUTOFF:]), 1), dtype='int32')

	similarity_testlabels = numpy.zeros((len(Traincombo[TRAIN_CUTOFF:]), 1), dtype='int32')

	random.shuffle(Traincombo)
	for idx, (left, right, similar_label) in enumerate(Traincombo[:TRAIN_CUTOFF]):
		X_left[idx, :], y_left[idx, 0] = left

		X_right[idx, :], y_right[idx, 0] = right

		similarity_labels[idx, 0] = similar_label

	for idx, (left, right, similar_label) in enumerate(Traincombo[TRAIN_CUTOFF:]):
		Xtest_left[idx, :], ytest_left[idx, 0] = left

		Xtest_right[idx, :], ytest_right[idx, 0] = right

		similarity_testlabels[idx, 0] = similar_label

	if devSplit is None:
		return ((X_left, y_left, X_right, y_right, similarity_labels), (None,),
		        (Xtest_left, ytest_left, Xtest_right, ytest_right, similarity_testlabels),)

	Xdev_left = numpy.zeros((len(Devcombo[:DEV_CUTOFF]), modelParameters.MaxLen_w), dtype='int32')
	Xdev_right = numpy.zeros((len(Devcombo[:DEV_CUTOFF]), modelParameters.MaxLen_w), dtype='int32')

	ydev_left = numpy.zeros((len(Devcombo[:DEV_CUTOFF]), 1), dtype='int32')
	ydev_right = numpy.zeros((len(Devcombo[:DEV_CUTOFF]), 1), dtype='int32')
	similarity_devlabels = numpy.zeros((len(Devcombo[:DEV_CUTOFF]), 1), dtype='int32')

	XdevKnn_left = numpy.zeros((len(Devcombo[DEV_CUTOFF:]), modelParameters.MaxLen_w), dtype='int32')
	XdevKnn_right = numpy.zeros((len(Devcombo[DEV_CUTOFF:]), modelParameters.MaxLen_w), dtype='int32')

	ydevKnn_left = numpy.zeros((len(Devcombo[DEV_CUTOFF:]), 1), dtype='int32')
	ydevKnn_right = numpy.zeros((len(Devcombo[DEV_CUTOFF:]), 1), dtype='int32')
	similarity_devKnnlabels = numpy.zeros((len(Devcombo[DEV_CUTOFF:]), 1), dtype='int32')

	random.shuffle(Devcombo)
	for idx, (left, right, similar_label) in enumerate(Devcombo[:DEV_CUTOFF]):
		Xdev_left[idx, :], ydev_left[idx, 0] = left

		Xdev_right[idx, :], ydev_right[idx, 0] = right

		similarity_devlabels[idx, 0] = similar_label

	for idx, (left, right, similar_label) in enumerate(Devcombo[DEV_CUTOFF:]):
		XdevKnn_left[idx, :], ydevKnn_left[idx, 0] = left

		XdevKnn_right[idx, :], ydevKnn_right[idx, 0] = right

		similarity_devKnnlabels[idx, 0] = similar_label

	print("finished building siamese input")

	return ((X_left, y_left, X_right, y_right, similarity_labels),
	        (Xdev_left, ydev_left, Xdev_right, ydev_right, similarity_devlabels),
	        (XdevKnn_left, ydevKnn_left, XdevKnn_right, ydevKnn_right, similarity_devKnnlabels),
	        (Xtest_left, ytest_left, Xtest_right, ytest_right, similarity_testlabels),)


if __name__ == '__main__':

	USEWORDS = True

	if USEWORDS:
		V = modelParameters.VocabSize_w
	else:
		V = modelParameters.VocabSize_c

	SKP = modelParameters.skip_top

	DEVSPLIT = 14

	(x, y) = build_design_matrix(vocab_size=V,
	                             use_words=USEWORDS,
	                             skip_top=SKP,
	                             dev_split=DEVSPLIT)
