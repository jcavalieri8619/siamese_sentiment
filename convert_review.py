"""
Created by John P Cavalieri

"""
from __future__ import print_function

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



# seed for consistency across calls
random.seed(1515)


def to_onehot_vector(reviewObject, one_hot_maps, use_words, skip_top=0, maxlen=None, **kwargs):
	"""

	:param reviewObject: tuple containing rating and movie review
	:param one_hot_maps: mapping from words to indices
	:param use_words: False to use chars
	:param skip_top: remove K most frequently occuring words seen during training i.e. the,is,a,...
	:param maxlen: upper limit on words per review; default value set in modelParameters module
	:param kwargs:
	:return: tuple containing a vector of 1hot indices representing the movie review and the movie rating
	"""
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
                        createValidationSet=True, verbose=True, **kwargs):
	"""

	:param vocab_size: size of vocabularly
	:param use_words: False to use chars
	:param skip_top: remove K most frequently occuring words seen during training i.e. the,is,a,...
	:param maxlen: upper limit on words per review; default value set in modelParameters module
	:param dev_split: int giving percent of training data to hold out for dev set
	:param createValidationSet: if True then splits off 700 examples from dev set for validation
	:param verbose:
	:param kwargs:
	:return: tuple returning design matrix and target for requested data sets: training,dev,validation
	"""
	testing_phase = kwargs.get('test_data', None)

	if verbose:
		print("pickled data not found, building it...")



	review_iterator = list()

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


		for review_file in os.listdir(TESTDIR):
			with open(os.path.join(TESTDIR, review_file)) as f:
				# review id and review text in tuple
				review_iterator.append((review_file[:-4], f.read()))

		if use_words:
			if verbose:
				print("building TEST word design matrix")

			designMatrix = numpy.zeros((modelParameters.testingCount,
			                            (maxlen if maxlen is not None
			                             else modelParameters.MaxLen_w)), dtype='int32')

		else:
			if verbose:
				print("building TEST char design matrix")

			designMatrix = numpy.zeros((modelParameters.testingCount,
			                            (maxlen if maxlen is not None
			                             else modelParameters.MaxLen_c)), dtype='int32')

		# for test data targets vector will hold review IDs; not ratings
		targets = numpy.zeros((modelParameters.testingCount, 1))



	else:
		# building TRAINING data

		sentiment_reviews = sentiment2reviews_map()

		if use_words:
			if verbose:
				print("building TRAINING word design matrix")

			designMatrix = numpy.zeros((modelParameters.trainingCount,
			                            (maxlen if maxlen is not None
			                             else modelParameters.MaxLen_w)), dtype='int32')
		else:
			if verbose:
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

	# now in common area where both test and training phase will execute

	word2index_mapping = generate_one_hot_maps(vocab_size, skip_top, use_words)

	MAXlen = (maxlen if maxlen is not None else
	          modelParameters.MaxLen_w if use_words else
	          modelParameters.MaxLen_c)

	func_to_one_hot = partial(to_onehot_vector,
	                          one_hot_maps=word2index_mapping,
	                          use_words=use_words,
	                          skip_top=skip_top,
	                          maxlen=MAXlen,
	                          )

	# design matrix built in parallel because why not
	workers = multiprocessing.Pool(multiprocessing.cpu_count())
	results = workers.map(func_to_one_hot,
	                      review_iterator)
	workers.close()
	workers.join()

	if dev_split is not None:

		assert isinstance(dev_split, int), "dev_split must be integer e.g. 14 implies 14%"

		if verbose:
			print("creating dev set")

		random.shuffle(results)

		split = int((float(dev_split) / 100.0) * modelParameters.trainingCount)
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

	if verbose:
		print("finished generating design matrices and target vectors")

	if not testing_phase and dev_split is not None:

		if not createValidationSet:
			# using dev set but no validation set
			return (
				(designMatrix, targets), (dev_designMatrix, dev_targets), (dev_designMatrix[:0, :], dev_targets[:0, :]))

		else:
			# return format is (trainingData),(devData),(testData)
			return ((designMatrix, targets), (dev_designMatrix[500:, :], dev_targets[500:, :]),
			        (dev_designMatrix[:500, :], dev_targets[:500, :]))

	else:

		return (designMatrix, targets)


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


def construct_designmatrix_pairs(VocabSize, useWords, skipTop=0, devSplit=None, verbose=True, **kwargs):
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

	((X_train, y_train), (X_dev, y_dev), _) = build_design_matrix(VocabSize,
	                                                              use_words=useWords,
	                                                              skip_top=skipTop,
	                                                              dev_split=devSplit,
	                                                              createValidationSet=False)

	if TRAIN_LOW_RAM_CUTOFF is not None:
		X_train = X_train[:TRAIN_LOW_RAM_CUTOFF]
		y_train = y_train[:TRAIN_LOW_RAM_CUTOFF]

		X_dev = X_dev[:DEV_LOW_RAM_CUTOFF]
		y_dev = y_dev[:DEV_LOW_RAM_CUTOFF]


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

	if verbose:
		print("building pairs of design matrices and target vectors for positive movie reviews")

	pcombo = list(combinations(posTrain, 2))
	count = 0
	random.shuffle(pcombo)

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
	random.shuffle(ncombo)

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
	random.shuffle(pdcombo)
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
	random.shuffle(ndcombo)
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

	if verbose:
		print("building pairs of dev design matrices and target vectors for both pos and neg movie reviews")


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

	if verbose:
		print("finished building pairs of movie review design matrices")

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
