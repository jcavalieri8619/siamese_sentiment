from __future__ import print_function

import datetime
import os

from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers import Input
from keras.layers.convolutional import Convolution1D, MaxPooling1D
from keras.layers.core import (Dense, Dropout,
                               Flatten)
from keras.layers.embeddings import Embedding
from keras.models import Model
from keras.regularizers import l2

import modelParameters
from convert_review import build_design_matrix

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

basename = "siamese_3_4".format(modelParameters.Margin)
suffix = datetime.datetime.now().strftime("%y%m%d_%I%M")
filename = "_".join([basename, suffix])

batch_size = 20

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

embedding_dims = 300

dense_dims1 = 1000
dense_dims2 = 150
dense_dims3 = 0
num_epochs = 4


def build_CNN_input(usewords=USEWORDS, skiptop=skipTop, devsplit=DEVSPLIT, verbose=True):
	"""

	:param usewords:
	:param skiptop:
	:param devsplit:
	:param verbose:
	:return:
	"""
	print('Building input')
	((X_train, y_train), (X_dev, y_dev), (X_test, y_test)) = build_design_matrix(VocabSize,
	                                                                             use_words=usewords,
	                                                                             skip_top=skiptop,
	                                                                             dev_split=devsplit)
	if verbose:
		print(len(X_train), 'train sequences')
		print(len(X_dev), 'test sequences')

		print('X_train shape:', X_train.shape)
		print('X_dev shape:', X_dev.shape)

		print('y_train shape:', y_train.shape)
		print('y_dev shape:', y_dev.shape)

	return X_train, y_train, X_dev, y_dev, X_test, y_test


def build_CNN_model(inputType, isIntermediate=False, **kwargs):
	"""

	:param inputType:
	:param kwargs:
	:return:
	"""
	assert inputType in ['embeddingMatrix', '1hotVector'], "unknown input type"

	if inputType == "1hotVector":

		review_input = Input(shape=(maxReviewLen,), dtype='int32', name="review")

		sharedEmbedding = Embedding(VocabSize + 2, embedding_dims,
		                            input_length=maxReviewLen)

		layer = sharedEmbedding(review_input)


	else:
		review_input = Input(shape=(maxReviewLen, embedding_dims), dtype="float32", name="review")
		layer = review_input

	sharedConv1 = Convolution1D(nb_filter=num_filters1,
	                            filter_length=filter_length1,
	                            border_mode='valid',
	                            activation='relu',
	                            subsample_length=stride_len1,
	                            init='uniform',
	                            input_length=maxReviewLen,
	                            batch_input_shape=(batch_size, maxReviewLen, embedding_dims))

	layer = sharedConv1(layer)

	layer = Dropout(0.25)(layer)

	layer = MaxPooling1D(pool_length=2)(layer)

	sharedConv2 = Convolution1D(nb_filter=num_filters2,
	                            filter_length=filter_length2,
	                            border_mode='valid',
	                            activation='relu',
	                            subsample_length=stride_len2,
	                            init='uniform'
	                            )

	layer = sharedConv2(layer)

	layer = Dropout(0.30)(layer)

	layer = MaxPooling1D(pool_length=2)(layer)

	sharedConv3 = Convolution1D(nb_filter=num_filters3,
	                            filter_length=filter_length3,
	                            border_mode='valid',
	                            activation='relu',
	                            subsample_length=stride_len3,
	                            init='uniform'
	                            )

	layer = sharedConv3(layer)

	layer = Dropout(0.35)(layer)

	layer = MaxPooling1D(pool_length=2)(layer)

	sharedConv4 = Convolution1D(nb_filter=num_filters4,
	                            filter_length=filter_length4,
	                            border_mode='valid',
	                            activation='relu',
	                            subsample_length=stride_len4,
	                            init='uniform',

	                            )

	layer = sharedConv4(layer)

	layer = Dropout(0.35)(layer)

	layer = MaxPooling1D(pool_length=2)(layer)

	layer = Flatten()(layer)

	# Dense layers default to 'glorot_normal' for init weights but that may not be optimal
	# for NLP tasks
	# init='uniform'
	sharedDense1 = Dense(dense_dims1, activation='relu',
	                     W_regularizer=l2(l=0.001))

	layer = sharedDense1(layer)

	layer = Dropout(0.35)(layer)

	sharedDense2 = Dense(dense_dims2, activation='relu',
	                     W_regularizer=l2(l=0.001))

	out_A = sharedDense2(layer)

	if isIntermediate:
		CNN_model = Model(input=[review_input], output=out_A, name="CNN_model")
		return CNN_model

	else:

		lastLayer = Dense(1, activation='sigmoid',
		                  W_regularizer=l2(l=0.001))

		out_B = lastLayer(out_A)

		CNN_model = Model(input=[review_input], output=out_B, name="CNN_model")

		CNN_model.compile(optimizer='rmsprop', loss='binary_crossentropy')

		return CNN_model


def train_CNN_model(model, X_train, y_train, X_dev, y_dev):
	weightPath = './model_data/saved_weights/' + filename
	checkpoint = ModelCheckpoint(weightPath + '_W.{epoch:02d}-{val_loss:.3f}.hdf5',
	                             verbose=1, )
	earlyStop = EarlyStopping(patience=1, verbose=1)

	call_backs = [checkpoint, earlyStop]

	hist = model.fit(X_train, y_train, batch_size=batch_size,
	          nb_epoch=num_epochs, verbose=1,
	          validation_data=(X_dev, y_dev),
	          callbacks=call_backs)

	with open(os.path.join('./model_data/model_specs', filename) + '.config', 'w') as f:
		f.write(str(model.get_config()))

	with open(os.path.join('./model_data/model_specs', filename + '.json'), 'w') as f:
		f.write(model.to_json())

	with open(os.path.join('./model_data/model_specs', filename) + '.hist', 'w') as f:
		f.write(str(hist.history))
