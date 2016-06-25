"""
Created by John P Cavalieri

"""
from __future__ import print_function

import datetime
import os

from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.engine.topology import merge
from keras.layers import Input
from keras.layers.core import (Dense)
from keras.models import Model

import modelParameters
from CNN_model import build_CNN_model
from convert_review import build_siamese_input
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

basename = "siamese_3_4".format( modelParameters.Margin )
suffix = datetime.datetime.now( ).strftime( "%y%m%d_%I%M" )
filename = "_".join( [ basename, suffix ] )

batch_size = 20

num_filters1 = 800
filter_length1 = 2
stride_len1 = 1
pool_len1 = 2

num_filters2 = 600
filter_length2 = 3
stride_len2 = 1
pool_len2 = 2

num_filters3 = 400
filter_length3 = 4
stride_len3 = 1
pool_len3 = 2

num_filters4 = 400
filter_length4 = 5
stride_len4 = 1
pool_len4 = 2

embedding_dims = 300

dense_dims1 = 1000
dense_dims2 = 300
dense_dims3 = 0
num_epochs = 4


def merged_outshape( inputShapes ):
	shape = list( inputShapes )
	assert len( shape ) == 2, "merged_outShape: len inputShapes != 2"
	return shape[ 0 ]


def build_siamese_model():
	print( 'building pairs of reviews for siamese model input...' )

	((trainingSets), (devSets), (devKNNsets), (testSets)) = build_siamese_input( VocabSize,
	                                                                             useWords = USEWORDS,
	                                                                             skipTop = skipTop,
	                                                                             devSplit = DEVSPLIT )

	# X_left and X_right are matrices with trainingSet rows and reviewLen columns
	# y_left and y_right are the corresponding sentiment labels i.e 0:negative 1:positive
	# similarity is 0 if X_left and X_right have same sentiment labels and 1 otherwise
	X_left, y_left, X_right, y_right, similarity = trainingSets

	# Xtest_left, ytest_left, Xtest_right, ytest_right, test_similarity = testSets

	# Xdev_left and Xdev_right are matrices with devSet rows and reviewLen columns
	Xdev_left, ydev_left, Xdev_right, ydev_right, dev_similarity = devSets

	print( len( X_left ), 'train sequences length' )
	print( len( Xdev_left ), 'dev sequences length' )

	print( len( devKNNsets[ 0 ] ), 'devKNN sequences length' )
	print( len( testSets[ 0 ] ), 'test sequences length' )

	print( 'train shape:', X_left.shape )
	print( 'dev shape:', Xdev_left.shape )
	print( 'devKNN shape:', devKNNsets[ 0 ].shape )
	print( 'test shape:', testSets[ 0 ].shape )


	print( 'Build model...' )

	# review_input = Input( shape = (maxReviewLen,), dtype = 'int32', name = "review" )
	#
	# # probability of positive sentiment for left input and right input;
	# # during training these are either 1 or 0 because we have that info in y_left and y_right
	# # but during testing its 0.5 indicating equal probability of positive or negative
	# #TODO currently not using but still thinking about how to use this information
	# #sentiment_prob_input = Input( shape = (1,), dtype = 'float32', name = "sentprob" )
	#
	# sharedEmbedding = Embedding( VocabSize, embedding_dims,
	#                              input_length = maxReviewLen )
	#
	# layer = sharedEmbedding( review_input )
	#
	# sharedConv1 = Convolution1D( nb_filter = num_filters1,
	#                              filter_length = filter_length1,
	#                              border_mode = 'valid',
	#                              activation = 'relu',
	#                              subsample_length = stride_len1,
	#                              init = 'uniform' )
	#
	# layer = sharedConv1( layer )
	#
	# layer = Dropout( 0.25 )( layer )
	#
	# layer = MaxPooling1D( pool_length = 2 )( layer )
	#
	# sharedConv2 = Convolution1D( nb_filter = num_filters2,
	#                              filter_length = filter_length2,
	#                              border_mode = 'valid',
	#                              activation = 'relu',
	#                              subsample_length = stride_len2,
	#                              init = 'uniform'
	#                              )
	#
	# layer = sharedConv2( layer )
	#
	# layer = Dropout( 0.30 )( layer )
	#
	# layer = MaxPooling1D( pool_length = 2 )( layer )
	#
	# sharedConv3 = Convolution1D( nb_filter = num_filters3,
	#                              filter_length = filter_length3,
	#                              border_mode = 'valid',
	#                              activation = 'relu',
	#                              subsample_length = stride_len3,
	#                              init = 'uniform'
	#                              )
	#
	# layer = sharedConv3( layer )
	#
	# layer = Dropout( 0.35 )( layer )
	#
	# layer = MaxPooling1D( pool_length = 2 )( layer )
	#
	# sharedConv4 = Convolution1D( nb_filter = num_filters4,
	#                              filter_length = filter_length4,
	#                              border_mode = 'valid',
	#                              activation = 'relu',
	#                              subsample_length = stride_len4,
	#                              init = 'uniform',
	#
	#                              )
	#
	# layer = sharedConv4( layer )
	#
	# layer = Dropout( 0.35 )( layer )
	#
	# layer = MaxPooling1D( pool_length = 2 )( layer )
	#
	# layer = Flatten( )( layer )
	#
	# # Dense layers default to 'glorot_normal' for init weights but that may not be optimal
	# # for NLP tasks
	# sharedDense1 = Dense( dense_dims1, init = 'uniform', activation = 'relu',
	#                       W_regularizer = l2( l = 0.0001 ) )
	#
	# layer = sharedDense1( layer )
	#
	# # layer = Dropout( 0.35 )( layer )
	#
	#
	#
	# sharedDense2 = Dense( dense_dims2, init = 'uniform', activation = 'relu',
	#                       W_regularizer = l2( l = 0.0001 ) )
	#
	# out = sharedDense2( layer )
	# #
	# # layer = Dropout( 0.35 )( layer )
	# #
	# # sharedDense3 = Dense( dense_dims3, activation = 'relu' )
	# #
	# # out = sharedDense3( layer )
	#
	# # TODO removed sentiment label info for now
	# #sentiment label is concatenated onto output vector of the prior fully connected layer
	# #out = merge( [ layer, sentiment_prob_input ], mode = 'concat',concat_axis = 1, name = "cnn_output" )
	#
	#
	# #TODO with sentiment label info added--model inputs are [review_input,sentiment_prob_input]
	#
	# CNN_model = Model( input = [ review_input ], output = out, name = "CNN_model" )

	CNN_model = build_CNN_model('1hotVector')


	Lreview = Input( shape = (maxReviewLen,), dtype = 'int32', name = "Lreview" )
	Rreview = Input( shape = (maxReviewLen,), dtype = 'int32', name = "Rreview" )


	#TODO removed sentiment label info for now
	#Lsentiment_prob = Input( shape = (1,), dtype = 'float32', name = "Lsentprob" )
	#TODO removed sentiment label info for now
	#Rsentiment_prob = Input( shape = (1,), dtype = 'float32', name = "Rsentprob" )




	#TODO with sentiment label info added--CNN_model is CNN_model([review,sentiment_prob])
	rightbranch = CNN_model  # ( [ Rreview ] )
	leftbranch = CNN_model  # ( [ Lreview ] )

	#first take the difference of the final feature representations from the CNN_model
	#represented by leftbranch and rightbranch
	merged_vector = merge( [ leftbranch, rightbranch ], mode = vectorDifference, output_shape = merged_outshape,
	                       name = 'merged_vector' )

	# then that difference vector is fed into the final fully connected layer that
	# outputs the energy i.e. squared euclidian distance ||leftbranch-rightbranch||
	siamese_out = Dense( 1, activation = squaredl2,
	                     name = 'energy_output' )( merged_vector )

	#TODO if sentiment label info included then inputs=[Lreview,Lsent_prob,Rreview,Rsent_prob]
	siamese_model = Model( input = [ Lreview, Rreview ], output = siamese_out,
	                       name = "siamese_model" )


	#TODO SGD is used in Lecunns paper; I am using RMSPROP instead for now
	#sgd = SGD( lr = 0.001, momentum = 0.0, decay = 0.0, nesterov = False )
	siamese_model.compile( optimizer = 'rmsprop', loss = contrastiveLoss )

	return { 'siamese': siamese_model, 'CNN': CNN_model,
	         'data'   : (trainingSets, devSets, devKNNsets, testSets) }


def train_siamese_model( model, trainingSets, devSets ):



	#X_left and X_right are matrices with trainingSet rows and reviewLen columns
	#y_left and y_right are the corresponding sentiment labels i.e 0:negative 1:positive
	#similarity is 0 if X_left and X_right have same sentiment labels and 1 otherwise
	X_left, y_left, X_right, y_right, similarity = trainingSets


	#Xdev_left and Xdev_right are matrices with devSet rows and reviewLen columns
	Xdev_left, ydev_left, Xdev_right, ydev_right, dev_similarity = devSets


	weightPath = './model_data/saved_weights/' + filename
	checkpoint = ModelCheckpoint( weightPath + '_W.{epoch:02d}-{val_loss:.3f}.hdf5',
	                              verbose = 1, )
	earlyStop = EarlyStopping( patience = 1, verbose = 1 )

	call_backs = [ checkpoint, earlyStop ]

	# TODO if sent label included then the input dictionary includes
	#{'Lsentprob': y_left, 'Rsentprob': y_right} and same for validation data inputs

	try:
		hist = model[ 'siamese' ].fit( { 'Lreview': X_left, 'Rreview': X_right, },
		                               { 'energy_output': similarity },
		                               batch_size = batch_size,
		                               nb_epoch = num_epochs,
		                               verbose = 1,
		                               validation_data =
		                               ({ 'Lreview': Xdev_left, 'Rreview': Xdev_right, },
		                                { 'energy_output': dev_similarity }),
		                               callbacks = call_backs
		                               )

		try:
			with open( os.path.join( './model_data/model_specs', filename + '.json' ), 'w' ) as f:
				f.write( model[ 'siamese' ].to_json( ) )
		except:
			print( "error writing model json" )
			pass

		try:
			with open( os.path.join( './model_data/model_specs', filename ) + '.hist', 'w' ) as f:
				f.write( str( hist.history ) )
		except:
			print( "error writing training history" )
			pass

	except KeyboardInterrupt:
		# hist is unitialized if here so return None in its place
		return trainingSets, devSets, None
	except:
		raise

	finally:

		with open( os.path.join( './model_data/model_specs', filename ) + '.config', 'w' ) as f:
			f.write( str( model[ 'siamese' ].get_config( ) ) )

		with open( os.path.join( './model_data/model_specs', filename ) + '.specs', 'w' ) as f:
			specs = """batch_size: {}\nembedding_dims: {}\ncontrast_margin: {}\n
			num_filters1: {}\nfilter_length1: {}\npool_len1: {}\n
			num_filters2: {}\nfilter_length2: {}\npool_len2: {}\n
			num_filters3: {}\nfilter_length3: {}\npool_len3: {}\n
			num_filters4: {}\nfilter_length4: {}\npool_len4: {}\n
			dense_dims1: {}\ndense_dims2: {}\ndense_dims3: {}\n""".format( batch_size,
			                                                               embedding_dims,
			                                                               modelParameters.Margin,
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
			f.write( specs )

	return trainingSets, devSets, hist
