"""
Created by John P Cavalieri

"""
from __future__ import print_function


from keras.models import Model
from keras.layers.core import (Dense, Dropout,
                               Activation, Flatten)
from keras.engine.topology import merge
from keras.layers.embeddings import Embedding
from keras.layers.convolutional import Convolution1D, MaxPooling1D
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers import Input
from keras.optimizers import SGD
from convert_review import build_design_matrix, build_siamese_input
from siamese_activations import vectorDifference,squaredl2,euclidDist
from loss_functions import contrastiveLoss
from siamese_utils import merged_outshape
import keras.backend as K
import numpy as np
import cPickle
import random
import modelParameters
import os
import datetime


DEVSPLIT = 14
USEWORDS = True

if USEWORDS:
	VocabSize = modelParameters.VocabSize_w
	maxReviewLen = modelParameters.MaxLen_w
	skipTop = modelParameters.skip_top
else:
	VocabSize = modelParameters.VocabSize_c
	maxReviewLen = modelParameters.MaxLen_c
	skipTop = 0







basename = "siamese_3_M{}".format(modelParameters.Margin)
suffix = datetime.datetime.now( ).strftime( "%y%m%d_%I%M" )
filename = "_".join( [ basename, suffix ] )

batch_size = 80

num_filters1 = 500
filter_length1 = 3
stride_len1 = 1
pool_len1 = 2

num_filters2 = 400
filter_length2 = 4
stride_len2 = 1
pool_len2 = 2

num_filters3 = 300
filter_length3 = 4
stride_len3 = 1
pool_len3 = 2

num_filters4 = 200
filter_length4 = 5
stride_len4 = 1
pool_len4 = 2

embedding_dims = 200

hidden_dims1 = 1000
hidden_dims2 = 250

num_epochs = 5


def build_siamese_model():


	print( 'Build model...' )

	review_input = Input( shape = (maxReviewLen,), dtype = 'int32', name = "review" )

	# probability of positive sentiment for left input and right input;
	# during training these are either 1 or 0 because we have that info in y_left and y_right
	# but during testing its 0.5 indicating equal probability of positive or negative
	#TODO currently not using but still thinking about how to use this information
	#sentiment_prob_input = Input( shape = (1,), dtype = 'float32', name = "sentprob" )

	sharedEmbedding = Embedding( VocabSize, embedding_dims,
	                             input_length = maxReviewLen )

	layer = sharedEmbedding( review_input )

	sharedConv1 = Convolution1D( nb_filter = num_filters1,
	                             filter_length = filter_length1,
	                             border_mode = 'same',
	                             activation = 'relu',
	                             subsample_length = stride_len1,
	                             init = 'uniform' )

	layer = sharedConv1( layer )

	layer = Dropout( 0.25 )( layer )

	layer = MaxPooling1D( pool_length = 2 )( layer )

	sharedConv2 = Convolution1D( nb_filter = num_filters2,
	                             filter_length = filter_length2,
	                             border_mode = 'same',
	                             activation = 'relu',
	                             subsample_length = stride_len2,
	                             init = 'uniform'
	                             )

	layer = sharedConv2( layer )

	layer = Dropout( 0.30 )( layer )

	layer = MaxPooling1D( pool_length = 2 )( layer )

	sharedConv3 = Convolution1D( nb_filter = num_filters3,
	                             filter_length = filter_length3,
	                             border_mode = 'same',
	                             activation = 'relu',
	                             subsample_length = stride_len3,
	                             init = 'uniform'
	                             )

	layer = sharedConv3( layer )

	layer = Dropout( 0.35 )( layer )

	layer = MaxPooling1D( pool_length = 2 )( layer )

	sharedConv4 = Convolution1D( nb_filter = num_filters4,
	                             filter_length = filter_length4,
	                             border_mode = 'same',
	                             activation = 'relu',
	                             subsample_length = stride_len4,
	                             init = 'uniform',

	                             )

	layer = sharedConv4( layer )

	layer = Dropout( 0.40 )( layer )

	layer = MaxPooling1D( pool_length = 2 )( layer )

	layer = Flatten( )( layer )

	sharedDense1 = Dense( hidden_dims1, activation = 'relu', )

	layer = sharedDense1( layer )

	layer = Dropout( 0.40 )( layer )

	sharedDense2 = Dense( hidden_dims2, activation = 'relu' )

	out = sharedDense2( layer )

	# TODO removed sentiment label info for now
	#sentiment label is concatenated onto output vector of the prior fully connected layer
	#out = merge( [ layer, sentiment_prob_input ], mode = 'concat',concat_axis = 1, name = "cnn_output" )


	#TODO with sentiment label info added--model inputs are [review_input,sentiment_prob_input]

	CNN_model = Model( input = [ review_input ], output = out, name = "CNN_model" )


	Lreview = Input( shape = (maxReviewLen,), dtype = 'int32', name = "Lreview" )
	Rreview = Input( shape = (maxReviewLen,), dtype = 'int32', name = "Rreview" )


	#TODO removed sentiment label info for now
	#Lsentiment_prob = Input( shape = (1,), dtype = 'float32', name = "Lsentprob" )
	#TODO removed sentiment label info for now
	#Rsentiment_prob = Input( shape = (1,), dtype = 'float32', name = "Rsentprob" )




	#TODO with sentiment label info added--CNN_model is CNN_model([review,sentiment_prob])
	rightbranch = CNN_model( [ Rreview ] )
	leftbranch = CNN_model( [ Lreview ] )

	#first take the difference of the final feature representations from the CNN_model
	#represented by leftbranch and rightbranch
	merged_vector = merge( [ leftbranch, rightbranch ], mode = vectorDifference, output_shape = merged_outshape,
	                       name = 'merged_vector' )

	#then that difference vector is fed into the final fully connected layer that
	#outputs the energy i.e. squared euclidian distance ||leftbranch-rightbranch||
	siamese_out = Dense( 1, activation = squaredl2, name = 'energy_output' )( merged_vector )



	#TODO if sentiment label info included then inputs=[Lreview,Lsent_prob,Rreview,Rsent_prob]
	siamese_model = Model( input = [ Lreview, Rreview ], output = siamese_out,
	                       name = "siamese_model" )


	#TODO SGD is used in Lecunns paper; I am using RMSPROP instead for now
	#sgd = SGD( lr = 0.001, momentum = 0.0, decay = 0.0, nesterov = False )
	siamese_model.compile( optimizer = 'rmsprop', loss = contrastiveLoss )

	return {'siamese':siamese_model,'CNN':CNN_model}





def train_siamese_model(model):

	print( 'Loading data...' )

	((trainingSets), (devSets)) = build_siamese_input( VocabSize,
	                                                   useWords = USEWORDS,
	                                                   skipTop = skipTop,
	                                                   devSplit = DEVSPLIT )


	#X_left and X_right are matrices with trainingSet rows and reviewLen columns
	#y_left and y_right are the corresponding sentiment labels i.e 0:negative 1:positive
	#similarity is 0 if X_left and X_right have same sentiment labels and 1 otherwise
	X_left, y_left, X_right, y_right, similarity = trainingSets

	#Xdev_left and Xdev_right are matrices with devSet rows and reviewLen columns
	Xdev_left, ydev_left, Xdev_right, ydev_right, dev_similarity = devSets

	print( len( X_left ), 'train sequences length' )
	print( len( Xdev_left ), 'dev sequences length' )

	print( 'train shape:', X_left.shape )
	print( 'dev shape:', Xdev_left.shape )

	weightPath = './model_data/saved_weights/' + filename
	checkpoint = ModelCheckpoint( weightPath + '_W.{epoch:02d}-{val_loss:.3f}.hdf5',
	                              verbose = 1, )
	earlyStop = EarlyStopping( patience = 1, verbose = 1 )

	call_backs = [ checkpoint, earlyStop ]



	#TODO is sent label included then the input dictions includes the following:
	#{'Lsentprob': y_left, 'Rsentprob': y_right} and same for validation data inputs

	hist = model['siamese'].fit( { 'Lreview': X_left, 'Rreview': X_right,  },
	                          { 'energy_output': similarity },
	                          batch_size = batch_size,
	                          nb_epoch = num_epochs,
	                          verbose = 1,
	                          validation_data =
	                          ({ 'Lreview'  : Xdev_left, 'Rreview': Xdev_right,  },
	                           { 'energy_output': dev_similarity }),
	                          callbacks = call_backs
	                          )

	with open( os.path.join( './model_data/model_specs', filename ) + '.config', 'w' ) as f:
		f.write( str( model['siamese'].get_config( ) ) )

	with open( os.path.join( './model_data/model_specs', filename + '.json' ), 'w' ) as f:
		f.write( model['siamese'].to_json( ) )

	with open( os.path.join( './model_data/model_specs', filename ) + '.hist', 'w' ) as f:
		f.write( str( hist.history ) )

	return trainingSets,devSets,hist
