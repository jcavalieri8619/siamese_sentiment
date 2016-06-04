"""
Created by John P Cavalieri on 5/31/16

"""

import keras.backend as K
import numpy as np


def identify_highprob_subset( model, X_dev, y_dev, subset_size ):
	"""

	:param model:
	:param X_dev:
	:param y_dev:
	:param subset_size:
	:return:
	"""
	probs = model.predict_proba( X_dev )

	indices = np.arange( start = 0, stop = len( X_dev ), ).reshape( (-1, 1) )

	prob_vs_label = np.concatenate( (probs.reshape( (-1, 1) ), y_dev, indices), axis = 1 )

	posExamples = prob_vs_label[ prob_vs_label[ :, 1 ] == 1, : ]

	posExamples.sort( 0 )

	X_subset = X_dev[ posExamples[ -1:-subset_size:-1, 2 ] ]

	remaining_X = X_dev[ posExamples[ -subset_size:0:-1, : ] ]
	remaining_y = y_dev[ posExamples[ -subset_size:0:-1, : ] ]

	return (X_subset, (remaining_X, remaining_y))


def get_network_layer_output( model, dataInput, layerNum, **kwargs ):
	get_output = K.function( [ model.layers[ 0 ].input, K.learning_phase( ) ],
	                         [ model.layers[ layerNum ].output ] )

	phase = kwargs.get( 'phase', None )

	if phase is None or phase == 'test':
		# output in test mode = 0
		layer_output = get_output( [ dataInput, 0 ] )[ 0 ]

	elif phase == 'train':
		# output in train mode = 1
		layer_output = get_output( [ dataInput, 1 ] )[ 0 ]

	return layer_output


def data_SGD( trained_model, perturb_data, training_data, loss_func, batch_size = 20, **kwargs ):
	"""


	:param trained_model: trained model that minimizes loss L(W;x,y) w.r.t weights W
	:param perturb_data: subset of data that was correctly classified with high probability
	by trained_model-list of embedded review matrices
	:param training_data:
	:param loss_func: loss function to be minimized w.r.t
	:param batch_size: batch size for stochastic gradient descent
	:return:
	"""

	# X_list = perturb_data
	# dim1,dim2 = X_list[0].shape
	# X_embed = np.zeros((len(X_list),dim1,dim2))
	#
	# for indx,matrix in enumerate(X_list):
	# 	X_embed[indx] = matrix


	# D = perturb_data[ (batch_itr) * batch_size: (batch_itr + 1) * batch_size ]
	#
	# Xembed_batch = np.stack( [ x[ 0 ] for x in D ] )
	# y_batch = np.asarray( [ x[ 1 ] for x in D ] )
	#
	# Xembed_batch = K.T._shared( Xembed_batch, name = 'Xembed_batch' )
	#
	# predictions = trained_model.predict_proba( Xbatch.get_value( ), batch_size )



	for X_perturb in perturb_data:
		for batch_itr in range( (len( training_data ) / batch_size) + 1 ):
			X_train, y_train = training_data[ (batch_itr) * batch_size: (batch_itr + 1) * batch_size ]
