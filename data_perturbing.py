"""
Created by John P Cavalieri on 5/31/16

"""

import keras.backend as K
import numpy as np


def identify_highprob_subset( model, X_test, y_test, subset_size ):
	"""

	:param model:
	:param X_test:
	:param y_test:
	:param subset_size:
	:return:
	"""
	probs = model.predict_proba( X_test )

	indices = np.arange( start = 0, stop = len( X_test ), ).reshape( (-1, 1) )

	prob_vs_label = np.concatenate( (probs.reshape( (-1, 1) ), y_test, indices), axis = 1 )

	posExamples = prob_vs_label[ prob_vs_label[ :, 1 ] == 1, : ]

	posExamples.sort( 0 )

	X_subset = X_test[ posExamples[ -1:-subset_size, 2 ] ]

	y_subset = np.ones_like( X_subset )

	return X_subset, y_subset


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


def data_SGD( trained_model, data, loss_func, batch_size, **kwargs ):
	"""


	:param trained_model: trained model that minimizes loss L(W;x,y) w.r.t weights W
	:param data: subset of data that was correctly classified with high probability
	by trained_model--list of tuples [(x,y)...]; x: data vector, y:label
	:param loss_func: loss function to be minimized w.r.t data x
	:param batch_size: batch size for stochastic gradient descent
	:return:
	"""

	for batch_itr in len( data ) / batch_size:
		D = data[ (batch_itr) * batch_size:(batch_itr + 1) * batch_size ]

		Xembed_batch = np.stack( [ x[ 0 ] for x in D ] )
		y_batch = np.asarray( [ x[ 1 ] for x in D ] )

		Xembed_batch = K.T._shared( Xembed_batch, name = 'Xembed_batch' )

		predictions = trained_model.predict_proba( Xbatch.get_value( ), batch_size )
