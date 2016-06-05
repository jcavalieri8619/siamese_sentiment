"""
Created by John P Cavalieri on 5/31/16

"""

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


import theano
import theano.tensor as T


def data_SGD( trained_model, best_inputs, loss_func, batch_size = 20, **kwargs ):



	# D = perturb_data[ (batch_itr) * batch_size: (batch_itr + 1) * batch_size ]
	#
	# Xembed_batch = np.stack( [ x[ 0 ] for x in D ] )
	# y_batch = np.asarray( [ x[ 1 ] for x in D ] )
	#
	# Xembed_batch = K.T._shared( Xembed_batch, name = 'Xembed_batch' )
	#
	# predictions = trained_model.predict_proba( Xbatch.get_value( ), batch_size )


	# X_train, y_train = training_data[ (batch_itr) * batch_size: (batch_itr + 1) * batch_size ]


	# shared_data=[]
	# for elem in perturb_data:
	# 	shared_data=theano.shared(elem,name = "X_perturb")
	#
	# for X_perturb in shared_data:

	# X = T.dmatrix( "X_dataSGD" )
	# y = T.dvector( "y_dataSGD" )

	X_list = best_inputs
	dim1, dim2 = X_list[ 0 ].shape
	X_embed = np.zeros( (len( X_list ), dim1, dim2) )
	y_embed = np.ones( (dim1, 1) )

	for indx, matrix in enumerate( X_list ):
		X_embed[ indx ] = matrix

	beta = 0.01
	alpha = 0.01

	X_embed = theano.shared( X_embed, name = 'X_embed' )

	p_1 = trained_model.output  # Probability that input is positive review
	prediction = p_1 > 0.5  # The prediction thresholded
	xent_loss = -y_embed * T.log( p_1 ) - (1 - y_embed) * T.log( 1 - p_1 )  # Cross-entropy loss function
	mean_xent_loss = -1 * xent_loss.mean( ) + beta * T.sqr( trained_model.input.norm( 2 ) )

	dL_dX = T.grad( mean_xent_loss, [ trained_model.input ] )

	train = theano.function(
			inputs = [ X_embed, y_embed ],
			outputs = [ prediction, mean_xent_loss ],
			updates = ((X_embed, X_embed - alpha * dL_dX),) )

	for batch_itr in range( (len( training_data ) / batch_size) + 1 ):
		pass
