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
import copy


def data_SGD( trained_model, highProb_subset, loss_func, batch_size = 20,
              **kwargs ):
	"""

	:param trained_model:
	:param highProb_subset:
	:param loss_func:
	:param batch_size:
	:param num_epochs:
	:param kwargs:
	:return:
	"""

	X_list = highProb_subset
	dim1, dim2 = X_list[ 0 ].shape
	X = np.zeros( (len( X_list ), dim1, dim2) )
	targets = np.ones( (dim1, 1) )

	for indx, matrix in enumerate( X_list ):
		X[ indx ] = matrix

	X_orig = copy.deepcopy( X )

	beta = 0.01
	alpha = 0.01

	X = theano.shared( X, name = 'X_designMatrix' )
	y = T.ivector( name = 'y_targetvect' )

	posClass_probability = trained_model.output  # Probability that input is positive class
	prediction = posClass_probability > 0.5  # The prediction thresholded at 0.50
	neg_crossEntropy_loss = -1 * (-y * T.log( posClass_probability ) - (1 - y) * T.log( 1 - posClass_probability )
	                              )
	mean_neg_crossEntropy_loss = neg_crossEntropy_loss.mean( ) + beta * (1)

	# gradient of mean negative cross entropy loss wrt inputs--not weights
	dLoss_dX = T.grad( mean_neg_crossEntropy_loss, [ trained_model.input ] )

	train = theano.function(
			inputs = [ trained_model.input, y ],
			outputs = [ prediction, mean_neg_crossEntropy_loss ],
			updates = ((X, X - alpha * dLoss_dX),),
			name = "train_data_func" )

	continuePerturbing = True
	while continuePerturbing:
		for batch_itr in range( (len( highProb_subset ) / (batch_size)) + 1 ):

			pred, loss = train( [ X[ batch_itr * batch_size:(batch_itr + 1) * batch_size, :, : ],
			                      targets[ batch_itr * batch_size:(batch_itr + 1) * batch_size, :, : ] ] )

			epsilon = 0.001
			if T.sqrt( T.sum( T.sqr( X_orig - X ) ) ) >= epsilon:
				continuePerturbing = False
				break
