"""
Created by John P Cavalieri on 5/31/16

"""

import numpy as np
import theano
import theano.tensor as T

from network_utils import get_network_layer_output


def get_embeddings(model, X):
	return get_network_layer_output(model, X, 0)


def identify_highprob_subset(model, X, y, subset_size):
	"""

	:param model:
	:param X:
	:param y:
	:param subset_size:
	:return:
	"""
	probs = model.predict_proba(X)

	indices = np.arange(start=0, stop=len(X), dtype='int32').reshape((-1, 1))

	prob_vs_index = np.concatenate((probs.reshape((-1, 1)), indices), axis=1)

	posTargets = y[y == 1].reshape((-1, 1))

	# posExamples = prob_vs_index[ np.isclose(prob_vs_index[ :, 1 ], 1), : ]
	posExamples = prob_vs_index[posTargets, :].reshape((-1, 1)).sort(axis=0)
	# posExamples.sort(axis=0)

	HPindices = posExamples[:, 1]

	X_subset = X[np.array(HPindices[-1:-subset_size:-1, 0], dtype='int32'), :, :]

	remaining_X = X[np.array(HPindices[-subset_size:0:-1, 0], dtype='int32'), :, :]
	remaining_y = y[np.array(HPindices[-subset_size:0:-1, 0], dtype='int32'), :]

	return (X_subset, (remaining_X, remaining_y))





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


	beta = 0.01
	alpha = 0.01

	X = theano.shared( X, name = 'X_designMatrix' )
	y = T.ivector( name = 'y_targetvect' )

	posClass_probability = trained_model.output  # Probability that input is positive class
	prediction = posClass_probability > 0.5  # The prediction thresholded at 0.50
	neg_crossEntropy_loss = -1 * (-y * T.log( posClass_probability ) - (1 - y) * T.log( 1 - posClass_probability )
	                              )
	cost = neg_crossEntropy_loss.mean()

	# gradient of mean negative cross entropy loss wrt inputs--not weights
	dCost_dX = T.grad(cost, [trained_model.input])

	train = theano.function(
			inputs = [ trained_model.input, y ],
		outputs=[prediction, cost],
		updates=((X, X - alpha * dCost_dX),),
			name = "train_data_func" )

	for batch_itr in range((len(highProb_subset) / (batch_size)) + 1):

		pred, cost = train([X[batch_itr * batch_size:(batch_itr + 1) * batch_size, :, :],
		                    targets[batch_itr * batch_size:(batch_itr + 1) * batch_size, :, :]])

		for p, c in zip(pred, cost):
			print((p, c) + '\n')


def dataSGD_test():
	from CNN_network_architecture import build_CNN_model
	fullModel, _, _, Xd, yd = build_CNN_model(weight_path='./CNN_weights.h5')
	Model = build_CNN_model(True, False, weight_path='./noembed.h5')

	embeddings = get_embeddings(fullModel, Xd)
