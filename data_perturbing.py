"""
Created by John P Cavalieri on 5/31/16

"""

import copy
import h5py
import os
import shutil

import keras.backend as K
import numpy as np
import theano
import theano.tensor as T
from keras.objectives import binary_crossentropy

from network_utils import get_network_layer_output


# def updateParameters(param,learning_rate,momentum,cost):
# 	updates=[]
#
# 	param_update = theano.shared(param.get_value()* 0.0, broadcastable=param.broadcastable)
#
# 	updates.append((param, param - learning_rate * param_update))
#
# 	updates.append((param_update, momentum * param_update + (1. - momentum) * T.grad(cost, param)))
#
# 	return updates



def get_embedding_matrices( model, X ):
	return get_network_layer_output( model, X, 1 )


def identify_highprob_subset( model, X, y, subset_size, **kwargs ):
	"""

	:param model: trained CNN model with inputType 1hotVectors
	:param X: design matrix
	:param y: targets vector
	:param subset_size: subset size of correctly predicted inputs w/ high probability
	:return: tuple with matrix of shape (subset_size,maxreviewLength) representing high probability inputs and
			remaining data pairs not included in high prob subset
	"""

	DEBUG_MODE = kwargs.get( "DEBUG", False )

	if DEBUG_MODE:
		print("feeding forward inputs to identifying most optimal subset")

	probs = model.predict( X )

	if DEBUG_MODE:
		print("finished feeding forward")

	indices = np.arange( start=0, stop=len( X ), dtype='int32' ).reshape( (-1, 1) )

	prob_vs_index = np.concatenate( (probs.reshape( (-1, 1) ), indices), axis=1 )

	targets = y.reshape( (-1,) )

	posExamples = prob_vs_index[targets == 1, :]
	posExamples.sort( axis=0 )

	negExamples = prob_vs_index[targets == 0, :]
	negExamples.sort( axis=0 )

	highProb_pos_indices = posExamples[:, 1]

	highProb_neg_indices = negExamples[:, 1]

	X_pos_subset = X[highProb_pos_indices[-1:-(subset_size / 2 + 1):-1].astype( 'int32' ), :]

	y_pos_subset = np.ones( (X_pos_subset.shape[0],), dtype='int32' )

	X_neg_subset = X[highProb_neg_indices[:subset_size / 2].astype( 'int32' ), :]

	y_neg_subset = np.zeros( (X_neg_subset.shape[0],), dtype='int32' )

	remaining_X = np.concatenate( (X[highProb_pos_indices[-(subset_size / 2 + 1):0:-1].astype( 'int32' ), :],
	                               X[highProb_neg_indices[subset_size / 2:].astype( 'int32' ), :]),
	                              axis=0 )

	remaining_y = np.concatenate( (y[highProb_pos_indices[-(subset_size / 2 + 1):0:-1].astype( 'int32' ), :],
	                               y[highProb_neg_indices[subset_size / 2:].astype( 'int32' ), :]),
	                              axis=0 )

	X_subset = np.concatenate( (X_pos_subset, X_neg_subset), axis=0 )

	y_subset = np.concatenate( (y_pos_subset, y_neg_subset), axis=0 )

	return (X_subset, y_subset), (remaining_X, remaining_y)


def batched_matrixNorm( A, B ):
	pass


def data_SGD( trained_model, optimal_subset, loss_func,
              num_epochs, batch_size, epsilon, **kwargs ):
	"""
	get embeddings for optimal_subset and use embedding type trained model

	:param trained_model: trained CNN model with inputType embeddingMatrix
	:param optimal_subset: tuple of high prob X and y's
	:param loss_func: objective function: f(ytrue,ypred) i.e. cross entropy, SSE, ...
	:param num_epochs: number of epochs to perturb data
	:param batch_size: number of examples in each training batch
	:param epsilon: || Xperturb - Xorig || < epsilon
	:param kwargs:
	:return: 3D tensor with shape (len(highProb_subset),maxreviewLength,embedding_dim) representing perturbation of
			high probability inputs constrained distance between perturbed inputs and original inputs
	"""

	DEBUG_MODE = kwargs.get( "DEBUG", False )

	targets = optimal_subset[1]
	# theano is complaining that targets needs to be float32 type instead of int32...why?
	targets = targets.reshape( (-1, 1) ).astype( 'float32' )

	original_input = optimal_subset[0].astype( theano.config.floatX )

	designMatrix = copy.deepcopy( original_input )
	designMatrix = designMatrix.astype( theano.config.floatX )

	# beta is a sort of regularization parameter for controlling importance of constraint ||Xperturb - Xorig|| < eps
	# alpha_X and alpha_L are learning rates for SGD wrt inputs X and lagrange multiplier
	beta = 0.01
	alpha_X = 0.01
	alpha_L = 0.01

	batched_sharedvar_inputs = list( )
	batched_reg_inputs = list( )

	lagrange_mult = theano.shared( 0.5, name="lagrange_mult" )

	# trained model output is the probability that input is in positive class
	posClass_probability = trained_model.layers[-1].output

	# predictions are thresholded at 0.5
	prediction = posClass_probability > 0.5

	for epoch in range( num_epochs ):

		for batch_itr in range( (len( targets ) / batch_size) ):

			start = batch_itr * batch_size
			end = (batch_itr + 1) * batch_size

			if not epoch:
				batched_sharedvar_inputs.append( theano.shared( designMatrix[start:end, :, :],
				                                                name='X_designMatrix_{}'.format( batch_itr ) ) )

				batched_reg_inputs.append( original_input[start:end, :, :] )

				if DEBUG_MODE:
					print(
						"batched design matrix shape: {}".format(
								batched_sharedvar_inputs[batch_itr].get_value( ).shape ))

			# computes tensor difference then sums along the batch dimension to result in matrix so that matrix norm
			# can be computed on the difference. this may only make sense for batch_size == 1 (on-line learning)
			dimreduc_differene = T.sum( (batched_sharedvar_inputs[batch_itr] - batched_reg_inputs[batch_itr]),
			                            axis=0, dtype=theano.config.floatX )

			# negates the loss so that minimizing loss is actually maximixing loss and constrained the squared
			# Frobenius
			# norm between perturbed inputs and original inputs to be less than epsilon for small epsilon.
			constrained_negated_loss = (-1 * loss_func( trained_model.targets[0], posClass_probability ) +
			                            beta * lagrange_mult *
			                            (T.nlinalg.trace( T.dot( dimreduc_differene,
			                                                     dimreduc_differene ) ) - epsilon ** 2))

			total_cost = T.mean( constrained_negated_loss, dtype=theano.config.floatX, )

			Dcost_Dlagrange = T.grad( cost=total_cost, wrt=lagrange_mult )

			DlayerOut_DlayerIn = [T.grad( cost=total_cost, wrt=trained_model.layers[-1].output )]

			# gradLayers = [0, 1, 3, 4, 6, 7, 9, 10, 12, 14, 16, 17]
			# gradLayers.reverse()

			num_gradients = len( trained_model.layers )

			# backpropagate the loss wrt inputs
			for itr in range( num_gradients - 1, 0, -1 ):
				if DEBUG_MODE:
					print("layer number: {}".format( itr ))

				layer_output = trained_model.layers[itr].output

				if layer_output.ndim == 0:
					element = theano.gradient.jacobian( layer_output, wrt=trained_model.layers[itr].input )
					if DEBUG_MODE:
						print("layer_output.ndim == 0")

				elif layer_output.ndim == 1:
					element = theano.gradient.jacobian( layer_output, wrt=trained_model.layers[itr].input )
					if DEBUG_MODE:
						print("layer_output.ndim == 1")

				elif layer_output.ndim == 2:
					# summing over batch dimension
					element = theano.gradient.jacobian(
							T.sum( layer_output, axis=0, dtype=theano.config.floatX, keepdims=False, ),
							wrt=trained_model.layers[itr].input )
					if DEBUG_MODE:
						print("layer_output.ndim == 2")

				elif layer_output.ndim == 3:
					element = theano.gradient.jacobian( expression=T.flatten( T.sum( layer_output, axis=0,
					                                                                 dtype=theano.config.floatX,
					                                                                 keepdims=False, ),
					                                                          ),
					                                    wrt=trained_model.layers[itr].input )

					if DEBUG_MODE:
						print("layer_output.ndim == 3")
				else:
					raise RuntimeError( "data_SGD: cannot handle ndim > 3" )

				DlayerOut_DlayerIn.append( element )

			# Dcost_DX = reduce( T.dot, DlayerOut_DlayerIn, 1.0 )

			layerGradients = []
			i = 0
			for elem in DlayerOut_DlayerIn:
				f = theano.function( [trained_model.layers[0].input, K.learning_phase( )], [elem] )
				layerGradients.append( f( batched_reg_inputs[i], 0 ) )
				i += 1

			# testFunc = theano.function( [trained_model.layers[0].input, trained_model.targets[0], K.learning_phase(
			#  )],
			#                             Dcost_Dlagrange, name="perturbTestFunc" )
			#
			# Dlagrange = testFunc( batched_reg_inputs[batch_itr], targets[:20], 0 )

			return layerGradients

		# SGDtrain = theano.function(
		# 		inputs=[trained_model.input, trained_model.targets[0], K.learning_phase()],
		# 		outputs=[prediction, constrained_negated_loss],
		# 		updates=((batched_sharedvar_inputs[batch_itr], batched_sharedvar_inputs[batch_itr] - alpha_X *
		# Dcost_DX),
		# 		         (lagrange_mult, lagrange_mult - alpha_L * Dcost_Dlagrange)),
		# 		name="SGDtrain", )
		#
		# predictionVect, lossVect = SGDtrain(batched_sharedvar_inputs[batch_itr], targets, 1)
		#
		# for pred, target, loss in zip(predictionVect, targets, lossVect):
		# 	print("prediction {},target {}, loss {}\n".format(pred, target, loss))

		# return batched_sharedvar_inputs


def convert_1hotWeights_to_embedWeights( weight_path ):
	"""
	the CNN model is always trained with input type 1hotVector because the model accepts
	movie reviews and these are naturally vectors with integer elements not real matrices so the
	embeddings must be learned. However, the only way to perturb movie reviews under the constraint
	||Xperturb - Xorig|| < epsilon is to start with a real embedding matrix--to achieve this the
	saved weights from 1hotVector type models are modified to accept a matrix as input and start
	with a convolutional layer.

	:param weight_path:
	:return:
	"""

	fileName = os.path.basename( weight_path )
	filePath = os.path.dirname( weight_path )

	fileName = "embed_mod_" + fileName

	new_path = os.path.join( filePath, fileName )

	shutil.copyfile( weight_path, new_path )

	hdf5_file = h5py.File( new_path, 'r+' )

	attr_key = hdf5_file.attrs.keys( )[0]
	attr_vals = hdf5_file.attrs.values( )[0].tolist( )

	attr_vals.remove( 'embedding_1' )
	attr_vals[0] = 'embedding_review'

	attr_array = np.asarray( attr_vals, 'str' )

	hdf5_file.attrs[attr_key] = attr_array

	embedded_review_group = hdf5_file.create_group( u'/embedding_review' )

	embedded_review_group.attrs[u'weight_names'] = np.asarray( [], theano.config.floatX )

	hdf5_file.pop( u'1hot_review' )

	hdf5_file.pop( u'embedding_1' )

	hdf5_file.flush( )
	hdf5_file.close( )

	return new_path


def perturb_testing( ):
	from CNN_model import build_CNN_input, build_CNN_model

	print("initializing test")
	_, _, X, y, _, _ = build_CNN_input( testSet=False )
	trained_1hotModel = build_CNN_model( '1hotVector', weight_path='./model_data/saved_weights/CNN_0.27.hdf5' )
	trained_embedModel = build_CNN_model( 'embeddingMatrix',
	                                      weight_path='./model_data/saved_weights/noembed_0.27.hdf5' )

	subset_size = 1000

	print("identifying an optimal subset of inputs (X,y) to perturb")

	highprob_subset, (Xleftover, yleftover) = identify_highprob_subset( trained_1hotModel, X, y, subset_size,
	                                                                    DEBUG=True )

	print("mapping optimal inputs X from integer 1hot-vectors to real embedded matrices")
	X_highprob_embeddings = get_embedding_matrices( trained_1hotModel, highprob_subset[0] )

	print("perturbing data by minimizing negated cross entropy loss via SGD with L2 constraint between\n"
	      "perturbed inputs and original inputs enforced")
	retval = data_SGD( trained_embedModel, (X_highprob_embeddings, highprob_subset[1]), loss_func=binary_crossentropy,
	                   num_epochs=5, batch_size=1, epsilon=0.001, DEBUG=True )

	return retval
