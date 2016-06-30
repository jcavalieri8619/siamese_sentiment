"""
Created by John P Cavalieri on 5/31/16

"""

import copy
import operator
from functools import reduce

import numpy as np
import theano
import theano.tensor as T

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



def get_embedding_matrices(model, X):
    return get_network_layer_output(model, X, 0)


def identify_highprob_subset(model, X, y, subset_size):
    """

    :param model: trained CNN model with inputType 1hotVectors
    :param X: design matrix
    :param y: targets vector
    :param subset_size: subset size of correctly predicted inputs w/ high probability
    :return: tuple with matrix of shape (subset_size,maxreviewLength) representing high probability inputs and
            remaining data pairs not included in high prob subset
    """
    probs = model.predict_proba(X)

    indices = np.arange(start=0, stop=len(X), dtype='int32').reshape((-1, 1))

    prob_vs_index = np.concatenate((probs.reshape((-1, 1)), indices), axis=1)

    posTargets = y.reshape((-1,))

    posExamples = prob_vs_index[posTargets == 1, :]
    posExamples.sort(axis=0)

    highProb_indices = posExamples[:, 1]

    X_subset = X[highProb_indices[-1:-subset_size:-1], :, :]

    remaining_X = X[highProb_indices[-subset_size:0:-1], :, :]
    remaining_y = y[highProb_indices[-subset_size:0:-1], :]

    return (X_subset, (remaining_X, remaining_y))


def data_SGD(trained_model, highProb_subset, loss_func, num_epochs, batch_size=20, epsilon=0.001,
             **kwargs):
    """
    get embeddings for highProb_subset and use trained model without embedding layer

    :param trained_model: trained CNN model with inputType embeddingMatrix
    :param highProb_subset: design matrix consisting of examples predicted correctly with high probability
    :param loss_func: objective function: f(ytrue,ypred) i.e. cross entropy, SSE, ...
    :param num_epochs: number of epochs to perturb data
    :param batch_size: number of examples in each training batch
    :param epsilon: || Xperturb - Xorig || < epsilon
    :param kwargs:
    :return: 3D tensor with shape (len(highProb_subset),maxreviewLength,embedding_dim) representing perturbation of
            high probability subset of inputs with constrained distance between perturbed inputs and original inputs
    """

    targets = np.ones((highProb_subset.shape[0], 1))
    designMatrix = copy.deepcopy(highProb_subset)

    beta = 0.01
    alpha_X = 0.01
    alpha_L = 0.01

    batched_inputs = list()
    lagrange_mult = theano.shared(0., name="lagrange_mult")
    y = T.ivector(name='y_targetvect')

    # trained model output is the probability that input is in positive class
    posClass_probability = trained_model.output
    # predictions are thresholded at 0.5
    prediction = posClass_probability > 0.5

    for epoch in num_epochs:

        for i, batch_itr in enumerate(range((len(highProb_subset) / batch_size) + 1)):

            batched_inputs.append(theano.shared(designMatrix[batch_itr * batch_size:(batch_itr + 1) * batch_size, :, :],
                                                name='X_designMatrix_{}'.format(i)))

            matrixDiff = (
            batched_inputs[i] - highProb_subset[batch_itr * batch_size:(batch_itr + 1) * batch_size, :, :])

            constrained_inverse_loss = (-1 * loss_func(trained_model.targets[0], posClass_probability) +
                                        beta * lagrange_mult * T.dot(matrixDiff, matrixDiff))

            total_cost = T.mean(constrained_inverse_loss, dtype='float32', acc_dtype='float32', )

            num_gradients = len(trained_model.layers)

            Dcost_Dlagrange = T.grad(cost=total_cost, wrt=[lagrange_mult])
            DlayerOut_DlayerIn = list()

            DlayerOut_DlayerIn.append(T.grad(cost=total_cost, wrt=[trained_model.layers[num_gradients - 1].input]))

            for itr in range(num_gradients - 1, 0, -1):
                out = trained_model.layers[itr].output

                if out.ndim == 0:
                    element = theano.gradient.jacobian(out, wrt=[trained_model.layers[itr - 1].input])

                elif out.ndim == 1:
                    element = theano.gradient.jacobian(out, wrt=[trained_model.layers[itr - 1].input])

                elif out.ndim == 2:
                    element = theano.gradient.jacobian(
                            T.sum(out, axis=1, dtype='float32', keepdims=True, acc_dtype='float32'),
                            wrt=[trained_model.layers[itr - 1].input])
                elif out.ndim == 3:
                    element = theano.gradient.jacobian(
                            T.sum(out, axis=2, dtype='float32', keepdims=True, acc_dtype='float32'),
                            wrt=[trained_model.layers[itr - 1].input])
                    # element = theano.gradient.jacobian(T.sum(T.sum(out, axis=0,dtype='float32',
                    #                                          keepdims=True,acc_dtype='float32'),
                    #                                          1,'float32',True,'float32'),
                    #                                    wrt=[trained_model.layers[itr - 1].input])
                else:
                    raise RuntimeError("data_SGD: cannot handle ndim > 3")

                DlayerOut_DlayerIn.append(element)

            Dcost_DX = reduce(operator.mul, DlayerOut_DlayerIn, 1.0)

            SGDtrain = theano.function(
                    inputs=[trained_model.input, trained_model.targets[0]],
                    outputs=[prediction, constrained_inverse_loss],
                    updates=((batched_inputs[i], batched_inputs[i] - alpha_X * Dcost_DX),
                             (lagrange_mult, lagrange_mult - alpha_L * Dcost_Dlagrange)),
                    name="SGDtrain", )







            #
            # for batch_itr in range((len(highProb_subset) / (batch_size)) + 1):
            #
            # 	pred, total_cost = train([X[batch_itr * batch_size:(batch_itr + 1) * batch_size, :, :],
            # 	                    targets[batch_itr * batch_size:(batch_itr + 1) * batch_size, :, :]])
            #
            # 	for p, c in zip(pred, total_cost):
            # 		print(str((p, c)) + '\n')


def dataSGD_test():
    pass
