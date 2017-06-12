"""
Created by John P Cavalieri on 5/31/16

"""

import copy
import os
import shutil

import h5py
import keras.backend as K
import numpy as np
import theano
import theano.tensor as T
from keras.objectives import constrained_loss

from network_utils import Batch_EarlyStopping
from network_utils import get_network_layer_output


def get_embedding_matrices(model, X):
    return get_network_layer_output(model, X, 1)


def identify_highprob_subset(model, X, y, subset_size, **kwargs):
    """

    :embedW model: trained CNN model with inputType 1hotVectors
    :embedW X: design matrix
    :embedW y: targets vector
    :embedW subset_size: subset size of correctly predicted inputs w/ high probability
    :return: tuple with matrix of shape (subset_size,maxreviewLength) representing high probability inputs and
            remaining data pairs not included in high prob subset
    """

    DEBUG_MODE = kwargs.get("DEBUG", False)

    if DEBUG_MODE:
        print("feeding forward inputs to identifying most optimal subset")

    probs = model.predict(X)

    if DEBUG_MODE:
        print("finished feeding forward")

    indices = np.arange(start=0, stop=len(X), dtype='int32').reshape((-1, 1))

    prob_vs_index = np.concatenate((probs.reshape((-1, 1)), indices), axis=1)

    targets = y.reshape((-1,))

    posExamples = prob_vs_index[targets == 1, :]
    CorrectPosExamples = posExamples[posExamples[:, 0] >= 0.5]

    # sorted from least probable to most probable--only want most probable
    CorrectPosExamples.sort(axis=0)

    if DEBUG_MODE:
        print("CorrectPosExamples shape: {}".format(CorrectPosExamples.shape))
        print(CorrectPosExamples[:50])
        print(CorrectPosExamples[-1:-50:-1])

    negExamples = prob_vs_index[targets == 0, :]
    CorrectNegExamples = negExamples[negExamples[:, 0] < 0.5]
    if DEBUG_MODE:
        print("CorrectNegExamples shape: {}".format(CorrectNegExamples.shape))
    # sorted from most probable to least probably--only want most probable
    CorrectNegExamples.sort(axis=0)

    highProb_pos_indices = CorrectPosExamples[:, 1]

    highProb_neg_indices = CorrectNegExamples[:, 1]

    X_pos_subset = X[highProb_pos_indices[-1:-(subset_size + 1):-1].astype('int32'), :]

    y_pos_subset = np.ones((X_pos_subset.shape[0],), dtype='int32')

    # X_neg_subset = X[highProb_neg_indices[:subset_size / 2].astype( 'int32' ), :]
    #
    # y_neg_subset = np.zeros( (X_neg_subset.shape[0],), dtype='int32' )

    remaining_X = np.concatenate((X[highProb_pos_indices[-(subset_size / 2 + 1):0:-1].astype('int32'), :],
                                  X[highProb_neg_indices[subset_size / 2:].astype('int32'), :]),
                                 axis=0)

    remaining_y = np.concatenate((y[highProb_pos_indices[-(subset_size / 2 + 1):0:-1].astype('int32'), :],
                                  y[highProb_neg_indices[subset_size / 2:].astype('int32'), :]),
                                 axis=0)

    # X_subset = np.concatenate( (X_pos_subset, X_neg_subset), axis=0 )
    #
    # y_subset = np.concatenate( (y_pos_subset, y_neg_subset), axis=0 )

    return (X_pos_subset, y_pos_subset), (remaining_X, remaining_y)


# todo data_SGD is broke--computing gradients of each layer's output wrt that layers input is proving difficult
def data_SGD(trained_model, optimal_subset, loss_func,
             num_epochs, batch_size, epsilon, **kwargs):
    """
    computes gradients of loss_fn function wrt input data X--the loss_fn function is designed to perturb the embedding
    matrix X such that the loss_fn can be arbitarily increased while maintaining the constrainst that the perturbed
    embedding matrix X_perturb is such that ||X_perturb - X|| < epsilon where X is the original unperturbed
    embedding matrix.

    :embedW trained_model: trained CNN model with inputType embeddingMatrix
    :embedW optimal_subset: set of pairs (X,y=1) not seen during training such that Prob{ y = 1 | CNN(X) } is approx 1
    :embedW loss_func: objective function: f(ytrue,ypred) i.e. cross entropy, SSE, ...
    :embedW num_epochs: number of epochs to perturb data
    :embedW batch_size: number of examples to perturb per batch of SGD
    :embedW epsilon: || X_perturb - X || < epsilon
    :embedW kwargs:
    :return: perturbed embedding matrix
    """

    DEBUG_MODE = kwargs.get("DEBUG", False)

    targets = optimal_subset[1]

    targets = targets.reshape((-1, 1)).astype(theano.config.floatX)

    original_input = optimal_subset[0].astype(theano.config.floatX)

    designMatrix = copy.deepcopy(original_input)
    designMatrix = designMatrix.astype(theano.config.floatX)

    # beta is a sort of regularization parameter for controlling importance of constrain_fn ||Xperturb - Xorig|| < eps
    # alpha_X and alpha_L are learning rates for SGD wrt inputs X and lagrange multiplier
    beta = 0.01
    alpha_X = 0.01
    alpha_L = 0.01

    batched_sharedvar_inputs = list()
    batched_reg_inputs = list()

    lagrange_mult = theano.shared(0.5, name="lagrange_mult")

    # trained model output is the probability that input is in positive class
    posClass_probability = trained_model.layers[-1].output

    # predictions are thresholded at 0.5
    prediction = posClass_probability > 0.5

    layerGradients_epoch = {}
    epoch_results = []
    for epoch in range(num_epochs):

        layerGradients_batch = []
        for batch_itr in range((len(targets) / batch_size)):

            start = batch_itr * batch_size
            end = (batch_itr + 1) * batch_size

            if not epoch:
                batched_sharedvar_inputs.append(theano.shared(designMatrix[start:end, :, :],
                                                              name='X_designMatrix_{}'.format(batch_itr)))

                batched_reg_inputs.append(original_input[start:end, :, :])

                if DEBUG_MODE:
                    print(
                        "batched design matrix shape: {}".format(
                            batched_sharedvar_inputs[batch_itr].get_value().shape))

            # computes tensor difference then sums along the batch dimension to result in matrix so that matrix norm
            # can be computed on the difference. this may only make sense for batch_size == 1 (on-line learning)
            dimreduc_differene = T.sum((batched_sharedvar_inputs[batch_itr] - batched_reg_inputs[batch_itr]),
                                       axis=0, dtype=theano.config.floatX)

            # negates the loss_fn so that minimizing loss_fn is actually maximixing loss_fn and constrained the squared
            # Frobenius
            # norm between perturbed inputs and original inputs to be less than epsilon for small epsilon.
            constrained_negated_loss = (-1 * loss_func(trained_model.targets[0], posClass_probability) +
                                        lagrange_mult * (beta *
                                                         (T.nlinalg.trace(T.dot(dimreduc_differene,
                                                                                dimreduc_differene)) -
                                                          epsilon ** 2)))

            total_cost = T.mean(constrained_negated_loss, dtype=theano.config.floatX, )

            Dcost_Dlagrange = T.grad(cost=total_cost, wrt=lagrange_mult)

            DlayerOut_DlayerIn = [T.grad(cost=total_cost, wrt=trained_model.layers[-1].output)]

            num_gradients = len(trained_model.layers)

            # backpropagate the loss_fn wrt inputs
            for itr in range(num_gradients - 1, 0, -1):
                if DEBUG_MODE:
                    print("layer number: {}".format(itr))

                layer_output = trained_model.layers[itr].output

                if layer_output.ndim == 0:
                    element = theano.gradient.jacobian(layer_output, wrt=trained_model.layers[itr].input)
                    if DEBUG_MODE:
                        print("layer_output.ndim == 0")

                elif layer_output.ndim == 1:
                    element = theano.gradient.jacobian(layer_output, wrt=trained_model.layers[itr].input)
                    if DEBUG_MODE:
                        print("layer_output.ndim == 1")

                elif layer_output.ndim == 2:
                    # summing over batch dimension
                    element = theano.gradient.jacobian(
                        T.sum(layer_output, axis=0, dtype=theano.config.floatX, keepdims=False, ),
                        wrt=trained_model.layers[itr].input)
                    if DEBUG_MODE:
                        print("layer_output.ndim == 2")

                elif layer_output.ndim == 3:
                    element = theano.gradient.jacobian(expression=T.flatten(T.sum(layer_output, axis=0,
                                                                                  dtype=theano.config.floatX,
                                                                                  keepdims=False, ),
                                                                            ),
                                                       wrt=trained_model.layers[itr].input)

                    if DEBUG_MODE:
                        print("layer_output.ndim == 3")
                else:
                    raise RuntimeError("data_SGD: cannot handle ndim > 3")

                DlayerOut_DlayerIn.append(element)

            Dcost_DX = reduce(T.dot, DlayerOut_DlayerIn, 1.0)

            for layerGrad in DlayerOut_DlayerIn:
                f = theano.function([trained_model.layers[0].input, K.learning_phase()], [layerGrad])
                layerGradients_batch.append(f(batched_reg_inputs[batch_itr], 0))

            SGDtrain = theano.function(
                inputs=[trained_model.input, trained_model.targets[0], K.learning_phase()],
                outputs=[prediction, constrained_negated_loss],
                updates=((batched_sharedvar_inputs[batch_itr], batched_sharedvar_inputs[batch_itr] - alpha_X *
                          Dcost_DX),
                         (lagrange_mult, lagrange_mult - alpha_L * Dcost_Dlagrange)),
                name="SGDtrain", )

            predictionVect, lossVect = SGDtrain(batched_sharedvar_inputs[batch_itr], targets, 1)

            epoch_results.append((predictionVect, lossVect))

            if DEBUG_MODE:
                for pred, target, loss_fn in zip(predictionVect, targets, lossVect):
                    print("prediction {},target {}, loss_fn {}\n".format(pred, target, loss_fn))

        layerGradients_epoch[epoch] = layerGradients_batch

        return batched_sharedvar_inputs, layerGradients_epoch


def _convert_1hotWeights_to_embedWeights(weight_path):
    """
    the CNN model is always trained with input type 1hotVector because the model accepts
    movie reviews and these are naturally vectors with integer elements not real matrices so the
    embeddings must be learned. However, the only way to perturb movie reviews under the constrain_fn
    ||Xperturb - Xorig|| < epsilon is to start with a real embedding matrix--to achieve this the
    saved weights from 1hotVector type models are modified to accept a matrix as input and start
    with a convolutional layer.

    :embedW weight_path:
    :return:
    """

    fileName = os.path.basename(weight_path)
    filePath = os.path.dirname(weight_path)

    fileName = "embed_mod_" + fileName

    new_path = os.path.join(filePath, fileName)

    shutil.copyfile(weight_path, new_path)

    hdf5_file = h5py.File(new_path, 'r+')

    attr_key = hdf5_file.attrs.keys()[0]
    attr_vals = hdf5_file.attrs.values()[0].tolist()

    attr_vals.remove('embeddingLayer')
    attr_vals[0] = 'embedding_review'

    attr_array = np.asarray(attr_vals, 'str')

    hdf5_file.attrs[attr_key] = attr_array

    embedded_review_group = hdf5_file.create_group(u'/embedding_review')

    embedded_review_group.attrs[u'weight_names'] = np.asarray([], theano.config.floatX)

    hdf5_file.pop(u'1hot_review')

    hdf5_file.pop(u'embeddingLayer')

    hdf5_file.flush()
    hdf5_file.close()

    return new_path


def perturb_data(loss_func, optimizer, model, optimal_subset, numEpochs, batchSize,
                 constrainWeight, epsilon, invertTargets, negateLoss,
                 ):
    """
    the model parameter is pre-trained model that minimizes loss. By freezing the weights of all layers except the
    embedding layer and training with either inverted target vector or negated loss, we can perturb the embedding
    layer weights to the input data as a perturbed embedding matrix--the one-hot vector input type cannot be
    perturbed because its values are integers whereas the embedding matrix is real. The embedding layer weight
    matrix functions as a lookup-table so that each row corresponds to a one-hot index--by perturbing the
    weight matrix/lookup-table we can construct perturbed embedding matrix inputs using the original one-hot vector
    input types and mapping them to perturbed embedding matrix input types via the perturbed
    weight matrix/lookup-table

    :param model:
    :param optimal_subset:
    :param loss_func:
    :param optimizer:
    :param numEpochs:
    :param constrainWeight:
    :param epsilon:
    :param invertTargets:
    :param negateLoss:
    :return:
    """

    from operator import neg

    def constrain_fn(embedW, embedW_orig, eps):
        return K.T.nlinalg.trace(K.dot(K.transpose(embedW_orig - embedW), embedW_orig - embedW)) - eps

    # freeze the weights of every layer except embedding layer
    for idx, L in enumerate(model.layers):
        if idx == 0:
            continue
        L.trainable = False

    assert model.layers[0].trainable is True, "embedding layer must be trainable"

    constrain_fn_args = [model.layers[0].W,
                         model.layers[0].W.get_value(), epsilon]

    loss = constrained_loss((lambda ytru, ypred: neg(loss_func(ytru, ypred)) if negateLoss else loss_func(ytru, ypred)),
                            constrain_fn, constrain_fn_args,
                            constraint_weight=constrainWeight)

    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

    targets = optimal_subset[1]
    onehotVectors = optimal_subset[0]

    if invertTargets:
        targets = np.logical_not(targets)

    callbacks = [Batch_EarlyStopping(verbose=1)]

    model.fit(onehotVectors, targets, nb_epoch=numEpochs, batch_size=batchSize, callbacks=callbacks)

    return model


# now that contrainWeight is equivalent to regularization param and not a learnable lagrange multiplier,
# no need for epsilon--set to zero for now but probably should just remove
def perturb_testing(loss_fn, optimizer, invertTargets, negateLoss, numEpochs, batchSize, subset_size,
                    weightPath, constrainWeight=5.0, epsilon=0.0, ):
    """

    :param loss_fn:
    :param optimizer:
    :param invertTargets:
    :param negateLoss:
    :param numEpochs:
    :param batchSize:
    :param subset_size:
    :param weightPath:
    :param constrainWeight:
    :param epsilon:
    :return:
    """

    from CNN_model import build_CNN_model
    from model_input_builders import build_CNN_input

    modelInputs = build_CNN_input()

    trained_1hotModel = build_CNN_model('1hotVector', load_weight_path=weightPath)
    trained_model_orig = build_CNN_model('1hotVector', load_weight_path=weightPath)

    print("identifying an optimal subset of inputs (X,y=1) to perturb")

    highprob_subset, _ = identify_highprob_subset(trained_1hotModel, modelInputs['dev'][0], modelInputs['dev'][1],
                                                  subset_size, DEBUG=True)

    perturbedModel = perturb_data(loss_fn, optimizer, trained_1hotModel, highprob_subset, numEpochs,
                                  batchSize, constrainWeight, epsilon, invertTargets, negateLoss)

    return perturbedModel, trained_model_orig, highprob_subset


def construct_perturbed_input(perturb_mapping, onehot_vectors):
    """

    :param perturb_mapping:
    :param onehot_vectors:
    :return:
    """

    return K.gather(perturb_mapping, onehot_vectors)
