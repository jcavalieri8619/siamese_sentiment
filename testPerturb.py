import keras.backend as K
import theano
from keras.losses import binary_crossentropy

from CNN_model import build_CNN_model
from model_input_builders import build_CNN_input

modelInputs = build_CNN_input(vocab_size=100000)
modelData = build_CNN_model('1hotVector', True, modelInputs)

model = modelData['model']

devData = modelInputs['dev']

Xdev = devData[0]
ydev = devData[1]

Xexample = Xdev[1]

Xexample = Xexample.reshape((1, len(Xexample)))
Xexample0 = Xexample.copy()


def frobNorm(A):
    return theano.tensor.nlinalg.trace(theano.tensor.dot(theano.tensor.transpose(A), A))


targets = model.targets[0]
Xorig = theano.tensor.constant(Xexample)
gamma = theano.tensor.constant(5.0, name='gamma', dtype='float32')

LOSS = binary_crossentropy(targets, model.output) + gamma * frobNorm(Xorig - model.input) ** 2

TrainFunc = theano.function([model.input, targets, K.learning_phase()], LOSS)
TrainFunc(Xexample0.astype('int32'), ydev[1].reshape((1, 1)), 0)
