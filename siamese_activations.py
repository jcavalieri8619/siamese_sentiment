"""
Created by John P Cavalieri on 5/27/16

"""
import keras.backend as K
from keras import initializations
from keras.engine.topology import Layer


def euclidDist( inputs ):
	assert len(inputs)==2,"euclidDist requires 2 inputs"
	l1 = inputs[ 0 ]
	l2 = inputs[ 1 ]
	x = l1 - l2
	output = K.batch_dot( x, x, axes = 1 )
	K.reshape( output, (1,) )
	return output


def vectorDifference( inputs ):
	assert len( inputs ) == 2, "vectorDifference requires 2 inputs"
	l1 = inputs[ 0 ]
	l2 = inputs[ 1 ]
	x = l1 - l2
	return x


def squaredl2( X ):
	output = K.batch_dot( X, X, axes = 1 )
	K.reshape( output, (1,) )
	return output


class MahalanobisDist( Layer ):
	'''
	# Input shape
		Arbitrary. Use the keyword argument `input_shape`
		(tuple of integers, does not include the samples axis)
		when using this layer as the first layer in a model.
	# Output shape
		Same shape as the input.
	# Arguments
		init: initialization function for the weights.
		weights: initial weights, as a list of a single numpy array.

	'''

	def __init__( self, init = 'zero', weights = None, **kwargs ):
		self.supports_masking = False
		self.init = initializations.get( init )
		self.initial_weights = weights
		super( MahalanobisDist, self ).__init__( **kwargs )

	def build( self, input_shape ):
		self.sigma = self.init( (input_shape[ 1 ], input_shape[ 1 ]),
		                        name = '{}_sigma'.format( self.name ) )
		self.trainable_weights = [ self.sigma ]

		if self.initial_weights is not None:
			self.set_weights( self.initial_weights )
			del self.initial_weights

	def call( self, x, mask = None ):
		output = K.T.transpose( x, axes = 1 ) * self.sigma * x
		# K.reshape( output, (1,) )
		return output

	def get_config( self ):
		config = { 'init': self.init.__name__ }
		base_config = super( MahalanobisDist, self ).get_config( )
		return dict( list( base_config.items( ) ) + list( config.items( ) ) )
