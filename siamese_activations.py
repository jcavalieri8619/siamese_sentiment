"""
Created by John P Cavalieri on 5/27/16

"""
import keras.backend as K


def euclidDist( inputs ):
	assert len( inputs ) == 2, "euclidDist requires 2 inputs"
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


