"""
Created by John P Cavalieri on 6/3/16

"""
import keras.backend as K


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
