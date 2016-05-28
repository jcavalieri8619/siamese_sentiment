"""
Created by John P Cavalieri on 5/27/16

"""
import keras.backend as K

from modelParameters import Margin


def contrastiveLoss( y_true, y_pred ):
	x1 = y_true * y_pred
	x2 = (1 - y_true) * K.T.power( (K.maximum( 0, Margin - K.T.sqrt( y_pred ) )), 2 )
	return K.T.mean( 0.5 * (x1 + x2) )
