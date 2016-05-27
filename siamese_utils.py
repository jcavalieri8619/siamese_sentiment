"""
Created by John P Cavalieri on 5/27/16

"""
def merged_outshape( inputShapes ):
	shape = list( inputShapes )
	assert len( shape ) == 2,"merged_outShape: len inputShapes != 2"
	return shape[0]
