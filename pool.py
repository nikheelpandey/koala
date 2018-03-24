import tensorflow as tf 
import numpy as np 

def pooling2d(inp, dim, steps, _type):

	"""
	inp = input 2Dlayer
	dim =  pool size
	steps = strides
	type =  max or avg
	"""
	if _type == "max":
		return(tf.layers.max_pooling2d(inputs = inp,
		pool_size = dim, strides = steps))

	else : 
		"""avgpooling"""
		return(tf.layers.average_pooling2d(inputs = inp,
		pool_size = dim, strides = steps))
