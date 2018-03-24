import tensorflow as tf 
import numpy as np 


""" Func1: 2D Convolutional layer
"""

def conv2(inp, filters, k_size, padding, activation):

	if activation == "relu":
		activation = tf.nn.relu

	return(tf.layers.conv2d(
			inputs = inp,
			filters = int(filters),
			kernel_size = k_size,
			padding = padding,
			activation= activation
			))
	 
def logit(inp, dim):
	return(tf.layers.dense(
					inputs = inp, units = dim))



def dense(inp, dim, activation):
	"""
	inp = input
	dim  = units

	"""
	if activation == "relu":
		activation = tf.nn.relu

	return(tf.layers.dense(
					inputs = inp, units = dim,
					activation = activation))

def dropout(inp, rate, mode):
	return(tf.layers.dropout(
	      inputs=inp, rate=rate, training=mode == tf.estimator.ModeKeys.TRAIN))