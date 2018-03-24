import tensorflow as tf 



def reshape(inp, shape):
	return(tf.reshape(inp, shape))


def prediction(inp, mode):
	predictions ={
	"classes": tf.argmax(input=inp, axis = 1),
	"probabilities": tf.nn.softmax(inp, name="softmax_tensor")
	}

	if mode == tf.estimator.ModeKeys.PREDICT:
		return (tf.estimator.EstimatorSpec(mode=mode, predictions=predictions),predictions)

	else: return(predictions) 


def loss_softmax(logits,onehot_labels):
	
	loss = tf.losses.softmax_cross_entropy(
		onehot_labels=onehot_labels, logits=logits)
	return loss 



def optimizer(function, lr, loss, mode):
	if mode == tf.estimator.ModeKeys.TRAIN:
		optimiser_  = tf.train.GradientDescentOptimizer(float(lr))
		train_op = optimiser_.minimize(loss=loss, 
			global_step = tf.train.get_global_step() )
		return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)





def metric(labels,loss,mode):

	pred = prediction(labels.astype(float32), mode)
	eval_ops = {
	"accuracy": tf.metrics.accuracy(
		labels = labels, predictions = pred)
	}

	return (tf.estimator.EstimatorSpec( mode = mode,
			loss = loss, eval_metric_ops= eval_ops))