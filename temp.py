 
from layer import *
from pool import *
from utils import *
import tensorflow as tf 
import numpy as np

mnist = tf.contrib.learn.dataset.load_dataset("mnist")

train_data = mnist.train.images
train_labels = np.asarray(mnist.train.labels, dtype = np.int32)

eval_data = mnist.test.images
eval_labels = np.asarray(mnist.test.labels, dtype = np.int32)
