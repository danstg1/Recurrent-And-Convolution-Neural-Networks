import sys
from packaging import version
import tensorflow as tf
t = tf.constant([[1., 2., 3.], [4., 5., 6.]]) # matrix

"""
Custom Hubler Loss Func.
"""
def huber_fn(y_true, y_pred):
    error = y_true - y_pred
    is_small_error = tf.abs(error) < 1
    squared_loss = tf.square(error) / 2
    linear_loss  = tf.abs(error) - 0.5
    return tf.where(is_small_error, squared_loss, linear_loss)

