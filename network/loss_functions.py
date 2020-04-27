# Author: Aqeel Anwar(ICSRL)
# Created: 2/22/2019, 4:57 PM
# Email: aqeel.anwar@gatech.edu
import numpy as np
import tensorflow as tf


def huber_loss(X, Y):
    err = X - Y
    loss = tf.where(tf.abs(err) < 1.0, 0.5 * tf.square(err), tf.abs(err) - 0.5)
    loss = tf.reduce_sum(loss)

    return loss


def mse_loss(X, Y):
    err = X - Y
    return tf.reduce_sum(tf.square(err))
