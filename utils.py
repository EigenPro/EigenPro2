from __future__ import absolute_import

import gc
import numpy as np
import tensorflow as tf

from keras import backend as K
from keras.layers import Lambda, Input


def enable_xla():
    """Enable XLA optimization in the default session of Keras."""
    config = tf.ConfigProto()
    config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1
    K.set_session(tf.Session(config=config))


def loadvar(array, trainable=False, name=None):
    """Load numpy array to tensorflow variable without 2GB limitation

    Arguments:
        array: numpy array.
        trainable: boolean.

    Returns:
        var: tensorflow variable.
        load_var: function to load array in given seesion.
    """
    placeholder = tf.placeholder(dtype=array.dtype, shape=array.shape)
    var = tf.Variable(placeholder, trainable=trainable,
                      collections=[], name=name)

    load_var = lambda sess: sess.run(var.initializer, feed_dict={placeholder: array})
    return var, load_var


def loadvar_in_sess(array, trainable=False, sess=None, name=None):
    var, load_var = loadvar(array, trainable, name)
    if sess is None:
        sess = K.get_session() # Keras default session
    load_var(sess)
    return var


def add_index(X):
    """Append sample index as the last feature to data matrix.

    Arguments:
        X: matrix of shape (n_sample, n_feat).

    Returns:
        matrix of shape (n_sample, n_feat+1).
    """
    inx = np.reshape(np.arange(X.shape[0]), (-1, 1))
    return np.hstack([X, inx])


def separate_index(IX):
    """Separate the index feature from the indexed tensor matrix.

    Arguments:
        IX: matrix of shape (n_sample, n_feat+1).

    Returns:
        X: matrix of shape (n_sample, n_feat).
        index: vector of shape (n_sample,).
    """
    X = Lambda(lambda x: x[:, :-1])(IX)
    index = Lambda(lambda x: x[:, -1])(IX)
    return X, K.cast(index, dtype='int32')


def reset():
    """Reset the Keras session and release the GPU memory."""
    K.clear_session()
    reload(K)
    gc.collect()
