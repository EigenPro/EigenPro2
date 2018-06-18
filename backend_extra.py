import tensorflow as tf
from tensorflow.python.client import device_lib

def scatter_update(ref, indices, updates):
    """Update the value of `ref` at indecies to `updates`.
    """
    return tf.scatter_update(ref, indices, updates)

def hasGPU():
    devs = device_lib.list_local_devices()
    return any([dev.device_type == u'GPU' for dev in devs])
