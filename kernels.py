import numpy as np

from keras import backend as K

def D2(X, Y, Y2=None, YT=None):
    """ Calculate the pointwise (squared) distance.
    
    Arguments:
    	X: of shape (n_sample, n_feature).
    	Y: of shape (n_center, n_feature).
        Y2: of shape (1, n_center).
        YT: of shape (n_feature, n_center). 
    
    Returns:
    	pointwise distances (n_sample, n_center).
    """
    X2 = K.sum(K.square(X), axis = 1, keepdims=True)
    if Y2 is None:
        if X is Y:
            Y2 = X2
        else:
            Y2 = K.sum(K.square(Y), axis = 1, keepdims=True)
        Y2 = K.reshape(Y2, (1, K.shape(Y)[0]))
    if YT is None:
        YT = K.transpose(Y)
    d2 = K.reshape(X2, (K.shape(X)[0], 1)) \
       + Y2 - 2 * K.dot(X, YT) # x2 + y2 - 2xy
    return d2

def Gaussian(X, Y, s, dist2_f=D2):
    """ Gaussian kernel.
    
    Arguments:
    	X: of shape (n_sample, n_feature).
    	Y: of shape (n_center, n_feature).
    	s: kernel bandwidth.
    
    Returns:
    	kernel matrix of shape (n_sample, n_center).
    """
    assert s > 0
    
    d2 = dist2_f(X, Y)
    gamma = np.float32(1. / (2 * s ** 2))
    G = K.exp(-gamma * K.clip(d2, 0, None))
    return G

def Laplacian(X, Y, s, dist2_f=D2):
    """ Laplacian kernel.
    
    Arguments:
    	X: of shape (n_sample, n_feature).
    	Y: of shape (n_center, n_feature).
    	s: kernel bandwidth.
    
    Returns:
    	kernel matrix of shape (n_sample, n_center).
    """
    assert s > 0
    
    d2 = K.clip(dist2_f(X, Y), 0, None)
    d = K.sqrt(d2)
    G = K.exp(- d / s)
    return G

def Cauchy(X, Y, s, dist2_f=D2):
    """ Cauchy kernel.
    
    Arguments:
    	X: of shape (n_sample, n_feature).
    	Y: of shape (n_center, n_feature).
    	s: kernel bandwidth.
    
    Returns:
    	kernel matrix of shape (n_sample, n_center).
    """
    assert s > 0
    
    d2 = dist2_f(X, Y)
    s2 = np.float32(s**2)
    G = 1 / K.exp( 1 + K.clip(d2, 0, None) / s2)
    return G


def Dispersal(X, Y, s, gamma, dist2_f=D2):
    """ Dispersal kernel.
    
    Arguments:
        X: of shape (n_sample, n_feature).
        Y: of shape (n_center, n_feature).
        s: kernel bandwidth.
		gamma: dispersal factor.
    
    Returns:
        kernel matrix of shape (n_sample, n_center).
    """
    assert s > 0

    d2 = K.clip(dist2_f(X, Y), 0, None)
    d = K.pow(d2, gamma / 2.)
    G = K.exp(- d / np.float32(s))
    return G
