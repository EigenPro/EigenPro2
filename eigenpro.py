import numpy as np
import scipy as sp
import tensorflow as tf
import time

from keras import backend as K
from keras.layers import Dense, Input, Lambda
from keras.models import Model

import utils
from backend_extra import scatter_update
from layers import KernelEmbedding
from optimizers import PSGD


def pre_eigenpro_f(feat, phi, k, n, mG, alpha=.9, seed=1):
    """Prepare gradient map f for EigenPro and calculate
    scale factor for step size such that the update rule,
        p <- p - eta * g
    becomes,
        p <- p - scale * eta * (g - f(g))

    Arguments:
        feat:   feature matrix.
        phi:    feature map or kernel function.
        k:      top-k eigensystem for constructing eigenpro iteration/kernel.
        n:      number of training points.
        mG:     maxinum batch size corresponding to GPU memory. 
        alpha:  exponential factor (<= 1) for eigenvalue ratio.

    Returns:
        f:      tensor function.
        scale:  factor that rescales step size.
        s1:     largest eigenvalue.
        beta:   largest k(x, x) for the EigenPro kernel.
    """

    np.random.seed(seed) # set random seed for subsamples
    start = time.time()
    n_sample, d = feat.shape

    if k is None:
        svd_k = min(n_sample - 1, 1000)
    else:
        svd_k = k

    _s, _V = nystrom_kernel_svd(feat, phi, n, svd_k)

    # Choose k such that the batch size is bounded by
    #   the subsample size and the memory size.
    #   Keep the original k if it is pre-specified.
    if k is None:
        k = np.sum(n / _s < min(n_sample, mG)) - 1

    _s, _sk, _V = _s[:k-1], _s[k-1], _V[:, :k-1]

    s = K.constant(_s)
    V = utils.loadvar_in_sess(_V.astype('float32'))
    sk = K.constant(_sk)

    scale = np.power(_s[0] / _sk, alpha, dtype='float32')
    D = (1 - K.pow(sk / s, np.float32(alpha))) / s
    pre_f = lambda g, kfeat: K.dot(
        V * D, K.transpose(K.dot(K.dot(K.transpose(g), kfeat), V)))
    s1 = _s[0] / n
    print("SVD time: %.2f, adjusted k: %d, s1: %.2f, new s1: %.2e" %
          (time.time() - start, k, _s[0] / n, s1 / scale))

    kxx = 1 - np.sum(_V ** 2, axis=1) / n * feat.shape[0]
    beta = np.max(kxx)

    return pre_f, scale, s1, beta


def asm_eigenpro_f(pre_f, kfeat, inx):
    """Assemble map for EigenPro iteration"""
    def eigenpro_f(p, g, eta):
        inx_t = K.constant(inx, dtype='int32')

        kinx = tf.gather(kfeat, inx_t, axis=1)
        pinx = K.gather(p, inx_t)
        update_p =  pinx + eta * pre_f(g, kinx)
        new_p = scatter_update(p, inx, update_p)
        return new_p
    return eigenpro_f


def nystrom_kernel_svd(X, kernel_f, n, k, bs=512):
    """Compute top eigensystem of kernel matrix using Nystrom method.

    Arguments:
        X: data matrix of shape (n_sample, n_feature).
        kernel_f: kernel tensor function k(X, Y).
        n: number of training points.
        k: top-k eigensystem.
        bs: batch size.

    Returns:
        s: top eigenvalues of shape (k).
        U: top eigenvectors of shape (n_sample, k).
    """

    m, d = X.shape

    # Assemble kernel function evaluator.
    input_shape = (d, )
    x = Input(shape=input_shape, dtype='float32',
              name='feat-for-nystrom')
    K_t  = KernelEmbedding(kernel_f, X)(x)
    kernel_tf = Model(x, K_t)
    
    K = kernel_tf.predict(X, batch_size=bs)
    D = np.float32(np.ones((m, 1)) * np.sqrt(n) / np.sqrt(m))
    W = D * K * D.T
    w, V = sp.linalg.eigh(W, eigvals=(m-k, m-1))
    U1r, sr = V[:, ::-1], w[::-1]
    s = sr[:k]
    NU = np.float32(D * U1r[:, :k])

    return s, NU


class EigenPro(object):

    def __init__(self, kernel, centers, n_label, mem_gb,
                 n_subsample=None, k=None, bs=None,
                 metric='accuracy', scale=.5, seed=1):
        """Assemble learner using EigenPro iteration/kernel.

        Arguments:
            kernel: kernel tensor function k(X, Y).
            centers: kernel centers of shape (n_center, n_feature).
            n_label: number of labels.
            mem_gb: GPU memory in GB.
            n_subsample: number of subsamples for preconditioner.
            k: top-k eigensystem for preconditioner.
            bs: mini-batch size.
            metric: keras metric, e.g., 'accuracy'.
            seed: random seed.
        """

        n, d = centers.shape
        if n_subsample is None:
            if n < 100000:
                n_subsample = 2000
            else:
                n_subsample = 10000

        mem_bytes = mem_gb * 1024**3 - 100 * 1024**2 # preserve 100MB
        # Has a factor 3 due to tensorflow implementation.
        mem_usages = (d + n_label + 3 * np.arange(n_subsample)) * n * 4
        mG = np.sum(mem_usages < mem_bytes) # device-dependent batch size

        # Calculate batch/step size for improved EigenPro iteration.
        np.random.seed(seed)
        pinx = np.random.choice(n, n_subsample, replace=False).astype('int32')
        kf, gap, s1, beta = pre_eigenpro_f(
            centers[pinx], kernel, k, n, mG, alpha=.95, seed=seed)
        new_s1 = s1 / gap

        if bs is None:
            bs = min(np.int32(beta / new_s1 + 1), mG)

        if bs < beta / new_s1 + 1:
        	eta = bs / beta
        elif bs < n:
        	eta = 2 * bs / (beta + (bs - 1) * new_s1)
        else:
        	eta = 0.95 * 2 / new_s1
        eta = .5 * eta # .5 for constant related to mse loss.

        print("n_subsample=%d, mG=%d, eta=%.2f, bs=%d, s1=%.2e, beta=%.2f" %
              (n_subsample, mG, eta, bs, s1, beta))
        eta = np.float32(eta * n_label) 
        
        # Assemble kernel model.
        ix = Input(shape=(d+1,), dtype='float32', name='indexed-feat')
        x, index = utils.separate_index(ix) # features, sample_id
        kfeat = KernelEmbedding(kernel, centers,
                                input_shape=(d,))(x)

        y = Dense(n_label, input_shape=(n,),
                  activation='linear',
                  kernel_initializer='zeros',
                  use_bias=False)(kfeat)
        model = Model(ix, y)
        model.compile(
            loss='mse',
            optimizer=PSGD(pred_t=y, index_t=index, eta=eta,
                           eigenpro_f=asm_eigenpro_f(kf, kfeat, pinx)),
            metrics=[metric])

        self.n_label = n_label
        self.seed = seed
        self.bs = bs
        self.model = model

    def fit(self, x_train, y_train, x_val, y_val, epochs,
            seed=1, n_train_sample=10000):
        """Train the model.

        Arguments:
            x_train: feature matrix of shape (n_train, n_feature).
            y_train: label matrix of shape (n_train, n_label).
            x_val:   feature matrix for validation.
            y_val:   label matrix for validation.
            epochs:  list of epochs when the error is calculated.

        Return:
            res:     dictionary with key: epoch, value: (train_error, test_error).
        """
        assert self.n_label == y_train.shape[1]
        np.random.seed(seed)

        x_train = utils.add_index(x_train)
        x_val = utils.add_index(x_val)

        bs = self.bs
        res = dict()

        initial_epoch=0
        train_sec = 0 # training time in seconds
        n, _ = x_train.shape

        # Subsample training data for fast estimation of training loss.
        inx = np.random.choice(n, min(n, n_train_sample), replace=False)
        x_sample, y_sample = x_train[inx], y_train[inx]

        for epoch in epochs:
            start = time.time()
            for _ in range(epoch - initial_epoch):
                epoch_ids = np.random.choice(n, n / bs * bs, replace=False)
                for batch_ids in np.array_split(epoch_ids, n / bs):
                    x_batch, y_batch = x_train[batch_ids], y_train[batch_ids]
                    self.model.train_on_batch(x_batch, y_batch)

            train_sec += time.time() - start
            tr_score = self.model.evaluate(x_sample, y_sample, batch_size=bs, verbose=0)
            tv_score = self.model.evaluate(x_val, y_val, batch_size=bs, verbose=0)

            print("train error: %.2f%%\tval error: %.2f%% (%d epochs, %.2f seconds)\t"
			      "train l2: %.2e\tval l2: %.2e" %
			      ((1 - tr_score[1]) * 100, (1 - tv_score[1]) * 100, epoch, train_sec,
			       tr_score[0], tv_score[0]))
            res[epoch] = (tr_score, tv_score)
            initial_epoch = epoch

        return res

    def predict(self, x_feat):
        """Predict regression scores.

        Argument:
            x_feat: feature matrix of shape (?, n_feature).

        Return:
            score matrix of shape (?, n_label).
        """
        return self.model.predict(utils.add_index(x_feat), batch_size=self.bs)
