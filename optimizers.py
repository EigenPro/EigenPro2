import numpy as np
from keras import backend as K
from keras.optimizers import Optimizer

from backend_extra import scatter_update


def nesterov(p0, p1, rmax=.95): 
    """Nesterov method.

    Arguments:
        p0: weight parameter tensor variable.
        p1: updated weight parameter tensor.
        rmax: maximum momentum term weight in [0, 1].

    Returns:
        p2: parameter tensor adjusted by Nesterov method.
        updates: a list of tensor updates.
    """

    p = K.variable(p0, name='nesterov.orig.p', dtype='float32')
    r = K.constant(-rmax, dtype='float32')

    p2 = (1 - r) * p1 + r * p
    updates = [K.update(p, p1)]
    return p2, updates


class PSGD(Optimizer):
    """Primal stochastic gradient descent optimizer.

    Arguments:
        pred_t: tensor. Prediction result.
        index_t: tensor. Mini-batch indices for primal updates.
        eta: float >= 0. Step size.
        eigenpro_f: Map grad of the original kernel to that of
                    the eigenpro kernel.
        nesterov_r: Nesterov parameter.
    """

    def __init__(self, pred_t, index_t, eta=0.01,
                 eigenpro_f=None, nesterov_r=None, **kwargs):
        super(PSGD, self).__init__(**kwargs)
        self.eta = K.variable(eta, name='eta')
        self.pred_t = pred_t
        self.index_t = index_t
        self.eigenpro_f = eigenpro_f
        self.nesterov_r = nesterov_r

    def get_updates(self, loss, params):
        self.updates = []
        grads = self.get_gradients(loss, [self.pred_t])

        eta = self.eta
        index = self.index_t
        eigenpro_f = self.eigenpro_f

        shapes = [K.get_variable_shape(p) for p in params]
        for p, g in zip(params, grads):
            update_p = K.gather(p, index) - eta * g
            new_p = scatter_update(p, index, update_p)
            
            if eigenpro_f:
                new_p = eigenpro_f(new_p, g, eta)

            if self.nesterov_r is not None:
                new_p, updates = nesterov(p, new_p, rmax=self.nesterov_r)
                self.updates += updates

            self.updates.append(K.update(p, new_p))
        return self.updates

    def get_config(self):
        config = {'eta': float(K.get_value(self.eta))}
        base_config = super(PSGD, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
