'''Train kernel methods on the MNIST dataset.
Require tensorflow (>=1.2.1) and GPU device.
Run command:
	CUDA_VISIBLE_DEVICES=0 python run_mnist.py --kernel=Gaussian --s=5 --mem_gb=12 --epochs 1 2 3 4 5
'''
from __future__ import print_function

import argparse
import keras
import numpy as np
import sys
import warnings

from distutils.version import StrictVersion

import kernels
import mnist
import utils
import wrapper

from eigenpro import EigenPro
from backend_extra import hasGPU


assert StrictVersion(keras.__version__) >= StrictVersion('2.0.8'), \
       "Requires Keras (>=2.0.8)."

if StrictVersion(keras.__version__) > StrictVersion('2.0.8'):
    warnings.warn('\n\nThis code has been tested with Keras 2.0.8. '
                   'If the\ncurrent version (%s) fails, ' 
                   'switch to 2.0.8 by command,\n\n'
                   '\tpip install Keras==2.0.8\n\n' %(keras.__version__), Warning)

assert keras.backend.backend() == u'tensorflow', \
       "Requires Tensorflow (>=1.2.1)."
assert hasGPU(), "Requires GPU."


parser = argparse.ArgumentParser(description='Run tests.')
parser.add_argument('--kernel', type=str, default='Gaussian',
                    help='kernel function (e.g. Gaussian, Laplacian, and Cauchy)', required=True)
parser.add_argument('-s', '--s', type=np.float32, help="bandwidth", required=True)
parser.add_argument('-mem_gb', '--mem_gb', type=np.float32, help="bandwidth", required=True)
parser.add_argument('-epochs', '--epochs', nargs='+', type=int,
                    help="epochs to calculate errors, e.g., --epochs 1 2 3 4 5", required=True)

parser.add_argument('-q', '--q', type=np.int32, default=None,
                    help="using the top-1 eigensystem for the eigenpro iteration/kernel")
parser.add_argument('-bs', '--bs', type=np.int32, default=None,
                    help="size of mini-batch")
parser.add_argument('-n_subsample', '--n_subsample', type=np.int32, default=None,
                    help="subsample size")

args = parser.parse_args()
args_dict = vars(args)


# Load dataset.
n_class = 10  # number of classes
(x_train, y_train), (x_test, y_test) = mnist.load()
y_train = keras.utils.to_categorical(y_train, n_class)
y_test = keras.utils.to_categorical(y_test, n_class)
x_train, y_train, x_test, y_test = x_train.astype('float32'), \
    y_train.astype('float32'), x_test.astype('float32'), y_test.astype('float32')

# Choose kernel functions.
s = args_dict['s'] # kernel bandwidth
if args_dict['kernel'] == 'Gaussian':
    kernel = wrapper.set_f_args(kernels.Gaussian, s=s)

elif args_dict['kernel'] == 'Laplacian':
    kernel = wrapper.set_f_args(kernels.Laplacian, s=s)

elif args_dict['kernel'] == 'Cauchy':
    kernel = wrapper.set_f_args(kernels.Cauchy, s=s)

else:
    raise Exception("Unknown kernel function - %s. \
                     Try Gaussian, Laplacian, or Cauchy"
                    % args_dict['kernel'])

# Initialize and train the model.
model = EigenPro(kernel, x_train, n_class,
                 mem_gb=args_dict['mem_gb'],
                 n_subsample=args_dict['n_subsample'],
                 q=args_dict['q'],
                 bs=args_dict['bs'])
model.fit(x_train, y_train,
          x_val=x_test, y_val=y_test,
          epochs=args_dict['epochs'])

utils.reset()
