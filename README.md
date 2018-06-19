# EigenPro2

## Introduction
The EigenPro2 is proposed to achieve very fast, scalable, and accurate training for kernel machines.

## Requirements: Tensorflow (>=1.2.1) and Keras (=2.0.8)
```
pip install tensorflow tensorflow-gpu keras
```
Follow the [Tensorflow installation guide](https://www.tensorflow.org/install/install_linux) for Virtualenv setup.


## Case 1: Quick Shell Script
For a quick test on the MNIST dataset, execute the following command in a bash shell,
```
CUDA_VISIBLE_DEVICES=0 python run_mnist.py --kernel=Gaussian --s=5 --mem_gb=12 --epochs 1 2 3 4 5
```

The arguments specify that we use Gaussian kernel with bandwidth 5 on a GPU with 12 GB memory.
The train and test (val here) errors are evaluated at the frist five epochs.
```
SVD time: 2.82, adjusted k: 277, s1: 0.15, new s1: 6.66e-04
n_subsample=2000, mG=2000, eta=751.35, bs=1432, s1=1.53e-01, delta=0.05
train error: 0.30%      val error: 1.53% (1 epochs, 1.71 seconds)       train l2: 4.99e-03      val l2: 7.88e-03
train error: 0.04%      val error: 1.43% (2 epochs, 3.25 seconds)       train l2: 2.79e-03      val l2: 6.60e-03
train error: 0.02%      val error: 1.30% (3 epochs, 4.70 seconds)       train l2: 1.78e-03      val l2: 6.04e-03
train error: 0.00%      val error: 1.23% (4 epochs, 6.24 seconds)       train l2: 1.21e-03      val l2: 5.66e-03
train error: 0.00%      val error: 1.28% (5 epochs, 7.68 seconds)       train l2: 9.40e-04      val l2: 5.60e-03
```

## Case 2: Interactive Python Console
When using a Python conosle, we can start by loading the dataset.
In this example, we will load the MNIST dataset and transform its multiclass (10 classes) label
into multiple (10) binary labels.
```
import keras, mnist
n_class = 10  # number of classes
(x_train, y_train), (x_test, y_test) = mnist.load()
y_train = keras.utils.to_categorical(y_train, n_class)
y_test = keras.utils.to_categorical(y_test, n_class)
x_train, y_train, x_test, y_test = x_train.astype('float32'), \
    y_train.astype('float32'), x_test.astype('float32'), y_test.astype('float32')
```
Then specify the kernel function (Gaussian kernel with bandwidth 5),
```
import kernels, wrapper
kernel = wrapper.set_f_args(kernels.Gaussian, s=5)
```

Next, we initialize a kernel machine using EigenPro iteration/kernel based on a given Gaussian kernel,
```
from eigenpro import EigenPro
model = EigenPro(kernel, x_train, n_class, mem_gb=12)
```
To train the model, call the fit method as follows 
```
res = model.fit(x_train=x_train, y_train=y_train, x_val=x_test, y_val=y_test, epochs=[1, 2, 5, 10])
```
Finally, to make prediction on any input feature, use the predict method as follows
```
scores = model.predict(x_test)
```
To calcuate the accuray, we can map the binary labels to multiclass label,
```
import numpy as np
np.mean(np.argmax(scores, axis=1) == np.argmax(y_test, axis=1))
```

