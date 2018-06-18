import numpy as np

from keras.datasets.mnist import load_data


def unit_range_normalize(X):
	min_ = np.min(X, axis=0)
	max_ = np.max(X, axis=0)
	diff_ = max_ - min_
	diff_[diff_<=0.0] = np.maximum(1.0, min_[diff_<=0.0])
	SX = (X - min_) / diff_
	return SX

def load():
	# input image dimensions
	img_rows, img_cols = 28, 28
	
	# the data, shuffled and split between train and test sets
	(x_train, y_train), (x_test, y_test) = load_data()
	
	x_train = x_train.reshape(x_train.shape[0], img_rows * img_cols)
	x_test = x_test.reshape(x_test.shape[0], img_rows * img_cols)
	
	x_train = x_train.astype('float32') / 255
	x_test = x_test.astype('float32') / 255
	
	x_train = unit_range_normalize(x_train)
	x_test = unit_range_normalize(x_test)
	print("Load MNIST dataset.")
	print(x_train.shape[0], 'train samples')
	print(x_test.shape[0], 'test samples')

	return (x_train, y_train), (x_test, y_test)
