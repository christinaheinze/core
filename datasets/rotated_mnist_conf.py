import os
import errno
import itertools
import scipy
import numpy as np
from six.moves import urllib
import pdb

import tensorflow as tf
from tensorflow.python.platform import gfile

from datasets import common, rotated_mnist2


def train(directory, filenames, buffer_size):
    """tf.data.Dataset object for rotated MNIST training data."""
    images_file = 'train-images-idx3-ubyte'
    labels_file = 'train-labels-idx1-ubyte'
    tfrecords_filename = 'rot-mnist-conf-train-tf'
    raw = 'rotmnist'
    directory = os.path.join(directory, raw)
    try:
        os.makedirs(os.path.join(directory))
    except OSError as e:
        if e.errno == errno.EEXIST:
            pass
        else:
            raise
    fname = rotated_mnist2.process(directory, images_file, labels_file,
                                   tfrecords_filename, n_sub=50000, c=25000,
                                   train=True, angles = (35,55),
                                   angles2 = (-55,-35), conf=True,
                                   rot_dig=(1,2,3,4,5), rot_dig2=(6,7,8,9,0))
    ds = common.img_dataset(directory, filenames, 28, 28, 1, tf.uint8,
                            buffer_size)
    return ds, fname


def test1(directory, filenames, buffer_size):
    images_file = 't10k-images-idx3-ubyte'
    labels_file = 't10k-labels-idx1-ubyte'
    tfrecords_filename = 'rot-mnist-conf-test1-tf'
    raw = 'rotmnist'
    directory = os.path.join(directory, raw)
    try:
        os.makedirs(os.path.join(directory))
    except OSError as e:
        if e.errno == errno.EEXIST:
            pass
        else:
            raise
    fname = rotated_mnist2.process(directory, images_file, labels_file,
                                   tfrecords_filename, n_sub=10000,
                                   c=0, train=True, angles=(35,55))
    ds = common.img_dataset(directory, filenames, 28, 28, 1, tf.uint8,
                            buffer_size)
    return ds, fname


def test2(directory, filenames, buffer_size):
    images_file = 't10k-images-idx3-ubyte'
    labels_file = 't10k-labels-idx1-ubyte'
    tfrecords_filename = 'rot-mnist-conf-test2-tf'
    raw = 'rotmnist'
    directory = os.path.join(directory, raw)
    try:
        os.makedirs(os.path.join(directory))
    except OSError as e:
        if e.errno == errno.EEXIST:
            pass
        else:
            raise
    fname = rotated_mnist2.process(directory, images_file, labels_file,
                                   tfrecords_filename, n_sub=10000,
                                   c=0, train=False, angles=(35,55))
    ds = common.img_dataset(directory, filenames, 28, 28, 1, tf.uint8,
                            buffer_size)
    return ds, fname


def test3(directory, filenames, buffer_size):
    images_file = 't10k-images-idx3-ubyte'
    labels_file = 't10k-labels-idx1-ubyte'
    tfrecords_filename = 'rot-mnist-conf-test3-tf'
    raw = 'rotmnist'
    directory = os.path.join(directory, raw)
    try:
        os.makedirs(os.path.join(directory))
    except OSError as e:
        if e.errno == errno.EEXIST:
            pass
        else:
            raise
    fname = rotated_mnist2.process(directory, images_file, labels_file,
                                   tfrecords_filename, n_sub=10000, c=0,
                                   train=False, angles=(-55,-35))
    ds = common.img_dataset(directory, filenames, 28, 28, 1, tf.uint8,
                            buffer_size)
    return ds, fname
