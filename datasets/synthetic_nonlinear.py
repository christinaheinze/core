import os

from datasets import common


def train(directory, filenames):
    tfrecords_filename = 'synthetic_nonlinear_train.tfrecords'
    raw = 'synthetic_nonlinear'
    fname = os.path.join(directory, raw, tfrecords_filename)
    ds = common.matrix_dataset(directory, filenames, 2)
    return ds, fname


def test1(directory, filenames):
    tfrecords_filename = 'synthetic_nonlinear_test1.tfrecords'
    raw = 'synthetic_nonlinear'
    fname = os.path.join(directory, raw, tfrecords_filename)
    ds = common.matrix_dataset(directory, filenames, 2)
    return ds, fname


def test2(directory, filenames):
    tfrecords_filename = 'synthetic_nonlinear_test2.tfrecords'
    raw = 'synthetic_nonlinear'
    fname = os.path.join(directory, raw, tfrecords_filename)
    ds = common.matrix_dataset(directory, filenames, 2)
    return ds, fname
