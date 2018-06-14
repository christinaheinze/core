import os
import errno
import itertools
import scipy
import numpy as np
from six.moves import urllib
import pdb

import tensorflow as tf
from tensorflow.python.platform import gfile

from datasets import common


def check_image_file_header(filename):
    """Validate that filename corresponds to images for the MNIST dataset."""
    with tf.gfile.Open(filename, 'rb') as f:
        magic = common.read32(f)
        rows = common.read32(f)
        cols = common.read32(f)
        if magic != 2051:
            raise ValueError('Invalid magic number %d in MNIST file %s' % (
                magic, f.name))
        if rows != 28 or cols != 28:
            raise ValueError(
                'Invalid MNIST file %s: Expected 28x28 images, found %dx%d' %
                (f.name, rows, cols))


def check_labels_file_header(filename):
    """Validate that filename corresponds to labels for the MNIST dataset."""
    with tf.gfile.Open(filename, 'rb') as f:
        magic = common.read32(f)
        if magic != 2049:
            raise ValueError('Invalid magic number %d in MNIST file %s' % (
                             magic, f.name))


def download(directory, filename):
    """Download (and unzip) a file from the MNIST dataset if not done."""
    filepath = os.path.join(directory, filename)
    zipped_filepath = filepath + '.gz'
    if tf.gfile.Exists(zipped_filepath):
        return zipped_filepath
    if not tf.gfile.Exists(directory):
        tf.gfile.MakeDirs(directory)
    # CVDF mirror of http://yann.lecun.com/exdb/mnist/
    url = 'https://storage.googleapis.com/cvdf-datasets/mnist/' + \
          filename + '.gz'
    print('Downloading %s to %s' % (url, zipped_filepath))
    urllib.request.urlretrieve(url, zipped_filepath)
    return zipped_filepath


def rotate_mnist(X, Y, n, m, train=True, angles=(35,55), angles2=(-55,-35),
                 conf=False, rot_dig=(1,2,3,4,5), rot_dig2=(6,7,8,9,0)):
    n_in = len(Y)
    X = X.reshape([n_in, 28, 28])
    if n_in < n:
        n = n_in
    total_sample_size = n+m
    X_ = np.zeros([total_sample_size, 28, 28], np.float32)
    labels_ = np.zeros(total_sample_size, np.int32)
    idx = np.zeros(total_sample_size, np.int32)

    # if m > 0: sample indices for which to add pair
    if not conf:
        idx_for_pairs = np.random.choice(np.arange(0, n), size=m, replace=False)
    else:
        rot_dig_all = rot_dig + rot_dig2
        sample_from = np.where(np.isin(Y[:n], rot_dig_all))[0]
        idx_for_pairs = np.random.choice(sample_from, size=m, replace=False)

    k = 0
    for i in range(n):
        # rotate
        if not train:
            rot = np.random.randint(low=angles[0],
                                    high=angles[1], size=1)[0]
            img = scipy.ndimage.rotate(X[i], rot, reshape=False)
            X_[k, :, :] = img
        else:
            X_[k, :, :] = X[i]

        idx[k] = i
        labels_[k] = Y[i]
        k += 1

        if i in idx_for_pairs:
            if Y[i] in rot_dig or not conf:
                rot = np.random.randint(low=angles[0],
                                        high=angles[1], size=1)[0]
            elif Y[i] in rot_dig2:
                rot = np.random.randint(low=angles2[0],
                                        high=angles2[1], size=1)[0]
            img = scipy.ndimage.rotate(X[i], rot, reshape=False)
            X_[k, :, :] = img
            labels_[k] = Y[i]
            idx[k] = i
            k += 1

    X_ = X_.reshape([total_sample_size, 28, 28, 1])
    X_ = X_.astype(np.uint8)
    return X_, labels_, idx


def process(directory, images_file, labels_file, tf_filename,
            n_sub, c, train, angles = (35,55), angles2 = (-55,-35), conf=False,
            rot_dig=(), rot_dig2=()):
    images_file = download(directory, images_file)
    labels_file = download(directory, labels_file)
    processed_folder = 'processed'
    fname = os.path.join(directory, processed_folder, tf_filename)
    if os.path.exists(fname):
        return fname

    print('Processing...')
    try:
        os.makedirs(os.path.join(directory, processed_folder))
    except OSError as e:
        if e.errno == errno.EEXIST:
            pass
        else:
            raise

    with gfile.Open(images_file, 'rb') as f:
        img = common.extract_images(f)

    with gfile.Open(labels_file, 'rb') as f:
        labels = common.extract_labels(f)

    images, labels, idx = rotate_mnist(img, labels, n_sub, c, train,
                                           angles=angles, angles2=angles2,
                                           conf=conf, rot_dig=rot_dig,
                                           rot_dig2=rot_dig2)
    zipped = [(i, wid, y) for i, wid, y in zip(images, idx, labels)]
    sorted_by_wid = sorted(zipped, key=lambda tup: tup[1])
    grouped_tmp = [(k, list(list(zip(*g))))
                   for k, g in
                   itertools.groupby(sorted_by_wid, lambda x: x[1])]
    grouped = [(v[1][0], v[2][0], v[0]) for k, v in grouped_tmp]
    # write to tfrecords
    writer = tf.python_io.TFRecordWriter(
                            os.path.join(directory,
                                         processed_folder,
                                         tf_filename))

    for idx, label, cf_list in grouped:
        n_cfs = len(cf_list)
        img_raws = [img_i.tostring() for img_i in cf_list]
        input_features = [common.bytes_feature(img_r) for img_r in img_raws]

        feature_list = {
              'inputs': tf.train.FeatureList(feature=input_features),
         }

        feature_lists = tf.train.FeatureLists(feature_list=feature_list)
        features = tf.train.Features(
                            feature={'num_cfs': common.int64_feature(n_cfs),
                                     'id': common.int64_feature(idx),
                                     'label': common.int64_feature(label)})
        example = tf.train.SequenceExample(feature_lists=feature_lists,
                                           context=features)

        writer.write(example.SerializeToString())

    writer.close()
    print('Done!')


def train(directory, filenames):
    """tf.data.Dataset object for rotated MNIST training data."""
    images_file = 'train-images-idx3-ubyte'
    labels_file = 'train-labels-idx1-ubyte'
    tfrecords_filename = 'rot-mnist-train-tf'
    raw = 'rotmnist'
    directory = os.path.join(directory, raw)
    try:
        os.makedirs(os.path.join(directory))
    except OSError as e:
        if e.errno == errno.EEXIST:
            pass
        else:
            raise
    fname = process(directory, images_file, labels_file, tfrecords_filename,
                    n_sub=50000, c=2000, train=True)
    ds = common.img_dataset(directory, filenames, 28, 28, 1, tf.uint8)
    return ds, fname


def test1(directory, filenames):
    images_file = 't10k-images-idx3-ubyte'
    labels_file = 't10k-labels-idx1-ubyte'
    tfrecords_filename = 'rot-mnist-test1-tf'
    raw = 'rotmnist'
    directory = os.path.join(directory, raw)
    try:
        os.makedirs(os.path.join(directory))
    except OSError as e:
        if e.errno == errno.EEXIST:
            pass
        else:
            raise
    fname = process(directory, images_file, labels_file, tfrecords_filename,
                    n_sub=10000, c=0, train=False)
    ds = common.img_dataset(directory, filenames, 28, 28, 1, tf.uint8)
    return ds, fname


def test2(directory, filenames):
    images_file = 't10k-images-idx3-ubyte'
    labels_file = 't10k-labels-idx1-ubyte'
    tfrecords_filename = 'rot-mnist-test2-tf'
    raw = 'rotmnist'
    directory = os.path.join(directory, raw)
    try:
        os.makedirs(os.path.join(directory))
    except OSError as e:
        if e.errno == errno.EEXIST:
            pass
        else:
            raise
    fname = process(directory, images_file, labels_file, tfrecords_filename,
                    n_sub=10000, c=0, train=True)
    ds = common.img_dataset(directory, filenames, 28, 28, 1, tf.uint8)
    return ds, fname
