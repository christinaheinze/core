import gzip
import numpy as np

import tensorflow as tf
# from tensorflow.python.platform import gfile

from utils import datahandler


def read32(bytestream):
    """Read 4 bytes from bytestream as an unsigned 32-bit integer."""
    dt = np.dtype(np.uint32).newbyteorder('>')
    return np.frombuffer(bytestream.read(4), dtype=dt)[0]


def extract_images(f):
    """Extract the images into a 4D uint8 numpy array [index, y, x, depth].
    Args:
      f: A file object that can be passed into a gzip reader.
    Returns:
      data: A 4D uint8 numpy array [index, y, x, depth].
    Raises:
      ValueError: If the bytestream does not start with 2051.
    """
    print('Extracting', f.name)
    with gzip.GzipFile(fileobj=f) as bytestream:
        magic = read32(bytestream)
        if magic != 2051:
            raise ValueError(
                    'Invalid magic number %d in MNIST image file: %s' %
                    (magic, f.name))
        num_images = read32(bytestream)
        rows = read32(bytestream)
        cols = read32(bytestream)
        buf = bytestream.read(rows * cols * num_images)
        data = np.frombuffer(buf, dtype=np.uint8)
        data = data.reshape(num_images, rows, cols, 1)
        return data


def extract_labels(f, num_classes=10):
    """Extract the labels into a 1D uint8 numpy array [index].
    Args:
    f: A file object that can be passed into a gzip reader.
    num_classes: Number of classes for the one hot encoding.
    Returns:
    labels: a 1D uint8 numpy array.
    Raises:
    ValueError: If the bystream doesn't start with 2049.
    """
    print('Extracting', f.name)
    with gzip.GzipFile(fileobj=f) as bytestream:
        magic = read32(bytestream)
        if magic != 2049:
            raise ValueError(
                        'Invalid magic number %d in MNIST label file: %s' %
                        (magic, f.name))
        num_items = read32(bytestream)
        buf = bytestream.read(num_items)
        labels = np.frombuffer(buf, dtype=np.uint8)
        return labels


def bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def float32_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def img_dataset(directory, filenames, height, width, depth, dtype):
    dataset = tf.data.TFRecordDataset(filenames)
    dataset = dataset.map(lambda x: datahandler.parse_function_img(
                                            x, height, width, depth,
                                            dtype_img=dtype))
    return dataset_reshape(dataset)


def matrix_dataset(directory, filenames, d_in):
    dataset = tf.data.TFRecordDataset(filenames)
    dataset = dataset.map(lambda x: datahandler.parse_function_simple(x, d_in))
    return dataset_reshape(dataset)


def dataset_reshape(dataset):
    dataset = dataset.map(
              lambda a, b, c: (a,
                               tf.tile(tf.reshape(b, [1]), [tf.shape(a)[0]]),
                               tf.tile(tf.reshape(c, [1]), [tf.shape(a)[0]])))
    dataset = dataset.shuffle(buffer_size=60000)
    dataset = dataset.flat_map(
                lambda a, b, c: tf.data.Dataset.from_tensor_slices((a, b, c)))
    return dataset
