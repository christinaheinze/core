import numpy as np

import tensorflow as tf

from datasets import rotated_mnist, synthetic_nonlinear


def load_data_set(args, filenames):
    if args.dataset == 'rotmnist':
        train_dataset, train_filenames = rotated_mnist.train(
                                                    args.data_path, filenames)
        test_dataset, test_filenames1 = rotated_mnist.test1(args.data_path,
                                                            filenames)
        _, test_filenames2 = rotated_mnist.test2(args.data_path, filenames)
        test_filenames = [test_filenames1, test_filenames2]
    elif args.dataset == 'synthetic_nonlinear':
        train_dataset, train_filenames = synthetic_nonlinear.train(
                                                    args.data_path, filenames)
        test_dataset, test_filenames1 = synthetic_nonlinear.test1(
                                                    args.data_path, filenames)
        _, test_filenames2 = synthetic_nonlinear.test2(args.data_path,
                                                       filenames)
        test_filenames = [test_filenames1, test_filenames2]
    else:
        raise ValueError("setting unknown")

    if not isinstance(train_filenames, list):
        train_filenames = [train_filenames]

    if not isinstance(test_filenames, list):
        test_filenames = [test_filenames]

    args.n_test_data_sets = len(test_filenames)

    train_dataset = train_dataset.batch(args.batch_size)
    test_dataset = test_dataset.batch(args.eval_batch_size)

    iterator_train = train_dataset.make_initializable_iterator()
    iterator_test = test_dataset.make_initializable_iterator()

    out_train = train_dataset, train_filenames, iterator_train
    out_test = test_dataset, test_filenames, iterator_test

    return out_train, out_test


def inspect_tfrecord_img(filename, n, img_size_h, img_size_w, img_size_d,
                         max_cf=2, dtype=np.float64):
    iterator = tf.python_io.tf_record_iterator(filename)
    X = np.ndarray([n, img_size_h, img_size_w, img_size_d])
    i = 0
    ids = np.ndarray([n], dtype=np.int64)
    labels = np.ndarray([n], dtype=np.int64)

    while i < n:
        serialized_example = next(iterator)
        example = tf.train.SequenceExample()
        example.ParseFromString(serialized_example)
        num_cfs = example.context.feature["num_cfs"].int64_list.value[0]
        idx = example.context.feature["id"].int64_list.value
        label = example.context.feature["label"].int64_list.value
        for cf in range(num_cfs):
            if cf >= max_cf:
                break
            im = example.feature_lists.feature_list["inputs"].feature[cf].bytes_list.value[0]
            arr_tmp = np.fromstring(im, dtype=dtype)
            if i == n:
                break
            X[i, :, :, :] = arr_tmp.reshape(
                                        [img_size_h, img_size_w, img_size_d])
            ids[i] = idx[0]
            labels[i] = label[0]
            i += 1

    return X, ids, labels


def parse_function_simple(example_proto, n_input):
    context_features = {
            'num_cfs': tf.FixedLenFeature(shape=[],
                                          dtype=tf.int64),
            'id': tf.FixedLenFeature(shape=[],
                                     dtype=tf.int64),
            'label': tf.FixedLenFeature(shape=[],
                                        dtype=tf.int64)
            }
    sequence_features = {
      'inputs': tf.FixedLenSequenceFeature(shape=[n_input],
                                           dtype=tf.float32,
                                           allow_missing=True)
      }

    context, parsed_features = tf.parse_single_sequence_example(
                                        example_proto,
                                        context_features=context_features,
                                        sequence_features=sequence_features)

    image = parsed_features["inputs"]
    length = tf.cast(context["num_cfs"], tf.int32)
    image = tf.reshape(image, [length, n_input])
    return image, context["label"], context["id"]


def parse_function_img(example_proto, img_size_h, img_size_w, img_size_d,
                       dtype_img=tf.float64):
    context_features = {
            'num_cfs': tf.FixedLenFeature(shape=[],
                                          dtype=tf.int64),
            'id': tf.FixedLenFeature(shape=[],
                                     dtype=tf.int64),
            'label': tf.FixedLenFeature(shape=[],
                                        dtype=tf.int64)
            }
    sequence_features = {
      'inputs': tf.FixedLenSequenceFeature(shape=[],
                                           dtype=tf.string,
                                           allow_missing=True)
      }

    context, parsed_features = tf.parse_single_sequence_example(
                                        example_proto,
                                        context_features=context_features,
                                        sequence_features=sequence_features)

    image = tf.decode_raw(parsed_features["inputs"], dtype_img)
    image = tf.cast(image, tf.float32)
    length = tf.cast(context["num_cfs"], tf.int32)
    image = tf.reshape(image, [length, img_size_h, img_size_w, img_size_d])
    label = tf.cast(context["label"], tf.int64)
    ids = tf.cast(context["id"], tf.int32)

    return image, label, ids
