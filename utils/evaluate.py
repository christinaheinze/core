import numpy as np

import tensorflow as tf


def compute_eval_err(sess, X, y, args, model, op2, op_opt=None):
    if args.architecture != "conv":
        batch = X.reshape(-1, args.n_input)
    else:
        batch = X

    if args.regression:
        input_dict = {model.eval_data: batch, model.response: y}
    else:
        input_dict = {model.eval_data: batch, model.labels: y}

    if op_opt is None:
        ops = [model.eval_error, op2]
        eval_err, s = sess.run(ops, feed_dict=input_dict)
        s2 = None
    else:
        ops = [model.eval_error, op2, op_opt]
        eval_err, s, s2 = sess.run(ops, feed_dict=input_dict)

    return eval_err, s, s2


def error_rate(predictions, labels, args, return_misclass_idx=False):
    if args.regression:
        return np.mean(np.square(np.subtract(predictions, labels)))
    else:
        mis_class_rate = 100.0 * np.mean(predictions != labels)
        if return_misclass_idx:
            idx = np.where(predictions != labels)[0]
            return mis_class_rate, idx
        else:
            return mis_class_rate


def error_rate_per_class(predictions, labels, args):
    rates = []
    y_unique, ytrue = np.unique(labels, return_inverse=True)
    for cl in y_unique:
        idx_y = np.where(ytrue == cl)[0]
        rates.append(100.0 * np.mean(predictions[idx_y] != labels[idx_y]))
    return rates


def eval_add_testsets(sess, args, model, iterator_test,
                      filenames, X_test, y_test, add_test_data_sets):
    errors_all = []
    for ts in range(len(add_test_data_sets)):
        sess.run(iterator_test.initializer,
                 feed_dict={filenames: [add_test_data_sets[ts]]})
        y_tmp = []
        y_hat_test = []
        while True:
            try:
                X_te, y_te = sess.run([X_test, y_test])
                input_dict = {model.eval_data: X_te,
                              model.labels: y_te}
                yh = sess.run(model.eval_prediction, feed_dict=input_dict)
                y_hat_test.extend(yh)
                y_tmp.extend(y_te)
            except tf.errors.OutOfRangeError:
                break
        y_hat_test = np.array(y_hat_test)
        y_tmp = np.array(y_tmp)
        err = error_rate(y_hat_test, y_tmp, args)
        if not args.regression:
            err_per_class = error_rate_per_class(y_hat_test, y_tmp, args)
        errors = []
        errors.append("\n### "+add_test_data_sets[ts])
        errors.append("\ntest error: {:.3f}".format(err))
        if not args.regression:
            err_per_class_str = "/".join(
                                    "{:.3f}".format(s) for s in err_per_class)
            errors.append(
                        "\ntest error per class: {}".format(err_per_class_str))
        errors_all.extend(errors)

    print('\n'.join([st for st in errors_all]))
    return errors_all
