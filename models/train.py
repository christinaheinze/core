import numpy as np
import os
import time
from collections import Counter

import tensorflow as tf

from models.CNN import Estimator
from utils import logging, evaluate, datahandler


def train(args, modelparam=""):
    filenames = tf.placeholder(tf.string, shape=[None])
    train, test = datahandler.load_data_set(args, filenames)
    train_dataset, training_filenames, iterator_train = train
    test_dataset, validation_filenames, iterator_test = test
    X_train, y_train, id_train = iterator_train.get_next()
    _, ids_within_batch_size, _ = tf.unique_with_counts(id_train)
    X_test, y_test, _ = iterator_test.get_next()
    class_model = Estimator(args)
    summariesClassOp = tf.summary.merge(class_model.summaries)

    # save configuration
    txt_log_dir, ckpt_dir, tb_log_dir = logging.create_log_dirs(
                                                            args, modelparam)
    checkpoint_path = os.path.join(ckpt_dir, 'model.ckpt')

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver(tf.global_variables())

        print("Trainable")
        names = [x.name for x in tf.trainable_variables()]
        [print(n) for n in names]

        # summary writer
        writer = tf.summary.FileWriter(tb_log_dir, sess.graph)

        # finalize graph
        tf.get_default_graph().finalize()

        global_step = 0

        # run training
        for e in range(args.num_epochs_class):
            sess.run(iterator_train.initializer,
                     feed_dict={filenames: training_filenames})

            print("\nEpoch " + str(e))

            # adjust learning rate
            new_lr = class_model.lr.eval() * (args.decay_rate ** e)
            sess.run(class_model.lr_update,
                     feed_dict={class_model.new_lr: new_lr})
            print("Learning rate: " + str(class_model.lr.eval()))

            # counterfactual loss annealing
            if args.cfl_annealing:
                if global_step > args.cfl_rate_rise_time and \
                        class_model.cfl_rate.eval() < 1:
                    new_cfl_rate = class_model.cfl_rate.eval() + \
                                    args.cfl_rate_rise_factor
                    sess.run(
                        class_model.cfl_rate_update,
                        feed_dict={class_model.new_cfl_rate: new_cfl_rate})
            print("cf loss ann rate: " + str(class_model.cfl_rate.eval()))

            while True:
                try:
                    global_step += 1
                    X_tr, y_tr, id_tr = sess.run(
                                    [X_train, y_train, ids_within_batch_size])

                    input_dict = {class_model.images_in: X_tr,
                                  class_model.labels: y_tr,
                                  class_model.group: id_tr}
                    start = time.time()
                    _, l, err_rate_mb, s = sess.run([class_model.train_op,
                                                     class_model.loss,
                                                     class_model.train_error,
                                                     summariesClassOp],
                                                    feed_dict=input_dict)
                    end = time.time()
                    if global_step % 10 == 0:
                        writer.add_summary(s, global_step)
                except tf.errors.OutOfRangeError:
                    # save model and visualize
                    saver.save(sess, checkpoint_path, global_step=global_step)
                    print("model saved to {}".format(checkpoint_path))

                    # compute error on counterfactual examples
                    cfs = (Counter(id_tr) - Counter(set(id_tr))).keys()
                    sel = [i for i in range(len(id_tr)) if id_tr[i] in cfs]
                    if len(sel) > 0:
                        X_sub = X_tr[sel, ...]
                        y_sub = y_tr[sel]
                        if args.save_img_sums:
                            opimg = class_model.cf_imgs
                        else:
                            opimg = None
                        eval_err_cfs, s, s2 = evaluate.compute_eval_err(
                                        sess, X_sub, y_sub, args, class_model,
                                        class_model.eval_sum_cfs, opimg)
                        writer.add_summary(s, global_step)
                        if args.save_img_sums:
                            writer.add_summary(s2, global_step)
                    else:
                        eval_err_cfs = np.nan

                    # compute error for test sets
                    eval_errs = []
                    for j in range(len(validation_filenames)):
                        sess.run(iterator_test.initializer,
                                 feed_dict={filenames:
                                            [validation_filenames[j]]})
                        Xtest_tmp, ytest_tmp = sess.run([X_test, y_test])
                        eval_err_tmp, s, _ = evaluate.compute_eval_err(
                                sess, Xtest_tmp, ytest_tmp, args, class_model,
                                class_model.eval_summaries[j])
                        eval_errs.append(
                            "eval_error{}: {:.3f}, ".format(j+1, eval_err_tmp))
                        writer.add_summary(s, global_step)

                    eval_err_str = ''.join([s for s in eval_errs])

                    out_str = ("\n{}/{} (epoch {}), loss = {:.3f}, "
                               "mb_error = {:.3f}, "
                               "cf_error = {:.3f}, "
                               "time/batch = {:.3f}, {}").format(
                                global_step,
                                global_step/(e+1)*args.num_epochs_class,
                                e, l, err_rate_mb, eval_err_cfs, end-start,
                                eval_err_str)
                    print(out_str)
                    with open(os.path.join(txt_log_dir,
                                           modelparam +
                                           "_output_tmp.txt"), "a") as f:
                        f.write(out_str)
                    break
        writer.flush()
        writer.close()

        sess.run(class_model.train_switch_update,
                 feed_dict={class_model.new_train_switch: False})
        print(sess.run(class_model.train_switch))
        tr_error = evaluate.eval_add_testsets(sess, args, class_model,
                                              iterator_train, filenames,
                                              X_train, y_train,
                                              training_filenames)
        add_errors = evaluate.eval_add_testsets(sess, args, class_model,
                                                iterator_test, filenames,
                                                X_test, y_test,
                                                validation_filenames)
        tr_error.extend(add_errors)
        text_file = open(
                    os.path.join(txt_log_dir, modelparam+"_output.txt"), "w")
        text_file.write('\n'.join([st for st in tr_error]))
        text_file.close()
