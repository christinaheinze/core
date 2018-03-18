import tensorflow as tf
from utils import ops


class Estimator():
    def __init__(self, args):
        self.args = args
        self.summaries = []
        self.eval_summaries = []

        self.train_switch = tf.Variable(
                                True, name='train_switch', trainable=False)
        self.new_train_switch = tf.placeholder(
                                    tf.bool, shape=[], name="new_train_switch")
        self.train_switch_update = tf.assign(
                                    self.train_switch, self.new_train_switch)

        with tf.name_scope("input_cnn"):
            self.group = tf.placeholder(tf.int32, [None], name="group")

            if self.args.architecture != "conv":
                self.images_in = tf.placeholder(tf.float32,
                                                [None, self.args.n_input],
                                                name="images_cnn")
            else:
                self.images_in = tf.placeholder(tf.float32,
                                                [None, self.args.img_size_h,
                                                 self.args.img_size_w,
                                                 self.args.n_channels],
                                                name="images_cnn")

            if self.args.normalize:
                self.images = (tf.cast(self.images_in, tf.float32)) / 255
            else:
                self.images = self.images_in

            if self.args.img_data:
                self.image_matrix = tf.reshape(self.images,
                                               [-1, self.args.img_size_h,
                                                self.args.img_size_w,
                                                self.args.n_channels])
                if self.args.save_img_sums:
                    self.summaries.append(
                        tf.summary.image('input_class', self.image_matrix, 6))

            self.labels = tf.placeholder(tf.int64, shape=(None,))
            self.response = tf.placeholder(tf.float32, shape=(None,))

            if self.args.architecture != "conv":
                self.eval_data = tf.placeholder(
                                            tf.float32,
                                            shape=(None, self.args.n_input),
                                            name="eval_data")
            else:
                self.eval_data = tf.placeholder(tf.float32,
                                                shape=(None,
                                                       self.args.img_size_h,
                                                       self.args.img_size_w,
                                                       self.args.n_channels),
                                                name="eval_data")
            if self.args.normalize:
                self.eval_data_casted = (tf.cast(self.eval_data,
                                                 tf.float32)) / 255
            else:
                self.eval_data_casted = self.eval_data

            if self.args.img_data:
                self.eval_image_matrix = tf.reshape(self.eval_data_casted,
                                                    [-1, self.args.img_size_h,
                                                     self.args.img_size_w,
                                                     self.args.n_channels])
                if self.args.save_img_sums:
                    self.cf_imgs = tf.summary.image('input_cfs',
                                                    self.eval_image_matrix, 6)

        with tf.name_scope("cnn"):
            with tf.variable_scope("cnn") as scope:

                if self.args.regression:
                    out_dim = 1
                else:
                    out_dim = self.args.num_classes

                if self. args.architecture == "linear":
                    self.logits = self.simple_model(self.images, out_dim)
                    scope.reuse_variables()
                    self.logits_eval = self.simple_model(self.eval_data_casted,
                                                         out_dim)
                elif self. args.architecture == "nonlinear":
                    self.logits = self.nonlinear_model(
                                self.images, out_dim, phase=self.train_switch,
                                bn=self.args.bn)
                    self.logits_eval = self.nonlinear_model(
                                self.eval_data_casted, out_dim, phase=False,
                                bn=self.args.bn, reuse=True)
                else:
                    if self.args.two_layer_CNN:
                        self.logits = self.cnn2(self.image_matrix, out_dim)
                        self.logits_eval = self.cnn2(self.eval_image_matrix,
                                                     out_dim, reuse=True)
                    else:
                        self.logits = self.cnn(
                                    self.image_matrix, out_dim,
                                    phase=self.train_switch, bn=self.args.bn)
                        self.logits_eval = self.cnn(
                                    self.eval_image_matrix, out_dim,
                                    phase=False, bn=self.args.bn, reuse=True)

        with tf.name_scope("eval_accuracy"):
            # with tf.variable_scope("eval_accuracy"):
                # Predictions for the current training minibatch.
                if self.args.regression:
                    self.train_prediction = tf.squeeze(self.logits)
                    self.train_error = tf.reduce_mean(tf.square(tf.subtract(
                        self.train_prediction, self.response)))
                    self.summaries.append(tf.summary.scalar("train_error",
                                                            self.train_error))

                    # Predictions for the test and validation
                    self.eval_prediction = tf.squeeze(self.logits_eval)
                    self.eval_error = tf.reduce_mean(tf.square(tf.subtract(
                        self.eval_prediction, self.response)))
                    self.eval_sum = tf.summary.scalar("eval_error",
                                                      self.eval_error)
                else:
                    self.train_prediction = tf.argmax(
                                                tf.nn.softmax(self.logits), 1)
                    self.train_error = 100.0*(1-tf.reduce_mean(
                                tf.cast(tf.equal(self.train_prediction,
                                                 self.labels), tf.float32)))
                    self.summaries.append(tf.summary.scalar("train_error",
                                                            self.train_error))

                    # Predictions for the test and validation
                    self.eval_probs = tf.nn.softmax(self.logits_eval)
                    self.eval_prediction = tf.argmax(self.eval_probs, 1)
                    self.eval_error = 100.0*(1-tf.reduce_mean(tf.cast(tf.equal(
                        self.eval_prediction, self.labels), tf.float32)))

                for env in range(self.args.n_test_data_sets):
                    self.eval_summaries.append(
                        tf.summary.scalar("eval_error"+str(env),
                                          self.eval_error))

                self.eval_sum_cfs = tf.summary.scalar("eval_error_cfs",
                                                      self.eval_error)

        with tf.name_scope("loss_cnn"):
            if self.args.cfl_annealing:
                self.cfl_rate = tf.Variable(
                                        0.0, trainable=False, dtype=tf.float32)
            else:
                self.cfl_rate = tf.Variable(1.0,
                                            trainable=False, dtype=tf.float32)
            self.new_cfl_rate = tf.placeholder(tf.float32, shape=[],
                                               name="new_rla_rate")
            self.cfl_rate_update = tf.assign(self.cfl_rate, self.new_cfl_rate)

            if self.args.classifier == "standard" or \
               self.args.classifier == "counterfactual":
                if self.args.regression:
                    self.loss = tf.reduce_mean(tf.square(
                            tf.subtract(self.train_prediction, self.response)))
                else:
                    self.loss = tf.reduce_mean(
                        tf.nn.sparse_softmax_cross_entropy_with_logits(
                                    labels=self.labels, logits=self.logits))
            else:
                raise ValueError('Unknown option for classifier provided.')

            tvars = tf.trainable_variables()
            regularizers = tf.add_n(
                [tf.nn.l2_loss(v) for v in tvars
                 if 'cnn' in v.name and 'bias' not in v.name])

            if self.args.classifier == "counterfactual":
                self.partitions_y = tf.dynamic_partition(
                                                        self.logits,
                                                        self.group,
                                                        self.args.batch_size)
                self.mean = [tf.cond(tf.shape(partition)[0] > 1,
                             lambda: tf.nn.moments(partition, axes=[0])[0],
                             lambda: tf.zeros(out_dim))
                             for partition in self.partitions_y]

                self.sum_non_zeros = tf.reduce_sum([tf.cond(
                                        tf.shape(partition)[0] > 1,
                                        lambda: tf.constant(1.0),
                                        lambda: tf.constant(0.0))
                                        for partition in self.partitions_y])
                self.summaries.append(tf.summary.scalar("num_cfs",
                                                        self.sum_non_zeros))

                self.mean_devs_over_m_i_instances = [tf.cond(
                                        tf.shape(partition[0])[0] > 1,
                                        lambda: tf.reduce_mean(
                                                tf.reduce_sum(tf.square(
                                                    tf.subtract(partition[0],
                                                                partition[1])),
                                                              axis=1)),
                                        lambda: tf.constant(0.0))
                                        for partition in list(
                                            zip(self.partitions_y, self.mean))]

                self.countfact_loss = self.cfl_rate * tf.cond(
                            tf.equal(self.sum_non_zeros, tf.constant(0.0)),
                            lambda: tf.constant(0.0),
                            lambda: tf.reduce_sum(
                                            self.mean_devs_over_m_i_instances
                                            )/self.sum_non_zeros)

                self.loss += self.args.weight_countfact_loss * \
                    self.countfact_loss

            # Add the regularization term to the loss.
            self.loss += self.args.lambda_reg * regularizers

            self.summaries.append(tf.summary.scalar("loss_class", self.loss))
            if self.args.classifier == "counterfactual":
                self.summaries.append(
                    tf.summary.scalar("countfact_loss", self.countfact_loss))

            self.lr = tf.Variable(self.args.learning_rate_class,
                                  trainable=False, dtype=tf.float32)
            self.new_lr = tf.placeholder(tf.float32, shape=[], name="new_lr")
            self.lr_update = tf.assign(self.lr, self.new_lr)

            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                self.train_op = tf.train.AdamOptimizer(
                                                self.lr).minimize(self.loss)

    def simple_model(self, data, output_dim):
        """The Model definition."""
        self.weights = tf.get_variable(
                       name='weights',
                       shape=[self.args.n_input, output_dim],
                       initializer=tf.random_normal_initializer(mean=0,
                                                                stddev=0.1))
        self.bias = tf.get_variable(name='bias',
                                    shape=[output_dim],
                                    initializer=tf.constant_initializer(0.0))
        apply_weights_OP = tf.matmul(data, self.weights, name="apply_weights")
        return tf.add(apply_weights_OP, self.bias, name="add_bias")

    def cnn2(self, data, output_dim, reuse=None):
        h1 = tf.layers.conv2d(
                  data,
                  filters=16,
                  kernel_size=[5, 5],
                  strides=2,
                  padding="same",
                  activation=tf.nn.relu,
                  name="conv_h1",
                  kernel_initializer=tf.contrib.layers.xavier_initializer(),
                  reuse=reuse) # 64x64x1 -> 32x32x16

        h2 = tf.layers.conv2d(
                  h1,
                  filters=32,
                  kernel_size=[5, 5],
                  strides=2,
                  padding="same",
                  activation=tf.nn.relu,
                  name="rec_conv_h2",
                  kernel_initializer=tf.contrib.layers.xavier_initializer(),
                  reuse=reuse) # 32x32x16 -> 16x16x32
        h2_flat = tf.reshape(h2,
                             [-1,
                              self.args.img_size_w//4 *
                              self.args.img_size_h//4*32])

        z = tf.layers.dense(h2_flat, units=output_dim, name="z", reuse=reuse)
        return z

    def nonlinear_model(self, input_images, dim, phase=True, bn=False,
                        reuse=None):
        h1 = tf.layers.dense(
         input_images,
         units=self.args.h1,
         activation=None if bn else tf.nn.softplus,
         kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),
         bias_initializer=tf.constant_initializer(0.0),
         name="h1_rec",
         reuse=reuse)

        if bn:
            h1bn = tf.layers.batch_normalization(h1,
                                                 center=True,
                                                 scale=True,
                                                 training=phase,
                                                 name='h1_rec_bn',
                                                 reuse=reuse)
            h1bnact = tf.nn.softplus(h1bn)
        else:
            h1bnact = h1

        h2 = tf.layers.dense(
         h1bnact,
         units=self.args.h2,
         activation=None if bn else tf.nn.softplus,
         kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),
         bias_initializer=tf.constant_initializer(0.01),
         name="h2_rec",
         reuse=reuse)

        if bn:
            h2bn = tf.layers.batch_normalization(h2,
                                                 center=True,
                                                 scale=True,
                                                 training=phase,
                                                 name='h2_rec_bn',
                                                 reuse=reuse)

            h2bnact = tf.nn.softplus(h2bn)
        else:
            h2bnact = h2

        z = tf.layers.dense(
            h2bnact,
            units=dim,
            kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),
            bias_initializer=tf.constant_initializer(0.0),
            name="z",
            reuse=reuse)

        return z

    def cnn(self, input_images, output_dim, phase=True, bn=False, reuse=None):
        h1 = tf.layers.conv2d(
                  input_images,
                  filters=16,
                  kernel_size=[5, 5],
                  strides=2,
                  padding="same",
                  activation=None if bn else ops.lrelu,
                  name="rec_conv_h1",
                  kernel_initializer=tf.contrib.layers.xavier_initializer(),
                  reuse=reuse) # 112x112x1 -> 56x56x16
        if bn:
            h1bn = tf.layers.batch_normalization(h1,
                                                 center=True,
                                                 scale=True,
                                                 training=phase,
                                                 name='rec_conv_h1_bn',
                                                 reuse=reuse)

            h1bnact = ops.lrelu(h1bn)
        else:
            h1bnact = h1

        h2 = tf.layers.conv2d(
                  h1bnact,
                  filters=32,
                  kernel_size=[5, 5],
                  strides=2,
                  padding="same",
                  activation=None if bn else ops.lrelu,
                  name="rec_conv_h2",
                  kernel_initializer=tf.contrib.layers.xavier_initializer(),
                  reuse=reuse) # 56x56x16 -> 28x28x32

        if bn:
            h2bn = tf.layers.batch_normalization(h2,
                                                 center=True,
                                                 scale=True,
                                                 training=phase,
                                                 name='rec_conv_h2_bn',
                                                 reuse=reuse)

            h2bnact = ops.lrelu(h2bn)
        else:
            h2bnact = h2

        h3 = tf.layers.conv2d(
                  h2bnact,
                  filters=64,
                  kernel_size=[5, 5],
                  strides=2,
                  padding="same",
                  activation=None if bn else ops.lrelu,
                  name="rec_conv_h3",
                  kernel_initializer=tf.contrib.layers.xavier_initializer(),
                  reuse=reuse) # 28x28x32 -> 14x14x64

        if bn:
            h3bn = tf.layers.batch_normalization(h3,
                                                 center=True,
                                                 scale=True,
                                                 training=phase,
                                                 name='rec_conv_h3_bn',
                                                 reuse=reuse)

            h3bnact = ops.lrelu(h3bn)
        else:
            h3bnact = h3

        h4 = tf.layers.conv2d(
                  h3bnact,
                  filters=128,
                  kernel_size=[5, 5],
                  strides=2,
                  padding="same",
                  activation=None if bn else ops.lrelu,
                  name="rec_conv_h4",
                  kernel_initializer=tf.contrib.layers.xavier_initializer(),
                  reuse=reuse) # 14x14x64 ->  7x7x128

        if bn:
            h4bn = tf.layers.batch_normalization(h4,
                                                 center=True,
                                                 scale=True,
                                                 training=phase,
                                                 name='rec_conv_h4_bn',
                                                 reuse=reuse)

            h4bnact = ops.lrelu(h4bn)
        else:
            h4bnact = h4

        h4_flat = tf.reshape(
                h4bnact,
                [-1, self.args.img_size_w//16*self.args.img_size_h//16*128])

        z = tf.layers.dense(h4_flat, units=output_dim,
                            name="z_rec", reuse=reuse)

        return z
