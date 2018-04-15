import numpy as np
import tensorflow as tf
from swissknife.files import SavingFolder
from swissknife.learning_rate import ExponentialDecay
from tensorflow.examples.tutorials.mnist import input_data

from basedir import MNIST_IMAGES


N_INPUTS = 28*28
N_HIDDEN = [300, 100]
N_OUTPUTS = 10


def dense(x, n_units, name, activation=None):
    """Custom implementation of dense layer."""

    with tf.name_scope(name):
        n_inputs = int(x.get_shape()[1])
        stddev = 2 / np.sqrt(n_inputs + n_units)
        init = tf.truncated_normal((n_inputs, n_units), stddev=stddev)
        w = tf.Variable(init, name='kernel')
        b = tf.Variable(tf.zeros([n_units]), name='bias')
        z = tf.matmul(x, w) + b
        if activation is not None:
            return activation(z, name='activation')
        else:
            return z


def main():
    x = tf.placeholder(tf.float32, shape=(None, N_INPUTS), name='X')
    y = tf.placeholder(tf.int64, shape=(None,), name='y')

    with tf.name_scope('dnn'):
        h = x
        for index, n_units in enumerate(N_HIDDEN):
            h = dense(h, n_units, name=f'hidden{index}', activation=tf.nn.relu)
        logits = dense(h, N_OUTPUTS, name='logits')

    with tf.name_scope('loss'):
        x_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=y, logits=logits)
        loss = tf.reduce_mean(x_entropy, name='loss')

    learning_rate = tf.placeholder(tf.float32, shape=[])
    with tf.name_scope('train'):
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            training_op = optimizer.minimize(loss)

    with tf.name_scope('eval'):
        correct = tf.nn.in_top_k(logits, y, 1)
        accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

    paths_manager = SavingFolder(model_name='dnn_tf_plain', model_ext='')
    paths_manager.create_model_dir(ask_on_rewrite=False)

    batch_size = 2048
    n_epochs = 200
    saver = tf.train.Saver()
    mnist = input_data.read_data_sets(MNIST_IMAGES)
    decay = ExponentialDecay(init_rate=1.0, decay_coef=0.075)

    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        for epoch in range(1, n_epochs + 1):
            lr = max(0.0005, decay(epoch))
            acc_train = []

            for iteration in range(mnist.train.num_examples // batch_size):
                x_batch, y_batch = mnist.train.next_batch(batch_size)
                feed = {x: x_batch,
                        y: y_batch,
                        learning_rate: lr}
                sess.run(training_op, feed)
                acc_train.append(accuracy.eval(feed))

            acc_train_avg = np.mean(acc_train)
            acc_val = accuracy.eval({x: mnist.validation.images,
                                     y: mnist.validation.labels})

            print('%03d' % epoch,
                  'Rate: %2.8f' % lr,
                  'Train accuracy:', acc_train_avg,
                  'Valid accuracy:', acc_val)

        checkpoint_path = paths_manager.model_path
        save_path = saver.save(sess, checkpoint_path)
        print('Model saved into path', save_path)


if __name__ == '__main__':
    main()
