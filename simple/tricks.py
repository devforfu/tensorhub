from os.path import join
from collections import namedtuple

import numpy as np
import tensorflow as tf
from swissknife.files import SavingFolder
from swissknife.learning_rate import ExponentialDecay
from tensorflow.examples.tutorials.mnist import input_data

from basedir import MNIST_IMAGES


N_INPUTS = 28*28
N_OUTPUTS = 10
N_HIDDEN = [300, 200, 100]


def create_inputs(input_size=N_INPUTS):
    x = tf.placeholder(tf.float32, shape=(None, input_size), name='x')
    y = tf.placeholder(tf.int64, shape=(None,), name='y')
    return x, y


def build_model(inputs, training, hidden=N_HIDDEN, n_classes=N_OUTPUTS):
    x = inputs
    with tf.name_scope('model'):
        for index, n in enumerate(hidden):
            layer_name = f'hidden{index}'
            init = tf.variance_scaling_initializer(mode='fan_avg')
            x = tf.layers.dense(x, n, kernel_initializer=init, name=layer_name)
            x = tf.layers.batch_normalization(x, momentum=0.9, training=training)
            x = tf.nn.elu(x)
        logits = tf.layers.dense(x, n_classes, name='logits')
    return logits


def create_optimizer(labels, logits, learning_rate):
    with tf.name_scope('metrics'):
        x_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=labels, logits=logits)
        loss = tf.reduce_mean(x_entropy, name='loss')
        correct = tf.nn.in_top_k(logits, labels, k=1)
        accuracy = tf.reduce_mean(tf.cast(correct, tf.float32), name='accuracy')

    with tf.name_scope('train'):
        opt = tf.train.GradientDescentOptimizer(learning_rate)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            training_op = opt.minimize(loss)

    result = namedtuple('Evaluated', 'loss accuracy training_op')
    return result(loss=loss, accuracy=accuracy, training_op=training_op)


def add_to_collection(name, *ops):
    for op in ops:
        tf.add_to_collection(name, op)


def main():
    training = tf.placeholder(tf.bool, name='training')
    learning_rate = tf.placeholder(tf.float32)
    x, y = create_inputs()
    logits = build_model(x, training)
    ops = create_optimizer(y, logits, learning_rate)

    add_to_collection('inputs', x, y, training, learning_rate)
    add_to_collection('metrics', ops.loss, ops.accuracy)
    add_to_collection('training', ops.training_op)

    n_epochs = 50
    batch_size = 2000
    mnist = input_data.read_data_sets(MNIST_IMAGES)
    scheduler = ExponentialDecay(init_rate=1.0, decay_coef=0.05)

    x_valid = mnist.validation.images
    y_valid = mnist.validation.labels
    n_batches = mnist.train.num_examples // batch_size

    paths_manager = SavingFolder(model_name='dnn_tf_tricks', model_ext='')
    paths_manager.create_model_dir(ask_on_rewrite=False)
    saver = tf.train.Saver()
    checkpoint = 'checkpoint_val_loss_{:2.4f}_epoch_{}'

    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        best_loss = np.infty
        no_improvement_threshold = 10
        no_improvement = 0

        for epoch in range(1, n_epochs + 1):
            current_lr = scheduler(epoch)
            train_losses = []
            for batch in range(n_batches):
                x_train, y_train = mnist.train.next_batch(batch_size)
                feed = {x: x_train,
                        y: y_train,
                        training: True,
                        learning_rate: current_lr}
                _, batch_loss = sess.run([ops.training_op, ops.loss], feed)
                train_losses.append(batch_loss)

            feed = {x: x_valid,
                    y: y_valid,
                    training: False,
                    learning_rate: current_lr}

            if epoch % 5 == 0:
                train_loss = np.mean(train_losses)
                valid_acc, valid_loss = sess.run([ops.accuracy, ops.loss], feed)
                print(f'{epoch:03} '
                      f'Learning Rate: {current_lr:2.4f} '
                      f'Train loss: {train_loss:2.8f} '
                      f'Valid loss: {valid_loss:2.8f} '
                      f'Valid accuracy: {valid_acc:2.2%}')
            else:
                valid_loss = ops.loss.eval(feed)
                if valid_loss < best_loss:
                    best_loss = valid_loss
                    best_model = join(
                        paths_manager.model_dir,
                        checkpoint.format(best_loss, epoch))
                    saver.save(sess, best_model)
                    no_improvement = 0
                elif no_improvement >= no_improvement_threshold:
                    print('Early stopping...')
                    break
                else:
                    no_improvement += 1

        final_model = paths_manager.model_path
        saver.save(sess, final_model)
        print('Final model:', final_model)

    with tf.Session() as sess:
        saver.restore(sess, final_model)
        feed = {x: mnist.test.images, y: mnist.test.labels, training: False}
        test_acc = ops.accuracy.eval(feed)
        print(f'Test accuracy: {test_acc:2.2%}')


if __name__ == '__main__':
    main()
