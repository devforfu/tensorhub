"""
Attempt to solve exercises 8-10 from Hands On Machine Learning (p. 313)
"""
import os
from os.path import join
from datetime import datetime
from collections import namedtuple

import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from swissknife.files import SavingFolder
from swissknife.plotting import plot_predictions

from basedir import MNIST_IMAGES


N_INPUTS = 784


def create_inputs(input_size=N_INPUTS):
    with tf.name_scope('inputs'):
        x = tf.placeholder(tf.float32, shape=(None, input_size), name='x')
        y = tf.placeholder(tf.int64, shape=(None,), name='y')
    return x, y


def build_model(inputs, layers, units, n_outputs, activation,
                rate=0.0, training=False):

    x = inputs
    with tf.name_scope('model'):
        for i in range(layers):
            with tf.name_scope(f'hidden{i}'):
                init = tf.variance_scaling_initializer(mode='fan_avg')
                x = tf.layers.dense(x, units, kernel_initializer=init)
                x = tf.layers.batch_normalization(x, training=training)
                x = tf.layers.dropout(x, rate=rate)
                x = activation(x)
        logits = tf.layers.dense(x, n_outputs, name='logits')
    return logits


def create_optimizer(labels, logits, learning_rate=0.01, opt_cls=None):
    with tf.name_scope('metrics'):
        x_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=labels, logits=logits)
        loss = tf.reduce_mean(x_entropy, name='loss')
        correct = tf.nn.in_top_k(logits, labels, k=1)
        match = tf.cast(correct, tf.float32, name='match')
        accuracy = tf.reduce_mean(match, name='accuracy')

    with tf.name_scope('train'):
        if opt_cls is None:
            opt_cls = tf.train.GradientDescentOptimizer
        opt = opt_cls(learning_rate)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            training_op = opt.minimize(loss)

    result = namedtuple('Evaluated', 'loss accuracy training_op')
    return result(loss=loss, accuracy=accuracy, training_op=training_op)


def create_file_writer(root_dir=None,
                       name=None,
                       date_format='%Y_%m_%d_%H_%M_%S',
                       subfolder='summary',
                       graph=None):

    if root_dir is None:
        root_dir = os.path.expandvars(os.environ.get('TF_LOGS_DIR'))
    if not name:
        name = datetime.utcnow().strftime(date_format)
    path = os.path.join(root_dir, name, subfolder)
    if graph is None:
        graph = tf.get_default_graph()
    writer = tf.summary.FileWriter(path, graph)
    return writer


def get_mnist(from_digit=None, to_digit=None):
    mnist = input_data.read_data_sets(MNIST_IMAGES)
    train_images = mnist.train.images
    valid_images = mnist.validation.images
    test_images = mnist.test.images
    train_labels = mnist.train.labels
    valid_labels = mnist.validation.labels
    test_labels = mnist.test.labels

    train_index = np.ones(train_labels.shape[0]).astype(bool)
    valid_index = np.ones(valid_labels.shape[0]).astype(bool)
    test_index = np.ones(test_labels.shape[0]).astype(bool)

    if from_digit:
        train_index &= (train_labels >= from_digit)
        valid_index &= (valid_labels >= from_digit)
        test_index &= (test_labels >= from_digit)

    if to_digit:
        train_index &= (train_labels <= to_digit)
        valid_index &= (valid_labels <= to_digit)
        test_index &= (test_labels <= to_digit)

    train_images = train_images[train_index]
    valid_images = valid_images[valid_index]
    test_images = test_images[test_index]

    train_labels = train_labels[train_index]
    valid_labels = valid_labels[valid_index]
    test_labels = test_labels[test_index]

    return ((train_images, train_labels),
            (valid_images, valid_labels),
            (test_images, test_labels))


class ArrayBatchGenerator:

    def __init__(self, *arrays, same_size_batches=False,
                 batch_size=32, infinite=False):

        assert same_length(arrays)
        self.same_size_batches = same_size_batches
        self.batch_size = batch_size
        self.infinite = infinite

        total = len(arrays[0])
        n_batches = total // batch_size
        if same_size_batches and (total % batch_size != 0):
            n_batches += 1

        self.current_batch = 0
        self.n_batches = n_batches
        self.arrays = list(arrays)

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def next(self):
        if self.current_batch == self.n_batches:
            if self.infinite:
                self.current_batch = 0
            else:
                raise StopIteration()
        start = self.current_batch * self.batch_size
        batches = [arr[start:(start + self.batch_size)] for arr in self.arrays]
        self.current_batch += 1
        return batches


def same_length(*arrays):
    first, *rest = arrays
    n = len(first)
    for arr in rest:
        if len(arr) != n:
            return False
    return True


class DNNClassifier:

    def __init__(self, n_layers, n_units, n_classes, activation=tf.nn.elu):
        self.n_layers = n_layers
        self.n_units = n_units
        self.n_classes = n_classes
        self.activation = activation

        self.graph = None
        self.training = None
        self.dropout = None
        self.x = None
        self.y = None
        self.logits = None
        self.loss = None
        self.accuracy = None
        self.training_op = None
        self.train_writer = None
        self.valid_writer = None
        self.write_op = None
        self.paths_manager = None
        self._built = False
        self._model_file = None

    def build(self, optimizer=tf.train.GradientDescentOptimizer):
        self.graph = tf.Graph()
        with self.graph.as_default():
            training = tf.placeholder(tf.bool)
            dropout = tf.placeholder(tf.float32)
            x, y = create_inputs()
            logits = build_model(
                inputs=x,
                layers=self.n_layers,
                units=self.n_units,
                n_outputs=self.n_classes,
                activation=self.activation)
            ops = create_optimizer(
                labels=y,
                logits=logits,
                opt_cls=optimizer)
            add_to_collection('inputs', x, y, training, dropout)
            add_to_collection('metrics', ops.loss, ops.accuracy)
            add_to_collection('training', logits, ops.training_op)

        self.training = training
        self.dropout = dropout
        self.logits = logits
        self.loss = ops.loss
        self.accuracy = ops.accuracy
        self.training_op = ops.training_op
        self.x = x
        self.y = y
        self._built = True

    def fit_generator(self,
                      generator,
                      epochs,
                      batches_per_epoch,
                      validation_data,
                      model_name='dnn'):

        if not self._built:
            raise RuntimeError('cannot fit model without building it')

        training_op, training, dropout, logits, loss, accuracy, x, y = (
            self.training_op,
            self.training,
            self.dropout,
            self.logits,
            self.loss,
            self.accuracy,
            self.x,
            self.y)

        x_valid, y_valid = validation_data

        self.paths_manager = SavingFolder(model_name=model_name, model_ext='')
        self.paths_manager.create_model_dir(ask_on_rewrite=False)
        checkpoint = 'checkpoint_val_loss_{:2.4f}_epoch_{}'
        history = []

        with self.graph.as_default():
            saver = tf.train.Saver()

        with tf.Session(graph=self.graph) as sess:
            tf.global_variables_initializer().run()
            best_loss = np.infty
            no_improvement_threshold = 5
            no_improvement = 0
            for epoch in range(1, epochs + 1):
                train_loss = 0.0
                batch_index = 0
                while True:
                    x_batch, y_batch = next(generator)
                    feed = {x: x_batch, y: y_batch,
                            training: False, dropout: 0.25}
                    _, batch_loss = sess.run([training_op, loss], feed)
                    train_loss += batch_loss
                    batch_index += 1
                    if batch_index == batches_per_epoch:
                        break
                train_loss /= batches_per_epoch
                feed = {x: x_valid, y: y_valid, training: False, dropout: 0.0}
                valid_acc, valid_loss = sess.run([accuracy, loss], feed)
                if valid_loss < best_loss:
                    best_loss = valid_loss
                    best_model = join(
                        self.paths_manager.model_dir,
                        checkpoint.format(best_loss, epoch))
                    saver.save(sess, best_model)
                    no_improvement = 0
                elif no_improvement == no_improvement_threshold:
                    print(f'Early stopping after epoch {epoch}')
                    break
                else:
                    no_improvement += 1
                print(f'{epoch:03} '
                      f'Train loss: {train_loss:2.8f} '
                      f'Valid loss: {valid_loss:2.8f} '
                      f'Valid accuracy: {valid_acc:2.2%}')
                record = {'epoch': epoch,
                          'loss': train_loss,
                          'val_loss': valid_loss,
                          'val_acc': valid_acc}
                history.append(record)
            final_model = self.paths_manager.model_path
            saver.save(sess, final_model)
            print('Final model:', final_model)

        self._model_file = final_model
        return history

    def score(self, X, y):
        if not self._built or self._model_file is None:
            raise RuntimeError('cannot predict while until is fit')
        saver, checkpoint = self._restore_graph()
        with tf.Session(graph=self.graph) as sess:
            saver.restore(sess, checkpoint)
            feed = {self.x: X,
                    self.y: y,
                    self.training: False,
                    self.dropout: 0.0}
            loss_value = self.loss.eval(feed)
        return loss_value

    def _restore_graph(self):
        graph = tf.Graph()
        paths_manager = self.paths_manager
        with graph.as_default():
            meta_path = f'{paths_manager.model_path}.meta'
            ckpt_path = tf.train.latest_checkpoint(paths_manager.model_dir)
            saver = tf.train.import_meta_graph(meta_path)
            (self.x, self.y,
             self.training,
             self.dropout) = tf.get_collection('inputs')
            self.loss, self.accuracy = tf.get_collection('metrics')
            self.logits, self.training_op = tf.get_collection('training')
        self.graph = graph
        return saver, ckpt_path


def train_zero_four_model():
    training = tf.placeholder(tf.bool)
    dropout = tf.placeholder(tf.float32)
    x, y = create_inputs()
    logits = build_model(
        inputs=x, layers=5, units=100,
        n_outputs=5, activation=tf.nn.elu)
    ops = create_optimizer(
        labels=y, logits=logits, opt_cls=tf.train.AdamOptimizer)

    train_writer = create_file_writer(name='log', subfolder='train')
    valid_writer = create_file_writer(name='log', subfolder='valid')
    _ = tf.summary.scalar('Loss', ops.loss)
    write_op = tf.summary.merge_all()

    paths_manager = SavingFolder(model_name='dnn_5_layers', model_ext='')
    paths_manager.create_model_dir(ask_on_rewrite=False)
    checkpoint = 'checkpoint_val_loss_{:2.4f}_epoch_{}'
    saver = tf.train.Saver()

    n_epochs = 200
    batch_size = 2000
    x_train, y_train, x_valid, y_valid = get_mnist(0, 4)
    n_batches = x_train.shape[0] // batch_size
    batches = ArrayBatchGenerator(
        x_train, y_train, batch_size=batch_size, infinite=True)

    print('Training model on digits from 0 to 4')
    print(f'Epochs: {n_epochs}, batch size: {batch_size}')

    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        best_loss = np.infty
        no_improvement_threshold = 5
        no_improvement = 0

        for epoch in range(1, n_epochs + 1):
            train_loss = 0.0
            batch_index = 0

            while True:
                x_batch, y_batch = next(batches)
                feed = {x: x_batch, y: y_batch, training: True, dropout: 0.25}
                _, batch_loss = sess.run([ops.training_op, ops.loss], feed)
                train_loss += batch_loss
                batch_index += 1
                if batch_index == n_batches:
                    break

            train_loss /= n_batches
            feed = {x: x_valid, y: y_valid, training: False, dropout: 0.0}
            valid_acc, valid_loss = sess.run([ops.accuracy, ops.loss], feed)

            if valid_loss < best_loss:
                best_loss = valid_loss
                best_model = join(
                    paths_manager.model_dir,
                    checkpoint.format(best_loss, epoch))
                saver.save(sess, best_model)
                no_improvement = 0
            elif no_improvement == no_improvement_threshold:
                print(f'Early stopping after epoch {epoch}')
                break
            else:
                no_improvement += 1

            print(f'{epoch:03} '
                  f'Train loss: {train_loss:2.8f} '
                  f'Valid loss: {valid_loss:2.8f} '
                  f'Valid accuracy: {valid_acc:2.2%}')
            summary = sess.run(write_op, {x: x_batch, y: y_batch})
            train_writer.add_summary(summary, epoch)
            summary = sess.run(write_op, {x: x_valid, y: y_valid})
            valid_writer.add_summary(summary, epoch)

        final_model = paths_manager.model_path
        saver.save(sess, final_model)
        print('Final model:', final_model)

    for writer in (train_writer, valid_writer):
        writer.close()

    return final_model


def add_to_collection(name, *ops):
    for op in ops:
        tf.add_to_collection(name, op)


def main():
    model = DNNClassifier(n_layers=5, n_units=100, n_classes=10)
    batch_size = 2000
    (x_train, y_train), validation_data, test_data = get_mnist()
    n_batches = x_train.shape[0] // batch_size
    batches = ArrayBatchGenerator(
        x_train, y_train, batch_size=batch_size, infinite=True)

    model.build()
    model.fit_generator(
        generator=batches,
        epochs=10,
        batches_per_epoch=n_batches,
        validation_data=validation_data)

    test_loss = model.score(*test_data)
    print(f'Test loss: {test_loss:2.6f}')


if __name__ == '__main__':
    main()
