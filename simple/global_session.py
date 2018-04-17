import os
import sys
from os import makedirs
from operator import gt, lt
from os.path import join, exists
from collections import namedtuple, OrderedDict

import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

from basedir import MNIST_IMAGES

N_INPUTS = 784


class DNNClassifier:
    def __init__(self, hidden_layers, n_classes, activation=tf.nn.elu):
        self.hidden_layers = hidden_layers
        self.n_classes = n_classes
        self.activation = activation
        self._graph = None
        self._session = None
        self._saver = None
        self.stop_training = False

    @property
    def session(self):
        return self._session

    @property
    def saver(self):
        return self._saver

    @property
    def graph(self):
        return self._graph

    @staticmethod
    def restore_model(model_dir) -> 'DNNClassifier':
        """Restores previously saved model from file."""

        graph = tf.Graph()
        latest_checkpoint = tf.train.latest_checkpoint(model_dir)
        meta_path = latest_checkpoint + '.meta'
        if not exists(meta_path):
            raise RuntimeError('cannot restore model: meta file not found')

        with graph.as_default():
            session = tf.Session(graph=graph)
            saver = tf.train.import_meta_graph(meta_path)
            saver.restore(session, latest_checkpoint)
            hidden_layers = (
                graph.
                    get_collection('variables', 'dense\d/kernel'))
            activation_type = (
                graph.
                    get_operation_by_name('model/activation1').
                    type.lower())
            n_outputs = (
                graph.
                    get_tensor_by_name('logits/kernel:0').
                    get_shape().as_list()[1])
            activation = getattr(tf.nn, activation_type)
            classifier = DNNClassifier(
                hidden_layers=len(hidden_layers),
                n_classes=n_outputs,
                activation=activation)
            classifier._saver = saver
            classifier._graph = graph
            classifier._session = session

        return classifier

    def build(self, graph=None, optimizer=None):
        if graph is None:
            graph = tf.Graph()
        with graph.as_default():
            x, y, training, dropout = create_inputs()
            logits = build_model(
                inputs=x,
                layers=self.hidden_layers,
                n_outputs=self.n_classes,
                activation=self.activation)
            ops = create_optimizer(
                labels=y,
                logits=logits,
                opt_cls=optimizer)
            add_to_collection('inputs', x, y, training, dropout)
            add_to_collection('metrics', ops.loss, ops.accuracy)
            add_to_collection('training', logits, ops.training_op)
        self._graph = graph

    def fit_generator(self,
                      generator,
                      epochs,
                      batches_per_epoch,
                      validation_data=None,
                      callbacks=None):
        """
        Fits model with generator yielding batches of (x, y) pairs.
        """
        if self._session is not None:
            self._session.close()

        graph = self._graph

        with graph.as_default():
            x, y, training, dropout = tf.get_collection('inputs')
            loss, accuracy = tf.get_collection('metrics')
            logits, training_op = tf.get_collection('training')
            init = tf.global_variables_initializer()
            self._saver = tf.train.Saver()

        self._session = session = tf.Session(graph=graph)
        monitor = CallbacksGroup(callbacks or [])
        monitor.set_model(self)
        monitor.on_start_training()

        init.run(session=session)
        for epoch in range(1, epochs + 1):
            if self.stop_training:
                break
            epoch_loss = 0.0
            for batch_index in range(batches_per_epoch):
                x_batch, y_batch = next(generator)
                feed = {x: x_batch, y: y_batch, training: True, dropout: 0.5}
                _, batch_loss = session.run([training_op, loss], feed)
                epoch_loss += batch_loss
                monitor.on_batch(epoch, batch_index)
            epoch_loss /= batches_per_epoch
            metrics = {'train_loss': epoch_loss}
            if validation_data:
                x_valid, y_valid = validation_data
                feed = {x: x_valid, y: y_valid, training: False, dropout: 0}
                val_acc, val_loss = session.run([accuracy, loss], feed)
                metrics['val_loss'] = val_loss
                metrics['val_acc'] = val_acc
            monitor.on_epoch(epoch, **metrics)
        monitor.on_end_training()

    def score(self, x, y):
        session = self._session

        if session is None:
            raise RuntimeError('cannot run predictions without trained model')

        with session.graph.as_default():
            x_, y_, training, dropout = tf.get_collection('inputs')
            loss, accuracy = tf.get_collection('metrics')

        feed = {x_: x, y_: y, training: False, dropout: 0}
        l, acc = self._session.run([loss, accuracy], feed)
        return {'loss': l, 'accuracy': acc}


def create_inputs(input_size=N_INPUTS):
    with tf.name_scope('inputs'):
        x = tf.placeholder(tf.float32, shape=(None, input_size), name='x')
        y = tf.placeholder(tf.int64, shape=(None,), name='y')
        dropout = tf.placeholder(tf.float32, name='dropout')
        training = tf.placeholder(tf.bool, name='training')
    return x, y, dropout, training


def build_model(inputs, layers, n_outputs, activation,
                rate=0.0, training=False, batch_norm=False,
                dropout=None):
    x = inputs
    with tf.name_scope('model'):
        for i, units in enumerate(layers):
            init = tf.variance_scaling_initializer(mode='fan_avg')
            x = tf.layers.dense(
                x, units, kernel_initializer=init, name=f'dense{i}')
            if batch_norm:
                x = tf.layers.batch_normalization(
                    x, training=training, name=f'batchnorm{i}')
            if dropout is not None:
                x = tf.layers.dropout(x, rate=rate, name=f'dropout{i}')
            x = activation(x, name=f'activation{i}')
        logits = tf.layers.dense(x, n_outputs, name='logits')
    return logits


def create_optimizer(labels, logits, learning_rate=0.01, opt_cls=None):
    """
    Creates an optimizer from labels and logits tensors.

    Args:
         labels: Tensor with training labels.
         logits: Tensor with logits.
         learning_rate: Optimizer learning rate.
         opt_cls: Optimizer's class (gradient descent optimizer is chosen
            by default if parameter not provided).

    Returns:
        loss, accuracy, training_op: Tensors required to train model and
            track progress.

    """
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


def add_to_collection(name, *ops):
    for op in ops:
        tf.add_to_collection(name, op)


class Callback:
    def __init__(self):
        self.observed_model = None

    def set_model(self, model):
        self.observed_model = model

    def on_start_training(self):
        pass

    def on_end_training(self):
        pass

    def on_epoch(self, epoch, **metrics):
        pass

    def on_batch(self, epoch, batch_index, **metrics):
        pass


class CallbacksGroup(Callback):
    def __init__(self, callbacks):
        super().__init__()
        self.callbacks = list(callbacks)

    def set_model(self, model):
        for callback in self.callbacks:
            callback.set_model(model)

    def on_start_training(self):
        for callback in self.callbacks:
            callback.on_start_training()

    def on_end_training(self):
        for callback in self.callbacks:
            callback.on_end_training()

    def on_epoch(self, epoch, **metrics):
        for callback in self.callbacks:
            callback.on_epoch(epoch, **metrics)

    def on_batch(self, epoch, batch_index, **metrics):
        for callback in self.callbacks:
            callback.on_batch(epoch, batch_index, **metrics)


class ModelSaver(Callback):
    _checkpoint_format = 'checkpoint_val_loss_{val_loss:2.4f}_epoch_{epoch:d}'

    def __init__(self,
                 model_name,
                 output_dir=None,
                 filename_format=None,
                 metric='val_loss',
                 minimize=True,
                 save_final_model=True):

        super().__init__()
        if not output_dir:
            output_dir = os.environ.get('TF_OUTPUT_DIR', '')
        self.model_name = model_name
        self.output_dir = output_dir
        self.model_dir = join(self.output_dir, self.model_name)
        self.filename_format = filename_format or self._checkpoint_format
        self.metric = metric
        self.minimize = minimize
        self.save_final_model = save_final_model
        self.observed_model = None
        self._op = lt if self.minimize else gt
        self._best_value = np.infty if self.minimize else -np.infty
        self._best_checkpoint = None
        self._final_model = None

    @property
    def best_checkpoint(self):
        return self._best_checkpoint

    @property
    def final_model(self):
        return self._final_model

    def on_start_training(self):
        """
        Creates a directory to save model checkpoints if it doesn't exist.
        """
        if not exists(self.model_dir):
            makedirs(self.model_dir, exist_ok=True)

    def on_epoch(self, epoch, **metrics):
        """
        Saves model if it has better `metric` then on previous epoch.
        """
        metric_value = metrics.get(self.metric)
        if not metric_value:
            return
        if self._not_better(metric_value, self._best_value):
            return
        params = {'epoch': epoch, self.metric: metric_value}
        filename = self.filename_format.format(**params)
        path = join(self.model_dir, filename)
        session = self.observed_model.session
        self.observed_model.saver.save(session, path)
        self._best_checkpoint = path

    def on_end_training(self):
        """
        Saves model at the end of training process if `save_final_model`
        parameter is True.
        """
        if self.save_final_model:
            path = join(self.model_dir, self.model_name + '.final')
            session = self.observed_model.session
            self.observed_model.saver.save(session, path)
            self._final_model = path

    def _not_better(self, a, b):
        return not self._op(a, b)


class EarlyStopping(Callback):
    def __init__(self, metric='val_loss', minimize=True, patience=10):
        super().__init__()
        self.metric = metric
        self.patience = patience
        self.minimize = minimize
        self._op = lt if self.minimize else gt
        self._no_improvement = 0
        self._prev_value = np.infty if minimize else -np.infty
        self._best_epoch = 0

    def on_epoch(self, epoch, **metrics):
        metric_value = metrics.get(self.metric)
        if not metric_value:
            return

        if self._better(metric_value, self._prev_value):
            self._prev_value = metric_value
            self._no_improvement = 0
            self._best_epoch = epoch
        else:
            self._no_improvement += 1
            if self._no_improvement < self.patience:
                return
            self.observed_model.stop_training = True

    def _better(self, a, b):
        return self._op(a, b)


class StreamLogger(Callback):
    def __init__(self, output=sys.stdout, stats_formats=None):
        super().__init__()

        if stats_formats is None:
            stats_formats = OrderedDict()
            stats_formats['epoch'] = '03d'
            stats_formats['train_loss'] = '2.6f'
            stats_formats['val_loss'] = '2.6f'
            stats_formats['val_acc'] = '2.2%'

        format_strings = [
            '%s: {%s:%s}' % (stat, stat, fmt)
            for stat, fmt in stats_formats.items()]
        format_string = ' - '.join(format_strings)

        self.output = output
        self.format_string = format_string

    def write(self, string):
        self.output.write(string)
        self.output.write('\n')
        self.output.flush()

    def on_start_training(self):
        self.write('Model training started')

    def on_end_training(self):
        self.write('Model training ended')

    def on_epoch(self, epoch, **metrics):
        metrics['epoch'] = epoch
        stats = self.format_string.format(**metrics)
        self.write(stats)


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


def main():
    bs = 1000
    epochs = 3
    (x_train, y_train), validation_data, test_data = get_mnist(0, 4)
    n_batches = x_train.shape[0] // bs
    layers = [100 for _ in range(5)]

    batches = ArrayBatchGenerator(
        x_train, y_train, batch_size=bs, infinite=True)

    callbacks = [
        ModelSaver(model_name='dnn_5_100'),
        EarlyStopping(patience=10),
        StreamLogger()]

    model = DNNClassifier(hidden_layers=layers, n_classes=5)
    model.build()
    model.fit_generator(
        generator=batches,
        epochs=epochs,
        batches_per_epoch=n_batches,
        validation_data=validation_data,
        callbacks=callbacks)

    restored = DNNClassifier.restore_model('/home/ck/tf_output/dnn_5_100')
    assert restored is not model
    scores = restored.score(*test_data)
    print('Test dataset scores: {loss:2.6f} - {accuracy:2.2%}'.format(**scores))


if __name__ == '__main__':
    main()
