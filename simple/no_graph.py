import tensorflow as tf
from swissknife.files import SavingFolder
from tensorflow.examples.tutorials.mnist import input_data

from basedir import MNIST_IMAGES


N_INPUTS = 28*28
N_OUTPUTS = 10
N_HIDDEN = [300, 200, 100]


def main():
    paths_manager = SavingFolder(model_name='dnn_tf_tricks', model_ext='')
    meta_path = paths_manager.model_path + '.meta'
    print('Loading meta data from:', meta_path)
    saver = tf.train.import_meta_graph(meta_path)

    x, y, training, _ = tf.get_collection('inputs')
    loss, accuracy = tf.get_collection('metrics')

    mnist = input_data.read_data_sets(MNIST_IMAGES)
    x_test = mnist.test.images
    y_test = mnist.test.labels

    with tf.Session() as sess:
        latest = tf.train.latest_checkpoint(paths_manager.model_dir)
        saver.restore(sess, latest)
        feed = {x: x_test, y: y_test, training: False}
        test_acc = accuracy.eval(feed)
        test_loss = loss.eval(feed)
        print(f'Test accuracy: {test_acc:2.2%}')
        print(f'Test loss: {test_loss:2.4f}')


if __name__ == '__main__':
    main()
