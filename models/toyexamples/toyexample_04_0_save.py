import tensorflow as tf
from django.template.smartif import prefix
from tensorflow.examples.tutorials.mnist import input_data
import os
from util.saver import PrefixSaver

"""
A model is created (in scope 'net') and all weights are saved by the saver.
Take a look at the main program - if you set load_weights=True the calculated weights are taken. To load the weights into another network have a look at toyexample_04_1_restore
"""


class ModelHandler:
    def __init__(self):
        ############# BEGIN SAVER #############
        self.prefix_saver = PrefixSaver("net", './private/models/toyexample_04')
        ############# END SAVER #############

    def create(self, X):
        # initialization
        # X = tf.placeholder(tf.float32, [None, 28, 28, 1])
        # placeholder for correct answers
        K = 200
        L = 100
        M = 60
        N = 30
        number = 1
        with tf.name_scope("net"):
            with tf.name_scope("l" + str(number)):
                W1 = tf.Variable(tf.truncated_normal([784, K], stddev=0.1))
                B1 = tf.Variable(tf.zeros([K]))
            number += 1
            with tf.name_scope("l" + str(number)):
                W2 = tf.Variable(tf.truncated_normal([K, L], stddev=0.1))
                B2 = tf.Variable(tf.zeros([L]))
            number += 1
            with tf.name_scope("l" + str(number)):
                W3 = tf.Variable(tf.truncated_normal([L, M], stddev=0.1))
                B3 = tf.Variable(tf.zeros([M]))
            number += 1
            with tf.name_scope("l" + str(number)):
                W4 = tf.Variable(tf.truncated_normal([M, N], stddev=0.1))
                B4 = tf.Variable(tf.zeros([N]))
            number += 1
            with tf.name_scope("l" + str(number)):
                W5 = tf.Variable(tf.truncated_normal([N, 10], stddev=0.1))
                B5 = tf.Variable(tf.zeros([10]))
        # model
        XX = tf.reshape(X, [-1, 28 * 28])  # Input Layer
        Y1 = tf.nn.sigmoid(tf.matmul(XX, W1) + B1)  # Hidden Layer 1
        Y2 = tf.nn.sigmoid(tf.matmul(Y1, W2) + B2)  # Hidden Layer 2
        Y3 = tf.nn.sigmoid(tf.matmul(Y2, W3) + B3)  # Hidden Layer 3
        Y4 = tf.nn.sigmoid(tf.matmul(Y3, W4) + B4)  # Hidden Layer 4
        HypLogit = tf.matmul(Y4, W5) + B5  # Output Layer
        Hyp = tf.nn.softmax(HypLogit)
        return HypLogit, Hyp

    def optimizer(self, hyp_logit, hyp, gt):
        # loss function
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(hyp_logit, gt)
        cross_entropy = tf.reduce_mean(cross_entropy)

        # % of correct answers found in batch
        correct_prediction = tf.equal(tf.argmax(gt, 1), tf.argmax(hyp, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        # training step
        # learning_rate = 0.003
        optimizer = tf.train.AdamOptimizer()
        loss = optimizer.minimize(cross_entropy)
        return loss, accuracy, cross_entropy

    def train(self, mnist, epochs=1000, load_weights=False):
        X = tf.placeholder(tf.float32, [None, 28, 28, 1])
        GT = tf.placeholder(tf.float32, [None, 10])
        Ylogits, Y = self.create(X)
        loss, accuracy, cross_entropy = self.optimizer(Ylogits, Y, GT)
        init = tf.global_variables_initializer()
        sess = tf.Session()
        sess.run(init)
        ############# BEGIN SAVER #############
        if load_weights:
            self.prefix_saver.restore(sess)
        ############# END SAVER #############
        for i in range(epochs):
            # Load batch of images and correct answers
            batch_X, batch_Y = mnist.train.next_batch(100)
            train_data = {X: batch_X, GT: batch_Y}

            # train
            sess.run(loss, feed_dict=train_data)

            # success? add code to print it
            if (i + 1) % 100 == 0:
                a, c = sess.run([accuracy, cross_entropy], feed_dict=train_data)
                print("Accuray on train set (i = " + str(i + 1) + "): " + str(a))
                ############# BEGIN SAVER #############
                self.prefix_saver.save(sess)  # , global_step=i + 1)
                ############# END SAVER #############
                # success on test data?
                test_data = {X: mnist.test.images, GT: mnist.test.labels}
                a, c = sess.run([accuracy, cross_entropy], feed_dict=test_data)
                print("Accuray on test set (i = " + str(i + 1) + "): " + str(a))


if __name__ == '__main__':
    os.chdir("../..")  # to root path
    mnist = input_data.read_data_sets("resources/MNIST_data/", one_hot=True, reshape=False, validation_size=0)
    mh = ModelHandler()
    # mh.train(mnist, epochs=3000, load_weights=False)
    mh.train(mnist, epochs=1000, load_weights=False)
