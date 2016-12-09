import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import os

os.chdir("../..")  # to root path
mnist = input_data.read_data_sets("resources/MNIST_data/", one_hot=True, reshape=False, validation_size=0)

K = 200
L = 100
M = 60
N = 30

# initialization
X = tf.placeholder(tf.float32, [None, 28, 28, 1])
# placeholder for correct answers
Y_ = tf.placeholder(tf.float32, [None, 10])

W1 = tf.Variable(tf.truncated_normal([784, K], stddev=0.1))
B1 = tf.Variable(tf.zeros([K]))
W2 = tf.Variable(tf.truncated_normal([K, L], stddev=0.1))
B2 = tf.Variable(tf.zeros([L]))
W3 = tf.Variable(tf.truncated_normal([L, M], stddev=0.1))
B3 = tf.Variable(tf.zeros([M]))
W4 = tf.Variable(tf.truncated_normal([M, N], stddev=0.1))
B4 = tf.Variable(tf.zeros([N]))
W5 = tf.Variable(tf.truncated_normal([N, 10], stddev=0.1))
B5 = tf.Variable(tf.zeros([10]))

# model
XX = tf.reshape(X, [-1, 28 * 28])  # Input Layer
Y1 = tf.nn.sigmoid(tf.matmul(XX, W1) + B1)  # Hidden Layer 1
Y2 = tf.nn.sigmoid(tf.matmul(Y1, W2) + B2)  # Hidden Layer 2
Y3 = tf.nn.sigmoid(tf.matmul(Y2, W3) + B3)  # Hidden Layer 3
Y4 = tf.nn.sigmoid(tf.matmul(Y3, W4) + B4)  # Hidden Layer 4
Ylogits = tf.matmul(Y4, W5) + B5  # Output Layer
Y = tf.nn.softmax(Ylogits)

# loss function
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(Ylogits, Y_)
cross_entropy = tf.reduce_mean(cross_entropy) * 100

# % of correct answers found in batch
correct_prediction = tf.equal(tf.argmax(Y, 1), tf.argmax(Y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# training step
learning_rate = 0.003
optimizer = tf.train.AdamOptimizer(learning_rate)
train_step = optimizer.minimize(cross_entropy)
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

for i in range(1001):
    # Load batch of images and correct answers
    batch_X, batch_Y = mnist.train.next_batch(100)
    train_data = {X: batch_X, Y_: batch_Y}

    # train
    sess.run(train_step, feed_dict=train_data)

    # success? add code to print it
    if i % 100 == 0:
        a, c = sess.run([accuracy, cross_entropy], feed_dict=train_data)
        print("Accuray on train set (i = " + str(i) + "): " + str(a))

        # success on test data?
        test_data = {X: mnist.test.images, Y_: mnist.test.labels}
        a, c = sess.run([accuracy, cross_entropy], feed_dict=test_data)
        print("Accuray on test set (i = " + str(i) + "): " + str(a))
