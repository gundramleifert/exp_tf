import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import os
import numpy as np


def variable_summaries(var, prefix):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
    with tf.name_scope(prefix):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)


os.chdir("..")
logdir = "private/logs/"
mnist = input_data.read_data_sets("./resources/MNIST_data/", one_hot=True, reshape=False)
K = 200
L = 100
M = 60
N = 30
C = 1.0
# initialization
X = tf.placeholder(tf.float32, [None, 28, 28, 1])
# placeholder for correct answers
Y_ = tf.placeholder(tf.float32, [None, 10])

W1 = tf.Variable(tf.truncated_normal([784, K], stddev=C/np.sqrt(784*1.0)))
B1 = tf.Variable(tf.zeros([K]))
W2 = tf.Variable(tf.truncated_normal([K, L], stddev=C/np.sqrt(K*1.0)))
B2 = tf.Variable(tf.zeros([L]))
W3 = tf.Variable(tf.truncated_normal([L, M], stddev=C/np.sqrt(L*1.0)))
B3 = tf.Variable(tf.zeros([M]))
W4 = tf.Variable(tf.truncated_normal([M, N], stddev=C/np.sqrt(M*1.0)))
B4 = tf.Variable(tf.zeros([N]))
W5 = tf.Variable(tf.truncated_normal([N, 10], stddev=C/np.sqrt(N*1.0)))
B5 = tf.Variable(tf.zeros([10]))

# model
XX = tf.reshape(X, [-1, 28 * 28])  # Input Layer
# summary1 = tf.summary.tensor_summary("INPUT", XX, "input of model")
with tf.name_scope("Hidden_1"):
    Y1 = tf.nn.sigmoid(tf.matmul(XX, W1) + B1)  # Hidden Layer 1
with tf.name_scope("Hidden_2"):
    Y2 = tf.nn.sigmoid(tf.matmul(Y1, W2) + B2)  # Hidden Layer 2
with tf.name_scope("Hidden_3"):
    Y3 = tf.nn.sigmoid(tf.matmul(Y2, W3) + B3)  # Hidden Layer 3
with tf.name_scope("Hidden_4"):
    Y4 = tf.nn.sigmoid(tf.matmul(Y3, W4) + B4)  # Hidden Layer 4
Ylogits = tf.matmul(Y4, W5) + B5  # Output Layer
Y = tf.nn.softmax(Ylogits)

variable_summaries(W1, "W_1")
variable_summaries(W2, "W_2")
variable_summaries(W3, "W_3")
variable_summaries(W4, "W_4")
variable_summaries(W5, "W_5")
variable_summaries(Y1, "Y_1")
variable_summaries(Y2, "Y_2")
variable_summaries(Y3, "Y_3")
variable_summaries(Y4, "Y_4")
variable_summaries(Y, "Y_out")
summary2 = tf.summary.tensor_summary("OUTPUT", Y)

# loss function
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(Ylogits, Y_)
cross_entropy = tf.reduce_mean(cross_entropy) * 100

# % of correct answers found in batch
err_prediction = tf.not_equal(tf.argmax(Y, 1), tf.argmax(Y_, 1))
error = tf.reduce_mean(tf.cast(err_prediction, tf.float32))

########## BEGIN: scalar summary #######################
cost_train = tf.scalar_summary("cost_train", cross_entropy)
cost_val = tf.scalar_summary("cost_val", cross_entropy)
acc_train = tf.scalar_summary("err_train", error)
acc_val = tf.scalar_summary("err_val", error)
########## END: scalar summary #######################

# training step
learning_rate = 0.003
optimizer = tf.train.AdamOptimizer(learning_rate)
train_step = optimizer.minimize(cross_entropy)
init = tf.global_variables_initializer()
merged = tf.merge_all_summaries()
sess = tf.Session()
sess.run(init)

########## BEGIN: initialize summary writer #######################
# Create a summary writer
writer = tf.train.SummaryWriter(logdir)
# add the 'graph' to the event file.
writer.add_graph(sess.graph)
########## END: initialize summary writer #######################

idx = 1;

for i in range(20000):
    # Load batch of images and correct answers
    batch_X, batch_Y = mnist.train.next_batch(100)
    train_data = {X: batch_X, Y_: batch_Y}

    # train
    _, ct, at, = sess.run([train_step, cost_train, acc_train], feed_dict=train_data)

    ########## BEGIN: scalar summary for training #######################
    writer.add_summary(ct, i)
    writer.add_summary(at, i)
    ########## END: scalar summary for training #######################

    # success? add code to print it
    if i % 100 == 99:
        # c, ct, at = sess.run([cross_entropy, cost_train, acc_train], feed_dict=train_data)
        # print("Accuracy on train set (i = " + str(i) + "): " + str(a))

        # success on test data?
        test_data = {X: mnist.test.images, Y_: mnist.test.labels}
        m, a = sess.run([merged, error], feed_dict=test_data)
        print("Error on test set (i = " + str(i+1) + "): " + str(a))

        ########## BEGIN: scalar summary for test #######################
        writer.add_summary(m, i)
        # writer.add_summary(av, i)
        ########## END: scalar summary for test #######################
print("DO THIS TO SEE THE RESULTS")
print(">source activate tensorflow")
print(">tensorboard --logdir " + logdir)
print("Then open http://0.0.0.0:6006/ into your web browser")
