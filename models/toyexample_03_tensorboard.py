import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True, reshape=False, validation_size=0)

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
# summary2 = tf.summary.tensor_summary("OUTPUT", Y, "output of model")

# loss function
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(Ylogits, Y_)
cross_entropy = tf.reduce_mean(cross_entropy) * 100
cost_train = tf.scalar_summary("cost_train", cross_entropy)
cost_val = tf.scalar_summary("cost_val", cross_entropy)

# % of correct answers found in batch
correct_prediction = tf.equal(tf.argmax(Y, 1), tf.argmax(Y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
acc_train = tf.scalar_summary("acc_train", accuracy)
acc_val = tf.scalar_summary("acc_val", accuracy)

# training step
learning_rate = 0.003
optimizer = tf.train.AdamOptimizer(learning_rate)
train_step = optimizer.minimize(cross_entropy)
init = tf.initialize_all_variables()
summary = tf.merge_all_summaries()
sess = tf.Session()
#### BEGIN ####
# Create a summary writer
writer = tf.train.SummaryWriter("./private/", flush_secs=1)
# add the 'graph' to the event file.
writer.add_graph(sess.graph)
# writer.add_graph(tf.get_default_graph())
#### END ####
sess.run(init)
idx = 1;

for i in range(100001):
    # Load batch of images and correct answers
    batch_X, batch_Y = mnist.train.next_batch(100)
    train_data = {X: batch_X, Y_: batch_Y}

    # train
    _, ct, at, = sess.run([train_step,cost_train,acc_train], feed_dict=train_data)
    writer.add_summary(ct, i)
    writer.add_summary(at, i)

    # success? add code to print it
    if i % 100 == 0:
        # c, ct, at = sess.run([cross_entropy, cost_train, acc_train], feed_dict=train_data)
        # print("Accuracy on train set (i = " + str(i) + "): " + str(a))

        # success on test data?
        test_data = {X: mnist.test.images, Y_: mnist.test.labels}
        a, c, cv, av = sess.run([accuracy,cross_entropy, cost_val, acc_val], feed_dict=test_data)
        writer.add_summary(cv, i)
        writer.add_summary(av, i)
        print("Accuracy on test set (i = " + str(i) + "): " + str(a))
