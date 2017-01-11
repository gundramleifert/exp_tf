import numpy as np
import tensorflow as tf
import tensorflow.python.debug as tf_debug

cluster = tf.train.ClusterSpec({"local": ["localhost:2222", "localhost:2223"]})
size = 2
x = tf.placeholder(tf.float32, size)
default = True

if default:
    first_batch = tf.slice(x, [0], [size / 2])
    mean1 = tf.reduce_mean(first_batch)
    second_batch = tf.slice(x, [size / 2], [-1])
    mean2 = tf.reduce_mean(second_batch)
    mean = (mean1 + mean2) / 2
else:
    with tf.device("/job:local/task:1"):
        first_batch = tf.slice(x, [0], [size / 2])
        mean1 = tf.reduce_mean(first_batch)

    with tf.device("/job:local/task:0"):
        second_batch = tf.slice(x, [size / 2], [-1])
        mean2 = tf.reduce_mean(second_batch)
        mean = (mean1 + mean2) / 2

print("before session")
if (False):
    sess1 = tf.Session("grpc://localhost:2222")
    sess = tf_debug.LocalCLIDebugWrapperSession(sess1)
else:
    sess = tf.Session()
# with tf_debug.LocalCLIDebugWrapperSession(sess1) as sess:
print("in session")
result = sess.run(mean, feed_dict={x: np.random.random(size)})
print(result)
