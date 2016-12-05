'''

Author: Gundram
'''

import tensorflow as tf

# for example an image
input1 = tf.placeholder(tf.float32, shape=[1, 2], name="input1")
# for example a second image
input2 = tf.placeholder(tf.float32, shape=[1, 2], name="input2")
# multiply them
output = tf.matmul(input1, input2, transpose_b=True)

with tf.Session() as sess:
    print(sess.run([output], feed_dict={input1: [[7., 4.]], input2: [[2., 4.]]}))
    print(sess.run([output], feed_dict={input1: [[1., 2.]], input2: [[3., 4.]]}))

    # output:
    # [array([ 14.], dtype=float32)]
