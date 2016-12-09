'''

Author: Gundram
'''

import tensorflow as tf

# Create a Constant op that produces a 1x2 matrix.  The op is
# added as a node to the default graph.
matrix1 = tf.constant([[3., 4.]])

# Create another Constant that produces a 2x1 matrix.
matrix2 = tf.constant([[2.],[2.]])

# Create a Matmul op that takes 'matrix1' and 'matrix2' as inputs.
# The returned value, 'product', represents the result of the matrix
# multiplication.
product = tf.matmul(matrix1, matrix2,transpose_a=True,transpose_b=True)
product2 = tf.matmul(matrix1, matrix2)

# Launch the default graph.
sess = tf.Session()

# To run the matmul op we call the session 'run()' method, passing 'product'
# which represents the output of the matmul op.  This indicates to the call
# that we want to get the output of the matmul op back.
#
# All inputs needed by the op are run automatically by the session.  They
# typically are run in parallel.
#
# The call 'run(product)' thus causes the execution of three ops in the
# graph: the two constants and matmul.
#
# The output of the matmul is returned in 'result' as a numpy `ndarray` object.
dict = {"prod1":product, "prod2":product2}

result = sess.run(dict)
# result = sess.run([product,product2])

print(result)
# ==> [[ 12.]]

# Close the Session when we're done.
sess.close()
