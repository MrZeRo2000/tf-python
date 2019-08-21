
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# constants
num_inputs = 2
num_neurons = 3

# placeholders
x0 = tf.compat.v1.placeholder(tf.float32, [None, num_inputs])
x1 = tf.compat.v1.placeholder(tf.float32, [None, num_inputs])

# variables
Wx = tf.Variable(tf.random.normal(shape=[num_inputs, num_neurons]))
Wy = tf.Variable(tf.random.normal(shape=[num_neurons, num_neurons]))

b = tf.Variable(tf.zeros([1, num_neurons]))

# graphs

y0 = tf.math.tanh(tf.matmul(x0, Wx) + b)
y1 = tf.math.tanh(tf.matmul(y0, Wy) + tf.matmul(x1, Wx) + b)

init = tf.compat.v1.global_variables_initializer()

# create data

# timestamp 0
x0_batch = np.array([[0, 1], [2, 3], [4, 5]])

# timestamp 1
x1_batch = np.array([[100, 101], [102, 103], [104, 105]])

with tf.compat.v1.Session() as sess:
    sess.run(init)

    y0_output_vals, y1_output_vals = sess.run([y0, y1], feed_dict={x0: x0_batch, x1: x1_batch})

print ("y0:")
print(y0_output_vals)

print ("y1:")
print(y1_output_vals)
