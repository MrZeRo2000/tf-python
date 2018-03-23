
import tensorflow as tf
import numpy as np

print(tf.__version__)

print("TF Hello")

hello = tf.constant("Hello")
print(type(hello))
print(hello)

with tf.Session() as sess:
    result = sess.run(hello)

print(result)

print("TF constants")

a = tf.constant(1)
b = tf.constant(2)

with tf.Session() as sess:
    result = sess.run(a + b)

print(result)

print("TF fill_mat")

fill_mat = tf.fill([4, 4], 9)

with tf.Session() as sess:
    result = sess.run(fill_mat)
    print("fill_mat")
    print(type(result))
    print(result)

    result = fill_mat.eval()
    print("fill_mat.eval")
    print(type(result))
    print(result)

print("TF matmul")

with tf.Session() as sess:
    a = tf.constant([[1, 2], [3, 4]])
    b = tf.constant([[5], [6]])

    print("matmul")
    print(sess.run(tf.matmul(a, b)))

    print("matmul via eval")
    print(tf.matmul(a, b).eval())

# TF Graphs

print("TF graphs")

n1 = tf.constant(1)
n2 = tf.constant(2)
n3 = n1 + n2

with tf.Session() as sess:
    result = sess.run(n3)
    print(result)

graph_two = tf.Graph()

with graph_two.as_default():
    print(graph_two is tf.get_default_graph())

print(graph_two is tf.get_default_graph())

print("TF variables")

with tf.Session() as sess:
    my_tensor = tf.random_uniform((4,4), 0, 1)
    my_var = tf.Variable(initial_value=my_tensor)

    # fails because variables need to be initialized
    # sess.run(my_var)

    init = tf.global_variables_initializer()
    sess.run(init)
    result = sess.run(my_var)
    print(result)

print("TF placeholders")

with tf.Session() as sess:
    ph = tf.placeholder(tf.float32, shape=(None,5))

print("TF linear NN")

np.random.seed(101)
tf.set_random_seed(101)

rand_a = np.random.uniform(0, 100, (5, 5))
rand_b = np.random.uniform(0, 100, (5, 1))

a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)

add_op = a + b
mul_op = a * b

with tf.Session() as sess:
    print("add_result")
    add_result = sess.run(add_op, feed_dict={a:rand_a, b:rand_b})
    print(add_result)
    print("mul_result")
    mult_result = sess.run(mul_op, feed_dict={a:rand_a, b:rand_b})
    print(mult_result)

print("Example Neural Network")

n_features = 10
n_dense_neurons = 3

x = tf.placeholder(tf.float32, (None, n_features))
W = tf.Variable(tf.random_normal([n_features, n_dense_neurons]))
b = tf.Variable(tf.ones([n_dense_neurons]))

xW = tf.matmul(x, W)

z = tf.add(xW, b)

# activation
a = tf.sigmoid(z)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)

    print("b")
    print(sess.run(b))

    layer_out = sess.run(a, feed_dict={x:np.random.random([1, n_features])})

    print("layer_out")
    print(layer_out)

print("Simple regression example")

x_data = np.linspace(0, 10, 10) + np.random.uniform(-1.5, 1.5, 10)
y_label = np.linspace(0, 10, 10) + np.random.uniform(-1.5, 1.5, 10)

import matplotlib.pyplot as plt

# plt.plot(x_data, y_label, '*')
# plt.show()

"""y = mx + b"""

rd = np.random.rand(2)

m = tf.Variable(rd[0])
b = tf.Variable(rd[1])

error = 0

for (x, y) in zip(x_data, y_label):
    print("X=" + str(x) + ", Y=" + str(y))
