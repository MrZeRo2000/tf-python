
import tensorflow as tf

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