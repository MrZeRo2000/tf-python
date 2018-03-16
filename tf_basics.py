
import tensorflow as tf

print(tf.__version__)

hello = tf.constant("Hello")
print(type(hello))
print(hello)

with tf.Session() as sess:
    result = sess.run(hello)

print(result)

a = tf.constant(1)
b = tf.constant(2)

with tf.Session() as sess:
    result = sess.run(a + b)

print(result)

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

with tf.Session() as sess:
    a = tf.constant([[1, 2], [3, 4]])
    b = tf.constant([[5], [6]])

    print("matmul")
    print(sess.run(tf.matmul(a, b)))

    print("matmul via eval")
    print(tf.matmul(a, b).eval())
