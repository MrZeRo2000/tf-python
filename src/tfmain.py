import tensorflow as tf

hello = tf.constant("hello")

sess = tf.compat.v1.Session()

print(sess.run(hello))