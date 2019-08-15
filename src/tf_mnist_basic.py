
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

from tensorflow.examples.tutorials.mnist import input_data

# deprecated

mnist = input_data.read_data_sets("MNIST/data", one_hot=True)

# placeholders
x = tf.compat.v1.placeholder(tf.float32, (None, 784))

# variables
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

# graphs
y = tf.matmul(x, W) + b

# loss function
y_true = tf.compat.v1.placeholder(tf.float32, [None, 10])
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=y))

# optimizer
optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=0.5)
train = optimizer.minimize(cross_entropy)

# session
init = tf.compat.v1.global_variables_initializer()

with tf.compat.v1.Session() as sess:
    sess.run(init)

    for step in range(1000):
        batch_x, batch_y = mnist.train.next_batch(100)
        sess.run(train, feed_dict={x: batch_x, y_true: batch_y})

    # evaluate the model
    correct_prediction = tf.equal(tf.math.argmax(y, 1), tf.math.argmax(y_true, 1))

    # true / false into 1/0
    acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    print(sess.run(acc, feed_dict={x: mnist.test.images, y_true: mnist.test.labels}))

pass


mnist2 = tf.keras.datasets.mnist
(X_train, y_train), (X_test, y_test) = mnist2.load_data()

# normalize
X_train = X_train.astype(np.float32) / 255

# change shape
X_train = X_train.reshape(X_train.shape[0], 784)
X_test = X_test.reshape(X_test.shape[0], 784)

# encode
y_train = tf.one_hot(y_train, 10)
y_test = tf.one_hot(y_test, 10)


# single_image = X_train[1]
# plt.imshow(single_image, cmap='gist_gray')
# plt.show()


# placeholders
x2 = tf.compat.v1.placeholder(tf.float32, (None, 784))

# variables
W2 = tf.Variable(tf.zeros([784, 10]))
b2 = tf.Variable(tf.zeros([10]))

# graphs
y2 = tf.matmul(x2, W2) + b2

# loss function
y_true2 = tf.compat.v1.placeholder(tf.float32, [None, 10])
cross_entropy2 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_true2, logits=y2))

# optimizer
optimizer2 = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=0.5)
train2 = optimizer.minimize(cross_entropy2)

# session
init2 = tf.compat.v1.global_variables_initializer()


with tf.compat.v1.Session() as sess:
    sess.run(init2)

    y_train = sess.run(y_train)
    y_test = sess.run(y_test)

    for step in range(1000):
        perm = np.arange(X_train.shape[0])
        np.random.shuffle(perm)

        batch_x2 = X_train[perm[:100]]
        batch_y2 = y_train[perm[:100]]

        sess.run(train2, feed_dict={x2: batch_x2, y_true2: batch_y2})

    # evaluate the model
    correct_prediction2 = tf.equal(tf.math.argmax(y2, 1), tf.math.argmax(y_true2, 1))

    # true / false into 1/0
    acc = tf.reduce_mean(tf.cast(correct_prediction2, tf.float32))

    print(sess.run(acc, feed_dict={x2: X_test, y_true2: y_test}))

