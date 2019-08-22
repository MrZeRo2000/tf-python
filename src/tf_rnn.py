
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


class TimeSeriesData:
    def __init__(self, num_points, xmin, xmax):
        self.num_points = num_points
        self.xmin = xmin
        self.xmax = xmax
        self.resolution = (xmax - xmin) / num_points

        self.x_data = np.linspace(xmin, xmax, num_points)
        self.y_true = np.sin(self.x_data)

    def ret_true(self, x_series):
        return np.sin(x_series)

    def next_batch(self, batch_size, steps, return_batch_ts=False):

        # Grab a random staring point
        rand_start = np.random.rand(batch_size, 1)

        # Convert to be on time series
        ts_start = rand_start * (self.xmax - self.xmin - (steps * self.resolution))

        # Create batch time series on x axis
        batch_ts = ts_start + np.arange(0.0, steps + 1) * self.resolution

        # Create Y data for time series
        y_batch = np.sin(batch_ts)

        # Formatting for RNN
        if return_batch_ts:
            return y_batch[:, :-1].reshape(-1, steps, 1), y_batch[:, 1:].reshape(-1, steps, 1), batch_ts
        else:
            # first is time series, second is time series shifted 1 step to future
            return y_batch[:, :-1].reshape(-1, steps, 1), y_batch[:, 1:].reshape(-1, steps, 1)


ts_data = TimeSeriesData(250, 0, 10)
# plt.plot(ts_data.x_data, ts_data.y_true)
# plt.show()

num_time_steps = 30
y1, y2, ts = ts_data.next_batch(1, num_time_steps, True)
# plt.plot(ts.flatten()[1:], y2.flatten(), '*')
# plt.show()

plt.plot(ts_data.x_data, ts_data.y_true, label="Sin(t)")
plt.plot(ts.flatten()[1:], y2.flatten(), '*', label="Single training instance")
plt.legend()
plt.tight_layout()
# plt.show()

train_ins = np.linspace(5, 5 + ts_data.resolution * (num_time_steps + 1), num_time_steps + 1)

plt.title('A TRAINING INSTANCE')
plt.plot(train_ins[:-1], ts_data.ret_true(train_ins[:-1]), 'bo', markersize=15, alpha=0.5, label='INSTANCE')
plt.plot(train_ins[1:], ts_data.ret_true(train_ins[1:]), 'ko', markersize=7, alpha=0.5, label='TARGET')
plt.legend()
# plt.show()

# Creating the model

tf.compat.v1.reset_default_graph()

# 1 feature in the time series
num_inputs = 1

# could be played around
num_neurons = 100

num_outputs = 1

# version 1
# learning_rate = 0.0001

# version 2
learning_rate = 0.001


num_train_iterations = 2000

batch_size = 1

# PLACEHOLDER

X = tf.compat.v1.placeholder(tf.float32, [None, num_time_steps, num_inputs])
y = tf.compat.v1.placeholder(tf.float32, [None, num_time_steps, num_outputs])

# RNN CELL LAYER

# version 1
# cell = tf.contrib.rnn.OutputProjectionWrapper(
#    tf.contrib.rnn.BasicRNNCell(num_units=num_neurons, activation=tf.compat.v1.nn.relu),
#    output_size=num_outputs
#)

# version 2
cell = tf.contrib.rnn.OutputProjectionWrapper(
    tf.contrib.rnn.GRUCell(num_units=num_neurons, activation=tf.compat.v1.nn.relu),
    output_size=num_outputs
)


outputs, states = tf.nn.dynamic_rnn(cell, X, dtype=tf.float32)

# MSE
loss = tf.reduce_mean(tf.square(outputs - y))

optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

train = optimizer.minimize(loss)

init = tf.compat.v1.global_variables_initializer()

# SESSION

# allows to saave a model
saver = tf.train.Saver()

with tf.compat.v1.Session() as sess:
    sess.run(init)

    for iteration in range(num_train_iterations):
        X_batch, y_batch = ts_data.next_batch(batch_size, num_time_steps)

        sess.run(train, feed_dict={X: X_batch, y: y_batch})

        if iteration % 100 == 0:
            mse = loss.eval(feed_dict={X: X_batch, y: y_batch})
            print(iteration, "\tMSE", mse)

    saver.save(sess, "./rnn_time_series_model_codealong")

with tf.compat.v1.Session() as sess:
    saver.restore(sess, "./rnn_time_series_model_codealong")

    X_new = np.sin(np.array(train_ins[:-1].reshape(-1, num_time_steps, num_inputs)))
    y_pred = sess.run(outputs, feed_dict={X: X_new})

plt.clf()

plt.title("TESTING THE MODEL")

# TRAINING INSTANCE
plt.plot(train_ins[:-1], np.sin(train_ins[:-1]), "bo", markersize=15, alpha=0.5, label="TRAINING INST")

# TARGET TO PREDICT
plt.plot(train_ins[1:], np.sin(train_ins[1:]), "ko", markersize=10, alpha=0.5, label="TARGET")

# MODELS PREDICTION
plt.plot(train_ins[1:], y_pred[0, :, 0], "r.", markersize=10, alpha=0.5, label="PREDICTIONS")

plt.xlabel("TIME")
plt.legend()
plt.tight_layout()
plt.show()