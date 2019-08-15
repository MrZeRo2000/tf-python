
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

x_data = np.linspace(0.0, 10.0, 1000000)
noise = np.random.randn(len(x_data))

y_true = (0.5 * x_data) + 5 + noise

x_df = pd.DataFrame(data=x_data, columns=['X Data'])
y_df = pd.DataFrame(data=y_true, columns=['Y'])

my_data = pd.concat([x_df, y_df], axis=1)

my_data.sample(n=250).plot(kind='scatter', x='X Data', y='Y')

batch_size = 10

rnd = np.random.rand(2)

m = tf.Variable(rnd[0], dtype=tf.float32)
b = tf.Variable(rnd[1], dtype=tf.float32)

xph = tf.compat.v1.placeholder(tf.float32, [batch_size])
yph = tf.compat.v1.placeholder(tf.float32, [batch_size])

y_model = m * xph + b

error = tf.reduce_sum(tf.square(yph - y_model))

optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=0.001)
train = optimizer.minimize(error)

init = tf.compat.v1.global_variables_initializer()

with tf.compat.v1.Session() as sess:
    sess.run(init)

    batches = 10000

    for i in range(batches):

        rand_ind = np.random.randint(len(x_data), size=batch_size)

        feed = {xph: x_data[rand_ind], yph: y_true[rand_ind]}

        sess.run(train, feed_dict=feed)

    model_m, model_b = sess.run([m, b])

print("m=" + str(model_m) + ", b=" + str(model_b))

y_hat = x_data * model_m + model_b
my_data.sample(250).plot(kind='scatter', x='X Data', y='Y')

plt.plot(x_data, y_hat, 'red')
plt.show()

"""TF estimator"""

feat_cols = [tf.feature_column.numeric_column('x', shape=[1])]
estimator = tf.compat.v1.estimator.LinearRegressor(feature_columns=feat_cols)

from sklearn.model_selection import train_test_split
x_train, x_eval, y_train, y_eval = train_test_split(x_data, y_true, test_size=0.3, random_state=101)

input_func = tf.compat.v1.estimator.inputs.numpy_input_fn({'x': x_train}, y_train,
                                                batch_size=8, num_epochs=None, shuffle=True)
train_input_func = tf.compat.v1.estimator.inputs.numpy_input_fn({'x': x_train}, y_train,
                                                      batch_size=8, num_epochs=1000, shuffle=False)
eval_input_func = tf.compat.v1.estimator.inputs.numpy_input_fn({'x': x_eval}, y_eval,
                                                      batch_size=8, num_epochs=1000, shuffle=False)

estimator.train(input_fn=input_func, steps=1000)

train_metrics = estimator.evaluate(input_fn=train_input_func, steps=1000)
eval_metrics = estimator.evaluate(input_fn=eval_input_func, steps=1000)

print('Training data metrics')
print(train_metrics)
print('Eval metrics')
print(eval_metrics)

brand_new_data = np.linspace(0, 10, 10)
predict_input_func = tf.compat.v1.estimator.inputs.numpy_input_fn({'x': brand_new_data}, shuffle=False)

predictions = []
for pred in estimator.predict(input_fn=predict_input_func):
    predictions.append(pred['predictions'])

my_data.sample(n=250).plot(kind='scatter', x='X Data', y='Y')
plt.plot(brand_new_data, predictions, 'r*')
plt.show()
