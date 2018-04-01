
# Introducing tf.placeholder()
# Node that don’t actually perform any computation, 
# it just output the data you tell its to output at runtime.

# Example:

# # shape(n_row, n_col)
# A = tf.placeholder(tf.float32, sahpe=(None, 3))
# B = A + 5

# with tf.Session() as sess:
#     # feed_dict give a value to the placeholder A
#     B_val_1 = B.eval(feed_dict={ A: [[1, 2, 3]] })
#     # -> [[6, 7, 8]]
#     B_val_2 = B.eval(feed_dict={ A: [[4, 5, 6], [7, 8, 9]] })
#     # -> [[9, 10, 11], [12, 13, 14]]
#     # B is evaluated usign the placeholder A set into eval()

import numpy as np
import tensorflow as tf

from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler

# MINI_BATCH GRADIENT DESCENT

# General variables
housing = fetch_california_housing()
m, n = housing.data.shape

scaled_housing_data = StandardScaler().fit_transform(housing.data)
scaled_housing_data_plus_bias = np.c_[np.ones((m, 1)), scaled_housing_data]

n_epochs = 20
learning_rate = 0.01


# Now set X and y as placeholders
X = tf.placeholder(tf.float32, shape=(None, n + 1), name="X")
y = tf.placeholder(tf.float32, shape=(None, 1), name="y")

theta = tf.Variable(tf.random_uniform([n + 1, 1], -1.0, 1.0, seed=42), name="theta")
y_pred = tf.matmul(X, theta, name="predictions")

error = y_pred - y
mse = tf.reduce_mean(tf.square(error), name="mse")

optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
training_op = optimizer.minimize(mse)

batch_size = 100
n_batches = int(np.ceil(m / batch_size))


def fetch_batch(epoch, batch_index, batch_size):
    np.random.seed(epoch * n_batches + batch_index)  
    indices = np.random.randint(m, size=batch_size)  

    # Select a random group of X and y
    X_batch = scaled_housing_data_plus_bias[indices] 
    y_batch = housing.target.reshape(-1, 1)[indices] 
    return X_batch, y_batch


init = tf.global_variables_initializer()
# saver token for save persistance the state of computation
# global save all variables:
#       saver = tf.train.Saver()
# Save precisely variables:
#       saver = tf.train.Saver({"weights": theta})


with tf.Session() as sess:
    sess.run(init)
    # instead of sess.run(init) call restore() in order to continue the
    # saved computation
    # saver.restore(sess, "/tmp/my_model_final.ckpt")

    for epoch in range(n_epochs):
        for batch_index in range(n_batches):
            # each epoch we change the placeholders X and y
            # fetch a random subset (batch) from all instances
            X_batch, y_batch = fetch_batch(epoch, batch_index, batch_size)
            sess.run(training_op, feed_dict={X: X_batch, y: y_batch})

        # Save checkpoint
        # if epoch % 100 == 0: # checkpoint every 100 epochs
        #     save_path = saver.save(sess, "/tmp/my_model.ckpt")

    best_theta = theta.eval()
    print("best theta\n", best_theta)

    # Persistance save result of computation 
    # save_path = saver.save(sess, "/tmp/my_model_final.ckpt")