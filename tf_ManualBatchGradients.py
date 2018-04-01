import tensorflow as tf
import numpy as np
from sklearn.datasets import fetch_california_housing

from sklearn.preprocessing import StandardScaler

housing = fetch_california_housing()
# print(housing.data.shape)   # -> (20640, 8)
# print(housing.feature_names) # -> ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup', 'Latitude', 'Longitude']

# get number of row entries (m) and num of columns features (n)
m, n = housing.data.shape
# np.ones((m, 1)): create array of m with array with one element that 
# contains 1 as value. [ [1] ... [1] ] output array of m element
# np.c_[first, second] : get an elem i from first array and concatenate with elem j from second array

# Summary add x_0 equal to 1 for intercept for each row in housing.data
# housing_data_plus_bias = np.c_[np.ones((m, 1)), housing.data]

# StandardScaler(): transform your data such that its distribution 
# will have a mean value 0 and standard deviation of 1 
#   -> help convergence of Gradient Descent
scaled_housing_data = StandardScaler().fit_transform(housing.data)

# Add x_0 = 1 for each row
#   -> know the number of columns = n + 1
scaled_housing_data_plus_bias = np.c_[np.ones((m, 1)), scaled_housing_data]


# MANUALLY COMPUTE GRADIENT

# Max iteration of computation of gradient
n_epochs = 1000
# How much a gradient step change position
learning_rate = 0.01

# Add data row to X tf constant
X = tf.constant(scaled_housing_data_plus_bias, dtype=tf.float32, name="X")
# -> [ [1, x1_1, ..., x1_N], [1, x2_1, ..., x2_N], ..., [1, xM_1..., xM_N] ]

# Add data target to y tf constant
# reshape(-1, 1): create for each element into target array an array of such 
#                 element, these array are insert into an array. 
#                 (-1 because we leave to numpy the responsibility)
y = tf.constant(housing.target.reshape(-1, 1), dtype=tf.float32, name="y")
# -> [ [y1], [y2], ... [yM] ]

# Add theta (weights to find)
# random_uniform(): outputs random values from a uniform distribution 
#                   between -1 and 1.
# -> [ [v_1] ... [v_n+1] ] 
# N.B. shape(9, 1) = array with 9 elements with 1 element
theta = tf.Variable(tf.random_uniform([n + 1, 1], 
                    minval=-1.0, maxval=1.0), 
                    name="theta")

y_pred = tf.matmul(X, theta, name="predictions")
# -> X * theta (vectors) = y_hat

error = y_pred - y
# .square(error) = error ^ 2
# .reduce_mean() = compute the mean of input array
mse = tf.reduce_mean(tf.square(error), name="mse")

# compute the batch gradient with all the instances X
# gradients = 2 / m * tf.matmul(tf.transpose(X), error)
# Now use tensorflow power: the gradients node will compute the gradient 
#                           vector of the MSE with regards to theta.
# gradients = tf.gradients(mse, [theta])[0]

# Gradient optimizer
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
optimizerMom = tf.train.MomentumOptimizer(learning_rate=learning_rate,
                                        momentum=0.9)

# assign(theta, value): assign to var theta the value
# training_op is a tensor that keep the reference
# training_op = tf.assign(theta, theta - learning_rate * gradients)
training_op = optimizerMom.minimize(mse)


init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)

    for epoch in range(n_epochs):
        if epoch % 100 == 0:
            print("Epoch", epoch, "MSE = ", mse.eval())
        sess.run(training_op)
    
    best_theta = theta.eval()
    print("Best theta", best_theta)