
# Gradient descent: generic optimization algorithm
# capable of finding optimal solutions to a wide range
# of problems

# Start with random value
# converge to a minimum

# Parameters
# learning rate

# MSE is a convex function -> no local minimum

# How to compute Gradient Descent?
# compute the gradient of the cost function
#   with regards to each model parameter theta_j.
#   -> partial derivative

# partial derivative of MSE(theta) respoct to theta_j
# this gives the gradient vectore
#   contains all the partial derivatives of
#   the cost function (one for each modela parameter)

# Now we need the Gradient Descent step
#   N.B. Gradient vector points uphill

# theta_(next step) = theta - lr * GradVector
#      lr: learning rate


# BATCH GRADIENT DESCENT
import numpy as np 

X = 2 * np.random.rand(100, 1)
# Real function use for create data: y = 4 + 3x + Gaussian noise
y = 4 + 3 * X + np.random.randn(100, 1)

X_b = np.c_[np.ones((100, 1)), X] # add x0 = 1 to each instance


eta = 0.1 # learning rate
n_iters = 1000
m = 100     # number of training instances

theta = np.random.randn(2, 1)

for iter in range(n_iters):
    gradients = 2 / m * X_b.T.dot(X_b.dot(theta) - y)
    theta = theta - eta * gradients

print(theta)

# Issue:
# uses the whole training set to compute the gradient
# at every step. Slow if dataset large

# STOCHASTIC GRADIENT DESCENT
# compute gradient based only on a single instance
# chose randomly

# Issue:
# randomness is good to escape from local optima, 
# but bad because it means that the algorithm can 
# never settle at the minimum.  
#   -> gradually reduce the learning rate.
#       -> simulated annealing (!!!)

# Now the problem is to find this function that
# determines the learning rate at each iteration

n_epochs = 50
t0 = 5
t1 = 50

# function for learning rate
def learning_schedule(t):
    return t0 / (t + t1)

theta = np.random.randn(2, 1)

for epoch in range(n_epochs):
    for i in range(m):
        # pick randomly one instance
        random_index = np.random.randint(m)
        xi = X_b[random_index:random_index + 1]
        yi = y[random_index:random_index + 1]

        gradients = 2 * xi.T.dot(xi.dot(theta) - yi)
        
        eta = learning_schedule(epoch * m + i)

        theta = theta - eta * gradients

print("Stochastic Gradient Descent:\n", theta)


# Using scikit-Learn
from sklearn.linear_model import SGDRegressor

sgd_reg = SGDRegressor(
    max_iter=50, penalty=None, eta0=0.1)
sgd_reg.fit(X, y.ravel())

print("With scikit:\n", sgd_reg.intercept_, 
    sgd_reg.coef_)


# MINI-BATCH GRADIENT DESCENT

# computes the gradients on small random sets of 
# instances called mini-batches

