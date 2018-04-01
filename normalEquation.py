import matplotlib.pyplot as plt
import numpy as np

X = 2 * np.random.rand(100, 1)
# Real function use for create data: y = 4 + 3x + Gaussian noise
y = 4 + 3 * X + np.random.randn(100, 1)

X_b = np.c_[np.ones((100, 1)), X] # add x0 = 1 to each instance

# X_b.T -> transpose X_b
# package linalg in numpy
theta_best = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)

# theta_best have to be near theta_0 = 4 and theta_1 = 3

# Make predictions
X_new = np.array([[0], [2]])
X_new_b = np.c_[np.ones((2, 1)), X_new]
y_predict = X_new_b.dot(theta_best)

print(y_predict)

# Graphical representation
plt.plot(X_new, y_predict, "r-")
plt.plot(X, y, "b.")
plt.axis([0, 2, 0, 15])
plt.show()


# Using Scikit-Learn
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X, y)
# get elements inside: 
# lin_reg.intercept_, lin_reg.coef_

result = lin_reg.predict(X_new)
print("\nSkikit-learn result: \n", result)


# COMPLEXITY
# Compute normal equation is O(n^3)
# compute a prediction is linear O(m)