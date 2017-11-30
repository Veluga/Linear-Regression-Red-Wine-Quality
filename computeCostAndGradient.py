import numpy as np

def compute_cost_and_gradient(theta, X, Y):
    m = X.shape[0]
    J = 1/(2*m)*np.sum(np.power(np.dot(X, theta)-Y, 2))
    print(J)
