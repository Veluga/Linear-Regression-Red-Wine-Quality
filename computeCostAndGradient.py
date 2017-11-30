import numpy as np

def compute_cost_and_gradient(theta, X, Y):
    # m = number of training examples
    m = X.shape[0]

    #compute cost
    J = np.sum(np.power(np.dot(X, theta)-Y, 2))/(2*m)
    
    #compute gradient
    