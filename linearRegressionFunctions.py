import numpy as np

def feature_scaling(X):
    m = X.shape[0]
    means = np.sum(X, 0)/m
    sd = np.sqrt(np.power(np.sum(X, 0),2)/(m-1))
    X = np.divide(np.subtract(X, means), sd)
    X[:, 0] = 1
    return X

def compute_cost_and_gradient(theta, X, Y):
    # m = number of training examples
    m = X.shape[0]

    #compute cost
    J = np.sum(np.power(np.dot(X, theta)-Y, 2))/(2*m)
    
    #compute gradient
    gradients = (np.dot(X.T,np.power(np.dot(X, theta)-Y, 2)))/m
    
    return J, gradients
    
    