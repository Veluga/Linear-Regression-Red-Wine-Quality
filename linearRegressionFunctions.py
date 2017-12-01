import numpy as np
import time

def feature_scaling(X):
    m = X.shape[0]
    means = np.mean(X, axis=0)
    X = X-means
    sd = np.std(X, axis=0)
    X = X/sd
    return X

def compute_cost_and_gradient(theta, X, Y):
    # m = number of training examples
    m = X.shape[0]
    #compute cost
    J = np.sum(np.power(np.dot(X, theta)-Y, 2))/(2*m)
    
    #compute gradient
    gradients = (np.dot((np.dot(X, theta)-Y).T,X).T)/m
    
    return J, gradients

def perform_gradient_descent(theta, X, Y, alpha, iterations,logging = False):
    for i in range(iterations+1):
        start = time.time()
        J, gradients = compute_cost_and_gradient(theta, X, Y)
        theta = theta - alpha * gradients
        if logging:
            log_progress(i, iterations, start, J)
    return J, theta



def log_progress(current_iteration, iterations, start_time, J):
    stop = time.time()
    print('Estimated time remaining: ' + str((stop-start_time)*1000*(iterations-current_iteration)) + 'ms')
    print('Current Cost: ' + str(J))
    print('Performing gradient descent (step ' + str(current_iteration) + '/' + str(iterations) + ')\n')
    
def predict(theta, X, Y):
    print ('- - - - - - - - - - - - - - - - - - - - - - - - - - - - -')
    m = X.shape[0]
    predictions = np.dot(X,theta)
    deviations = np.abs(predictions-Y)

    tolerances = [0.25,0.5,1]
    accuracies = [0,0,0]

    for x in tolerances:
        for i in range(0,m):
            if deviations[i] <= x:
                accuracies[tolerances.index(x)] += 1
        print('Accuracy (tolerance = ' + str(x) + '): ' + format(float(accuracies[tolerances.index(x)])/m*100, '.2f') + '%')
            
    mad = np.mean(np.abs(predictions-Y))
    print('Mean Average Deviation (MAD) on test set: ' + str(mad) + '\n')