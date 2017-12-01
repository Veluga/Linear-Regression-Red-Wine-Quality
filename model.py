import numpy as np

import linearRegressionFunctions as functionality

# Importing dataset from CSV file, splitting data into train/test set
filename = 'winequality-red.csv' 
raw_file = open(filename, 'rt')
raw_data = np.loadtxt(raw_file, delimiter=';')
(number_examples, feature)= raw_data.shape

#Shuffle Array to prevent bias in the training/test set
np.random.shuffle(raw_data)

data_train = raw_data[0:1119, 0:11]
data_test = raw_data[1119:number_examples, 0:11]

Y_train = raw_data[0:1119, 11:12]
Y_test = raw_data[1119:number_examples, 11:12]

#setting hyperparameters
alpha = 0.1
max_iterations = 1000

#perform feature scaling
data_train = functionality.feature_scaling(data_train)
data_test = functionality.feature_scaling(data_test)

#Extend both data_train/data_test with X0 = 1, respectively
x0 = np.ones((data_train.shape[0], 1))
data_train = np.append(x0, data_train, axis=1)
x0 = np.ones((data_test.shape[0], 1))
data_test = np.append(x0, data_test, axis=1)

theta = np.zeros((data_train.shape[1], 1))

#testing only
log = True
J, final_theta = functionality.perform_gradient_descent(theta, data_train, Y_train, alpha, max_iterations, log)

functionality.predict(final_theta, data_test, Y_test)