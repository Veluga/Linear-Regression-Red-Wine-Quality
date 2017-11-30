import numpy as np

import computeCostAndGradient as costGradient

# Importing dataset from CSV file, splitting data into train/test set
filename = 'winequality-red.csv' 
raw_file = open(filename, 'rt')
raw_data = np.loadtxt(raw_file, delimiter=';')
(number_examples, feature)= raw_data.shape
data_train = raw_data[0:1119, 0:11]
data_test = raw_data[1119:number_examples, 0:11]

Y_train = raw_data[0:1119, 11:12]
Y_test = raw_data[1119:number_examples, 11:12]

#Extend both data_train/data_test with X0 = 1, respectively
x0 = np.ones((data_train.shape[0], 1))
data_train = np.append(x0, data_train, axis=1)
data_test = np.append(x0, data_train, axis=1)

theta = np.ones((data_train.shape[1], 1))

costGradient.compute_cost_and_gradient(theta, data_train, Y_train)

