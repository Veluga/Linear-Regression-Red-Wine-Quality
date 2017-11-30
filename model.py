import csv
import numpy as np

filename = 'winequality-red.csv' 
raw_file = open(filename, 'rt')
raw_data = np.loadtxt(raw_file, delimiter=';')

data_train = raw_data[0:959, 0:11]
print(data_train[1,:])
Y = raw_data[:, 11]