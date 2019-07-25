import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import csv
import math

# k nearest neighbours
k = 996
inputs = "X.csv"
labels = "Y.csv"

x = pd.read_csv(inputs,header = None)
y = pd.read_csv(labels,header = None)

# transpose array x
x = np.asarray(x).T
y = np.asarray(y)

# test
x_test = np.array([[1,1],[1,-1],[-1,1],[-1,-1]]);
y_test =[]

# the outer loop is running 4 times since there are 4 test inputs
for j in range(0,4):
	# stores Euclidean distances
	distances = []
	# corresponding index
	indices = []
	# calculate euclidean distance of j^th test input from each point x
	for i in range(0,1000):
		distances.append(math.sqrt(np.square(x[i][0]-x_test[j][0]) + np.square(x[i][1]-x_test[j][1])))
		indices.append(i)

	# bubble sort
	for m in range(0,1000):
		for n in range(0,999-m):
			# if element n is greater than element n+1, swap them
			# since we require the corresponding indices, we are swapping those too
			if distances[n]>distances[n+1]:
				distances[n], distances[n+1] = distances[n+1], distances[n]
				indices[n], indices[n+1] = indices[n+1], indices[n]
				
	# calculate which class is occuring more often in those k nearest neighbours
	y_hat = 0
	for i in range(0,k):
		y_hat = y_hat + y[indices[i]]
	y_hat = y_hat/k
	# store the class value that occurs most
	if y_hat > 0:
		y_test.append(1)
	else:
		y_test.append(-1)

print(y_test)