import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import csv
import math


inputs = "X.csv"
labels = "Y.csv"

x = pd.read_csv(inputs,header = None)
y = pd.read_csv(labels,header = None)

# transpose array x
x = np.asarray(x).T
y = np.asarray(y)

# elements in column 1 of x corresponding to label 1
x11 = []
# elements in column 2 of x corresponding to label 1
x21 = []
# elements in column 1 of x corresponding to label -1
x1_1 = []
# elements in column 2 of x corresponding to label -1
x2_1 = []
# to store number of ones
y1 = 0
# to store number of negative ones
y_1 = 0

x_test = np.array([[1,1],[1,-1],[-1,1],[-1,-1]]);
y_test =[]

for i in range(0,1000):
	if y[i] == 1:
		x11.append(x[i][0])
		x21.append(x[i][1])
		y1 = y1 + 1
	else:
		x1_1.append(x[i][0])
		x2_1.append(x[i][1])
		y_1 = y_1 + 1


# mean and standard deviation of the respective distributions explained in the previous comments

mu11,std11 = np.mean(x11), np.std(x11)
mu1_1,std1_1 = np.mean(x1_1), np.std(x1_1)
mu21,std21 = np.mean(x21), np.std(x21)
mu2_1,std2_1 = np.mean(x2_1), np.std(x2_1)

# probability(y=1)
p_y1 = y1/1000.0
# probability(y=-1)
p_y_1 = y_1/1000.0

# function to calculate the probability for a normal distribution with mean mu and variance std squared
def prob(mu,std,x):
	return np.exp(-0.5*np.square(x-mu)/np.square(std))/math.sqrt(2*math.pi*np.square(std))

# p(y=k|x) is proportional to p(x|y=k)*p(y=k)
# Naive assumption on p(x|y=k) is elements of x become independent when conditioned on y
for i in range(0,4):
	# p1 and p_1 are probabilities of labels being 1 and -1 respectively
	p1 = (prob(mu11,std11,x_test[i][0])*prob(mu21,std21,x_test[i][1]))*p_y1
	p_1 = (prob(mu1_1,std1_1,x_test[i][0])*prob(mu2_1,std2_1,x_test[i][1]))*p_y_1
	if p1> p_1:
		y_test.append(1)
	else:
		y_test.append(-1)

print (y_test)
