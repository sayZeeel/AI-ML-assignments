import numpy as np
import csv
import math
import pandas as pd 
import cvxpy as cvx

inputs = "Xsvm.csv"
labels = "ysvm.csv"


x = pd.read_csv(inputs,header = None)
y = pd.read_csv(labels,header = None)


x = np.asarray(x)
y = np.asarray(y)

N = len(y)

# create an array of length N
a = cvx.Variable(N)

# print(a.shape, y.shape)

# the terms of the final equation in SVM. They are put together in LD
Q = cvx.matmul(cvx.diag(a),y)
R = cvx.matmul(x.T,Q)
S = cvx.norm(R)**2
T = cvx.matmul(a.T,y)

LD = cvx.sum(a) - 0.5*S

# constraints on LD
constraints = [a>=0 , T == 0 ]

# objective is to maximize LD
obj= cvx.Maximize(LD)
prob = cvx.Problem(obj,constraints)
# Remove verbose = True if you don't want to see the iterations
prob.solve(verbose = True)
a = np.asarray(a.value)
# print( a)

w = [0,0]
for i in range(0,500):
    w = w + a[i]*y[i]*x[i]
    # for the KKT condition
    if(a[i]>1e-3):
        j=i


# KKT condition
w0 = 1/y[j] - np.dot(w,x[j])

# print(w0,w)

x_test = np.asarray([[2,0.5],[0.8,0.7],[1.58,1.33],[0.008,0.001]])
y_test = []

for i in range(0,4):
    if(w[0]*x_test[i][0] + w[1]*x_test[i][1] + w0>=0):
        y_test.append(1)
    else:
        y_test.append(-1)

print(x_test)
print(y_test)