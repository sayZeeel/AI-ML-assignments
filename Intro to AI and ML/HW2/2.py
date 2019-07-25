import numpy as np 
import math


print("No of hidden layer nodes: ")
M = int(input())
print("No of training samples per input pair: ")
N = int(input())
print("No of epochs: ")
epochs = int(input())
print("Learning rate: ")
gamma = input()
print("Enter the option corresponding to the operation to be trained")
print("1. XOR")
print("2. AND")
print("3. OR")
option = int(input())
# standard deviation
std = 0.001

Input = np.asarray([[0,0],[0,1],[1,0],[1,1]])
if(option == 1): # XOR
	Output = np.asarray([[0],[1],[1],[0]])
if(option == 2): # AND
	Output = np.asarray([[0],[0],[0],[1]])
if(option == 3): # OR
	Output = np.asarray([[0],[1],[1],[1]])

X= []
Y= []

# Add noise for training examples
for i in range(0,4):
	for j in range(0,N):
		X.append(Input[i] + np.random.normal(0,std,(1,2)))
		Y.append(Output[i] + np.random.normal(0,std))

X = np.asarray(X)
Y = np.asarray(Y)

# Weights for input layer
W1 = np.random.normal(0,1,(2,M))
# Weights for hidden layer
W2 = np.random.normal(0,1,(M,1))
# Bias for input layer
W0_1 = np.random.normal(0,1,(M,1))
# Bias for hidden layer
W0_2 = np.random.normal(0,1,(1,1))

# sigmoid function
def sigmoid(x):
	return 1/(1+ np.exp(-x))

# differentiating sigmoid function
def ddx_sigmoid(x):
	return sigmoid(x) - sigmoid(x)**2

def Loss(x,X):
	return (x-X)**2

for i in range(0,epochs):
	# after differentiation
	ddx_w1 = np.zeros((2,M))
	ddx_w2 = np.zeros((M,1))
	ddx_w0_1 = np.zeros((M,1))
	ddx_w0_2 = np.zeros((1,1))

	for j in range(0,len(Y)):
		y1 = np.matmul(W1.T,X[j].T) + W0_1
		z = sigmoid(y1)
		y2 = np.matmul(W2.T,z) + W0_2
		y = sigmoid(y2)
		
		L = Loss(y,Y[j])
		

		ddx_w0_2 += 2*(y - Y[j])*ddx_sigmoid(y2)
		ddx_w2 += 2*(y - Y[j])*ddx_sigmoid(y2)*z

		for k in range(0,M):
			ddx_w0_1[k] += (2*(y - Y[j])*ddx_sigmoid(y2)*W2[k]*ddx_sigmoid(y1[k])).reshape(1,)
			ddx_w1[:,k] += (2*(y - Y[j])*ddx_sigmoid(y2)*W2[k]*ddx_sigmoid(y1[k])*X[j]).reshape(2,)

	print("loss: ",L)
	W1 -= gamma*ddx_w1
	W2 -= gamma*ddx_w2
	W0_1 -= gamma*ddx_w0_1
	W0_2 -= gamma*ddx_w0_2

print (W1)
print (W2)

op = 1

while(op!=0):
	new_in = []
	print("Enter input operand 1: ")
	new_in.append(int(input()))
	print("Enter input operand 2: ")
	new_in.append(int(input()))
	new_in = (np.asarray(new_in)).reshape((1,2))
	print('input: ' ,new_in)

	y1 = np.matmul(W1.T,new_in.T) + W0_1

	z = sigmoid(y1)
	y2 = np.matmul(W2.T,z) + W0_2

	out = sigmoid(y2)


	print(out)
	if(out>=0.5):
		print('output: ',1)
	else:
		print('output: ',0)

	print('enter 1 to continue and 0 to exit: ')
	op = int(input())
