import numpy as np
import matplotlib.pyplot as plt
# Number of training samples
N = 10
# Generate equispaced floats in the interval [0, 2pi]
x1 = np.linspace(0, 2*np.pi, N)
# Generate noise
mean = 0
std = 0.05
# Generate some numbers from the sine function
y = np.sin(x1)
# Add noise
y += np.random.normal(mean, std, N)

# Creating a Nx2 matrix with column 1 full of ones for the bias
x2 = np.array([np.ones(N),x1])
x = np.transpose(x2)

# optimum weight vector W. The formula is taken from class notes
w = np.matmul(np.linalg.inv(np.matmul(np.transpose(x),x)),np.matmul(np.transpose(x),y))

# y_hat = X
y1 = np.matmul(x,np.transpose(w))

plt.plot(x1,y,'ro', label = 'training samples')
plt.plot(x1,y1,'bo',label = 'test samples')

plt.xlabel('x')
plt.ylabel('sin(x)')
plt.show()