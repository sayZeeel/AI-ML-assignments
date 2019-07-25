import numpy as np
import matplotlib.pyplot as plt
# Number of training samples
N = 10
degree = 10
# alpha
a = 0.02
# Generate equispaced floats in the interval [0, 2pi]
x1 = np.linspace(0, 2*np.pi, N)
# Generate noise
mean = 0
std = 0.05
# lambda = (sigma/alpha)^2
l = np.square(std/a)
# Generate some numbers from the sine function
y = np.sin(x1)
# Add noise
y += np.random.normal(mean, std, N)

# A polynomial basis function of degree ___ 

def phi (M,p):
	L = []
	for i in range(0,p+1):
		L.append(np.power(M,i))

	L= np.asarray(L)
	return np.transpose(L)

phi1 = phi(x1,degree)


# optimum weight vector W. The formula is taken from class notes
w = np.matmul(np.linalg.inv(np.matmul(np.transpose(phi1),phi1)-l*np.identity(degree +1)),np.matmul(np.transpose(phi1),y))


# actual sin(x) without noise
M = 100
x2 = np.linspace(0, 2*np.pi, M)
y2 = np.sin(x2)


plt.plot(x1,y,'ro', label = 'training samples')
plt.plot(x2,y2,'-b',label = 'test samples')

plt.xlabel('x')
plt.ylabel('sin(x)')
plt.show()





