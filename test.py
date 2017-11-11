import numpy as np
import mnist

a = np.array([[1,8,3],[4,5,6],[2,5,9]])
print(a)
b = a.argmax(axis=0).reshape(1, -1)
c = np.array([[1, 1, 2]])
print(np.sum(b == c))

