import numpy as np
import pandas as pd

a = np.array([12, 1231, 41, 2, 3])
b = np.array([1322, 7, 441, 52, 73])
print(a)
## (ARANAGE) creates range of elements
print(np.arange(6))
##specifying first ,last numbers and step size
print(np.arange(1, 39, 2))

# creates an array that is from 1 to 50 that has 5 size.
print(np.linspace(1, 50, 5))

##(SORT) sorts the array.
print(np.sort(a))

##(ADD) adding arrays to each other

print(np.concatenate((a, b)))


array_example = np.array([[[0, 1, 2, 3],
                           [4, 5, 6, 7]],

                          [[0, 1, 2, 3],
                           [4, 5, 6, 7]],

                          [[0 ,1 ,2, 3],
                           [4, 5, 6, 7]]])

##NUMBER OF DIMENSION
print(array_example.ndim)
##SIZE
print(array_example.size)
##SHAPE
print(array_example.shape)

dates=pd.date
