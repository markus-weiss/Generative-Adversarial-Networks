#### Weiteres in Python ####

# List Comprehension

## List mit n zahlen 
my_list = [1, 2, 3, 4, 5]

##
for i in range(100):
    my_list.append(i)
#print(my_list)

## Short way
my_list_comp = [i for i in range(100)]
## Das quadrat der zahl
my_list_comp = [i**4 for i in range(10)]
## ich speicher nur wenn die zahl druch 2 teilbar ist
my_list_comp = [i**2 for i in range(10) if i % 2 == 0]
my_list_comp2 = [i for i in range(10) if i % 2 == 0]

# print(my_list_comp)
# print(my_list_comp2)

import numpy as np
# Weiteres zu Numpy
# zweidimensionale Matix
m = np.array([1,0,0,1])

m = np.array([[1,0],[0,1]])
m = np.array([[1,0,1,0],[1,0,0,1],[1,1,1,1],[0,0,0,0]])
m = np.array([1,0,0,1])
print(m)
# Dimensionen drehen
m = np.reshape(m, (4,1))
print(m)
# Gibt die Dimensionen an 
# print(m.shape)

# m.np.reshape(m (2,2))
# print(m.shape)
# print(m)


