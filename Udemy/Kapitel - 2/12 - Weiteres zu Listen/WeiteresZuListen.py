# Weiteres zu Listen 

import matplotlib.pyplot as plt

# x 1 , x 2
x = [[1,4,3,9],[3,1,5,2]]
 
y = [[1,4,3,9],[3,1,5,2]]

# Slicing
# for index in range(4):
#     plt.scatter(x[0][index], x[1][index])
# plt.show()

plt.scatter(x[0][:], y[0][:])
plt.show()

w = [1,3,6,9,7,4]
print(w)
# Nur bis Element 3 ausgeben!""
w_prime = [val for val in w[:3]]
print(w_prime)