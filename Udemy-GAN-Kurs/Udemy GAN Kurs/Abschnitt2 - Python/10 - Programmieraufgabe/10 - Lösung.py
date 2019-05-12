import numpy as np
import matplotlib.pyplot as plt

# Aufgabe 1
freundebuch = {"Peter": 21, "Jan": 24, "Dieter": 44, "Dennis": 27, "Daniel": 33}

def get_friends_by_age(freundebuch, age):
    friends = []
    for friends_name, friends_age in freundebuch.items():
        if friends_age > age:
            friends.append(friends_name)
    return friends

print(get_friends_by_age(freundebuch, 22))

# Aufgabe 2
M = [[1, 2], [3, 4]]

def transpose(M):
    M_t = [[0, 0], [0, 0]]
    for i in range(2):
        for j in range(2):
            M_t[j][i] = M[i][j]
    return M_t

print(transpose(M))

# Aufgabe 3
def e_function(a, b):
    values = []
    for x in range(a, b+1):
        values.append(np.exp(x))
    return values

a, b = 1, 5
exp_vals = e_function(a, b)

plt.plot(range(a, b+1), exp_vals)
plt.show()