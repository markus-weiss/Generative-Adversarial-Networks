### Numpy in Python ### 

import numpy as np

noten = [100, 89, 44, 78, 45, 22, 15]

###

noten_np = np.array(noten, dtype=np.int8)
print(noten_np)

listen_arg_min = np.argmin(noten_np) 
listen_arg_max = np.argmax(noten_np)

print(listen_arg_min)
print(listen_arg_max)

listen_min = np.min(noten_np) 
listen_max = np.max(noten_np)

print(listen_min)
print(listen_max)

listen_mean = np.mean(noten_np)
listen_median = np.median(noten_np)

print(listen_mean)
print(listen_median)





# print(listen_arg_max,listen_arg_max)

# listen_min = 0
# listen_max = 0
