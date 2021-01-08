import numpy as np
import random
import matplotlib.pyplot as plt
import os

total_point = 10000
select_num=6
x_range=[0, 1]
noise = np.random.normal(0, 0.05 * x_range[1], select_num)

datasave_path = '/home/mcz/LocalData/x2noise/'

list_index = [i for i in range(total_point)]
select = sorted(random.sample(list_index, select_num))

# print('select', select)
x_label = np.asarray([x * 1.0 / (total_point/x_range[1]) for x in range(total_point)])
x_select = x_label[select]
y_raw_select = (x_select-x_range[1]/2.0) ** 2
y_select = y_raw_select + noise
index = [i for i in range(select_num)]

print(x_select)
plt.plot(x_select, y_select, 'r')
plt.plot(x_select, y_raw_select, 'b')
plt.show()

data = [[i, x, y, z] for i, x, y, z in zip(index, x_select, y_select, y_raw_select)]
data = np.asarray(data)
np.save(os.path.join(datasave_path, 'x2noise{}.npy'.format(x_range[1])), data)
print('data shape', data.shape)

y_raw_select = 1 / (1 + np.exp(-y_raw_select))
y_select = 1 / (1 + np.exp(-y_select))

# plt.plot(x_select, y_select, 'r')
# plt.plot(x_select, y_raw_select, 'b')
# plt.show()

data = [[i, x, y, z] for i, x, y, z in zip(index, x_select, y_select, y_raw_select)]
data = np.asarray(data)
np.save(os.path.join(datasave_path, 'x2noise_sigmoid{}.npy'.format(x_range[1])), data)
print('data shape', data.shape)
