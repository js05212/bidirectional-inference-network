import numpy as np
import random
import matplotlib.pyplot as plt
import os

total_point = 1000
x_range=[-1, 1]
select_num=20
#select_num=10
noise = np.random.normal(0, x_range[1], select_num)

#datasave_path = '/home/mcz/LocalData/x1noise/'
datasave_path = 'data'
os.makedirs(datasave_path, exist_ok=True)


list_index = [i for i in range(total_point)]
select = sorted(random.sample(list_index, select_num))

x = np.asarray([x for x in range(total_point)], dtype=np.float32)
x = x / total_point * (x_range[1] - x_range[0]) + x_range[0]

x = x[select]
y = 3 * x + 1
y_noise = y + noise

plt.plot(x, y, 'r')
plt.plot(x, y_noise, 'b')
plt.show()

data = [[ii, xx, yy, zz] for ii, xx, yy, zz in zip(range(select_num), x, y_noise, y)]
data = np.asarray(data)
#np.save(os.path.join(datasave_path, 'x1noise{}_{}-.npy'.format(x_range[1], select_num)), data)
np.save(os.path.join(datasave_path, 'x1noise{}_{}.npy'.format(x_range[1], select_num)), data)
print('data shape', data.shape)
