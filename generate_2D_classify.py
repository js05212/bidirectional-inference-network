import numpy as np
import random
import matplotlib.pyplot as plt
import os

total_point = 1000
x_range=[-1, 1]
select_num=30
noise = np.random.normal(0, x_range[1]*3, select_num)

datasave_path = '/home/mcz/LocalData/x1noise_classify/'
os.makedirs(datasave_path, exist_ok=True)


list_index = [i for i in range(total_point)]
select = sorted(random.sample(list_index, select_num))

x = np.asarray([x for x in range(total_point)], dtype=np.float32)
x = x / total_point * (x_range[1] - x_range[0]) + x_range[0]

x = x[select]
y = 3 * x + 1
y_noise = y + noise
y_class = np.asarray([0 if each < 1 else 1 for each in y_noise], dtype=np.float32)
y_class_real = np.asarray([0 if each < 1 else 1 for each in y], dtype=np.float32)

print(x)
print(y_class)
print(y_class_real)
# plt.plot(x, y, 'r')
# plt.plot(x, y_noise, 'b')
# plt.show()

data = [[ii, xx, yy, zz] for ii, xx, yy, zz in zip(range(select_num), x, y_class, y_class_real)]
data = np.asarray(data)
np.save(os.path.join(datasave_path, 'x1noise{}_{}.npy'.format(x_range[1], select_num)), data)
print('data shape', data.shape)
