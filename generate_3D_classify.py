import numpy as np
import random
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
import config

total_point = 1000
x_range = [-1, 1]
select_num = 10

# by adding larger noise and let network overfit it good, we are able to get many local optimal
noise = np.random.normal(0, x_range[1], select_num)
noise_2 = np.random.normal(0, x_range[1], select_num)

datasave_path = config.X1noise_3d_classify
os.makedirs(datasave_path, exist_ok=True)


list_index = [i for i in range(total_point)]
select = sorted(random.sample(list_index, select_num))

x = np.linspace(x_range[0], x_range[1], total_point)

BB = x[select]
V1 = 3 * BB + 1 + noise
V2 = 0.5 * V1 - BB + 1 + noise_2
data = [[ii, xx, 0 if yy < 1 else 1, 0 if zz < 1.5 else 1] for ii, xx, yy, zz in zip(range(select_num), BB, V1, V2)]
data = np.asarray(data)

print(x.shape, type(x))
BB_g = x
V1_g = 3 * BB_g + 1
V2_g = 0.5 * V1_g - BB_g + 1
print(V1_g.shape)

# fig = plt.figure()
# ax = fig.gca(projection='3d')
# ax.plot(x, V1_g, V2_g, 'g')
# ax.plot(BB, V1, V2, 'r')
# plt.show()

# fig = plt.figure()
# ax = fig.gca(projection='3d')
# ax.plot(BB, V1, V2)
# plt.show()

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x, V1_g, V2_g, c='g')
ax.scatter(data[:, 0], data[:, 1], data[:, 2], c='r')
plt.show()



print(data)
np.save(os.path.join(datasave_path, 'x1noise{}_{}.npy'.format(x_range[1], select_num)), data)
print('data shape', data.shape)
