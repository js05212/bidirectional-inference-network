import numpy as np
import os
# import config
import matplotlib.pyplot as plt


root_path = '/home/mcz/Pytorch_Project/ICMLplot/'
subfolder = 'dim2'
filenames = os.listdir(os.path.join(root_path, subfolder))
for each in filenames:
    path_ = os.path.join(root_path, subfolder, each)
    data = np.load(path_).item()

    fig_xy = data['fig_loss']
    V1 = fig_xy[0]
    Loss = fig_xy[1]
    plt.plot(V1, Loss)
    plt.title(each)
    plt.show()




