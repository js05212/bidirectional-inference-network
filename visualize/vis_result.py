import numpy as np
import matplotlib.pyplot as plt
import os

savepath = '/home/mcz/Pytorch_Project/Log/'

filelist = os.listdir(savepath)

train_list=[]
test_list=[]
for each in filelist:
    if 'train_loss' in each:
        train_list.append(each)
    elif 'test_loss' in each:
        test_list.append(each)

file_1_train = '01-28-10-21train_loss6_0_300.npy'
file_1_test = '01-28-10-21test_loss6_0_300.npy'

file_2_train = '01-28-10-27train_loss6_3_300.npy'
file_2_test = '01-28-10-27train_loss6_3_300.npy'

f1_tr = np.load(os.path.join(savepath, file_1_train))
f1_test = np.load(os.path.join(savepath, file_1_test))

f2_tr = np.load(os.path.join(savepath, file_2_train))
f2_test = np.load(os.path.join(savepath, file_2_test))

print(f1_test)
print(f2_test)

f, ax = plt.subplots(1, 2, sharey=True)
ax[0].plot(f1_tr, 'r')
ax[0].plot(f2_tr, 'g')

ax[1].plot(f1_test, 'r')
ax[1].plot(f2_test, 'g')

plt.show()