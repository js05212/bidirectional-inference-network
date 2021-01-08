from __future__ import print_function
from torch.utils import data
import numpy as np
import random



class SimulateDataset2D(data.Dataset):
    def __init__(self, path):

        self.data = np.load(path)

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        index = index % self.data.shape[0]
        pid = self.data[index][0]
        x = np.asarray([self.data[index][1], self.data[index][2]], dtype=np.float32)
        ground_truth = np.asarray([self.data[index][3]], dtype=np.float32)

        return pid, x, ground_truth

class SimulateDataset3D(data.Dataset):
    def __init__(self, path):
        self.data = np.load(path)

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        index = index % self.data.shape[0]
        pid = self.data[index][0]
        x = np.asarray([self.data[index][1], self.data[index][2], self.data[index][3]], dtype=np.float32)
        ground_truth = np.asarray([self.data[index][3]], dtype=np.float32)

        return pid, x, ground_truth


# class SimulateDataset2D(data.Dataset):
#     def __init__(self, path, select_index, raw_data_len):
#
#         self.data = np.load(path)
#         assert self.data.shape[0] == raw_data_len
#         self.select_index = select_index
#
#         self.combo_select = self.data[select_index]
#         print('combo shape', self.combo_select.shape)
#
#
#     def __len__(self):
#         return self.combo_select.shape[0]
#
#     def __getitem__(self, index):
#         index = index % len(self.select_index)
#         pid = self.combo_select[index][0]
#         x = np.asarray([self.combo_select[index][1], self.combo_select[index][2]], dtype=np.float32)
#         ground_truth = np.asarray([self.combo_select[index][1], self.combo_select[index][3]], dtype=np.float32)
#
#         return pid, x, ground_truth

#
# class SimulateDataset2D_flexible(data.Dataset):
#     def __init__(self, path, select_num, x_range, ):
#         self.data = np.load(path)
#
#
#     def __len__(self):
#         return self.data.shape[0]
#
#     def __getitem__(self, index):
#         index = index % self.data.shape[0]
#         pid = self.data[index][0]
#         x = np.asarray([self.data[index][1], self.data[index][2]], dtype=np.float32)
#         ground_truth = np.asarray([self.data[index][3]], dtype=np.float32)
#
#         return pid, x, ground_truth
