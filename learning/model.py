from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import *
from learning.utils import *

class Enc(nn.Module):
    def __init__(self):
        super(Enc, self).__init__()
        width = 64
        self.fc1 = nn.Linear(1, width)
        self.fc2 = nn.Linear(width, width)
        # self.fc3 = nn.Linear(width, width)
        # self.fc4 = nn.Linear(width, width)
        self.fc5 = nn.Linear(width, 1)

    def forward(self, x):
        # print('pre input x', x)
        input_x = x[:, 0].unsqueeze(1)  # v1
        label_y = x[:, 1].unsqueeze(1)  # v2

        x = F.relu(self.fc1(input_x))
        x = F.relu(self.fc2(x))
        # x = F.relu(self.fc3(x))
        # x = F.relu(self.fc4(x))
        x = self.fc5(x)
        loss = full_mse_loss(x, label_y) # v2

        return x, loss


class Enc_inf(nn.Module):
    def __init__(self):
        super(Enc_inf, self).__init__()
        width = 64
        self.fc1 = nn.Linear(1, width)
        self.fc2 = nn.Linear(width, width)
        # self.fc3 = nn.Linear(width, width)
        # self.fc4 = nn.Linear(width, width)
        self.fc5 = nn.Linear(width, 1)

    def forward(self, x):
        x, target = x

        label_y = x[:, 1].unsqueeze(1) # v2
        x = F.relu(self.fc1(target)) # U1
        x = F.relu(self.fc2(x))
        # x = F.relu(self.fc3(x))
        # x = F.relu(self.fc4(x))
        x = self.fc5(x) # v2
        loss = full_mse_loss(x, label_y)
        return x, loss

class Enc_3(nn.Module):
    def __init__(self, args):
        super(Enc_3, self).__init__()
        width = 64
        self.fc1_1 = nn.Linear(1, width)
        self.fc3_1 = nn.Linear(width, width)
        # self.fc4 = nn.Linear(width, width)
        self.fc5_1 = nn.Linear(width, 1)

        self.fc1_2 = nn.Linear(2, width)
        self.fc3_2 = nn.Linear(width, width)
        # self.fc4 = nn.Linear(width, width)
        self.fc5_2 = nn.Linear(width, 1)
        self.lambda_a = args.lambda_a

    def forward(self, x):
        # print('pre input x', x)
        BB = x[:, 0].unsqueeze(1)
        V1 = x[:, 1].unsqueeze(1)
        V2 = x[:, 2].unsqueeze(1)

        v1_r = F.relu(self.fc1_1(BB))
        v1_r = F.relu(self.fc3_1(v1_r))
        v1_r = self.fc5_1(v1_r)
        loss1 = full_mse_loss(v1_r, V1)

        v2_r = torch.cat((V1, BB), 1)
        v2_r = F.relu(self.fc1_2(v2_r))
        v2_r = F.relu(self.fc3_2(v2_r))
        v2_r = self.fc5_2(v2_r)
        loss2 = full_mse_loss(v2_r, V2) #Caution, for MSE(x, y) loss in pytorch, it only compute the gradient for x

        return v1_r, v2_r, self.lambda_a * loss1 + loss2

class Enc_inf_3(nn.Module):
    def __init__(self, args):
        super(Enc_inf_3, self).__init__()
        width = 64
        self.fc1_1 = nn.Linear(1, width)
        self.fc3_1 = nn.Linear(width, width)
        # self.fc4 = nn.Linear(width, width)
        self.fc5_1 = nn.Linear(width, 1)

        self.fc1_2 = nn.Linear(2, width)
        self.fc3_2 = nn.Linear(width, width)
        # self.fc4 = nn.Linear(width, width)
        self.fc5_2 = nn.Linear(width, 1)
        self.lambda_a = args.lambda_a

    def forward(self, x):
        x, target = x
        BB = x[:, 0].unsqueeze(1)
        V1 = x[:, 1].unsqueeze(1)
        V2 = x[:, 2].unsqueeze(1)

        v1_r = F.relu(self.fc1_1(BB))
        v1_r = F.relu(self.fc3_1(v1_r))
        v1_r = self.fc5_1(v1_r)
        v1_r = v1_r.detach()
        loss1 = full_mse_loss(target, v1_r)

        v2_r = torch.cat((target, BB), 1)
        v2_r = F.relu(self.fc1_2(v2_r))
        v2_r = F.relu(self.fc3_2(v2_r))
        # x = F.relu(self.fc2(x))
        # x = F.relu(self.fc3(x))
        # x = F.relu(self.fc4(x))
        v2_r = self.fc5_2(v2_r)
        loss2 = full_mse_loss(v2_r, V2)
        return v1_r, self.lambda_a * loss1 + loss2
        # remain discussion, the lambda_a


# class Enc(nn.Module):
#     def __init__(self):
#         super(Enc, self).__init__()
#         width = 512
#         self.fc1 = nn.Linear(1, width)
#         self.fc2 = nn.Linear(width, width)
#         self.fc3 = nn.Linear(width, width)
#         # self.fc4 = nn.Linear(width, width)
#         self.fc5 = nn.Linear(width, 1)
#
#     def forward(self, x):
#         # print('pre input x', x)
#         input_x = x[:, 0].unsqueeze(1)
#         label_y = x[:, 1].unsqueeze(1)
#
#         x = F.relu(self.fc1(input_x))
#         x = F.relu(self.fc2(x))
#         x = F.relu(self.fc3(x))
#         # x = F.relu(self.fc4(x))
#         x = self.fc5(x)
#         loss = full_mse_loss(x, label_y)
#
#         return x, loss


# class Enc_inf(nn.Module):
#     def __init__(self):
#         super(Enc_inf, self).__init__()
#         width = 512
#         self.fc1 = nn.Linear(1, width)
#         self.fc2 = nn.Linear(width, width)
#         self.fc3 = nn.Linear(width, width)
#         # self.fc4 = nn.Linear(width, width)
#         self.fc5 = nn.Linear(width, 1)
#
#     def forward(self, x):
#         x, target = x
#
#         input_x = x[:, 0].unsqueeze(1)
#         x = F.relu(self.fc1(input_x))
#         x = F.relu(self.fc2(x))
#         x = F.relu(self.fc3(x))
#         # x = F.relu(self.fc4(x))
#         x = self.fc5(x)
#         x = x.detach()
#         loss = full_mse_loss(x, target)
#         return x, loss


class Enc_C(nn.Module):
    def __init__(self):
        super(Enc_C, self).__init__()
        width = 64
        self.fc1 = nn.Linear(1, width)
        self.fc2 = nn.Linear(width, width)
        # self.fc3 = nn.Linear(width, width)
        # self.fc4 = nn.Linear(width, width)
        self.fc5 = nn.Linear(width, 1)
        self.loss = nn.BCELoss()

    def forward(self, x):
        # print('pre input x', x)
        input_x = x[:, 0].unsqueeze(1)  # v1
        label_y = x[:, 1].unsqueeze(1)  # v2

        x = F.relu(self.fc1(input_x))
        x = F.relu(self.fc2(x))
        # x = F.relu(self.fc3(x))
        # x = F.relu(self.fc4(x))
        x = F.sigmoid(self.fc5(x))
        loss = self.loss(x, label_y.detach()) # v2

        return x, loss


class Enc_inf_C(nn.Module):
    def __init__(self):
        super(Enc_inf_C, self).__init__()
        width = 64
        self.fc1 = nn.Linear(1, width)
        self.fc2 = nn.Linear(width, width)
        # self.fc3 = nn.Linear(width, width)
        # self.fc4 = nn.Linear(width, width)
        self.fc5 = nn.Linear(width, 1)
        self.loss = nn.BCELoss()

    def forward(self, x):
        x, target = x

        label_y = x[:, 1].unsqueeze(1) # v2
        x = F.relu(self.fc1(target)) # U1
        x = F.relu(self.fc2(x))
        # x = F.relu(self.fc3(x))
        # x = F.relu(self.fc4(x))
        x = F.sigmoid(self.fc5(x)) # v2
        loss = self.loss(x, label_y)
        return x, loss


class Enc_3_C(nn.Module):
    def __init__(self, args):
        super(Enc_3_C, self).__init__()
        width = 64
        self.fc1_1 = nn.Linear(1, width)
        self.fc3_1 = nn.Linear(width, width)
        # self.fc4 = nn.Linear(width, width)
        self.fc5_1 = nn.Linear(width, 1)

        self.fc1_2 = nn.Linear(2, width)
        self.fc3_2 = nn.Linear(width, width)
        # self.fc4 = nn.Linear(width, width)
        self.fc5_2 = nn.Linear(width, 1)
        self.lambda_a = args.lambda_a
        self.loss1 = nn.BCELoss()
        self.loss2 = nn.BCELoss()

    def forward(self, x):
        # print('pre input x', x)
        BB = x[:, 0].unsqueeze(1)
        V1 = x[:, 1].unsqueeze(1)
        V2 = x[:, 2].unsqueeze(1)

        v1_r = F.relu(self.fc1_1(BB))
        v1_r = F.relu(self.fc3_1(v1_r))
        v1_r = F.sigmoid(self.fc5_1(v1_r))
        # V1 = V1.clamp(0, 1)  ###
        print('v1_r', v1_r)
        print('V1', V1)
        loss1 = self.loss1(v1_r, V1.detach())
        # loss1 = full_mse_loss(v1_r, V1) #loss2 = full_mse_loss(v2_r, V2)

        v2_r = torch.cat((V1, BB), 1)
        v2_r = F.relu(self.fc1_2(v2_r))
        v2_r = F.relu(self.fc3_2(v2_r))
        v2_r = F.sigmoid(self.fc5_2(v2_r))
        loss2 = self.loss2(v2_r, V2.detach()) #Caution, for MSE(x, y) loss in pytorch, it only compute the gradient for x
        # loss2 = full_mse_loss(v2_r, V2)
        print('loss 1', loss1.data[0], 'loss2', loss2.data[0])

        return v1_r, v2_r, self.lambda_a * loss1 + loss2

class Enc_inf_3_C(nn.Module):

    def __init__(self, args):
        super(Enc_inf_3_C, self).__init__()
        width = 64
        self.fc1_1 = nn.Linear(1, width)
        self.fc3_1 = nn.Linear(width, width)
        # self.fc4 = nn.Linear(width, width)
        self.fc5_1 = nn.Linear(width, 1)

        self.fc1_2 = nn.Linear(2, width)
        self.fc3_2 = nn.Linear(width, width)
        # self.fc4 = nn.Linear(width, width)
        self.fc5_2 = nn.Linear(width, 1)
        self.lambda_a = args.lambda_a
        self.loss1 = nn.BCELoss()
        self.loss2 = nn.BCELoss()

    def forward(self, x):
        x, target = x

        # target = target.clamp(1e-5, 1 - 1e-5)


        # print('after clip', target.data)
        # temp = target != target
        # print('is nan', torch.transpose(temp.data, 0, 1))
        # print('target', torch.transpose(target.data, 0, 1))
        # target[target != target] = 0


        BB = x[:, 0].unsqueeze(1)
        V1 = x[:, 1].unsqueeze(1)
        V2 = x[:, 2].unsqueeze(1)

        v1_r = F.relu(self.fc1_1(BB))
        v1_r = F.relu(self.fc3_1(v1_r))
        v1_r = F.sigmoid(self.fc5_1(v1_r))
        v1_r = v1_r.detach()
        # print('target', target)
        # print('v1_r', v1_r)
        # loss1 = self.loss1(v1_r, target)
        loss1 = torch.sum(- target * torch.log(v1_r + 1e-5) - (1 - target) * torch.log(1 - v1_r + 1e-5)) / v1_r.size()[0]
        # log zero results in inf, need the 1e-5
        # loss1 = full_mse_loss(target, v1_r)


        v2_r = torch.cat((target, BB), 1)
        v2_r = F.relu(self.fc1_2(v2_r))
        v2_r = F.relu(self.fc3_2(v2_r))
        # x = F.relu(self.fc2(x))
        # x = F.relu(self.fc3(x))
        # x = F.relu(self.fc4(x))
        # print('before sigmoid', v2_r)
        v2_r = self.fc5_2(v2_r)
        v2_r = F.sigmoid(v2_r)
        # print('trigger, v2_r', v2_r)
        # print('V2', V2)
        loss2 = self.loss2(v2_r, V2)
        # loss2 = full_mse_loss(v2_r, V2)
        # print('loss 1', loss1.data[0], 'loss2', loss2.data[0])

        return v1_r, self.lambda_a * loss1 + loss2
        # remain discussion, the lambda_a
