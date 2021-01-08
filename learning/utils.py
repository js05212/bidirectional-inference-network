from config import *
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import zipfile, tempfile, glob
import shutil
import numpy as np



def set_dropout_mode(models, train):
    """ set the mode of all dropout layers """
    for name, model in models._modules.items():
        if model is None:
            continue
        if model.__class__.__name__.find('Dropout') != -1:
            if train:
                model.train()
            else:
                model.eval()
        set_dropout_mode(model, train)


def full_mse_loss(pred, label):
    loss = F.mse_loss(pred, label, size_average=False)
    loss = loss / pred.size(1) / pred.size(0)
    return loss

def find_target_y_index(pid, pid_dic):
    # search for the position index pid in pid_dic, and return the position

    ###check ?
    position=[]
    for each in pid:
        flag=0
        for i, dic in enumerate(pid_dic):
            if dic == each:
                flag=1
                position.append(i)
                break
        assert flag==1
    return position
