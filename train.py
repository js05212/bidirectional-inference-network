import numpy as np
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torch.utils import data
import random
import config
import utils
import os, time
from learning.datasets import SimulateDataset2D, SimulateDataset3D
from learning.model import Enc, Enc_inf, Enc_3, Enc_inf_3, Enc_C, Enc_inf_C, Enc_3_C, Enc_inf_3_C
from learning.loops import *
from learning.loops_3d import train_loop_infer_3d
from learning.loops_classify import train_loop_infer_class
from learning.loops_3d_classify import train_loop_infer_3d_C
import argparse

parser = argparse.ArgumentParser(description='Simulation Script')
parser.add_argument('--lr', default=5e-3, type=float, help='learning rate')
parser.add_argument('--lr_infer', default=1e-1, type=float, help='infer learning rate')
parser.add_argument('--train_epoch', default=100, type=int, help='train times')
parser.add_argument('--train_num', default=100, type=int, help='')
parser.add_argument('--num_loop', default=10, type=int, help='infer iteration times')
parser.add_argument('--test_infer_num_loop', default=3, type=int, help='infer iteration times during test')
parser.add_argument('--lambda_c', default=0, type=float, help='ratio for estimation loss')
parser.add_argument('--lambda_a', default=1, type=float, help='ratio for estimation loss')
parser.add_argument('--infer_start', default=3, type=int, help='when to start infer')
parser.add_argument('--optim', default='adam', type=str, help='infer optimization method')
parser.add_argument('--plot', default=False, type=bool, help='plot or not')
parser.add_argument('--datafile', default='x1noise1_10.npy', type=str, help='plot or not')
parser.add_argument('--seed', default=12345, type=int, help='manual seed')
#parser.add_argument('--dimension', default=3, type=int, help='dimension of toy dataset')
parser.add_argument('--dimension', default=2, type=int, help='dimension of toy dataset')
#parser.add_argument('--classify', default=True, type=bool, help='Classifier/ Regression')
parser.add_argument('--classify', default=False, type=bool, help='Classifier/ Regression')
parser.add_argument('--output', default=time.strftime('%m-%d-%H-%M'),
                    type=str, help='folder to output model checkpoints')

parser.set_defaults(augment=True)
args = parser.parse_args()

if args.lambda_c == 0:
    args.num_loop = 1

np.random.seed(args.seed)
random.seed(args.seed)
torch.manual_seed(int(args.seed))

savepath = config.savepath
if args.dimension == 3:
    args.output = os.path.join(savepath, args.output + '_infer_{}_dim_{}_a{}_c{}'.format(args.num_loop, args.dimension,
                                                                                        args.lambda_a, args.lambda_c
                                                                                        ))
elif args.dimension ==2:
    args.output = os.path.join(savepath, args.output + '_infer_{}_dim_{}_c{}'.format(args.num_loop, args.dimension,
                                                                                         args.lambda_c
                                                                                         ))
    # there is no a when 2 dimension

if args.classify:
    args.output = args.output + 'cl'
#args.output = os.path.join(config.LOCAL_PATH, args.output) # local
if os.path.exists(args.output):
    if utils.query_yes_no('overwrite previous folder of %s?' % args.output):
        shutil.rmtree(args.output)
        print(args.output + ' removed.\n')
    else:
        raise RuntimeError('Output folder {} already exists'.format(args.output))

os.makedirs(args.output, mode=0o770)
# utils.copy_key_src('.', os.path.join(args.output, 'src'))
#shutil.copytree('.', os.path.join(args.output, 'src'))

if args.dimension==2:
    train_dataset = SimulateDataset2D(
        path = os.path.join(config.X1noise if not args.classify else config.X1noise_classify, args.datafile)
    )

    infer_dataset_tr = SimulateDataset2D(
        path = os.path.join(config.X1noise if not args.classify else config.X1noise_classify, args.datafile)
    )
elif args.dimension==3:
    train_dataset = SimulateDataset3D(
        path=os.path.join(config.X1noise_3d if not args.classify else config.X1noise_3d_classify, args.datafile)
    )

    infer_dataset_tr = SimulateDataset3D(
        path=os.path.join(config.X1noise_3d if not args.classify else config.X1noise_3d_classify, args.datafile)
    )

train_dataloader = data.DataLoader(
    dataset=train_dataset,
    batch_size=1,
    shuffle=True,
    num_workers=1,
    pin_memory=False
)

infer_dataloader_tr = data.DataLoader(
    dataset=train_dataset,
    batch_size=len(infer_dataset_tr),  ###?
    shuffle=False,
    num_workers=1,
    pin_memory=False
)
if not args.classify:
    if args.dimension == 2:
        predictor = Enc().cuda()
        predictor_c = Enc_inf().cuda()
    elif args.dimension == 3:
        predictor = Enc_3(args).cuda()
        predictor_c = Enc_inf_3(args).cuda()
elif args.classify:
    if args.dimension == 2:
        predictor = Enc_C().cuda()
        predictor_c = Enc_inf_C().cuda()
    elif args.dimension == 3:
        predictor = Enc_3_C(args).cuda()
        predictor_c = Enc_inf_3_C(args).cuda()

model_names = ["predictor", "predictor_c"]
models =[predictor, predictor_c]

if args.optim == 'adam':
    opt = optim.Adam(predictor.parameters(), lr=args.lr)
elif args.optim == 'SGD':
    opt = optim.SGD(predictor.parameters(), lr=args.lr, momentum=0.9)
elif args.optim == 'adadelta':
    opt = optim.Adadelta(predictor.parameters(), lr=args.lr, eps=1e-7)

lr_scheduler = [lr_scheduler.ExponentialLR(optimizer=opt, gamma=0.5 ** (1/500))]


target = []
all_total=[]
all_front=[]
all_Estimate=[]
all_test=[]
all_U1=[]
for epoch in range(args.train_epoch):
    if not args.classify:
        if args.dimension == 2:
            target, loss_total, loss_front, loss_Estimate, test_loss = train_loop_infer(models, train_dataloader,
                                                                                        infer_dataloader_tr, opt,
                                                                                        lr_scheduler, epoch + 1, args,
                                                                                        target=target)
            all_total.append(loss_total)
            all_front.append(loss_front)
            all_Estimate.append(loss_Estimate)
            all_test.append(test_loss)

        elif args.dimension == 3:
            target, loss_total, loss_front, loss_Estimate, test_loss = train_loop_infer_3d(models, train_dataloader,
                                                                                        infer_dataloader_tr, opt,
                                                                                        lr_scheduler, epoch + 1, args,
                                                                                        target=target)
            all_total.append(loss_total)
            all_front.append(loss_front)
            all_Estimate.append(loss_Estimate)
            all_test.append(test_loss)
    else:
        if args.dimension == 2:
            target, loss_total, loss_front, loss_Estimate, test_loss = train_loop_infer_class(models, train_dataloader,
                                                                                        infer_dataloader_tr, opt,
                                                                                        lr_scheduler, epoch + 1, args, all_U1,
                                                                                        temp=target)
            all_total.append(loss_total)
            all_front.append(loss_front)
            all_Estimate.append(loss_Estimate)
            all_test.append(test_loss)

        elif args.dimension == 3:
            target, loss_total, loss_front, loss_Estimate, test_loss = train_loop_infer_3d_C(models, train_dataloader,
                                                                                           infer_dataloader_tr, opt,
                                                                                           lr_scheduler, epoch + 1,
                                                                                           args, all_U1,
                                                                                           temp=target)
            all_total.append(loss_total)
            all_front.append(loss_front)
            all_Estimate.append(loss_Estimate)
            all_test.append(test_loss)


np.save(os.path.join(args.output, 'train_loss_{}_{}.npy'.format(args.lambda_c, args.train_epoch)), all_total)
np.save(os.path.join(args.output, 'test_loss_{}_{}.npy'.format(args.lambda_c, args.train_epoch)), all_test)

utils.save_model(model_names, models, args.output, args.train_epoch, loss_total, 'last_model')

import matplotlib.pyplot as plt
plt.plot(all_total,'b')
plt.plot(all_test,'r')
plt.title('lam_c_{}infer_{}_'.format(args.lambda_c, args.num_loop))
plt.show()
plt.savefig(os.path.join(args.output, 'loss_curve.png'))
