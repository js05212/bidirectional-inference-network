from __future__ import print_function
import torch.nn
import time, math
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from learning.utils import *
import matplotlib.pyplot as plt
import config


def infer_loop_variable_class_full(models, infer_data_loader, epoch, args, mode='train'):

    predictor, predictor_c = models
    predictor_c.load_state_dict(predictor.state_dict())

    predictor_c.eval()
    set_dropout_mode(predictor_c, False)

    num_loop = args.num_loop

    for iter in range(num_loop):

        if iter == 0:
            # print('init target')
            target_variable = Variable(torch.rand(infer_data_loader.batch_size, 1).cuda(), requires_grad=True)
            # if len(target) == 0:
            #     target_variable = Variable(torch.rand(infer_data_loader.batch_size, 1).cuda(), requires_grad=True)
            #     target = np.random.normal(0, 1, (infer_data_loader.batch_size, 1)) ###############
            # else:
            #     target_variable = Variable(torch.FloatTensor(target).cuda(), requires_grad=True)
            #     target = np.random.normal(0, 1, (infer_data_loader.batch_size, 1)) ################

            if args.optim == 'adam':
                optim_inf = optim.Adam([target_variable], lr=args.lr_infer)
            elif args.optim == 'adadelta':
                optim_inf = optim.Adadelta([target_variable], lr=args.lr_infer, eps=1e-7)
            elif args.optim == 'SGD':
                optim_inf = optim.SGD([target_variable], lr=args.lr_infer, momentum=0.5)
            else:
                print('Optimizer error')
            lr_scheduler_inf = lr_scheduler.ExponentialLR(optimizer=optim_inf, gamma=0.5 ** (1 / 20))

        for idx, input_data in enumerate(infer_data_loader, 1):
            pid, x, _ = input_data
            x = Variable(x.cuda())
            y_pred, loss_inf = predictor_c.forward((x, target_variable))

            predictor_c.zero_grad()
            lr_scheduler_inf.step()
            loss_inf.backward()

            optim_inf.step()
            # print('loss grad', loss.grad)
            # print('target grad', target_variable.grad, type(target_variable), target_variable.requires_grad, target_variable.is_leaf)

            # y_pred_np = y_pred.data.cpu().numpy()
            # grad_target = 2*(target - y_pred_np)
            # loss = np.sum((target - y_pred_np)**2) / target.shape[0]
            # target -= grad_target * args.lr_infer
            # print('infer loop loss:',iter, loss)
        if num_loop>2:
            print('estimate loss', iter, loss_inf.data[0])
        # print('estimate loss', iter, loss)
        # print('target value', target_variable)
    if epoch % 10 == 0:
        print('infer loop loss:', loss_inf.data[0])


    return pid, target_variable


def train_loop_infer_class(models, train_dataloader, infer_dataloader, opt, lr_schedulers, epoch, args, all_U1, temp):
    total_loss_value = 0
    loss_front = 0
    loss_E_val = 0
    cnt = 0

    for model in models:
        model.train()
        set_dropout_mode(model, True)

    predictor, predictor_c = models
    lr_scheduler_non_discr = lr_schedulers[0]

    lr_scheduler_non_discr.step()

    gpu_time = 0
    start_time = time.time()
    num_per_epoch = min(len(train_dataloader), args.train_num)  # get number per epoch

    ##### first infer the estimation of the prediction variable
    if args.infer_start <= epoch:
        # print('infer variable')
        pid_inf, target = infer_loop_variable_class_full(models, infer_dataloader, epoch, args)

    # then train the network based on input and label
    for idx, input_data in enumerate(train_dataloader, 1):
        if idx > num_per_epoch:
            break
        pid, x, groundtruth = input_data

        batch_size = x.size()[0]
        gpu_time -= time.time()
        cnt += batch_size

        # build PyTorch Variables, and transfer to GPU
        x = Variable(x.cuda())

        y_pred, loss = predictor.forward(x)
        loss_front += batch_size * loss.data[0]
        # the ordinary loss

        if args.infer_start <= epoch:
            # caculate the estimated loss

            x = Variable(x.data.clone(), requires_grad=False)
            target_y_index = find_target_y_index(pid, pid_inf)
            x[:, 0] = target[target_y_index, :]  ## U1 -> v1, previous error code: might be U1 -> v2

            augment_data = x.data.cpu().numpy()
            all_U1.append(augment_data)

            # y_new = Variable(y.clone(), requires_grad=False)
            # print('x require gradient', x.requires_grad)  # turn out to print True
            # x.requires_grad = False
            y_pred, loss_E = predictor.forward(x)

            loss = loss + float(args.lambda_c) * loss_E
            loss_E_val += batch_size * loss_E.data[0]

        else:
            loss_E_val=0
        total_loss_value += batch_size * loss.data[0]

        predictor.zero_grad()
        loss.backward()
        opt.step()

    total_point = 1000
    x_range = [-1, 1]
    # X = np.asarray([[x * 1.0 / (total_point/x_range[1]), (x * 1.0 / (total_point/x_range[1]) - 0.5*x_range[1])**2] for x in range(total_point)], dtype=np.float32)
    xx = np.asarray([x for x in range(total_point)], dtype=np.float32)
    xx = xx / total_point * (x_range[1] - x_range[0]) + x_range[0]
    X = np.asarray([[x, 0 if x < 0 else 1] for x in xx], dtype=np.float32)

    input_x = Variable(torch.FloatTensor(X).cuda(), requires_grad=False)
    y, loss_test = predictor.forward(input_x)
    y_val = y.data.cpu().numpy()
    # print(X.shape, y_val.shape)
    x = np.reshape(X[:, 0], X.shape[0])
    y_groundtruth = np.reshape(X[:, 1], X.shape[0])
    y = np.reshape(y_val, y_val.shape[0])

    if epoch == args.train_epoch-1:

        datasave = {}
        datasave['prediction'] = [x, y]
        datasave['args'] = args

        temp = datasave
    if epoch == args.train_epoch:
        groundtruth = np.load(os.path.join(config.X1noise_classify, args.datafile))
        xg = groundtruth[:, 1]
        yg = groundtruth[:, 2]

        datasave = temp
        datasave['data_aug'] = all_U1
        datasave['raw_data'] = [xg, yg]

        [x, y] = datasave['prediction']

        x_aug = []
        y_aug = []
        for eh in all_U1:
            x_aug.append(eh[0][0])
            y_aug.append(eh[0][1])
        cut = 16

        f, ax = plt.subplots(1, 2)
        ax[0].plot(x, y, 'b')
        ax[0].plot(x, y_groundtruth, 'g')
        ax[0].set_title('lam_c_{}infer_{}_'.format(args.lambda_c, args.num_loop))
        ax[0].scatter(x_aug[-cut:], y_aug[-cut:], c='r')

        y_i = 1
        loss = (y_i - y) ** 2
        loss_BCE = - np.log(y)
        datasave['fig_loss'] = [x, loss]
        datasave['fig_BCEloss'] = [x, loss_BCE]
        ax[1].plot(x, loss, 'g')
        ax[1].plot(x, loss_BCE, 'r')
        plt.show()

        print('Is this loss what you want? y|n')
        choice = input().lower()
        if choice == 'y':

            np_savepath = config.plot_ICML_path
            subfolder = 'Cdim2_m'
            if args.dimension == 2:
                os.makedirs(os.path.join(np_savepath, subfolder), exist_ok=True)
                np_savepath = os.path.join(np_savepath, subfolder, args.output.split('/')[-1])
            import scipy.io
            scipy.io.savemat(np_savepath + 'mm.mat', mdict=datasave)
            print('save to {} success'.format(np_savepath + 'mm.mat'))

        if epoch == args.train_epoch:
            plt.savefig(os.path.join(args.output, 'plot_grad.png'))

    if epoch % 10==0:
        print('Epoch: ', epoch, ' Loss total:', total_loss_value / cnt, '\t Loss front :', loss_front / cnt,
              '\t Loss Estimate: ', loss_E_val/ cnt, '\t Test loss: ', loss_test.data[0])
        # print('bingo target', target)

    return temp, total_loss_value / cnt, loss_front / cnt, loss_E_val/ cnt, loss_test.data[0]


