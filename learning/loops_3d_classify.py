from __future__ import print_function
import torch.nn
import time, math
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from learning.utils import *

import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import config


def infer_loop_variable_full(models, infer_data_loader, epoch, args, mode='train'):

    predictor, predictor_c = models
    predictor_c.load_state_dict(predictor.state_dict())

    predictor_c.eval()
    set_dropout_mode(predictor_c, False)

    if mode == 'train':
        num_loop = args.num_loop

    # target = Variable(torch.rand(infer_data_loader.batch_size, 1).cuda(), requires_grad=True)
    target = Variable(torch.FloatTensor(np.random.uniform(low=0.25, high=0.75, size=(infer_data_loader.batch_size, 1))).cuda(),
                      requires_grad=True)

    for iter in range(num_loop):
        for idx, input_data in enumerate(infer_data_loader, 1):
            pid, x, _ = input_data
            x = Variable(x.cuda())

            var_init, loss_inf = predictor_c.forward((x, target))
            # print('target init',var_init.data)

            if iter != 0:
                predictor_c.zero_grad()
                lr_scheduler_inf.step()
                loss_inf.backward()
                optim_inf.step()

            if iter == 0:
                target = Variable(var_init.data.clone(), requires_grad=True)

                if args.optim == 'adam':
                    optim_inf = optim.Adam([target], lr=args.lr_infer)
                elif args.optim == 'adadelta':
                    optim_inf = optim.Adadelta([target], lr=args.lr_infer, eps=1e-7)
                elif args.optim == 'SGD':
                    optim_inf = optim.SGD([target], lr=args.lr_infer, momentum=0.5)
                else:
                    print('Optimizer error')
                lr_scheduler_inf = lr_scheduler.ExponentialLR(optimizer=optim_inf, gamma=0.5 ** (1 / 20))

            truth = x[:, 1]
        if iter > 1:
            tt = target.data.cpu().numpy()
            # if np.sum(np.isnan(tt))>0:
            print('target', np.transpose(tt))
            print('loss infer', iter, loss_inf.data[0])

    return pid, target


def infer_loop_target(models, x_in, args):
    predictor, predictor_c = models
    predictor_c.load_state_dict(predictor.state_dict())

    predictor_c.eval()
    set_dropout_mode(predictor_c, False)

    target = Variable(torch.rand(x_in.shape[0], 1).cuda(), requires_grad=True)
    x_in = Variable(torch.FloatTensor(x_in).cuda())

    for iter in range(args.test_infer_num_loop):

        var_init, loss_inf = predictor_c.forward((x_in, target))

        if iter != 0:

            predictor_c.zero_grad()
            lr_scheduler_inf.step()
            loss_inf.backward()
            optim_inf.step()

        if iter == 0:
            print('infer loop target init target')
            target = Variable(var_init.data.clone(), requires_grad=True)
            # target_val = target.data.cpu().numpy()
            # flag_nan = np.isnan(target_val)
            # target_val[flag_nan] = 1 - 1e-5
            #
            # flag_1 = target_val > 1
            # flag_0 = target_val < 0
            # target_val[flag_0] = 1e-5
            # target_val[flag_1] = 1 - 1e-5
            #
            # target = Variable(torch.FloatTensor(target_val).cuda(), requires_grad=True)

            if args.optim == 'adam':
                optim_inf = optim.Adam([target], lr=args.lr_infer)
            elif args.optim == 'adadelta':
                optim_inf = optim.Adadelta([target], lr=args.lr_infer, eps=1e-7)
            elif args.optim == 'SGD':
                optim_inf = optim.SGD([target], lr=args.lr_infer, momentum=0)
            else:
                print('Optimizer error')
            lr_scheduler_inf = lr_scheduler.ExponentialLR(optimizer=optim_inf, gamma=0.5 ** (1 / 20))
    return target


def train_loop_infer_3d_C(models, train_dataloader, infer_dataloader, opt, lr_schedulers, epoch, args, all_U1, temp):
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
        pid_inf, target = infer_loop_variable_full(models, infer_dataloader, epoch, args)

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

        v1_pred, v2_pred, loss = predictor.forward(x)
        loss_front += batch_size * loss.data[0]
        # the ordinary loss

        if args.infer_start <= epoch:
            # caculate the estimated loss

            x = Variable(x.data.clone(), requires_grad=False)
            target_y_index = find_target_y_index(pid, pid_inf)

            # target_val = target.data.cpu().numpy()
            # flag_nan = np.isnan(target_val)
            # target_val[flag_nan] = 1 - 1e-5
            #
            # flag_1 = target_val > 1
            # flag_0 = target_val < 0
            # target_val[flag_0] = 0
            # target_val[flag_1] = 1
            # target = Variable(torch.FloatTensor(target_val).cuda(), requires_grad=True)

            x[:, 1] = target[target_y_index, :]  ##### V1 estiamte

            augment_data = x.data.cpu().numpy()
            all_U1.append(augment_data)  # for each train point, get its estimation point

            # y_new = Variable(y.clone(), requires_grad=False)
            v1E_pred, v2E_pred, loss_E = predictor.forward(x)

            loss = loss + float(args.lambda_c) * loss_E
            loss_E_val += batch_size * loss_E.data[0]

        else:
            loss_E_val=0
        total_loss_value += batch_size * loss.data[0]

        predictor.zero_grad()
        loss.backward()
        opt.step()
        # Loop ends

    # training stop, start testing
    total_point = 20
    BB_range = [-1, 1]

    # using ground truth to test
    BB = np.linspace(BB_range[0], BB_range[1], total_point)
    V1 = 3 * BB + 1
    V2 = 0.5 * V1 - BB + 1
    x_g = np.asarray([[bb, 0 if v1<1 else 1, 0 if v2<1.5 else 1] for bb, v1, v2 in zip(BB, V1, V2)], dtype=np.float32)
    # x_g = Variable(torch.FloatTensor(x_g).cuda(), requires_grad=False)

    target = infer_loop_target(models, x_g, args)

    target_val = target.data.cpu().numpy()
    flag_nan = np.isnan(target_val)
    target_val[flag_nan] = 0

    flag_1 = target_val > 1
    flag_0 = target_val < 0
    target_val[flag_0] = 0
    target_val[flag_1] = 1

    target = Variable(torch.FloatTensor(target_val).cuda(), requires_grad=False)

    x_g_e = Variable(torch.FloatTensor(x_g).cuda(), requires_grad=False)
    x_g_e[:, 1] = target ##### V1 estiamte
    _, _, loss_E = predictor.forward(x_g_e)

    # target = target.data.cpu().numpy()
    # target = np.reshape(target, target.shape[0])
    # print('target test', target)
    # test_loss = -np.sum(V2_ * np.log(target + 1e-5) + (1 - V2_) * np.log(1 - target + 1e-5)) / target.shape[0]
    test_loss = loss_E.data[0]

    #visualize
    if epoch == args.train_epoch - 1:
        datasave = {}

        # get BB predict V1 prediction curve, using feedforward
        x_g_t = Variable(torch.FloatTensor(x_g).cuda())
        v1_pred_use, _, _ = predictor.forward(x_g_t)
        datasave['BB'] = BB ## BB, V1
        datasave['BB_V1_pred'] = v1_pred_use.data.cpu().numpy()[:, 0] ## BB, V1
        # print('bingo datasave BB_V1_pred', datasave['BB_V1_pred'])

        # get V1 pred using infer, mesh array
        V2_array = np.zeros((total_point, total_point), dtype=np.float32)

        for i, each_v1 in enumerate(V1):
            x_g = np.asarray([[bb, each_v1, v2] for bb, v2 in zip(BB, V2)], dtype=np.float32)
            x_g_t = Variable(torch.FloatTensor(x_g).cuda())
            _, v2_pred, _ = predictor.forward(x_g_t) ###############################
            v2_pred = v2_pred.data.cpu().numpy()
            v2_pred = np.reshape(v2_pred, v2_pred.shape[0])
            V2_array[i] = v2_pred

        datasave['v2_mesh'] = V2_array
        datasave['BB_mesh'] = BB
        datasave['V1_mesh'] = V1

        fig = plt.figure()
        ax = fig.gca(projection='3d')
        surf = ax.plot_surface(BB, V1, V2_array, cmap=cm.coolwarm, linewidth=0, antialiased=False)
        fig.colorbar(surf, shrink=0.5, aspect=5)
        plt.show()

        # get loss curve
        V1 = np.linspace(0, 1, total_point)
        BB = np.ones((total_point,), dtype=np.float32) * (-0.37337337)
        V2 = np.ones((total_point,), dtype=np.float32) * 1  # 0.79357764

        X_val = np.asarray([[bb, v1, v2] for bb, v1, v2 in zip(BB, V1, V2)], dtype=np.float32)
        # take BB and V1 as known, predict v2, plot the grad loss curve
        X = Variable(torch.FloatTensor(X_val).cuda(), requires_grad=False)
        v1_pred, v2_pred, loss = predictor.forward(X)

        v1_pred = v1_pred.data.cpu().numpy()
        v2_pred = v2_pred.data.cpu().numpy()
        v1_pred = np.reshape(v1_pred, v1_pred.shape[0])
        v2_pred = np.reshape(v2_pred, v2_pred.shape[0])
        # print('v1_pred.shape, V1.shape, v2_pred.shape, V2.shape', v1_pred.shape, V1.shape, v2_pred.shape, V2.shape)
        # loss = args.lambda_a * (V1 - v1_pred) ** 2 + (V2 - v2_pred) ** 2
        # loss_1 = args.lambda_a * (V1 - v1_pred) ** 2
        # loss_2 = (V2 - v2_pred) ** 2

        # loss = args.lambda_a * (- np.log(v1_pred)) + (- np.log(v2_pred))
        # loss_1 = args.lambda_a * (- np.log(v1_pred))
        # loss_2 = (- np.log(v2_pred))
        loss_1 = args.lambda_a * (-V1 * np.log(v1_pred + 1e-5) - (1-V1) * np.log(1 - v1_pred + 1e-5))
        loss_2 = - V2 * np.log(v2_pred + 1e-5) - (1 - V2) * np.log(1 - v2_pred + 1e-5)
        loss = loss_1 + loss_2
        datasave['fig_loss'] = [V1, loss]
        datasave['fig_loss1'] = [V1, loss_1]
        datasave['fig_loss2'] = [V1, loss_2]

        temp = datasave

    if epoch == args.train_epoch:

        datasave = temp
        # data aug point
        datasave['all_U1'] = all_U1

        BB = datasave['BB']
        v1_pred = datasave['BB_V1_pred']

        plt.plot(BB, v1_pred)
        plt.title('P(V1|BB)')
        plt.show()
        # # plot prediction
        # fig = plt.figure()
        # ax = fig.gca(projection='3d')
        # ax.plot(target, BB, V2, 'r', label='prediction_c{}_a{}'.format(args.lambda_c, args.lambda_a))
        # ax.plot(V1, BB, V2, 'g', label='ground truth')  ## v1, bb, v2
        # ax.legend()
        # plt.show()

        # train data results
        data = np.load(os.path.join(config.X1noise_3d, args.datafile))
        x_g = data[:, 1:]
        print(x_g.shape)
        target = infer_loop_target(models, x_g, args)
        target = target.data.cpu().numpy()
        target = np.reshape(target, target.shape[0])

        BB = x_g[:, 0]
        V1 = x_g[:, 1]
        V2 = x_g[:, 2]

        BB = np.reshape(BB, BB.shape[0])
        V1 = np.reshape(V1, V1.shape[0])
        V2 = np.reshape(V2, V2.shape[0])

        datasave['data_raw'] = [BB, V1, V2]
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        ax.plot(target, BB, V2, 'r', label='prediction_c{}_a{}'.format(args.lambda_c, args.lambda_a))
        ax.plot(V1, BB, V2, 'g', label='ground truth') # v1, bb, v2
        ax.legend()
        plt.show()

        # plot loss curve, optimize grad

        [V1, loss] = datasave['fig_loss']
        [V1, loss_1] = datasave['fig_loss1']
        [V1, loss_2] = datasave['fig_loss2']

        plt.plot(V1, loss)
        plt.plot(V1, loss_1,'g')
        plt.plot(V1, loss_2,'r')
        plt.title('prediction_c{}_a{}'.format(args.lambda_c, args.lambda_a))
        plt.show()

        print('Is this loss what you want? y|n')
        choice = input().lower()
        if choice == 'y':
            datasave['args'] = args

            np_savepath = '/home/mcz/Pytorch_Project/ICMLplot/'
            os.makedirs(os.path.join(np_savepath, 'dim3_half_Class'), exist_ok=True)
            np_savepath = os.path.join(np_savepath, 'dim3_half_Class', args.output.split('/')[-1])
            import scipy.io
            scipy.io.savemat(np_savepath + 'mm.mat', mdict=datasave)
            print('save to {} success'.format(np_savepath + 'mm.mat'))

    # one epoch of training is over
    if epoch % 10==0:
        print('Epoch: ', epoch, ' Loss total:', total_loss_value / cnt, '\t Loss front :', loss_front / cnt,
              '\t Loss Estimate: ', loss_E_val/ cnt, '\t Test loss: ', test_loss)
        # print('bingo target', target)

    return temp, total_loss_value / cnt, loss_front / cnt, loss_E_val/ cnt, test_loss



