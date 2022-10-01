#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import copy
import pickle
import numpy as np
from numpy.core.fromnumeric import shape
import pandas as pd
import torch
import time
import random
import sys
import math
import os
import pdb
import matplotlib.pyplot as plt
import scipy.stats as stats

from utils.energy import EnergyCon_up
from utils.energy import EnergyCon_up_model
from utils.energy import EnergyCon_down
from utils.energy import EnergyComp_local

from regression_fitting import pareto, gamma, exponential
from regression_fitting import earlystop_round, fitting_pareto, fitting_gamma

from utils.options import args_parser
from utils.train_utils import get_data, get_model
from models.Update import LocalUpdate
from models.test import test_img



if __name__ == '__main__':
    total_start = time.time()

    # parse args
    args = args_parser()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
    
    base_dir = './save/{}/{}_iid{}_num{}_C{}_le{}/shard{}/{}/'.format(
        args.dataset, args.model, args.iid, args.num_users, args.frac, args.local_ep, args.shard_per_user, args.results_save)
    if not os.path.exists(os.path.join(base_dir, 'fed')):
        os.makedirs(os.path.join(base_dir, 'fed'), exist_ok=True)

    dataset_train, dataset_test, dict_users_train, dict_users_test = get_data(args)
    dict_save_path = os.path.join(base_dir, 'dict_users.pkl')
    with open(dict_save_path, 'wb') as handle:
        pickle.dump((dict_users_train, dict_users_test), handle)

    net_glob = get_model(args)
    net_glob.train()

    # training
    results_save_path = os.path.join(base_dir, 'fed/results.csv')

    loss_train = []
    net_best = None
    best_loss = None
    best_acc = None
    best_epoch = None

    lr = args.lr
    results = []

    # 紀錄還剩下可以training的users
    worker_energy_consump_list = []
    n0 = []
    number_can_upload = args.num_users
    upload_round = []
    upload_list = []
    users_list = []
    drop_user_list = []
    drop_count = []
    trigger_times = []
    for i in range(args.num_users):
        worker_energy_consump_list.append(0)
        n0.append(0)
        upload_round.append(args.epochs)
        users_list.append(i)
        drop_count.append(0)
        trigger_times.append(0)

    # 創numpy紀錄每個worker的loss
    worker_loss = np.zeros((args.num_users, args.epochs))
    worker_energy_consump = np.zeros((args.num_users, args.epochs))
    drop_round = np.zeros(args.num_users)
    for i in range(args.num_users):
        drop_round[i] = args.epochs
        for j in range(args.epochs):
            worker_loss[i, j] = 3

    # 紀錄worker被選到的round
    worker_selected_round = np.zeros((args.num_users, args.epochs))

    # 創list記錄train一個epoch時間
    time_list = []
    straggle_list = []
    time_difference = []
    count = 0

    # 創dictionary紀錄各項計算能源消耗所需要的變數
    energy_consump_up = []
    energy_consump_down = []
    dictE = {}
    dictE['B_max'] = 1000
    dictE['G_up'] = 10**(-4)
    dictE['G_down'] = 10**(-4)

    dictE['n_up'] = []
    dictE['n_down'] = []
    dictE['fi'] = []
    dictE['mi'] = []
    dictE['power_up'] = []
    dictE['power_down'] = []
    dictE['data_size'] = []
    dictE['Elocal'] = []
    dictE["ci"] = []

    #dictE['time_differ'] = []
    #dictE['time_differ'] = [[0]*10 for i in range(1000)]
    
    # 隨機指派device變數: fi, ci, channel_gain, power to upload & download
    for i in range(args.num_users):

        dictE['n_up'].append(10**(-8))
        dictE['n_down'].append(10**(-8))

        f = random.uniform(10000, 80000)
        dictE['fi'].append(f)

        mi = random.uniform(10**(-4), 10**(-3))
        dictE['mi'].append(mi)

        seq = [0.6, 3]
        dictE['power_up'].append(random.choice(seq))
        dictE['power_down'].append(random.choice(seq))
        #dictE['ci'].append(random.choice(seq))

        s = sys.getsizeof(dict_users_train[i])
        dictE['data_size'].append(s)
    
    # 為每一個worker配置一個pareto distribution
    pareto_model = np.empty((args.num_users, args.epochs))
    predict_drop_list = np.full(args.num_users, args.epochs)
    predict_drop_list_when_upload = np.full(args.num_users, args.epochs+1)

    #print(predict_drop_list, len(predict_drop_list))
   
    server_capacity = args.server_capacity
    count_len = 0
    final_epoch = args.epochs

    transmission_flag = True
    iter_can_trans = 0
    count_qualification = 0
    for iter in range(args.epochs):

        if(iter == iter_can_trans): transmission_flag = True

        w_glob = None
        loss_locals = []
        m = max(int(args.frac * args.num_users), 1)
        #idxs_users = np.random.choice(range(args.num_users), m, replace=False)
        if len(users_list) >= 50:
            t = random.sample(users_list, m)
        else:
            t = random.sample(users_list, len(users_list))
        idxs_users = np.array(t)
        print("\nRound {:3d}, lr: {:.6f}, {}".format(iter, lr, idxs_users))

        count = 0
        # 此回合沒被選到就讓其loss和能耗與上回合一樣
        for i in range(args.num_users):
            if i not in idxs_users:
                worker_loss[i, iter] = worker_loss[i, iter-1]
                worker_energy_consump[i, iter] = worker_energy_consump[i, iter-1]

            ############### New Algo #################################################################
        if(transmission_flag):
            for user in users_list:
                if(iter>2):
                    rv , MSE, fitting_parameters = fitting_pareto(user, iter, worker_loss)
                    # print('第{}個worker在{}回合的MSE: {}'.format(user, iter, MSE))
                    # print('第{}個worker在{}回合的柏拉圖分佈: {}'.format(user, iter, rv))
                    if(MSE<=0.1):
                        predict_drop_list[user], _ = earlystop_round(fitting_parameters)
                        # print('第{}個worker在{}回合的predict_drop: {}'.format(user, iter, predict_drop_list[user]))
                    else:
                        predict_drop_list[user] = 350
                        count_qualification+=1
                else:
                    predict_drop_list[user] = 350
            print(count_qualification)
            ##########################################################################################

            ####################Algo##################################################################
            if number_can_upload > args.num_users - server_capacity:
                for i in range(args.num_users):
                    if n0[i] == float('-inf'):
                        n0[i] = float('-inf')
                    else:
                        if predict_drop_list[i] >= iter:
                            n0[i] = (EnergyComp_local(dictE, i, 1)) * (predict_drop_list[i] - iter)
                        else:
                            #n0[i] = (EnergyComp_local(dictE, i, 1)) * (iter - predict_drop_list[i])
                            n0[i] = float('-inf')

                tmp = max(n0)
                if tmp != float('-inf'):
                    index = n0.index(tmp)
                    upload_round[index] = iter
                    n0[index] = float('-inf')
                    number_can_upload = number_can_upload - 1
                    worker_energy_consump_list[index] += EnergyCon_up(dictE, index, 1, 1)
                    for k in range(iter, args.epochs):
                        worker_energy_consump[index, k] = worker_energy_consump_list[index]
                    
                    upload_list.append(index)
                    predict_drop_list_when_upload[index] = predict_drop_list[index]
                    print("worker#{} 會在第{}回合上傳local data, 上傳時預測的round:{}".format(index, iter, predict_drop_list_when_upload[index]))
                    transmission_flag = False
                    iter_can_trans = iter+2
                    if iter >= 75:
                        fp , _ , _ = fitting_pareto(index, iter, worker_loss)
                        for i in range(len(fp)):
                            pareto_model[index, i] = fp[i]
            ##########################################################################################

        for idx in idxs_users:
            # record workers' energy consumption
            if upload_round[idx] > iter: 
                worker_energy_consump_list[idx] += EnergyComp_local(dictE, idx, 1)
                worker_energy_consump_list[idx] += EnergyCon_up_model(dictE, idx, 0.02, 1)
                worker_energy_consump[idx, iter] = worker_energy_consump_list[idx]

            start = time.time()
            worker_selected_round[idx, iter] = 1 # 紀錄worker被選到的round
            local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users_train[idx])
           
            net_local = copy.deepcopy(net_glob)
            #print(dict_users_train[idx])

            w_local, loss = local.train(net=net_local)
            #print(loss)
            worker_loss[idx, iter] = 0.75*worker_loss[idx, iter-1] + 0.25*loss # 用moving avg來紀錄worker的loss
            delta = (worker_loss[idx, iter-1] - worker_loss[idx, iter]) / worker_loss[idx, iter] # 紀錄loss的delta
            #print('delta of worker#{:2d} in round{}: {}'.format(idx, iter, delta))
            loss_locals.append(copy.deepcopy(loss))
            
            # data size
            '''
            print('data: {}'.format(sys.getsizeof(dict_users_train[idx]))) # 獲得每個user的data大小
            print('weight: {}'.format(sys.getsizeof(w_local))) # weight大小
            #print(sys.getsizeof(loss_locals))
            print('loss: {}'.format(sys.getsizeof(loss))) # loss大小
            total_params = sum(p.numel() for p in net_glob.parameters())
            print('total_params: {}'.format(total_params))
            '''

            start_weight1 = time.time()
            if w_glob is None:
                w_glob = copy.deepcopy(w_local)
            else:
                for k in w_glob.keys():
                    w_glob[k] += w_local[k]
            end_weight1 = time.time()

            end = time.time()
            total = end - start

            if count < 10:
                total += 0.1
            else:
                total = total
            time_list.append(total)
            #print(total)
            count += 1
            # loss低於一個threshold就drop掉該worker
            if len(users_list) > int(args.frac * args.num_users): # if len(users_list) > 50
                if worker_loss[idx, iter] > worker_loss[idx, iter-1]:
                    trigger_times[idx] += 1
                    if trigger_times[idx] >= 2 and worker_loss[idx, iter] < 1.477759:
                        users_list.remove(idx)
                        drop_user_list.append(idx)
                        drop_round[idx] = iter
                        number_can_upload = number_can_upload + 1
                        n0[idx] = float('-inf')
                        print('Drop worker#{:2d}, round:{}, delta: {}'.format(idx, iter, delta))
                else:
                    trigger_times[idx] = 0 
        
        #print(worker_loss)
        print('剩餘workers數量: {}'.format(len(users_list)))

        straggle = max(time_list)
        straggle_list.append(straggle)
        time_list.clear()
        print('train一個epoch要花時間： ' + str(straggle))

        lr *= args.lr_decay

        start_weight2 = time.time()
        # update global weights
        for k in w_glob.keys():
            w_glob[k] = torch.div(w_glob[k], m)
            #print("更新weight大小: {}".format(sys.getsizeof(w_glob[k])))#model weight的大小
        end_weight2 = time.time()
        agg = end_weight2 - start_weight2 + end_weight1 - start_weight1
        print('server做aggregate的時間: {}'.format(agg))

        # copy weight to net_glob
        net_glob.load_state_dict(w_glob)

        # print loss
        loss_avg = sum(loss_locals) / len(loss_locals)
        loss_train.append(loss_avg)

        if (iter + 1) % args.test_freq == 0:
            net_glob.eval()
            acc_test, loss_test = test_img(net_glob, dataset_test, args)
            print('Round {:3d}, Average loss {:.3f}, Test loss {:.3f}, Test accuracy: {:.2f}'.format(
                iter, loss_avg, loss_test, acc_test))

            if best_acc is None or acc_test > best_acc:
                net_best = copy.deepcopy(net_glob)
                best_acc = acc_test
                best_epoch = iter

            # if (iter + 1) > args.start_saving:
            #     model_save_path = os.path.join(base_dir, 'fed/model_{}.pt'.format(iter + 1))
            #     torch.save(net_glob.state_dict(), model_save_path)

            results.append(np.array([iter, loss_avg, loss_test, acc_test, best_acc]))
            final_results = np.array(results)
            final_results = pd.DataFrame(final_results, columns=['epoch', 'loss_avg', 'loss_test', 'acc_test', 'best_acc'])
            final_results.to_csv(results_save_path, index=False)

        if (iter + 1) % 50 == 0:
            best_save_path = os.path.join(base_dir, 'fed/best_{}.pt'.format(iter + 1))
            model_save_path = os.path.join(base_dir, 'fed/model_{}.pt'.format(iter + 1))
            torch.save(net_best.state_dict(), best_save_path)
            torch.save(net_glob.state_dict(), model_save_path)

        '''
        if len(users_list) == int(args.frac * args.num_users):
            count_len += 1
            if count_len >= 50:
                final_epoch = iter
                break
        else:
            count_len = 0
        '''
        count_qualification = 0
        
    print('Best model, iter: {}, acc: {}'.format(best_epoch, best_acc))
    total_end = time.time()
    total_total = total_end - total_start
    print('總共耗時: {}'.format(total_total))
    print('workers能源消耗: {}'.format(worker_energy_consump_list))


    # 為了畫能耗曲線圖，將dictionary存下來
    a = dictE
    with open('./pickle/filename_num{}_sc{}.pickle'.format(args.num_users, server_capacity), 'wb') as handle:
        pickle.dump(a, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # 為了畫worker的loss圖，紀錄下worker_loss
    b = worker_loss
    with open('./pickle/worker_loss_num{}_sc{}.pickle'.format(args.num_users, server_capacity), 'wb') as handle:
        pickle.dump(b, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # 紀錄下畫出loss預測的model相關file
    c = drop_round # worker實際被early stop的round
    with open('./pickle/drop_round_num{}_sc{}.pickle'.format(args.num_users, server_capacity), 'wb') as handle:
        pickle.dump(c, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    d = predict_drop_list # worker運用柏拉圖分布預測出來會被drop的round
    with open('./pickle/predict_drop_list_num{}_sc{}.pickle'.format(args.num_users, server_capacity), 'wb') as handle:
        pickle.dump(d, handle, protocol=pickle.HIGHEST_PROTOCOL)

    e = pareto_model # worker的柏拉圖分布
    with open('./pickle/pareto_model_num{}_sc{}.pickle'.format(args.num_users, server_capacity), 'wb') as handle:
        pickle.dump(e, handle, protocol=pickle.HIGHEST_PROTOCOL)

    g = drop_user_list # worker會在第幾回合被drop掉(index)
    with open('./pickle/drop_user_list_num{}_sc{}.pickle'.format(args.num_users, server_capacity), 'wb') as handle:
        pickle.dump(g, handle, protocol=pickle.HIGHEST_PROTOCOL)    

    h = upload_round # worker上傳的那個round
    with open('./pickle/upload_round_num{}_sc{}.pickle'.format(args.num_users, server_capacity), 'wb') as handle:
        pickle.dump(h, handle, protocol=pickle.HIGHEST_PROTOCOL)

    k = worker_selected_round # worker會在第幾個round中被選中
    with open('./pickle/worker_selected_round_num{}_sc{}.pickle'.format(args.num_users, server_capacity), 'wb') as handle:
        pickle.dump(k, handle, protocol=pickle.HIGHEST_PROTOCOL)

    l = worker_energy_consump # worker會在第幾個round中被選中
    with open('./pickle/worker_energy_consump_num{}_sc{}.pickle'.format(args.num_users, server_capacity), 'wb') as handle:
        pickle.dump(l, handle, protocol=pickle.HIGHEST_PROTOCOL)

    m = upload_list # 哪些workers有被上傳
    with open('./pickle/upload_list_num{}_sc{}.pickle'.format(args.num_users, server_capacity), 'wb') as handle:
        pickle.dump(m, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    n = predict_drop_list_when_upload # 上傳時的預測round
    with open('./pickle/predict_drop_list_when_upload_num{}_sc{}.pickle'.format(args.num_users, server_capacity), 'wb') as handle:
        pickle.dump(n, handle, protocol=pickle.HIGHEST_PROTOCOL)    


    f = open('message.txt','a+')
    for line in f:
        print(line)
    f.close()
    




