from xml.dom.minidom import TypeInfo
import matplotlib.pyplot as plt
import numpy as np
import pickle

from utils.energy import EnergyCon_up
from utils.energy import EnergyCon_down
from utils.energy import EnergyComp_local
from utils.energy import EnergyCon_up_model

with open('pickle/filename_num100_sc50.pickle', 'rb') as handle:
    dictE = pickle.load(handle)

with open('pickle/worker_loss_num100_sc50.pickle', 'rb') as handle:
    worker_loss = pickle.load(handle)

with open('pickle/drop_round_num100_sc50.pickle', 'rb') as handle:
    drop_round = pickle.load(handle)

with open('pickle/predict_drop_list_num100_sc50.pickle', 'rb') as handle:
    predict_drop_list = pickle.load(handle)

with open('pickle/pareto_model_num100_sc50.pickle', 'rb') as handle:
    pareto_model = pickle.load(handle)

with open('pickle/upload_round_num100_sc50.pickle', 'rb') as handle:
    upload_round = pickle.load(handle)

with open('pickle/worker_selected_round_num100_sc50.pickle', 'rb') as handle:
    worker_selected_round = pickle.load(handle)

n0 = []
upload_round_gt = []
count = 0
users_num = len(pareto_model)
epoch_num = len(worker_loss[0])
number_can_upload = users_num

for i in range(users_num):
    n0.append(0)
    upload_round_gt.append(epoch_num)

for j in range(epoch_num):
    ####################Algo ground truth####################
    if number_can_upload > 90:
        for i in range(users_num):
            if n0[i] == float('-inf'):
                n0[i] = float('-inf')
            else:
                if drop_round[i] >= j:
                    n0[i] = (EnergyComp_local(dictE, i, 1)) * (drop_round[i] - j)
                else:
                    #n0[i] = (EnergyComp_local(dictE, i, 1)) * (j - drop_round[i])
                    n0[i] = float('-inf')
                
            '''
            else:
                n0[i] = (EnergyComp_local(dictE, i, 1)) * abs(drop_round[i] - j)
            '''
        
        tmp = max(n0)
        if tmp != float('-inf'):  
            index = n0.index(tmp)
            upload_round_gt[index] = j
            n0[index] = float('-inf')
            number_can_upload = number_can_upload - 1
            print("woker#{} 會在第{}回合上傳local data".format(index, j))
            count +=1
        
    for w in range(users_num):
        if drop_round[w] == j:
            n0[w] = float('-inf')
            number_can_upload += 1


print(count)

#print(EnergyComp_local(dictE, i, 0))
a = np.zeros((len(worker_loss[0]),))
b = np.zeros((len(worker_loss[0]),))
c = np.zeros((len(worker_loss[0]),))
d = np.zeros((len(worker_loss[0]),))
e = np.zeros((len(worker_loss[0]),))
f = np.zeros((len(worker_loss[0]),))

x = np.linspace(0, len(worker_loss[0])-1, len(worker_loss[0]))# 這是我慢慢觀察圖形，所調整出來的範圍

# 有上傳的能源消耗
for w in range(len(pareto_model)):
    if worker_selected_round[w, 0] == 1:
        if upload_round[w] == 0:
            b[0] = EnergyCon_up(dictE, w, 1, 1)
        else:
            b[0] = EnergyCon_up_model(dictE, w, 0.02, 1) + EnergyComp_local(dictE, w, 1)
    for i in range(1, len(worker_loss[0])):
        if i < upload_round[w]:
            b[i] = b[i-1] + EnergyCon_up_model(dictE, w, 0.02, 1) + EnergyComp_local(dictE, w, 1)

        elif i == upload_round[w]:
            if i != 350:
                b[i] = b[i-1] + EnergyCon_up(dictE, w, 1, 1)
            else:
                b[i] = b[i-1]

        elif i > upload_round[w]:
            b[i] = b[i-1]
    d += b

# 有上傳的能源消耗(ground truth)
for w in range(len(pareto_model)):
    if worker_selected_round[w, 0] == 1:
        if upload_round_gt[w] == 0:
            a[0] = EnergyCon_up(dictE, w, 1, 1)
        else:
            a[0] = EnergyCon_up_model(dictE, w, 0.02, 1) + EnergyComp_local(dictE, w, 1)
    for i in range(1, len(worker_loss[0])):
        if i < upload_round_gt[w]:
            a[i] = a[i-1] + EnergyCon_up_model(dictE, w, 0.02, 1) + EnergyComp_local(dictE, w, 1)

        elif i == upload_round_gt[w]:
            if i != 350:
                a[i] = a[i-1] + EnergyCon_up(dictE, w, 1, 1)
            else:
                a[i] = a[i-1]

        elif i > upload_round_gt[w]:
            a[i] = a[i-1]
    c += a

y1 = c
y2 = d

print('差距的percentage:{}, ground truth的能耗:{}, 預測的能耗:{}'.format((d[349]-c[349])/d[349], c[349], d[349])) # /c[349]

for pos in ['right', 'top']:
    plt.gca().spines[pos].set_visible(False)
plt.plot(x, y1, color = 'limegreen', label = 'Total E.C. of Ground Truth')
plt.plot(x, y2, color = 'c', label = 'total E.C. of Our Approach')

plt.title('Total Energy Consumption Curve of Workers')
plt.legend(loc = 'best', frameon=False)

plt.ylabel('Energy Consumption (J)')
plt.xlabel('Round Number')

plt.savefig('./graph_experimental_results/total_power_consumption_curve_with_ground_truth.png')
plt.savefig('./eps/total_power_consumption_curve_with_ground_truth.eps')