from xml.dom.minidom import TypeInfo
import matplotlib.pyplot as plt
import numpy as np
import pickle
import random

from utils.energy import EnergyCon_up
from utils.energy import EnergyCon_down
from utils.energy import EnergyComp_local
from utils.energy import EnergyCon_up_model

with open('pickle/filename_num100_sc50.pickle', 'rb') as handle:
    dictE = pickle.load(handle)

with open('pickle/worker_loss_num100_sc50.pickle', 'rb') as handle:
    worker_loss = pickle.load(handle)
#print(len(worker_loss[0]))
#print(worker_loss)

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

w = 6 #worker number

#print(EnergyComp_local(dictE, i, 0))
a = np.zeros((len(worker_loss[0]),))
b = np.zeros((len(worker_loss[0]),))
c = np.zeros((len(worker_loss[0]),))

x = np.linspace(0, len(worker_loss[0])-1, len(worker_loss[0]))# 這是我慢慢觀察圖形，所調整出來的範圍

#initial
if worker_selected_round[w, 0].astype(int) == 1:
    c[0] = EnergyCon_up_model(dictE, w, 0.01, 1) + EnergyComp_local(dictE, w, 1)
    a[0] = EnergyCon_up_model(dictE, w, 0.01, 1) + EnergyComp_local(dictE, w, 1)
    if upload_round[w] == 1:
        b[0] = EnergyCon_up(dictE, w, 0.01, 1)
    else:
        b[0] = EnergyCon_up_model(dictE, w, 0.01, 1) + EnergyComp_local(dictE, w, 1)
else:
    a[0] = b[0] = c[0] = 0

#沒有上傳的能源消耗(無early stop)
for i in range(1, len(worker_loss[0])):
    if i < drop_round[w]:
        if worker_selected_round[w, i].astype(int) == 1:
            c[i] = c[i-1] + EnergyCon_up_model(dictE, w, 0.02, 1) + EnergyComp_local(dictE, w, 1)
        else:
            c[i] = c[i-1]
    else:
        if(random.choice((0,1))):
            c[i] = c[i-1] + EnergyCon_up_model(dictE, w, 0.02, 1) + EnergyComp_local(dictE, w, 1)
        else:
            c[i] = c[i-1]

#沒有上傳的能源消耗(有early stop)
for i in range(1, len(worker_loss[0])):
    if worker_selected_round[w, i].astype(int) == 1:
        if i < drop_round[w]:
            a[i] = a[i-1] + EnergyCon_up_model(dictE, w, 0.02, 1) + EnergyComp_local(dictE, w, 1)
        else:
            a[i] = a[i-1]
    else:
        a[i] = a[i-1]

#有上傳的能源消耗
for i in range(1, len(worker_loss[0])):
    upload_round[w] = 205
    if worker_selected_round[w, i].astype(int) == 1:
        if i < upload_round[w]:
            b[i] = b[i-1] + EnergyCon_up_model(dictE, w, 0.02, 1) + EnergyComp_local(dictE, w, 1)

        elif i == upload_round[w]:
            b[i] = b[i-1] + EnergyCon_up(dictE, w, 1, 1)

        elif i > upload_round[w]:
            b[i] = b[i-1]
    else:
        b[i] = b[i-1]

y1 = a
y2 = b
y3 = c 

for pos in ['right', 'top']:
    plt.gca().spines[pos].set_visible(False)

plt.plot(x, y1, color = 'limegreen', label = 'E.C. w/o Offloading')
plt.plot(x, y2, color = 'c', label = 'E.C. w/ Offloading')
plt.plot(x, y3, color = 'darkorange', linestyle = 'dashed', label = 'E.C. of Original Model')

predict = predict_drop_list[w]
d = drop_round[w].astype(int)
u = upload_round[w]
plt.axvline(x = predict, linestyle = 'dashed', color = 'gray', label = 'Prediction of Convergent Round')
plt.axvline(x = d, linestyle = 'dashed', color = 'r', label = 'Real Convergent Round')
plt.axvline(x = u, linestyle = 'dashed', color = 'y', label = 'Uploaded Round') #uploaded round

plt.title('Power Consumption Curve of Worker#{}'.format(w))
plt.legend(loc = 'best', frameon=False)

plt.ylabel('Power Consumption (J)')
plt.xlabel('Round Number')

plt.savefig('./graph_experimental_results/energy_consumption_curve_solo.png')
plt.savefig('./eps/energy_consumption_curve_solo.eps')
for i in range(100):
    print('預測drop round:{}, 實際drop round:{}, 上傳round:{}'.format(predict_drop_list[i], drop_round[i].astype(int), upload_round[i]))