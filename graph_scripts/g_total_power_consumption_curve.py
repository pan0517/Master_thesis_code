from xml.dom.minidom import TypeInfo
import matplotlib.pyplot as plt
import numpy as np
import pickle

from utils.energy import EnergyCon_up
from utils.energy import EnergyCon_down
from utils.energy import EnergyComp_local
from utils.energy import EnergyCon_up_model

with open('pickle/filename_num100_sc20.pickle', 'rb') as handle:
    dictE = pickle.load(handle)

with open('pickle/worker_loss_num100_sc20.pickle', 'rb') as handle:
    worker_loss = pickle.load(handle)
#print(len(worker_loss[0]))
#print(worker_loss)

with open('pickle/drop_round_num100_sc20.pickle', 'rb') as handle:
    drop_round = pickle.load(handle)

with open('pickle/predict_drop_list_num100_sc20.pickle', 'rb') as handle:
    predict_drop_list = pickle.load(handle)

with open('pickle/pareto_model_num100_sc20.pickle', 'rb') as handle:
    pareto_model = pickle.load(handle)

with open('pickle/upload_round_num100_sc20.pickle', 'rb') as handle:
    upload_round = pickle.load(handle)

with open('pickle/worker_selected_round_num100_sc20.pickle', 'rb') as handle:
    worker_selected_round = pickle.load(handle)

#print(EnergyComp_local(dictE, i, 0))
a = np.zeros((len(worker_loss[0]),))
b = np.zeros((len(worker_loss[0]),))
c = np.zeros((len(worker_loss[0]),))
d = np.zeros((len(worker_loss[0]),))
e = np.zeros((len(worker_loss[0]),))
f = np.zeros((len(worker_loss[0]),))

first = upload_round.index(1)

x = np.linspace(0, len(worker_loss[0])-1, len(worker_loss[0]))# 這是我慢慢觀察圖形，所調整出來的範圍

#原model的能源消耗
for w in range(len(pareto_model)):
    if worker_selected_round[w, 0] == 1:
        e[0] = EnergyCon_up_model(dictE, w, 0.02, 1) + EnergyComp_local(dictE, w, 1)
    for i in range(1, len(worker_loss[0])):
        e[i] = e[i-1] + EnergyCon_up_model(dictE, w, 0.02, 1) + EnergyComp_local(dictE, w, 1)
    f += e

'''
for w in range(len(pareto_model)):
    a[0] += EnergyCon_up_model(dictE, w, 0.01, 1) + EnergyComp_local(dictE, w, 1)/len(worker_loss[0])
    if w == upload_round.index(1):
        b[0] += EnergyCon_up(dictE, w, 0.01, 1)
    else:
        b[0] += EnergyCon_up_model(dictE, w, 0.01, 1) + EnergyComp_local(dictE, w, 1)/len(worker_loss[0])
print(a[0], b[0])
'''
#沒有上傳的能源消耗
for w in range(len(pareto_model)):
    if worker_selected_round[w, 0] == 1:
        a[0] = EnergyCon_up_model(dictE, w, 0.02, 1) + EnergyComp_local(dictE, w, 1)
    for i in range(1, len(worker_loss[0])):
        if i < drop_round[w]:
            a[i] = a[i-1] + EnergyCon_up_model(dictE, w, 0.02, 1) + EnergyComp_local(dictE, w, 1)
        else:
            a[i] = a[i-1]
    c += a

#有上傳的能源消耗
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
            b[i] = b[i-1] + EnergyCon_up(dictE, w, 1, 1)

        elif i > upload_round[w]:
            b[i] = b[i-1]
    d += b

y1 = c/1000
y2 = d/1000
y3 = f/1000
print((c[349]-d[349])/c[349])

for pos in ['right', 'top']:
    plt.gca().spines[pos].set_visible(False)
plt.plot(x, y1, color = 'limegreen', label = 'Total E.C. w/o Offloading')
plt.plot(x, y2, color = 'c', label = 'Total E.C. of Our Approach')
plt.plot(x, y3, color = 'darkorange', linestyle = 'dashed', label = 'Total E.C. of Orginal Model')

plt.title('Total Power Consumption Curve of Workers')
plt.legend(loc = 'best', frameon=False)

plt.ylabel('Energy Consumption (kJ)')
plt.xlabel('Round Number')

plt.savefig('./graph_experimental_results/total_power_consumption_curve.png')
plt.savefig('./eps/total_power_consumption_curve.eps')