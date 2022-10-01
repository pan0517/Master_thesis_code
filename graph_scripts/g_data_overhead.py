from xml.dom.minidom import TypeInfo
import matplotlib.pyplot as plt
import numpy as np
import pickle
import random

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


original = np.zeros(len(pareto_model))
offload = np.zeros(len(pareto_model))
early_stop = np.zeros(len(pareto_model))

x = np.linspace(0, len(pareto_model)-1, len(pareto_model))# 這是我慢慢觀察圖形，所調整出來的範圍

#原model的DATA
for w in range(len(pareto_model)):
    for i in range(0, len(worker_loss[0])):
            if i < drop_round[w]:
                if worker_selected_round[w, i].astype(int) == 1:
                    original[w] += 24 + 880 + 10000
                else:
                    original[w] = original[w]
            else:
                original[w] += 0.5*(24 + 880 + 10000)

#沒有上傳的DATA(有early stop)
for w in range(len(pareto_model)):
    for i in range(0, len(worker_loss[0])):
        if worker_selected_round[w, i].astype(int) == 1:
            early_stop[w] += 24 + 880 + 10000
        else:
            early_stop[w] = early_stop[w]

#有上傳的DATA
for w in range(len(pareto_model)):
    for i in range(0, len(worker_loss[0])):
        if worker_selected_round[w, i].astype(int) == 1:
            if i < upload_round[w]:
                offload[w] += 24 + 880 + 10000

            elif i == upload_round[w]:
                offload[w] += 32992

            elif i > upload_round[w]:
                offload[w] = offload[w]
        else:
            offload[w] = offload[w]

y1 = np.sort(early_stop)
y2 = np.sort(offload)
y3 = np.sort(original)

Early=0
Off=0
Ori=0
for w in range(len(pareto_model)):
    Early+=early_stop[w]
    Off+=offload[w]
    Ori+=original[w]

print('early:{}, Off:{}, Ori:{}'.format(Early, Off, Ori))

for pos in ['right', 'top']:
    plt.gca().spines[pos].set_visible(False)
plt.scatter(x, y1, marker="x", s = 20, linewidths = 1, color = 'limegreen', label = 'Data Overhead w/o Offloading')
plt.scatter(x, y2, marker="o", s = 20, linewidths = 0.3, color = 'r', label = 'Data Overhead of Our Design')
plt.scatter(x, y3, marker="^", s = 20, linewidths = 0.3, color = 'gold', linestyle = 'dashed', label = 'Data Overhead of Orignal Model')

plt.title('Total Data Overhead')
plt.legend(loc = 'best', frameon=False)

plt.ylabel('Data Overhead (bits)')
plt.xlabel('Number of Worker')

plt.show()
plt.savefig('./graph_experimental_results/data_overhead.png')
plt.savefig('./eps/data_overhead.eps')