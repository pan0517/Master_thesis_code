
from xml.dom.minidom import TypeInfo
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pickle

from utils.energy import EnergyCon_up
from utils.energy import EnergyCon_down
from utils.energy import EnergyComp_local
from utils.energy import EnergyCon_up_model

#from torch import threshold

with open('pickle/filename_num100_sc30.pickle', 'rb') as handle:
    dictE = pickle.load(handle)

with open('pickle/worker_loss_num100_sc30.pickle', 'rb') as handle:
    worker_loss = pickle.load(handle)

with open('pickle/drop_round_num100_sc30.pickle', 'rb') as handle:
    drop_round = pickle.load(handle)

with open('pickle/predict_drop_list_num100_sc30.pickle', 'rb') as handle:
    predict_drop_list = pickle.load(handle)

with open('pickle/pareto_model_num100_sc30.pickle', 'rb') as handle:
    pareto_model = pickle.load(handle)

with open('pickle/upload_round_num100_sc30.pickle', 'rb') as handle:
    upload_round = pickle.load(handle)

with open('pickle/worker_selected_round_num100_sc30.pickle', 'rb') as handle:
    worker_selected_round = pickle.load(handle)

a = np.zeros((len(worker_loss[0]),))
b = np.zeros((len(worker_loss[0]),))

x = np.linspace(0, len(drop_round)-1, len(drop_round))
x1 = x.astype(int)

for w in range(len(pareto_model)):
    b[w] = (EnergyComp_local(dictE, w, 1) + EnergyCon_up_model(dictE, w, 0.02, 1))*(drop_round[w]) + EnergyCon_up(dictE, w, 1, 1)
    a[w] = (EnergyComp_local(dictE, w, 1) + EnergyCon_up_model(dictE, w, 0.02, 1))*(predict_drop_list[w]) + EnergyCon_up(dictE, w, 1, 1)

#print("誤差最大到 {} percent".format(threshold))

#plt.scatter(x1, a[x1] - b[x1], color = 'gold')
df = pd.DataFrame(a[x1] - b[x1],columns=['server_capacity = 30'])
f = df.boxplot(sym = 'o',
              vert = True,
              whis = 1.5,
              patch_artist = False,
              meanline = False,
              showmeans = True,
              showbox = True,
              showcaps = True,
              showfliers = False,
              notch = False, 
              return_type = 'dict'
              )

plt.ylabel('Difference of Energy Consumption (J)')
plt.title('Error between Ground Truth and Our Approach in Power Consumption')

plt.savefig('./graph_experimental_results/error_between_ground_truth_and_our_approach_in_power_consumption.png')
plt.savefig('./eps/error_between_ground_truth_and_our_approach_in_power_consumption.eps')