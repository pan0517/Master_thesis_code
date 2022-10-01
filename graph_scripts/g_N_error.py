
from xml.dom.minidom import TypeInfo
import matplotlib.pyplot as plt
import numpy as np
import pickle
from regression_fitting import pareto, gamma, exponential
from regression_fitting import earlystop_round, fitting_pareto

#from torch import threshold

with open('pickle/drop_round_num100_sc40.pickle', 'rb') as handle:
    drop_round = pickle.load(handle)


with open('pickle/predict_drop_list_num100_sc40.pickle', 'rb') as handle:
    predict_drop_list = pickle.load(handle)

with open('pickle/worker_loss_num100_sc40.pickle', 'rb') as handle:
    worker_loss = pickle.load(handle)

for i in range(100):
    _, _, rv = fitting_pareto(i, 350, worker_loss)
    predict_drop_list[i], _ = earlystop_round(rv)

x = np.linspace(0, len(drop_round)-1, len(drop_round))
x1 = x.astype(int)

y = (predict_drop_list[x1]-drop_round[x1])/350
y1 = sorted(y)

#print(type(y))
threshold = np.max(abs(y))
print("誤差最大到 {} percent".format(threshold))

for pos in ['right', 'top']:
    plt.gca().spines[pos].set_visible(False)
plt.scatter(x1, y1, s = 15, linewidths = 0.3, edgecolor ="red", color = 'gold')

plt.xlabel('Worker Number')
plt.ylabel('Percentage of Error between Prediction and Ground Truth')
plt.title('Error of N')
plt.savefig('./graph_experimental_results/N_error.png')
plt.savefig('./eps/N_error.eps')