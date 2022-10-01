
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

for pos in ['right', 'top']:
    plt.gca().spines[pos].set_visible(False)
# sort the data:
y_sorted = np.sort(y)

# calculate the proportional values of samples
p = 1. * np.arange(len(y)) / (len(y) - 1)

# plot the sorted data:
plt.plot(y_sorted, p, label = 'CDF of Error')

#plt.xlabel('Worker Number')
plt.ylabel('Cumulative Percentage of Error between Ground Truth')
plt.title('Error of N')
plt.legend(loc = 'best', frameon=False)

plt.axvline(x = 0.2, linestyle = 'dashed', color = 'darkorange')
plt.axvline(x = -0.2, linestyle = 'dashed', color = 'darkorange')

plt.savefig('./graph_experimental_results/N_error_cdf.png')
plt.savefig('./eps/N_error_cdf.eps')




