from turtle import color
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.optimize import curve_fit
import pandas as pd
import numpy as np
import pickle
import scipy.stats as stats 

with open('pickle/worker_loss_num100_sc50.pickle', 'rb') as handle:
    worker_loss = pickle.load(handle)

with open('pickle/drop_round_num100_sc50.pickle', 'rb') as handle:
    drop_round = pickle.load(handle)

def pareto(x, alpha, beta, s):
    rv = s*stats.pareto.pdf(x, alpha, -25, beta) + 1.477
    return rv

def gamma(x, alpha, beta, s):
    rv = s*stats.gamma.pdf(x, alpha, -5, beta) + 1.477
    return rv

w = 10
y = []
for round in range(350):
    if round<8: 
        y.append(0)
    else:
        x_flex = np.linspace(0, round-1, round).astype(int)
        y_flex = worker_loss[w, x_flex]
        fitting_parameters_flex, covariance_flex = curve_fit(gamma, x_flex, y_flex)
    
        x_350 = np.linspace(0, 349, 350).astype(int)
        k_flex = gamma(x_350, *fitting_parameters_flex)

        MSE_flex = 0
        for i in range(round):
            MSE_flex += (k_flex[i]-worker_loss[w, i])**2
        MSE_flex/=round
        y.append(MSE_flex)

predict_flex = len(k_flex)
for i in range(len(k_flex)):
    if k_flex[i] < 1.477759 and i > 50:
        predict_flex = i
        break


fig, axs = plt.subplots()
fig.suptitle('MSE per Epoch', fontsize=14)

axs.plot(x_350, y, linewidth=2, color = 'orange', label = 'MSE of Work#{}'.format(w))
# axs.axhline(y=0.004, linestyle = 'dashed', color = 'limegreen')
mpl.rcParams['lines.linewidth'] = 0.5

for spine in ['top', 'right']:
    axs.spines[spine].set_visible(False)

leg1 = axs.legend(fontsize=8, frameon=False)
leg1.set_draggable(state=True)

axs.set_ylabel('MSE')
axs.set_xlabel('Sample Round')
# axs.set_ylabel('MSE', color='dimgray')
# axs.set_xlabel('Sample Round', color='dimgray')

plt.show()
plt.savefig('./graph_experimental_results/MSE_per_epoch.png')
plt.savefig('./eps/MSE_per_epoch.eps')