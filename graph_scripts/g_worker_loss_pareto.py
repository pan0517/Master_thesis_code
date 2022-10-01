import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import pandas as pd
import numpy as np
import pickle
import scipy.stats as stats 

with open('pickle/worker_loss_num100_sc50.pickle', 'rb') as handle:
    worker_loss = pickle.load(handle)

with open('pickle/drop_round_num100_sc50.pickle', 'rb') as handle:
    drop_round = pickle.load(handle)
w = 8
round = 350
x = np.linspace(0, round-1, round)
x1 = x.astype(int)
x2 = np.linspace(1, round-2, round-1).astype(int)
y = worker_loss[w, x1]


# pareto distribution
def pareto(x, alpha, beta, s):
    rv = s*stats.pareto.pdf(x, alpha, -25, beta) + 1.477
    return rv

popt3, pcov3 = curve_fit(pareto, x1, y)

f3 = pareto(x1, *popt3)
print(f3)

predict = len(f3)
for i in range(len(f3)):
    if f3[i] <= 1.47729 and i > 50:
        predict = i
        break

d = drop_round[w].astype(int)
plt.axvline(x = d, linestyle = 'dashed', color = 'r', label = 'Real Convergent Round')
plt.axvline(x = predict, linestyle = 'dashed', color = 'y', label = 'Prediction of Convergent Round')

plt.plot(x1, y, linewidth=2, color = 'limegreen', label = 'Losses of Worker#{}'.format(w))
plt.plot(x1, pareto(x1, *popt3), 'blue',linestyle = 'dotted',
         label='Pareto Distribution')

for pos in ['right', 'top']:
    plt.gca().spines[pos].set_visible(False)
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.title('Losses w/ the Pareto Regression')
plt.legend(loc = 'best', frameon=False)
plt.show()
plt.savefig('worker_loss_pareto.png')
plt.savefig('./graph_experimental_results/worker_loss_pareto.png')
plt.savefig('./eps/worker_loss_pareto.eps')