import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import pandas as pd
import numpy as np
import pickle
import scipy.stats as stats 

with open('pickle/worker_loss_num100_sc10.pickle', 'rb') as handle:
    worker_loss = pickle.load(handle)

with open('pickle/drop_round_num100_sc10.pickle', 'rb') as handle:
    drop_round = pickle.load(handle)
w = 4
round = 350
x = np.linspace(0, round-1, round)
x1 = x.astype(int)
x2 = np.linspace(1, round-2, round-1).astype(int)
y = worker_loss[w, x1]

def func(x, a, b, c):
    # rv = a * np.exp(-b * x) + c
    rv = a*stats.expon.pdf(x, 0, b) + 1.4
    return rv

popt, pcov = curve_fit(func, x1, y)

f = func(x1, *popt)
print(f)

# noncentral F distribution
def func_f(x, a, b, s):
    rv = s*stats.f.pdf(x, a, b) + 1.477
    return rv

popt1, pcov1 = curve_fit(func_f, x1, y)

f1 = func_f(x1, *popt1)
print(f1)

# log-logistic distribution
def func_l(x, a, b, s):
    rv = s*stats.fisk.pdf(x,a) + b
    return rv

popt2, pcov2 = curve_fit(func_l, x1, y)

f2 = func_l(x1, *popt2)
print(f2)

# pareto distribution
def pareto(x, alpha, beta, s):
    rv = s*stats.pareto.pdf(x, alpha, -25, beta) + 1.477
    #rv = s*(alpha*beta**alpha / x**(alpha+1)) + 1.477
    return rv

popt3, pcov3 = curve_fit(pareto, x1, y)

f3 = pareto(x1, *popt3)
print(f3)

# log distribution
def func_ln(x, a, b, s):
    rv = s*stats.lognorm.pdf(x, a) + b
    return rv

popt4, pcov4 = curve_fit(func_ln, x1, y)

f4 = func_ln(x1, *popt4)
print(f4)


plt.plot(x1, y, linewidth=2, color = 'limegreen', label = 'Losses of Worker#{}'.format(w))

plt.plot(x1, pareto(x1, *popt3), 'r',linestyle = '--',
         label='Pareto Distribution')

plt.plot(x1, func(x1, *popt), 'darkorange',linestyle = '-.',
         label='Exponential Distribution')

# plt.plot(x1, func_f(x1, *popt1), 'r',linestyle = 'dotted',
#          label='Noncentral F Distribution')

# plt.plot(x1, func_l(x1, *popt2), 'y',linestyle = 'dotted',
#          label='Log-logistic Distribution')

plt.plot(x1, func_ln(x1, *popt4), 'royalblue',linestyle = 'dotted',
         label='Log Distribution')

for pos in ['right', 'top']:
    plt.gca().spines[pos].set_visible(False)
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.title('Losses with Different Distribution Function')
plt.legend(frameon=False)
plt.show()
plt.savefig('./graph_experimental_results/regression_combination.png')
plt.savefig('./eps/regression_combination.eps')