from turtle import color
import matplotlib.pyplot as plt
import matplotlib
from scipy.optimize import curve_fit
import pandas as pd
import numpy as np
import pickle
import scipy.stats as stats 

with open('pickle/worker_loss_num100_sc50.pickle', 'rb') as handle:
    worker_loss = pickle.load(handle)

with open('pickle/drop_round_num100_sc50.pickle', 'rb') as handle:
    drop_round = pickle.load(handle)

round = 5
x_flex = np.linspace(0, round-1, round).astype(int)

x_50 = np.linspace(0, 24, 25).astype(int)
x_100 = np.linspace(0, 49, 50).astype(int)
x_250 = np.linspace(0, 99, 100).astype(int)
x_350 = np.linspace(0, 349, 350).astype(int)

w = 5
y_flex = worker_loss[w, x_flex]
y_50 = worker_loss[w, x_50]
y_100 = worker_loss[w, x_100]
y_250 = worker_loss[w, x_250]
y_350 = worker_loss[w, x_350]
# print(y-1.477)

def pareto(x, alpha, beta, s):
    rv = s*stats.pareto.pdf(x, alpha, -25, beta) + 1.477
    #rv = s*(alpha*beta**alpha / x**(alpha+1)) + 1.477
    return rv

fitting_parameters_flex, covariance_flex = curve_fit(pareto, x_flex, y_flex)
fitting_parameters_50, covariance_50 = curve_fit(pareto, x_50, y_50)
fitting_parameters_100, covariance_100 = curve_fit(pareto, x_100, y_100)
fitting_parameters_250, covariance_250 = curve_fit(pareto, x_250, y_250)
fitting_parameters_350, covariance_350 = curve_fit(pareto, x_350, y_350)

perr_50 = np.sqrt(np.diag(covariance_50))
perr_100 = np.sqrt(np.diag(covariance_100))
perr_250 = np.sqrt(np.diag(covariance_250))
perr_350 = np.sqrt(np.diag(covariance_350))
# alpha, beta, s= fitting_parameters
# print(alpha, beta, s)

# print("cov50:{}, 100:{}, 250:{}, 350:{}".format(covariance_50, covariance_100, covariance_250, covariance_350))
# print("per50:{}, 100:{}, 250:{}, 350:{}".format(perr_50, perr_100, perr_250, perr_350))

k_flex = pareto(x_350, *fitting_parameters_flex)
k_350 = pareto(x_350, *fitting_parameters_350)
k_50 = pareto(x_350, *fitting_parameters_50)
k_100 = pareto(x_350, *fitting_parameters_100)
k_250 = pareto(x_350, *fitting_parameters_250)
# print(k)

MSE_50 = 0
for i in range(25):
    MSE_50 += (k_50[i]-worker_loss[w, i])**2
MSE_50/=25

MSE_100 = 0
for i in range(50):
    MSE_100 += (k_100[i]-worker_loss[w, i])**2
MSE_100/=50

MSE_250 = 0
for i in range(100):
    MSE_250 += (k_250[i]-worker_loss[w, i])**2
MSE_250/=100

MSE_350 = 0
for i in range(350):
    MSE_350 += (k_350[i]-worker_loss[w, i])**2
MSE_350/=350

MSE_flex = 0
for i in range(round):
    MSE_flex += (k_flex[i]-worker_loss[w, i])**2
MSE_flex/=round

print('MSE_5:{}, 50:{}, 100:{}, 250:{}, 350:{}'.format(MSE_flex, MSE_50, MSE_100, MSE_250, MSE_350))

predict = len(k_350)
for i in range(len(k_350)):
    if (k_350[i-1] - k_350[i])/k_350[i] < 0.000000001 and i > 50:
        predict = i
        break

predict_flex = len(k_flex)
for i in range(len(k_flex)):
    if k_flex[i] < 1.477759 and i > 50:
        predict_flex = i
        break
predict2 = len(k_350)
for i in range(len(k_350)):
    if k_350[i] < 1.477759 and i > 50:
        predict2 = i
        break
predict3 = len(k_50)
for i in range(len(k_50)):
    if k_50[i] < 1.477759 and i > 50:
        predict3 = i
        break
predict4 = len(k_100)
for i in range(len(k_100)):
    if k_100[i] < 1.477759 and i > 50:
        predict4 = i
        break
predict5 = len(k_250)
for i in range(len(k_250)):
    if k_250[i] < 1.477759 and i > 50:
        predict5 = i
        break

d = drop_round[w].astype(int)
print("預測1drop掉的round: {}, 預測2drop掉的round: {}, 實際drop掉的round: {}".format(predict, predict2, d))

plt.rc('font', size=10)
fig, axs = plt.subplots(2,2, figsize=(8,6))

fig.suptitle('Regression with Different Number of Samples', fontsize=14)

axs[0][0].plot(x_350, y_350, linewidth=2, color = 'limegreen', label = 'Loss of Worker#{}'.format(w))
axs[0][1].plot(x_350, y_350, linewidth=2, color = 'limegreen', label = 'Loss of Worker#{}'.format(w))
axs[1][0].plot(x_350, y_350, linewidth=2, color = 'limegreen', label = 'Loss of Worker#{}'.format(w))
axs[1][1].plot(x_350, y_350, linewidth=2, color = 'limegreen', label = 'Loss of Worker#{}'.format(w))

axs[0][0].plot(x_350, pareto(x_350, *fitting_parameters_flex), linestyle = 'dotted', label='Pareto Distribution')
axs[0][1].plot(x_350, pareto(x_350, *fitting_parameters_50), linestyle = 'dotted', label='Pareto Distribution')
axs[1][0].plot(x_350, pareto(x_350, *fitting_parameters_100), linestyle = 'dotted', label='Pareto Distribution')
axs[1][1].plot(x_350, pareto(x_350, *fitting_parameters_250), linestyle = 'dotted', label='Pareto Distribution')

# title
axs[0][0].set_title('Sample Rounds = {}'.format(round), fontsize=10)
axs[0][1].set_title('Sample Rounds = 25', fontsize=10)
axs[1][0].set_title('Sample Rounds = 50', fontsize=10)
axs[1][1].set_title('Sample Rounds = 100', fontsize=10)


axs[1][1].axvline(x = d, linestyle = 'dashed', color = 'r', label = 'Real Convergent Round')
# predict drop round
axs[1][1].axvline(x = predict5, linestyle = 'dashdot', color = 'y', label = 'Prediction of Convergent Round')
axs[0][0].axvline(x = predict_flex, linestyle = 'dashdot', color = 'y', label = 'Prediction of Convergent Round')
axs[0][1].axvline(x = predict3, linestyle = 'dashdot', color = 'y', label = 'Prediction of Convergent Round')
axs[1][0].axvline(x = predict4, linestyle = 'dashdot', color = 'y', label = 'Prediction of Convergent Round')

for spine in ['top', 'right']:
    axs[0][0].spines[spine].set_visible(False)
    axs[0][1].spines[spine].set_visible(False)
    axs[1][0].spines[spine].set_visible(False)
    axs[1][1].spines[spine].set_visible(False)

tmp = 5.9
leg1 = axs[0][0].legend(fontsize=tmp, frameon=False)
leg1.set_draggable(state=True)
leg2 = axs[0][1].legend(fontsize=tmp, frameon=False)
leg2.set_draggable(state=True)
leg3 = axs[1][0].legend(fontsize=tmp, frameon=False)
leg3.set_draggable(state=True)
leg4 = axs[1][1].legend(fontsize=tmp, frameon=False, bbox_to_anchor=(0.7,1))
leg4.set_draggable(state=True)


axs[0][0].set_ylabel('Loss', color='dimgray')
axs[0][1].set_ylabel('Loss', color='dimgray')
axs[1][0].set_ylabel('Loss', color='dimgray')
axs[1][1].set_ylabel('Loss', color='dimgray')
axs[0][0].set_xlabel('Epoch', color='dimgray')
axs[0][1].set_xlabel('Epoch', color='dimgray')
axs[1][0].set_xlabel('Epoch', color='dimgray')
axs[1][1].set_xlabel('Epoch', color='dimgray')

plt.subplots_adjust(wspace=0.25, hspace=0.4)
plt.savefig('./graph_experimental_results/regression_with_different_samples.png')
plt.savefig('./eps/regression_with_different_samples.eps')
plt.show()