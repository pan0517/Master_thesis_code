#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt

np1 = np.genfromtxt('results.csv', delimiter=',')
np2 = np.genfromtxt('results_original.csv', delimiter=',')
print(np1)
print(np2)

x = np.linspace(0, 349, 350)

acc = np1[1:, 3]
acc_o = np2[1:, 3]

print(len(acc))
print(acc_o)

for pos in ['right', 'top']:
    plt.gca().spines[pos].set_visible(False)
plt.plot(x, acc, color='limegreen', label='w/ Early Stopping')
plt.plot(x, acc_o, color='dimgray', linestyle = 'dashed', label='w/o Early Stopping')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Accuracy per Epoch')
plt.legend(loc = 'best', frameon=False)
plt.show()
plt.savefig('./eps/accuracy_per_epoch.eps')
plt.savefig('./graph_experimental_results/accuracy_per_epoch.png')