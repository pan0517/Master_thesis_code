from xml.dom.minidom import TypeInfo
import matplotlib.pyplot as plt
import numpy as np
import pickle
import math

sc = [10, 20, 30, 40, 50, 60, 70, 80, 90]
#y = [0.0544, 0.0643, 0.0654, 0.17627]
percentage = []
y_gt = [464737.9757, 389148.2213, 316117.2273, 255614.5837, 206206.1642, 162790.5410, 131062.5675, 107461.4635, 85887.0581]
y_pd = [541197.4360, 459963.9104, 416594.8749, 356321.7428, 325823.1734, 276933.9378, 247475.7508, 170113.9938, 105508.9635]
for i in range(len(y_pd)):
    percentage.append((y_pd[i]-y_gt[i]))

x = np.arange(len(sc))
width = 0.3
plt.rcParams["figure.figsize"] = (8, 6)
for pos in ['right', 'top']:
    plt.gca().spines[pos].set_visible(False)
plt.bar(x, y_pd, width, color='green', hatch='o', label='algorithm')
plt.bar(x + width, y_gt, width, color='blue', hatch='/', label='ground truth')
plt.plot(x, percentage, marker='^', color='red', label='difference')
plt.xticks(x + width / 2, sc)
plt.xlabel('Server Capacity')
plt.ylabel('Energy Consumption (J)')
plt.title('Difference of E.C. between Ground Truth and Our Approach')
#plt.legend(bbox_to_anchor=(1,1), loc='upper left')
plt.legend(loc = 'best', frameon=False)

plt.savefig('./graph_experimental_results/difference_of_energy_consumption_between_ground_truth_and_our_approach.png')
plt.savefig('./eps/difference_of_energy_consumption_between_ground_truth_and_our_approach.eps')

