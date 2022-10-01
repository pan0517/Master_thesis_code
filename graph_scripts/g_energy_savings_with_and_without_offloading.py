from xml.dom.minidom import TypeInfo
import matplotlib.pyplot as plt
import numpy as np
import pickle
import math

sc = [10, 20, 30, 40, 50, 60, 70, 80, 90]
y = [0.0529590452, 0.187128454464, 0.26, 0.3518898548, 0.48, 0.52, 0.5774561781, 0.68, 0.82]

for pos in ['right', 'top']:
    plt.gca().spines[pos].set_visible(False)
plt.plot(sc, y, linewidth=2, color = 'm', marker = 'o')
plt.grid(True)

plt.ylabel('Difference in Percentage')
plt.xlabel('Server Capacity')
plt.title('Energy Savings w/ and w/o Offloading (%)')
plt.savefig('energy_savings_with_and_without_offloading')
plt.savefig('./graph_experimental_results/energy_savings_with_and_without_offloading.png')
plt.savefig('./eps/denergy_savings_with_and_without_offloading.eps')
