from pty import slave_open
from xml.dom.minidom import TypeInfo
import matplotlib.pyplot as plt
import numpy as np
import pickle
import math

from utils.energy import EnergyCon_up
from utils.energy import EnergyCon_down
from utils.energy import EnergyComp_local
from utils.energy import EnergyCon_up_model

with open('filename.pickle', 'rb') as handle:
    dictE = pickle.load(handle)

id = 1

print(dictE['data_size'][id])

bandwidth = 2 
tmp = (dictE['power_up'][id] * (dictE['G_up']**2)) / (dictE['n_up'][id] * bandwidth)
tmp += 1
data_rate_up = bandwidth * math.log(tmp, 2)

time_offload = dictE['data_size'][id] / data_rate_up
time_offload /= 10000

time_upload_model = 10000 / data_rate_up
time_upload_model /= 10000

tmp = (dictE['power_down'][id] * (dictE['G_down']**2)) / (dictE['n_down'][id] * bandwidth)
tmp += 1
data_rate_down = bandwidth * math.log(tmp, 2)

#E_down = d['power_down'][id] * (a*8*d['data_size'][id]/data_rate_down)
time_down = 10000 / data_rate_down
time_down /= 10000

print("offloading要花的時間: {}, 上傳weight時間: {}, 廣播weight時間: {}".format(time_offload, time_upload_model, time_down))
print(EnergyCon_up(dictE, id, 1, 1), EnergyCon_up_model(dictE, id, 0.02, 1) )
print(dictE['data_size'][id], EnergyComp_local(dictE, id, 1), dictE['fi'][id])