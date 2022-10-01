from xml.dom.minidom import TypeInfo
import matplotlib.pyplot as plt
import numpy as np
import pickle

from utils.energy import EnergyCon_up
from utils.energy import EnergyCon_down
from utils.energy import EnergyComp_local
from utils.energy import EnergyCon_up_model

with open('pickle/filename_num100_sc50.pickle', 'rb') as handle:
    dictE = pickle.load(handle)

with open('pickle/worker_loss_num100_sc50.pickle', 'rb') as handle:
    worker_loss = pickle.load(handle)

with open('pickle/drop_round_num100_sc50.pickle', 'rb') as handle:
    drop_round = pickle.load(handle)

with open('pickle/predict_drop_list_num100_sc50.pickle', 'rb') as handle:
    predict_drop_list = pickle.load(handle)

with open('pickle/pareto_model_num100_sc50.pickle', 'rb') as handle:
    pareto_model = pickle.load(handle)

with open('pickle/upload_round_num100_sc50.pickle', 'rb') as handle:
    upload_round = pickle.load(handle)

with open('pickle/worker_selected_round_num100_sc50.pickle', 'rb') as handle:
    worker_selected_round = pickle.load(handle)

with open('pickle/worker_energy_consump_num100_sc50.pickle', 'rb') as handle:
    worker_energy_consump = pickle.load(handle)

with open('pickle/upload_list_num100_sc50.pickle', 'rb') as handle:
    upload_list = pickle.load(handle)

with open('pickle/predict_drop_list_when_upload_num100_sc50.pickle', 'rb') as handle:
    predict_drop_list_when_upload = pickle.load(handle)

print(pareto_model)
print(upload_list)
print(len(upload_list))
print(predict_drop_list_when_upload)
print(drop_round)
print(predict_drop_list)