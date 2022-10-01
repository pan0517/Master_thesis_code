from re import A
import numpy as np
from numpy.core.fromnumeric import shape
import pandas as pd
import math
import pdb

#計算energy_comsuption公式
def EnergyCon_up(d, id, bandwidth, a):

    if(bandwidth == 0):
        data_rate_up = 0
        E_up = 0
    else:
        bandwidth *= 10

        tmp = (d['power_up'][id] * (d['G_up']**2)) / (d['n_up'][id] * bandwidth)
        tmp += 1
        data_rate_up = bandwidth * math.log(tmp, 2)

        E_up = d['power_up'][id] * (a*d['data_size'][id] / data_rate_up)
        E_up /= 10000
    
    #print('data rate: {}'.format(data_rate_up))

    return E_up

def EnergyCon_up_model(d, id, bandwidth, a):

    if(bandwidth == 0):
        data_rate_up = 0
        E_up_d = 0
    else:
        bandwidth *= 10

        tmp = (d['power_up'][id] * (d['G_up']**2)) / (d['n_up'][id] * bandwidth)
        tmp += 1
        data_rate_up = bandwidth * math.log(tmp, 2)

        E_up_d = d['power_up'][id] * (10**4 / data_rate_up)
        E_up_d /= 10000
    
    #print('data rate: {}'.format(data_rate_up))

    return E_up_d

def EnergyCon_down(d, id, bandwidth, a):

    if(bandwidth == 0):
        data_rate_down = 0
        E_down = 0
    else:
        bandwidth *= 10

        tmp = (d['power_down'][id] * (d['G_down']**2)) / (d['n_down'][id] * bandwidth)
        tmp += 1
        data_rate_down = bandwidth * math.log(tmp, 2)

        #E_down = d['power_down'][id] * (a*8*d['data_size'][id]/data_rate_down)
        E_down = d['power_down'][id] * (10000 / data_rate_down)
        E_down /= 10000

    return E_down

def EnergyComp_local(d, id, a):

    #E_local = (10**(-11)) * a * d['data_size'][id] * (d['fi'][id]**2)
    #E_local = (10**(-3)) * a * d['data_size'][id] # 新論文
    E_local = a * d['data_size'][id] * d['mi'][id]
    return E_local


#找出函數最小值
def ternary(d, id, bandwidth):

    L = 0
    R = 1
    while(R-L > 1e-5):
        midl = L + (R-L)/3
        midr = R - (R-L)/3
        v1 = EnergyCon_up(d, id, bandwidth, midl) + EnergyCon_down(d, id, bandwidth, midl)
        v2 = EnergyCon_up(d, id, bandwidth, midr) + EnergyCon_down(d, id, bandwidth, midr)
        if ( v1 < v2 ):
            R = midr
        else:
            L = midl
    return L
