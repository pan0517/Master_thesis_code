from scipy.optimize import curve_fit
import pandas as pd
import numpy as np
import pickle
import scipy.stats as stats


def pareto(x, alpha, beta, s):
    rv = s*stats.pareto.pdf(x, alpha, -25, beta) + 1.477
    return rv

def gamma(x, alpha, beta, s):
    rv = s*stats.gamma.pdf(x, alpha, -5, beta) + 1.477
    return rv

def exponential(x, alpha, beta, c):
    rv = alpha * np.exp(-beta * x) + c
    #rv = a*stats.expon.pdf(x, 0, b) + 1.477
    return rv

def fitting_pareto(w, round, worker_loss):
    x = np.linspace(0, round-1, round)
    x_int = x.astype(int)
    y = worker_loss[w, x_int]
    fitting_parameters, covariance = curve_fit(pareto, x_int, y)

    rv = pareto(x_int, *fitting_parameters)
    
    MSE = 0
    for i in range(round):
        MSE += (rv[i]-worker_loss[w, i])**2
    MSE/=round

    return rv, MSE, fitting_parameters

def fitting_gamma(w, round, worker_loss):
    x = np.linspace(0, round-1, round)
    x_int = x.astype(int)
    y = worker_loss[w, x_int]
    fitting_parameters, covariance = curve_fit(gamma, x_int, y)

    rv = gamma(x_int, *fitting_parameters)
    cov = covariance

    return rv, cov

def earlystop_round(fitting_parameters):
    x = np.linspace(0, 349, 350)
    model = pareto(x, *fitting_parameters)
    
    predict_delta = len(model)
    for i in range(len(model)):
        if (model[i-1] - model[i])/model[i] < 0.000000001 and i > 50:
            predict_delta = i
            break

    predict = len(model)
    for i in range(len(model)):
        if model[i] < 1.477759 and i > 50:
            predict = i
            break
    return predict, predict_delta    

# def earlystop_round(model):
#     predict_delta = len(model)
#     for i in range(len(model)):
#         if (model[i-1] - model[i])/model[i] < 0.000000001 and i > 50:
#             predict_delta = i
#             break

#     predict = len(model)
#     for i in range(len(model)):
#         if model[i] < 1.477759 and i > 50:
#             predict = i
#             break
#     return predict, predict_delta    

