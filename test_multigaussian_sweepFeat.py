import numpy as np
import matplotlib.pyplot as plt

import glob
import sys

from math import *

import torch
import torch.nn as nn
from torch.nn import Parameter
from torch.nn import functional as F
import torch.optim
from torch.autograd import Variable

import time
import copy

import seaborn

from problem import problemGenerator
from architecture import ClassifierGenerator, NetworkSKL, tovar, toivar
from testing import evalClassifier, compareMethodsOnProblem

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import roc_auc_score
import xgboost as xgb

import warnings

def fxn():
    warnings.warn("deprecated", DeprecationWarning)
        
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    fxn()

net32_16 = ClassifierGenerator(32, 16, 384).cuda()
net32_16.load_state_dict(torch.load("models/classifier-generator-32-16.pth"))

net128_16 = ClassifierGenerator(128, 16, 384).cuda()
net128_16.load_state_dict(torch.load("models/classifier-generator-128-16.pth"))

methods = [ 
lambda: SVC(kernel='linear', C=1, probability=True), 
lambda: SVC(kernel='rbf', C=1, probability=True),
RandomForestClassifier,
lambda: xgb.XGBClassifier(n_jobs = 64),
KNeighborsClassifier,
lambda: NetworkSKL(net32_16, ensemble=30),
lambda: NetworkSKL(net128_16, ensemble=30)]

# Sweep Feat
f = open("results/Feat-128-16-100.txt","wb")
f.write("N;LSVC;;SVC;;RF;;XGB;;KNN(N=5);;$CG_{32,16}^{100,0.2}$;;$CG_{128,16}^{100,0.2}$;\n")
f.close()

for features in range(4, 384, 4):
	results = compareMethodsOnProblem(methods, 16, features, 0.4, N=100, samples=20)
	f = open("results/Feat-128-16-100.txt","a")
	f.write("%d" % (features))
	for i in range(len(results)):
		f.write(";%.3g;%.3g" % (results[i][1],results[i][3]))
	f.write("\n")
	f.close()
