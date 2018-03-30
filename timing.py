import numpy as np

import torch
import torch.nn as nn
from torch.nn import Parameter
from torch.nn import functional as F
import torch.optim
from torch.autograd import Variable

from math import *

from architecture import ClassifierGenerator, NetworkSKL
from problem import problemGenerator
from testing import evalClassifier

from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
import xgboost as xgb

import time

def timeMethodsOnProblem(methods, classes, features, sigma, N=100, samples=20, NTEST=400):
    alltrain_x = []
    alltrain_y = []
    
    alltest_x = []
    alltest_y = []
    
    for i in range(samples):
        data_x, data_y = problemGenerator(N+NTEST, classes, features, sigma)
        data_y = np.argmax(data_y,axis=1)

        # Make sure we have examples of all of the classes included
        for j in range(classes):
            k = np.where(data_y[classes:]==j)[0][0] + classes
            data_x[[j,k]] = data_x[[k,j]]
            data_y[[j,k]] = data_y[[k,j]]

        mu = np.mean(data_x,axis=0,keepdims=True)
        std = np.std(data_x,axis=0,keepdims=True) + 1e-16

        train_x = (data_x[0:N]-mu)/std
        test_x = (data_x[N:]-mu)/std

        train_y = data_y[0:N]
        test_y = data_y[N:]

        alltrain_x.append(train_x)
        alltrain_y.append(train_y)
        alltest_x.append(test_x)
        alltest_y.append(test_y)

    t0 = time.time()
    results = [evalClassifier(m, alltrain_x, alltrain_y, alltest_x, alltest_y) for m in methods]
    t1 = time.time()
    
    return t1-t0
    
net = ClassifierGenerator(128,16,384).cpu()
net.load_state_dict(torch.load("models/classifier-generator-128-16.pth"))
netskl = NetworkSKL(net)

methods = [lambda: SVC(kernel='linear', probability=True), \
	lambda: SVC(kernel='rbf', probability=True), \
	RandomForestClassifier, \
	lambda: xgb.XGBClassifier(n_jobs = 12), \
	KNeighborsClassifier, \
	lambda: NetworkSKL(net,cuda=False), \
	lambda: NetworkSKL(net,cuda=True)]

values = [ [100, 400], [200,400], [400,400], [100,800], [100,1600], [100,3200], [100,6400] ]

f = open("results/timings.tex","wb")

f.write("$N_{train}$ & $N_{test}$ & LSVC & SVC & RF & XGB & KNN & CG (CPU) & CG (GPU) \\\\\n")
f.write("\\midrule\n")

f.close()

for val in values:
    f = open("results/timings.tex","a")
    nTrain = val[0]
    nTest = val[1]
    f.write("%d & %d " % (nTrain, nTest))
    for i in range(len(methods)):
        t = timeMethodsOnProblem([methods[i]], 16, 128, 1.0, nTrain, 10, nTest)/10.0
        f.write("& %.3g " % t)
    f.write("\\\\\n")
    f.close()
