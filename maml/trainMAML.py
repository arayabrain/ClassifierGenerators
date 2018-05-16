import numpy as np
import matplotlib.pyplot as plt

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

from architecture import MAMLNet, MAMLSKL, tovar, toivar, normalizeAndProject
from problem import problemGenerator
from testing import evalClassifier, compareMethodsOnSet

def trainingStep(net, NTRAIN, min_difficulty = 1.0, max_difficulty = 1.0, min_sparseness = 0, max_sparseness = 0, min_imbalance = 0, max_imbalance = 0, feature_variation = True, class_variation = True, BS = 20):
	FEATURES = net.FEATURES
	CLASSES = net.CLASSES

	net.zero_grad()
	err = []

	for i in range(BS):
		if feature_variation:
			feat = np.random.randint(2.5*FEATURES) + FEATURES//2
		else:
			feat = FEATURES
		
		if class_variation:
			classes = np.random.randint(CLASSES-2) + 2
		else:
			classes = CLASSES
			
		xd,yd = problemGenerator(N=NTRAIN+100, FEATURES=feat, CLASSES=classes, 
								 sigma = np.random.rand()*(max_difficulty - min_difficulty) + min_difficulty,
								 sparseness = np.random.rand()*(max_sparseness - min_sparseness) + min_sparseness,
								 imbalance = np.random.rand()*(max_imbalance - min_imbalance) + min_imbalance)
		
		if classes<CLASSES:
			yd = np.pad(yd, ( (0,0), (0,CLASSES-classes)), 'constant', constant_values=0)
		xd = normalizeAndProject(xd, NTRAIN, FEATURES)
		
		yd = np.argmax(yd,axis=1)
		
		trainset_x = tovar(xd[0:NTRAIN])
		trainset_y = toivar(yd[0:NTRAIN])
		testset = tovar(xd[NTRAIN:])
		labelset = toivar(yd[NTRAIN:])
		
		idx = torch.arange(100).cuda().long()
		
		p = net.fullpass(trainset_x, trainset_y, testset, classes)
		loss = -torch.mean(p[idx,labelset[idx]])
		loss.backward()
		err.append(loss.cpu().detach().item())

	net.adam.step()

	return np.mean(err)
	
# Echocardiogram, blood transfusion, autism
echocardio = np.load("data/echocardiogram.npz")
bloodtransfusion = np.load("data/bloodtransfusion.npz")
autism = np.load("data/autism.npz")
	
net = MAMLNet(FEATURES=128, CLASSES=16, NETSIZE=128).cuda()

difficulty_level = 0.0125
errs = []

err = 0
err_count = 0

for i in range(100000):	
	err += trainingStep(net, 100, min_difficulty = difficulty_level * 0.5, max_difficulty = difficulty_level * 1.5)
	err_count += 1
	
	if i%10000 == 5000:
		torch.save(net.state_dict(),open("ckpt/maml-128-16-ckpt%d.pth" % i,"wb"))
		
	if err_count >= 50:
		err = err/err_count
		errs.append(err)
		
		"""
		methods = [lambda: MAMLSKL(net)]
		results1 = compareMethodsOnSet(methods, echocardio['x'], echocardio['y'].astype(np.int32), samples=200)
		auc1 = results1[0][1]
		results2 = compareMethodsOnSet(methods, bloodtransfusion['x'], bloodtransfusion['y'].astype(np.int32), samples=200)
		auc2 = results2[0][1]
		results3 = compareMethodsOnSet(methods, autism['x'], autism['y'].astype(np.int32), samples=200)
		auc3 = results3[0][1]
		"""
		f = open("maml128-16.txt","a")
		f.write("%d %.6g %.6g\n" % (i, err, difficulty_level))
		f.close()
	
		# Curriculum
		if err<0.7 and difficulty_level<0.2:
			difficulty_level *= 2.0
		
		err = 0
		err_count = 0
		
		torch.save(net.state_dict(),open("maml-128-16.pth","wb"))
