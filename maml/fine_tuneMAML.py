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

import glob

from architecture import MAMLNet, MAMLSKL, tovar, toivar, normalizeAndProject
from problem import problemGenerator
from testing import evalClassifier, compareMethodsOnSet

def trainingStep(net, NTRAIN, NTEST, data_x, data_y, BS=200):
	FEATURES = net.FEATURES
	CLASSES = net.CLASSES

	net.zero_grad()
	err = []

	for i in range(BS):
		j = np.random.randint(len(data_x))
		feat = data_x[j].shape[1]
		classes = np.unique(data_y[j]).shape[0]
		
		xd = data_x[j].copy()
			
		# Data augmentation
		f_idx = np.random.permutation(feat)
		xd = xd[:,f_idx]
		
		c_idx = np.random.permutation(classes)
		yd = np.zeros((data_y[j].shape[0], classes))
		yd[np.arange(data_y[j].shape[0]), c_idx[data_y[j][np.arange(data_y[j].shape[0])]]] = 1

		idx = np.random.permutation(xd.shape[0])
		xd = xd[idx]
		yd = yd[idx]
				
		if classes<CLASSES:
			yd = np.pad(yd, ( (0,0), (0,CLASSES-classes)), 'constant', constant_values=0)
		xd = normalizeAndProject(xd, NTRAIN, FEATURES)		
		yd = np.argmax(yd,axis=1)
		
		trainset_x = tovar(xd[0:NTRAIN])
		trainset_y = toivar(yd[0:NTRAIN])
		testset = tovar(xd[NTRAIN:NTRAIN+NTEST])
		labelset = toivar(yd[NTRAIN:NTRAIN+NTEST])
		
		idx = torch.arange(NTEST).cuda().long()
		
		p = net.fullpass(trainset_x, trainset_y, testset, classes)
		loss = -torch.mean(p[idx,labelset[idx]])
		loss.backward()
		err.append(loss.cpu().detach().item())

	net.adam.step()

	return np.mean(err)

echocardio = np.load("data/echocardiogram.npz")
bloodtransfusion = np.load("data/bloodtransfusion.npz")
autism = np.load("data/autism.npz")

data_names = []
data_x = []
data_y = []	

for file in glob.glob("data/*.npz"):
	data = np.load(file)
	if np.unique(data['y']).shape[0]<=16:
		data_names.append(file[5:-4])
		data_x.append(data['x'].copy())
		data_y.append(data['y'].copy().astype(np.int32))

for didx in range(len(data_names)):	
	net = MAMLNet(FEATURES=128, CLASSES=16, NETSIZE=128).cuda()
	net.load_state_dict(torch.load("maml-128-16.pth"))
	tdx = []
	tdy = []
	
	for didx2 in range(len(data_names)):
		if didx2 != didx:
			if data_x[didx2].shape[0]>=120:
				tdx.append(data_x[didx2])
				tdy.append(data_y[didx2])
	
	ecol = 0
	for i in range(500):
		err = trainingStep(net, 100, 20, tdx, tdy)
		ecol = ecol + err
		if i%10 == 9:
			methods = [lambda: MAMLSKL(net)]
			results1 = compareMethodsOnSet(methods, echocardio['x'], echocardio['y'].astype(np.int32), samples=20)
			auc1 = results1[0][1]
			results2 = compareMethodsOnSet(methods, bloodtransfusion['x'], bloodtransfusion['y'].astype(np.int32), samples=20)
			auc2 = results2[0][1]
			results3 = compareMethodsOnSet(methods, autism['x'], autism['y'].astype(np.int32), samples=20)
			auc3 = results3[0][1]
			
			f = open("finetuning-%s.txt" % data_names[didx], "a")
			f.write("%d %.6g %.6g %.6g %.6g\n" % (i, ecol/10.0, auc1, auc2, auc3))
			f.close()
			ecol = 0
		
	torch.save(net.state_dict(), open("maml-%s.pth" % data_names[didx], "wb"))
