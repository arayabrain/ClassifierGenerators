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

from architecture import ClassifierGenerator, NetworkSKL, tovar, toivar, normalizeAndProject
from problem import problemGenerator
from testing import evalClassifier, compareMethodsOnSet

def trainingStep(net, NTRAIN, NTEST, data_x, data_y, BS = 200):
	FEATURES = net.FEATURES
	CLASSES = net.CLASSES
	
	net.zero_grad()
	batch_mem = []
	batch_test = []
	batch_label = []
	class_count = []
	
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
		
		trainset = np.hstack([xd[0:NTRAIN],yd[0:NTRAIN]])
		testset = xd[NTRAIN:NTRAIN+NTEST]
		labelset = yd[NTRAIN:NTRAIN+NTEST]

		batch_mem.append(trainset)
		batch_test.append(testset)
		batch_label.append(labelset)
		class_count.append(classes)

	batch_mem = tovar(np.array(batch_mem).transpose(0,2,1).reshape(BS,1,FEATURES+CLASSES,NTRAIN))
	batch_test = tovar(np.array(batch_test).transpose(0,2,1).reshape(BS,1,FEATURES,NTEST))
	batch_label = tovar(np.array(batch_label).transpose(0,2,1))
	class_count = torch.cuda.FloatTensor(np.array(class_count))
	
	net.zero_grad()
	p = net.forward(batch_mem, batch_test, class_count)
	loss = -torch.sum(p*batch_label,1).mean()
	loss.backward()
	net.adam.step()
	err = loss.cpu().data.numpy()[0]
	
	return err

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
	net = ClassifierGenerator(FEATURES=128, CLASSES=16, NETSIZE=384).cuda()
	net.load_state_dict(torch.load("models/classifier-generator-128-16.pth"))
	
	tdx = []
	tdy = []
	
	for didx2 in range(len(data_names)):
		if didx2 != didx:
			if data_x[didx2].shape[0]>=120:
				tdx.append(data_x[didx2])
				tdy.append(data_y[didx2])
	
	for i in range(20):
		err = trainingStep(net, 100, 20, tdx, tdy)
		f = open("training_curves/finetuning-%s.txt" % data_names[didx], "a")
		f.write("%d %.6g\n" % (i, err))
		f.close()
		
	torch.save(net.state_dict(), open("models/classifier-generator-128-16-%s.pth" % data_names[didx], "wb"))
