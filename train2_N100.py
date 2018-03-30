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

from architecture import ClassifierGenerator, NetworkSKL, tovar, toivar, normalizeAndProject
from problem import problemGenerator
from testing import evalClassifier, compareMethodsOnSet

def trainingStep(net, NTRAIN, min_difficulty = 1.0, max_difficulty = 1.0, min_sparseness = 0, max_sparseness = 0, min_imbalance = 0, max_imbalance = 0, feature_variation = True, class_variation = True, BS = 200):
	FEATURES = net.FEATURES
	CLASSES = net.CLASSES
	
	net.zero_grad()
	batch_mem = []
	batch_test = []
	batch_label = []
	class_count = []
	
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
		
		trainset = np.hstack([xd[0:NTRAIN],yd[0:NTRAIN]])
		testset = xd[NTRAIN:]
		labelset = yd[NTRAIN:]

		batch_mem.append(trainset)
		batch_test.append(testset)
		batch_label.append(labelset)
		class_count.append(classes)

	batch_mem = tovar(np.array(batch_mem).transpose(0,2,1).reshape(BS,1,FEATURES+CLASSES,NTRAIN))
	batch_test = tovar(np.array(batch_test).transpose(0,2,1).reshape(BS,1,FEATURES,100))
	batch_label = tovar(np.array(batch_label).transpose(0,2,1))
	class_count = torch.cuda.FloatTensor(np.array(class_count))
	
	net.zero_grad()
	p = net.forward(batch_mem, batch_test, class_count)
	loss = -torch.sum(p*batch_label,1).mean()
	loss.backward()
	net.adam.step()
	err = loss.cpu().data.numpy()[0]
	
	return err
		
net = ClassifierGenerator(FEATURES=2, CLASSES=4, NETSIZE=384).cuda()

difficulty_level = 1.0
errs = []

err = 0
err_count = 0

for i in range(40000):
	err += trainingStep(net, 100, min_difficulty = difficulty_level, max_difficulty = difficulty_level, feature_variation = False, class_variation = False)
	err_count += 1
			
	if err_count >= 50:
		err = err/err_count
		errs.append(err)
				
		f = open("training_curves/training2-N100.txt","a")
		f.write("%d %.6g %.6g\n" % (i, err, difficulty_level))
		f.close()
		
		err = 0
		err_count = 0
		
		torch.save(net.state_dict(),open("models/classifier-generator-2-4-N100.pth","wb"))
