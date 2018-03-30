import numpy as np

from math import *

import torch
import torch.nn as nn
from torch.nn import Parameter
from torch.nn import functional as F
import torch.optim
from torch.autograd import Variable

def problemGenerator(N=200, CLASSES = 8, FEATURES = 8, sigma = 1.0, sparseness = 0, imbalance = 0):
	rclass = np.random.randn(CLASSES)
	pclass = np.exp(-imbalance*rclass)
	pclass = pclass/np.sum(pclass)

	covariances = torch.FloatTensor(CLASSES, FEATURES, FEATURES).cuda().normal_() * sigma
	means = torch.FloatTensor(CLASSES, FEATURES).cuda().normal_()

	for i in range(FEATURES):
		if np.random.rand()<sparseness:
			means[:,i] = torch.mean(means[:,i],keepdim=True)
			for j in range(FEATURES):
				covariances[:,i,j] = torch.mean(covariances[:,i,j],keepdim=True)
				covariances[:,j,i] = torch.mean(covariances[:,j,i],keepdim=True)
	
	cls = torch.LongTensor(np.random.choice(np.arange(CLASSES), p=pclass, size=(N,))).cuda()
	z = torch.FloatTensor(N,1,FEATURES).cuda().normal_()

	i = torch.arange(N).cuda().long()
	xd = means[cls[i]] + torch.bmm(z, covariances[cls[i]])[:,0,:]
	yd = torch.zeros(N,CLASSES).cuda()
	yd[i,cls[i]] = 1

	return xd.cpu().numpy(), yd.cpu().numpy()
