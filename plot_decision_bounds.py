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
from architecture import ClassifierGenerator, tovar, toivar

import warnings

def fxn():
    warnings.warn("deprecated", DeprecationWarning)
        
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    fxn()

def plotDecisionBoundary(x,y,net):
	x = (x-np.mean(x,axis=0,keepdims=True))/(1e-16 + np.std(x,axis=0,keepdims=True))

	x = x.reshape((1,1,x.shape[0],x.shape[1])).transpose(0,1,3,2)
	y = y.reshape((1,1,y.shape[0],y.shape[1])).transpose(0,1,3,2)

	trainset = np.concatenate([x,y],axis=2)

	xx,yy = np.meshgrid(np.arange(-3.0,3.05,0.05), np.arange(-3.0, 3.05, 0.05))
	XR = xx.shape[0]

	xx = xx.reshape((1,1,1,XR*XR))
	yy = yy.reshape((1,1,1,XR*XR))

	testset = np.concatenate([xx,yy],axis=2)
		
	p = np.exp(net.forward(tovar(trainset), tovar(testset), torch.cuda.FloatTensor(np.array([4]))).cpu().data.numpy())

	p = p.reshape((4,XR,XR)).transpose(1,2,0)
	xx = xx.reshape((XR,XR))
	yy = yy.reshape((XR,XR))

	colors = np.array([ [0.7,0.2,0.2], [0.2,0.7,0.2], [0.2, 0.2, 0.7], [0.7, 0.2, 0.7]])

	im = np.zeros((XR,XR,3))
	for j in range(4):
		im += p[:,:,j].reshape((XR,XR,1))*np.array(colors[j]).reshape((1,1,3))
	
	yl = np.argmax(y,axis=2)[0,0]
	
	plt.imshow(im,extent=[-3,3,3,-3])
	for j in range(4):
		plt.scatter(x[0,0,0,yl==j],x[0,0,1,yl==j], c=colors[j], edgecolors='k', lw=1,s=10)

	plt.xticks([])
	plt.yticks([])
	plt.xlim(-3,3)
	plt.ylim(-3,3)
                    			
net2_4_400_1 = ClassifierGenerator(2, 4, 384).cuda()
net2_4_400_1.load_state_dict(torch.load("models/classifier-generator-2-4-base.pth"))

net2_4_20_1 = ClassifierGenerator(2, 4, 384).cuda()
net2_4_20_1.load_state_dict(torch.load("models/classifier-generator-2-4-N20.pth"))

net2_4_100_1 = ClassifierGenerator(2, 4, 384).cuda()
net2_4_100_1.load_state_dict(torch.load("models/classifier-generator-2-4-N100.pth"))

net2_4_100_4 = ClassifierGenerator(2, 4, 384).cuda()
net2_4_100_4.load_state_dict(torch.load("models/classifier-generator-2-4-diff4.pth"))

net2_4_gen = ClassifierGenerator(2, 4, 384).cuda()
net2_4_gen.load_state_dict(torch.load("models/classifier-generator-2-4-general.pth"))

np.random.seed(12345)
torch.manual_seed(12345)

xd1, yd1 = problemGenerator(100, CLASSES=4, FEATURES=2, sigma=0.25)
xd2, yd2 = problemGenerator(100, CLASSES=4, FEATURES=2, sigma=1)

def rollGenerator(N,CLASSES):
	yl = np.random.randint(CLASSES,size=(N,))
	y = np.zeros((N,CLASSES))
	y[np.arange(N), yl[np.arange(N)]] = 1
	
	u = np.random.rand(N)
	v = np.random.randn(N,2)
	
	r = 0.5+2.5*u
	theta = (2*pi/CLASSES)*yl + 3*(2*pi/CLASSES)*u
	
	x = np.array([ r*np.cos(theta), r*np.sin(theta) ]).transpose(1,0) + 0.1*v
	
	return x,y
	
xd3, yd3 = rollGenerator(100,4)

plt.subplot(3,5,1)
plt.title("$N=20, \sigma=1$")
plt.ylabel("$\\sigma=0.25$")
plotDecisionBoundary(xd1,yd1,net2_4_20_1)

plt.subplot(3,5,2)
plt.title("$N=100, \\sigma=1$")
plotDecisionBoundary(xd1,yd1,net2_4_100_1)

plt.subplot(3,5,3)
plt.title("$N=400, \\sigma=1$")
plotDecisionBoundary(xd1,yd1,net2_4_400_1)

plt.subplot(3,5,4)
plt.title("$N=100, \\sigma=4$")
plotDecisionBoundary(xd1,yd1,net2_4_100_4)

plt.subplot(3,5,5)
plt.title("$N=20-400, \\sigma=0.25-4$")
plotDecisionBoundary(xd1,yd1,net2_4_gen)

plt.subplot(3,5,6)
plt.ylabel("$\\sigma=1$")
plotDecisionBoundary(xd2,yd2,net2_4_20_1)

plt.subplot(3,5,7)
plotDecisionBoundary(xd2,yd2,net2_4_100_1)

plt.subplot(3,5,8)
plotDecisionBoundary(xd2,yd2,net2_4_400_1)

plt.subplot(3,5,9)
plotDecisionBoundary(xd2,yd2,net2_4_100_4)

plt.subplot(3,5,10)
plotDecisionBoundary(xd2,yd2,net2_4_gen)

plt.subplot(3,5,11)
plt.ylabel("Roll")
plotDecisionBoundary(xd3,yd3,net2_4_20_1)

plt.subplot(3,5,12)
plotDecisionBoundary(xd3,yd3,net2_4_100_1)

plt.subplot(3,5,13)
plotDecisionBoundary(xd3,yd3,net2_4_400_1)

plt.subplot(3,5,14)
plotDecisionBoundary(xd3,yd3,net2_4_100_4)

plt.subplot(3,5,15)
plotDecisionBoundary(xd3,yd3,net2_4_gen)

plt.gcf().set_size_inches((15,9))
plt.savefig("decision.pdf")
