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

from architecture import ClassifierGenerator, NetworkSKL, MAMLSKL, MAMLNet, tovar, toivar
from testing import evalClassifier, compareMethodsOnSet

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

dataset_descriptions = {
	"data/immunotherapy.npz": "Immunotherapy\\cite{khozeimeh2017expert, khozeimeh2017intralesional}",
	"data/foresttype.npz": "Forest type\\cite{johnson2012using}",
	"data/winetype.npz" : "Wine type\\cite{forina1990parvus}",
	"data/cryotherapy.npz" : "Cryotherapy\\cite{khozeimeh2017expert, khozeimeh2017intralesional}",
	"data/chronic-kidney.npz" : "Chronic kidney\\cite{chronickidney}",
	"data/echocardiogram.npz" : "Echocardiogram\\cite{echocardiogram}",
	"data/haberman.npz" : "Haberman\\cite{haberman1976generalized}",
	"data/iris.npz" : "Iris\\cite{fisher1936use}",
	"data/hcc-survival.npz" : "HCC Survival\\cite{santos2015new}",
	"data/horse-colic.npz" : "Horse Colic\\cite{horsecolic}",
	"data/lung-cancer.npz" : "Lung cancer\\cite{hong1991optimal}",
	"data/hepatitis.npz" : "Hepatitis\\cite{hepatitis}",
	"data/bloodtransfusion.npz" : "Blood transfusion\\cite{yeh2009knowledge}",
	"data/autism.npz" : "Autism\\cite{thabtah2017autism}",
	"data/cervical_cancer.npz" : "Cervical cancer\\cite{fernandes2017transfer}",
        "data/winequality_red.npz" : "Wine quality (red)",
        "data/winequality_white.npz" : "Wine quality (white)",
        "data/dermatology.npz" : "Dermatology"
	}
	
f = open("results/auctable_maml_ft.tex","w")
f.write("Dataset & N & $\sigma$ & FTMAML \\\\ \n")
f.write("\\midrule\n")
f.close()

avg10 = []
avg20 = []
avg50 = []

for file in glob.glob("data/*.npz"):
	mamlnet = MAMLNet(128,16,128).cuda()
	mamlnet.load_state_dict(torch.load("maml-%s.pth" % file[5:-4]))
	
	methods = [ lambda: MAMLSKL(mamlnet) ]
	data = np.load(file)
	print(file)
	data_x = data['x'].astype(np.float32)
	data_y = data['y'].astype(np.int32)
	
	if np.unique(data_y).shape[0]<=16:
		f = open("results/auctable_maml_ft.tex","a")
		f.write("\\multirow{3}{*}{%s} " % (dataset_descriptions[file]))
				
		if data_x.shape[0]>=20:
			f.write("& 10 ")
			results10 = np.array(compareMethodsOnSet(methods, data_x, data_y, N=10, samples=800))
			stdev = np.mean(results10[:,3])
			maxval = np.max(results10[:,1])
			f.write("& %.3g " % stdev)
			for i in range(results10.shape[0]):
				if abs(maxval-results10[i,1])<stdev:
					f.write("& \\bf{%.3g} " % (results10[i,1]))
				else:
					f.write("& %.3g " % (results10[i,1]))
			f.write("\\\\ \n")
			avg10.append(results10)
			
		if data_x.shape[0]>=60:
			f.write("& 50 ")
			results50 = np.array(compareMethodsOnSet(methods, data_x, data_y, N=50, samples=800))
			stdev = np.mean(results50[:,3])
			maxval = np.max(results50[:,1])
			f.write("& %.3g " % stdev)
			for i in range(results50.shape[0]):
				if abs(maxval-results50[i,1])<stdev:
					f.write("& \\bf{%.3g} " % (results50[i,1]))
				else:
					f.write("& %.3g " % (results50[i,1]))
                        
			f.write("\\\\ \n")
			avg50.append(results50)
		f.write("\\midrule \n")
		f.close()		        
        
	print("   %d,%d" % (data_x.shape[0], np.unique(data_y).shape[0]))

avg10 = np.array(avg10).mean(axis=0)
avg50 = np.array(avg50).mean(axis=0)

f = open("results/auctable_maml_ft.tex","a")
f.write("\\midrule \n")
f.write("\multirow{3}{*}{Average} & 10 & %.3g " % (avg10[i,3]/sqrt(18)))
for i in range(len(results10)):
	f.write("& %.3g " % (avg10[i,1]))
f.write("\\\\ \n & 50 & %.3g " % (avg50[i,3]/sqrt(18)))
for i in range(len(results50)):
	f.write("& %.3g " % (avg50[i,1]))
f.write("\\\\ \n")
f.close()
