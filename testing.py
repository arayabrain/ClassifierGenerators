import numpy as np

from math import *
from sklearn.metrics import roc_auc_score
from problem import problemGenerator
from architecture import NetworkSKL

def evalClassifier(classifier, alltrain_x, alltrain_y, alltest_x, alltest_y):
	acc = []
	auc = []
	
	for i in range(len(alltrain_x)):
		classes = np.unique(alltrain_y[i]).shape[0]
		test_y = np.zeros((alltest_x[i].shape[0], classes))
		x = np.arange(alltest_x[i].shape[0]).astype(np.int32)
		test_y[x, alltest_y[i][x]] = 1

		p = np.zeros((alltest_x[i].shape[0], classes))

		# This is a bit of a hack, because we want to use GPU+Batching to accelerate ensembling
		# for the classifier generator, but we need to ensemble the sklearn models as well
		# So we check if this is a network and if so just do one pass, otherwise ensemble
		#
		# Unfortunately, means some code duplication with normalizeAndProject
		
		if isinstance(classifier(), NetworkSKL):
			clf = classifier()
			clf.fit(alltrain_x[i],alltrain_y[i])
			p = p + clf.predict_proba(alltest_x[i])
		else:
			for j in range(30):
				proj = np.random.randn(alltrain_x[i].shape[1], 128)
				mu = np.mean(alltrain_x[i],axis=0,keepdims=True)
				std = np.std(alltrain_x[i],axis=0,keepdims=True)+1e-16
				
				train_x = np.matmul( (alltrain_x[i]-mu)/std, proj)
				test_x = np.matmul( (alltest_x[i]-mu)/std, proj)

				mu = np.mean(train_x,axis=0,keepdims=True)
				std = np.std(train_x,axis=0,keepdims=True)+1e-16

				train_x = (train_x-mu)/std
				test_x = (test_x-mu)/std
				
				clf = classifier()
				clf.fit(train_x, alltrain_y[i])
				p = p + clf.predict_proba(test_x)/30.0
			
		acc.append(np.mean(np.argmax(p,axis=1) == alltest_y[i]))
					
		auc.append(roc_auc_score(test_y, p))
		
	return np.mean(acc), np.mean(auc), np.std(acc)/sqrt(len(alltrain_x)), np.std(auc)/sqrt(len(alltrain_x))

def compareMethodsOnSet(methods, data_x, data_y, samples=100, N=10):
	CLASSES = np.unique(data_y).shape[0]
	FEATURES = data_x.shape[1]
	
	alltrain_x = []
	alltrain_y = []
	
	alltest_x = []
	alltest_y = []
	
	for i in range(samples):
		idx = np.random.permutation(data_x.shape[0])
		
		# Make sure we have examples of all of the classes included
		while np.unique(data_y[idx[0:N]]).shape[0]<CLASSES or np.unique(data_y[idx[N:]]).shape[0]<CLASSES:
			idx = np.random.permutation(data_x.shape[0])
		
		mu = np.mean(data_x,axis=0,keepdims=True)
		std = np.std(data_x,axis=0,keepdims=True) + 1e-16
		
		train_x = (data_x[idx[0:N]]-mu)/std
		test_x = (data_x[idx[N:]]-mu)/std
		
		train_y = data_y[idx[0:N]]
		test_y = data_y[idx[N:]]
		
		alltrain_x.append(train_x)
		alltrain_y.append(train_y)
		alltest_x.append(test_x)
		alltest_y.append(test_y)
	
	results = [evalClassifier(m, alltrain_x, alltrain_y, alltest_x, alltest_y) for m in methods]
				
	return results

def compareMethodsOnProblem(methods, classes, features, sigma, N=100, samples=20):		
	alltrain_x = []
	alltrain_y = []
	
	alltest_x = []
	alltest_y = []
	
	for i in range(samples):
		data_x, data_y = problemGenerator(N+400, classes, features, sigma)
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
	
	results = [evalClassifier(m, alltrain_x, alltrain_y, alltest_x, alltest_y) for m in methods]
				
	return results
