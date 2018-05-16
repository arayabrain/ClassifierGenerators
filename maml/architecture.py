import numpy as np

from math import *

import torch
import torch.nn as nn
from torch.nn import Parameter
from torch.nn import functional as F
import torch.optim
from torch.autograd import Variable

def tovar(x):
	return Variable(torch.FloatTensor(x).cuda(), requires_grad = False)

def toivar(x):
	return Variable(torch.LongTensor(x).cuda(), requires_grad = False)
		
class Attention(nn.Module):
	def __init__(self, Nfield, Nquery, Nkey, Nval):
		super(Attention,self).__init__()
		
		self.field_to_key = nn.Conv1d(Nfield, Nkey, 1)
		self.field_to_val = nn.Conv1d(Nfield, Nval, 1)        
		self.query_to_key = nn.Conv1d(Nquery, Nkey, 1)
		
		self.nkey = Nkey
		self.nval = Nval
		
	def forward(self, field, query):
		s = field.size()
		fkeys = self.field_to_key(field)
		fvals = self.field_to_val(field)
		
		hkeys = self.query_to_key(query) # Batch * Key Size * Queries
		
		z = torch.bmm(fkeys.transpose(1,2), hkeys)/sqrt(self.nkey)
		w = torch.exp(torch.clamp(z,-30,30)) # Batch * # Keys * Queries
		w = w/(torch.sum(w,1,keepdim=True) + 1e-16)
		
		y = torch.bmm(fvals, w) # Batch * Val Size * Queries
		return y

class ClassifierGenerator(nn.Module):
	def __init__(self, FEATURES, CLASSES, NETSIZE=512):
		super(ClassifierGenerator,self).__init__()
		
		self.FEATURES = FEATURES
		self.CLASSES = CLASSES
		
		self.emb1a = nn.Conv1d(FEATURES,NETSIZE,1)
		self.emb2a = nn.Conv1d(NETSIZE,NETSIZE,1)

		self.emb1b = nn.Conv1d(FEATURES+CLASSES,NETSIZE,1)
		self.emb2b = nn.Conv1d(NETSIZE,NETSIZE,1)
		
		self.attn1a = Attention(NETSIZE,NETSIZE,32,NETSIZE//4)
		self.attn1b = Attention(NETSIZE,NETSIZE,32,NETSIZE//4)
		self.attn1c = Attention(NETSIZE,NETSIZE,32,NETSIZE//4)
		self.attn1d = Attention(NETSIZE,NETSIZE,32,NETSIZE//4)
		
		self.emb3a = nn.Conv1d(NETSIZE,NETSIZE,1)
		self.emb3b = nn.Conv1d(NETSIZE,NETSIZE,1)
		
		self.attn2a = Attention(NETSIZE,NETSIZE,32,NETSIZE//4)
		self.attn2b = Attention(NETSIZE,NETSIZE,32,NETSIZE//4)
		self.attn2c = Attention(NETSIZE,NETSIZE,32,NETSIZE//4)
		self.attn2d = Attention(NETSIZE,NETSIZE,32,NETSIZE//4)
		
		self.emb4a = nn.Conv1d(NETSIZE,NETSIZE,1)
		self.emb4b = nn.Conv1d(NETSIZE,NETSIZE,1)
		
		self.attn3a = Attention(NETSIZE,NETSIZE,32,NETSIZE//4)
		self.attn3b = Attention(NETSIZE,NETSIZE,32,NETSIZE//4)
		self.attn3c = Attention(NETSIZE,NETSIZE,32,NETSIZE//4)
		self.attn3d = Attention(NETSIZE,NETSIZE,32,NETSIZE//4)
		
		self.emb5a = nn.Conv1d(NETSIZE,NETSIZE,1)
		self.emb5b = nn.Conv1d(NETSIZE,NETSIZE,1)
		
		self.attn4a = Attention(NETSIZE,NETSIZE,32,NETSIZE//4)
		self.attn4b = Attention(NETSIZE,NETSIZE,32,NETSIZE//4)
		self.attn4c = Attention(NETSIZE,NETSIZE,32,NETSIZE//4)
		self.attn4d = Attention(NETSIZE,NETSIZE,32,NETSIZE//4)
		
		self.emb6 = nn.Conv1d(NETSIZE,NETSIZE,1)        
		self.emb7 = nn.Conv1d(NETSIZE,NETSIZE,1)
		self.emb8 = nn.Conv1d(NETSIZE,NETSIZE,1)
		self.emb9 = nn.Conv1d(NETSIZE,CLASSES,1)
		
		self.adam = torch.optim.Adam(self.parameters(), lr=1e-5)
		
	def forward(self, mem, test, classes):
		ts = test.size()
		
		mempts = mem.squeeze(1)
		testpts = test.squeeze(1)
		
		# Scaling here improves initial training speed
		x = 10*F.relu(self.emb2b(F.relu(self.emb1b(mempts))))
		y = 10*F.relu(self.emb2a(F.relu(self.emb1a(testpts))))
				
		z1 = self.attn1a(x,x)
		z2 = self.attn1b(x,x)
		z3 = self.attn1c(x,x)
		z4 = self.attn1d(x,x)
		
		z = torch.cat([z1,z2,z3,z4],1)
		x = x + self.emb3b(F.relu(self.emb3a(z)))
		
		z1 = self.attn2a(x,x)
		z2 = self.attn2b(x,x)
		z3 = self.attn2c(x,x)
		z4 = self.attn2d(x,x)
		
		z = torch.cat([z1,z2,z3,z4],1)
		xm = x + self.emb4b(F.relu(self.emb4a(z)))
		
		z1 = self.attn3a(xm,y)
		z2 = self.attn3b(xm,y)
		z3 = self.attn3c(xm,y)
		z4 = self.attn3d(xm,y)
		
		z = torch.cat([z1,z2,z3,z4],1)
		
		z = y + self.emb5b(F.relu(self.emb5a(z)))
		
		z1 = self.attn4a(xm,z)
		z2 = self.attn4b(xm,z)
		z3 = self.attn4c(xm,z)
		z4 = self.attn4d(xm,z)
		
		z = torch.cat([z1,z2,z3,z4],1)
		
		y = self.emb9(F.relu(self.emb8(F.relu(self.emb7(F.relu(self.emb6(z)))))))

		# Mask out classes that are known to not be present in the dataset
		mask = classes.unsqueeze(1).unsqueeze(2).expand(ts[0],self.CLASSES,ts[3])
		idx_y = torch.arange(self.CLASSES).cuda().unsqueeze(0).unsqueeze(2).expand(ts[0],self.CLASSES,ts[3])
		
		mask = Variable(-30*torch.ge(idx_y, mask).float(), requires_grad=False)
		y = y + mask 
		
		return F.log_softmax(y,dim=1)

class metaLinear(nn.Module):
    def __init__(self, Nin, Nout):
        super(metaLinear,self).__init__()
        
        self.betaW = Parameter(torch.FloatTensor(np.ones(1)*1e-2))
        self.betaB = Parameter(torch.FloatTensor(np.ones(1)*1e-2))
        
        self.layer = nn.Linear(Nin, Nout)
    
    def forward(self, x, meta=False):
        if meta:
            return F.linear(x,
                            weight=self.layer.weight + self.dweight, 
                            bias=self.layer.bias + self.dbias)
        else:
            self.dweight = torch.zeros_like(self.layer.weight)
            self.dbias = torch.zeros_like(self.layer.bias)
        
            return F.linear(x,
                            weight=self.layer.weight, 
                            bias=self.layer.bias)
    
    def update(self, loss):
        gradW, gradB = torch.autograd.grad([loss], [self.layer.weight, self.layer.bias], 
                                           retain_graph=True, create_graph=True)
        #gradW = Variable(gradW.data)
        #gradB = Variable(gradB.data)
        self.dweight = (self.dweight - self.betaW[0] * gradW)
        self.dbias = (self.dbias - self.betaB[0] * gradB)
        
class MAMLNet(nn.Module):
	def __init__(self, FEATURES, CLASSES, NETSIZE=128):
		super(MAMLNet,self).__init__()
		
		self.l1 = metaLinear(FEATURES,NETSIZE)
		self.l2 = metaLinear(NETSIZE,NETSIZE)
		self.l3 = metaLinear(NETSIZE,NETSIZE)	
		self.l4 = metaLinear(NETSIZE,NETSIZE)
		self.l5 = metaLinear(NETSIZE,CLASSES)
		
		self.steps = 1
		self.CLASSES = CLASSES
		self.FEATURES = FEATURES
		self.adam = torch.optim.Adam(self.parameters(), lr=5e-5)

	def loss(self, p, y):
		idx = torch.arange(p.size()[0]).long()
		
		return -torch.mean(p[idx,y[idx]])

	def forward(self, x, classes, meta=False):
		z = F.relu(self.l1(x,meta))
		z = F.relu(self.l2(z,meta))
		z = F.relu(self.l3(z,meta))
		z = F.relu(self.l4(z,meta))
		z = self.l5(z,meta)
		z[:,classes:] -= 30
		
		z = F.log_softmax(z,dim=1)
		
		return z

	def update(self, loss):
		for module in self.modules():
			if type(module) == metaLinear:
				module.update(loss)
			
	def fullpass(self, x_train, y_train, x_test, classes):
		p_train = self.forward(x_train, classes, meta=False)        
		loss = self.loss(p_train, y_train)
		self.update(loss)
		
		for i in range(self.steps-1):
			p_train = self.forward(x_train, classes, meta=True)
			loss = self.loss(p_train, y_train)
			self.update(loss)
		
		p_test = self.forward(x_test, classes, meta=True)
		
		return p_test
	
# Transform a dataset into the canonical number of features
def normalizeAndProject(xd, NTRAIN, FEATURES):
	feat = xd.shape[1]
	txd = tovar(xd)
	mu = torch.mean(txd[0:NTRAIN],0,keepdim=True)
	std = torch.std(txd[0:NTRAIN],0,keepdim=True) + 1e-6
	txd = (txd-mu)/std
	
	projection = torch.zeros(feat,FEATURES).cuda()
	projection.normal_()
	projection = projection / sqrt(FEATURES+feat)
	txd = torch.matmul(txd, projection)

	# Normalize before and after to prevent features with extreme scale 
	mu = torch.mean(txd[0:NTRAIN],0,keepdim=True)
	std = torch.std(txd[0:NTRAIN],0,keepdim=True) + 1e-6
	txd = (txd-mu)/std

	return txd.cpu().detach().numpy()

# Fake SKLearn wrapper for the network
class NetworkSKL():
	def __init__(self, net, ensemble=30):
		self.net = net
		self.ensemble = ensemble
	
	def fit(self, x, y):
		self.x = x
		self.y = y
		pass
	
	def predict_proba(self, x):
		train_x = self.x
		train_y = self.y
		test_x = x
		net = self.net
		ensemble = self.ensemble
		
		CLASSES = net.CLASSES
		FEATURES = net.FEATURES
		
		# This isn't necessarily accurate, for training data that doesn't contain one of each class, but we ensure that when making the training/test sets anyhow
		classes = np.unique(train_y).shape[0]

		trainlabels = np.zeros((train_x.shape[0],CLASSES))
		x = np.arange(train_x.shape[0])
		trainlabels[x,train_y[x]] = 1
	
		classtensor = torch.cuda.FloatTensor(np.array([classes]))
		
		traindata = []
		testdata = []
		for i in range(ensemble):
			# Need to transform everything together to make sure we use the same projection
			xd = np.vstack([train_x, test_x])
			xd = normalizeAndProject(xd, train_x.shape[0], FEATURES)
			ptrain_x = xd[0:train_x.shape[0]]
			ptest_x = xd[train_x.shape[0]:]
				
			traindata.append(tovar(np.hstack([ptrain_x,trainlabels]).reshape((1,1,train_x.shape[0],FEATURES+CLASSES)).transpose(0,1,3,2)))
			testdata.append(tovar(ptest_x.reshape((1,1,ptest_x.shape[0],FEATURES)).transpose(0,1,3,2)))
			
		traindata = torch.cat(traindata,0)
		testdata = torch.cat(testdata,0)

		preds = np.exp(net.forward(traindata, testdata, classes=classtensor).cpu().data.numpy()).mean(axis=0)
		
		# We need to strictly project to the right number of classes and maintain probabilities
		preds = preds.transpose(1,0)[:,:classes]
		preds = preds/np.sum(preds,axis=1,keepdims=True)
		return preds

# Fake SKLearn wrapper for the MAML network
class MAMLSKL():
	def __init__(self, net, ensemble=30):
		self.net = net
		self.ensemble = ensemble
	
	def fit(self, x, y):
		self.x = x
		self.y = y
		pass
	
	def predict_proba(self, x):
		train_x = self.x
		train_y = self.y
		test_x = x
		net = self.net
		ensemble = self.ensemble
		
		CLASSES = net.CLASSES
		FEATURES = net.FEATURES
		
		# This isn't necessarily accurate, for training data that doesn't contain one of each class, but we ensure that when making the training/test sets anyhow
		classes = np.unique(train_y).shape[0]
			
		traindata = []
		testdata = []
		preds = []
		for i in range(ensemble):
			# Need to transform everything together to make sure we use the same projection
			xd = np.vstack([train_x, test_x])
			xd = normalizeAndProject(xd, train_x.shape[0], FEATURES)
			ptrain_x = tovar(xd[0:train_x.shape[0]])
			ptest_x = tovar(xd[train_x.shape[0]:])
			
			p = np.clip(net.fullpass(ptrain_x, toivar(train_y), ptest_x, classes).cpu().data.numpy(),-50,0)
			if np.sum(np.isnan(p))>0:
				print(xd[0])
			preds.append(np.exp(p))
		
		preds = np.array(preds).mean(axis=0)
		# We need to strictly project to the right number of classes and maintain probabilities
		preds = preds[:,:classes]
		preds = preds/(np.sum(preds,axis=1,keepdims=True)+1e-8)
		
		return preds
