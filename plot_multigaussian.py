import numpy as np
import matplotlib.pyplot as plt
import seaborn
import pandas as pd

seaborn.set()

def plotCurve(xvar, data, label, linestyle, linewidth):
    nd = np.array(data)
    mu = nd[:,0]
    std = nd[:,1]
    
    plt.errorbar(xvar, mu, std, label=label, ls=linestyle, lw=linewidth)

plt.subplot(1,3,1)    
plt.title("Dependency on amount training data")
points = pd.read_csv("results/N-128-16-100.txt",sep=";")
labels = np.array(points.columns[1::2])

order = [5,6,0,1,2,3,4]
Nv = np.array(points.iloc[:,0])
points = np.array(points.iloc[:,1:])
labels = labels[order]

for i in range(labels.shape[0]):
	if i>=2:
		style='--'
		width=1
	else:
		style='-'
		width=2
	plotCurve(Nv, points[:,2*order[i]:2*order[i]+2], labels[i], style, width)
 
plt.ylabel("AUC")
plt.xlabel("$N_{train}$")
plt.legend()

plt.subplot(1,3,2)    
plt.title("Dependency on number of features")
points = pd.read_csv("results/Feat-128-16-100.txt",sep=";")
labels = np.array(points.columns[1::2])

order = [5,6,0,1,2,3,4]
Nv = np.array(points.iloc[:,0])
points = np.array(points.iloc[:,1:])
labels = labels[order]

for i in range(labels.shape[0]):
	if i>=2:
		style='--'
		width=1
	else:
		style='-'
		width=2
	plotCurve(Nv, points[:,2*order[i]:2*order[i]+2], labels[i], style, width)
 
plt.ylabel("AUC")
plt.xlabel("$N_F$")
plt.legend()

plt.subplot(1,3,3)    
plt.title("Dependency on problem difficulty")
points = pd.read_csv("results/Sigma-128-16-100.txt",sep=";")
labels = np.array(points.columns[1::2])

order = [5,6,7,8,0,1,2,3,4]
Nv = np.array(points.iloc[:,0])
points = np.array(points.iloc[:,1:])
labels = labels[order]

for i in range(labels.shape[0]):
	if i>=4:
		style='--'
		width=1
	else:
		style='-'
		width=2
	plotCurve(Nv, points[:,2*order[i]:2*order[i]+2], labels[i], style, width)
 
plt.ylabel("AUC")
plt.xlabel("$\sigma$")
plt.legend()

plt.gcf().set_size_inches((18,6))
plt.savefig("sweeps.pdf")
