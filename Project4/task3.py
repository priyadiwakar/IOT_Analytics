"""

ECE 592 IOT Analytics
Project 4
Task 3
DBSCAN

"""
from sklearn.neighbors import NearestNeighbors
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D 
from sklearn.cluster import DBSCAN
import os
import pandas as pd
import numpy as np


#Read Data
dir_1="C:/Users/priya/Desktop/IOT/project4/"
filename="psdiwaka.csv"
path=os.path.join(dir_1,filename)
data = pd.read_csv(path, header=None)
data.columns=["X1","X2","X3"]
X=data.values

#Distance graph
ns=6
nbrs = NearestNeighbors(n_neighbors=ns).fit(data)
distances, indices = nbrs.kneighbors(data)
distanceDec = sorted(distances[:,ns-1], reverse=True)
imgname="distancegraph"+str(ns)+".png"
plt.plot(indices[:,0], distanceDec)
plt.title("Distance Graph")
plt.xlabel("Data Index")
plt.ylabel("Distance")
#plt.savefig(imgname)
plt.show()

#epsilon chosen from distance graph
eps=24
# Compute DBSCAN
db = DBSCAN(eps=eps, min_samples=ns).fit(X)
core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True
labels = db.labels_

# Number of clusters in labels, ignoring noise if present.
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)

print('Estimated number of clusters: %d' % n_clusters_)


#plot clusters of data
unique_labels = set(labels)
colors = [plt.cm.Spectral(each)
          for each in np.linspace(0, 1, len(unique_labels))]

imgname="clusterfor"+str(ns)+"_"+str(eps)+".png"
fig = plt.figure()
ax = plt.axes(projection='3d')
ax.set_xlabel('Feature X1')
ax.set_ylabel('Feature X2')
ax.set_zlabel('Feature X3')
for k, col in zip(unique_labels, colors):
    if k!=-1:
        label="Cluster "+str(k)
        class_member_mask = (labels == k)
        
        xy = X[class_member_mask & core_samples_mask]
        ax.scatter3D(xy[:, 0], xy[:, 1], xy[:, 2],  'o',label=label)
ax.legend()
plt.title('Estimated number of clusters: %d' % n_clusters_)
#plt.savefig(imgname)
plt.show()