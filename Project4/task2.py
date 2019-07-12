"""

ECE 592 IOT Analytics
Project 4
Task 2
K Means Clustering

"""
from matplotlib import pyplot as plt
from sklearn.metrics import silhouette_score
from mpl_toolkits.mplot3d import Axes3D 
from sklearn.cluster import KMeans
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

#Elbow method and silhoutte score
ssd = []
silscore = []
K = range(2,15)
for k in K:
    km = KMeans(n_clusters=k)
    km = km.fit(X)
    labels = km.predict(X)
    ssd.append(km.inertia_)
    silhouette_avg = silhouette_score(X, labels)
    silscore.append(silhouette_avg)
  
plt.figure()
plt.plot(K, ssd, 'bx-')
plt.xlabel('k')
plt.ylabel('Sum of squared distances')
plt.title('Sum of squared distances vs k')
#plt.savefig("Elbowtest.png")
plt.show()

plt.figure()
plt.plot(K, silscore, 'bx-')
plt.xlabel('k')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Score vs k')
#plt.savefig("Elbowtest1.png")
plt.show()

#Determine k from Silhoutte score as not very clear elbow
k=np.argmax(silscore)
print("The maximum average silhouette_score is :", silscore[k],"For n_clusters =", K[k])
n_clusters=K[k]
#K means Clustering
kmeans = KMeans(n_clusters=n_clusters)
kmeans = kmeans.fit(X)
labels = kmeans.labels_
C = kmeans.cluster_centers_

colors = ['r','g','b','c','m']


title="Visualize Data Clusters for k = "+str(n_clusters)
fig = plt.figure()
ax = plt.axes(projection='3d')
ax.set_title(title)
ax.set_xlabel('Feature X1')
ax.set_ylabel('Feature X2')
ax.set_zlabel('Feature X3')

for i in range(n_clusters):
    label="Cluster "+str(i)
    c=colors[i]
    idx = X[labels==i]
    ax.scatter(idx[:,0], idx[:,1],idx[:,2], c=c,label=label)
ax.legend()
#plt.savefig("Hierarchical.png")
plt.show()