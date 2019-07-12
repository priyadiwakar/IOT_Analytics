"""

ECE 592 IOT Analytics
Project 4
Task 1 
Hierarchical Clustering

"""

import pandas as pd 
from matplotlib import pyplot as plt
import os
from mpl_toolkits.mplot3d import Axes3D 
from scipy.cluster.hierarchy import linkage,dendrogram
from scipy.cluster.hierarchy import cophenet
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import fcluster


def dendrogram_1(*args, **kwargs):
    max_d = kwargs.pop('max_d', None)
    if max_d and 'color_threshold' not in kwargs:
        kwargs['color_threshold'] = max_d
    annotate_above = kwargs.pop('annotate_above', 0)

    ddata = dendrogram(*args, **kwargs)

    if not kwargs.get('no_plot', False):
        
        plt.title('Hierarchical Clustering Dendrogram (truncated)')
        plt.xlabel('Sample index or (Cluster size)')
        plt.ylabel('Distance')
        for i, d, c in zip(ddata['icoord'], ddata['dcoord'], ddata['color_list']):
            x = 0.5 * sum(i[1:3])
            y = d[1]
            if y > annotate_above:
                plt.plot(x, y, 'o', c=c)
                plt.annotate("%.3g" % y, (x, y), xytext=(0, -5),
                             textcoords='offset points',
                             va='top', ha='center')
        if max_d:
            plt.axhline(y=max_d, c='k')
    return ddata

#Read Data
dir_1="C:/Users/priya/Desktop/IOT/project4/"
filename="psdiwaka.csv"
path=os.path.join(dir_1,filename)
data = pd.read_csv(path, header=None)
data.columns=["X1","X2","X3"]

#Plot data
fig = plt.figure()
ax = plt.axes(projection='3d')

ax.scatter3D(data.X1, data.X2, data.X3, c='r', marker='o')
ax.set_title("Visualize Data")
ax.set_xlabel('Feature X1')
ax.set_ylabel('Feature X2')
ax.set_zlabel('Feature X3')
#plt.savefig("plotdata.png")
plt.show()

method='complete'
# create dendrogram
Z=linkage(data, method=method)

#Plot Dendrogram
plt.figure()
da=dendrogram_1(
    Z,
    truncate_mode='lastp',
    p=12,
    leaf_rotation=90.,
    leaf_font_size=12.,
    show_contracted=True,
    annotate_above=10)
#plt.savefig("dendrogram.png")
plt.show()

#Hierarchical CLustering
#Number of clusters k chosen from dendrogram
k=3
clusters=fcluster(Z, k, criterion='maxclust')

X=data.values
colors = ['r','g','b','c','k','y','m']

title="Visualize Data Clusters for k = "+str(k)
fig = plt.figure()
ax = plt.axes(projection='3d')
ax.set_title(title)
ax.set_xlabel('Feature X1')
ax.set_ylabel('Feature X2')
ax.set_zlabel('Feature X3')

for i in range(1,k+1):
    label="Cluster "+str(i)
    c=colors[i-1]
    idx = X[clusters==i]
    ax.scatter(idx[:,0], idx[:,1],idx[:,2], c=c,label=label)
ax.legend()
#plt.savefig("Hierarchical.png")
plt.show()
