"""

ECE 592 IOT Analytics
Project 4
GMM

"""
from sklearn.mixture import GaussianMixture as GMM
from matplotlib import pyplot as plt
import os
import pandas as pd
import numpy as np
import visualization

    
#Read Data
dir_1="C:/Users/priya/Desktop/IOT/project4/"
filename="psdiwaka.csv"
path=os.path.join(dir_1,filename)
data = pd.read_csv(path, header=None)
data.columns=["X1","X2","X3"]
X=data.values

#For different values of k fit the model and get maximum likelihood
n_components = np.arange(1,10)
models = [GMM(n, covariance_type='diag', random_state=22).fit(X)
          for n in n_components]

score=[m.score(X) for m in models]

#Select k based on maximum value of the maximum likelihood
k=n_components[np.argmax(np.asarray(score))]

#Plot maximum likelihood vs k
plt.figure()
plt.plot(n_components, score)
plt.title("Maximum Likelihood vs k")
plt.xlabel('k')
plt.ylabel('Maximum Likelihood')
#plt.savefig("gmm_Scorevsk.png")
plt.show()

#Obtain projections for each plane
XY_plane=X[:,[0,1]]
XZ_plane=X[:,[0,2]]
YZ_plane=X[:,[1,2]]

gmm = GMM(n_components=k, covariance_type='diag',random_state=22)
#fit the gaussian model

#XY plane
gmm.fit(XY_plane)
p=gmm.predict(XY_plane)
imgname="XY"
#visualize
visualization.visualize_2D_gmm(XY_plane, gmm.weights_, gmm.means_.T, np.sqrt(gmm.covariances_).T,p,imgname)

#XZ plane
gmm.fit(XZ_plane)
p=gmm.predict(XZ_plane)
imgname="XZ"
#visualize
visualization.visualize_2D_gmm(XZ_plane, gmm.weights_, gmm.means_.T, np.sqrt(gmm.covariances_).T,p,imgname)

#YZ plane
gmm.fit(YZ_plane)
p=gmm.predict(YZ_plane)
imgname="YZ"
#visualize
visualization.visualize_2D_gmm(YZ_plane, gmm.weights_, gmm.means_.T, np.sqrt(gmm.covariances_).T,p,imgname)


#3D projection
gmm.fit(X)
p=gmm.predict(X)

#visualize
visualization.visualize_3d_gmm(X, gmm.weights_, gmm.means_.T, np.sqrt(gmm.covariances_).T,p)

