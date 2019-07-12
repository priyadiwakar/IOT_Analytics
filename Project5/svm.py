"""

ECE 592 IOT Analytics
Project 5
SVM

"""

import pandas as pd 
from matplotlib import pyplot as plt
import os
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
from mpl_toolkits import mplot3d

#Read Data
dir_1="C:/Users/priya/Desktop/IOT/project5/"
filename="psdiwaka.csv"
path=os.path.join(dir_1,filename)
data = pd.read_csv(path, header=None)
data.columns=["X1","X2","Y"]
X=np.append(np.reshape(data.X1.values,[-1,1]),np.reshape(data.X2.values,[-1,1]),axis=1)
Y=data.Y.values
labels=np.unique(data.Y.values)
colors = ['r','b']
#Plot data
fig = plt.figure()
ax = plt.axes()
ax.set_title("Visualize Data")
ax.set_xlabel('Feature X1')
ax.set_ylabel('Feature X2')

for i in labels:
    label="Label "+str(i)
    c=colors[i-1]
    idx = X[data.Y.values==i]
    ax.scatter(idx[:,0], idx[:,1], c=c,label=label)
ax.legend()
#plt.savefig("plotdata.png")
plt.show()

#Scale data
scaler = MinMaxScaler()
X = scaler.fit_transform(X)

#Range of C and gamma values for the initial coarse grid search
C_range = 2. ** np.arange(-1, 18,2)
gamma_range = 2. ** np.arange(-12, 8,2)

#Dictionary of C and gamma
param_grid = dict(gamma=gamma_range, C=C_range)

#Coarse Grid Search
grid = GridSearchCV(SVC(), param_grid=param_grid,scoring='accuracy', cv=StratifiedKFold(5,random_state=42, shuffle=True),return_train_score=True)
grid.fit(X, Y)
results=grid.cv_results_
accuracy=results['mean_test_score']*100

#maximum cross validation accuracy
m=np.max(accuracy)
print("Coarse Grid Search Results : ")
print("The maximum cross validation accuracy obtained is :",m)
print("\n")
print("The best parameters are : ",grid.best_params_)
print("\n")
print("\n")

#get all pairs of C and gamma values
params=results['params']

C=np.log2([l['C'] for l in params])
gamma=np.log2([l['gamma'] for l in params])


#3D plot to visualize accuracy
fig = plt.figure()
ax = plt.axes(projection='3d')
ax.set_xlabel('log2 gamma')
ax.set_ylabel('log2 C')
ax.set_zlabel('Accuracy')

surf=ax.plot_trisurf(gamma,C, accuracy, cmap='viridis',edgecolor='none')
fig.colorbar(surf)
plt.title("Coarse Grid Search")
#plt.savefig("grid1.png")
plt.show()

#Range of C and gamma values for the fine grid search
C_range = 2. ** np.arange(0, 2.5,0.25)
gamma_range = 2. ** np.arange(-1, 1,0.25)

#Dictionary of C and gamma
param_grid = dict(gamma=gamma_range, C=C_range)

#Fine Grid Search
grid = GridSearchCV(SVC(), param_grid=param_grid,scoring='accuracy', cv=StratifiedKFold(5,random_state=42, shuffle=True),return_train_score=True)
grid=grid.fit(X, Y)
results=grid.cv_results_
accuracy=results['mean_test_score']*100
#maximum cross validation accuracy
m=np.max(accuracy)
print("Fine Grid Search Results : ")
print("The maximum cross validation accuracy obtained is :",m)
print("\n")
print("The best parameters are : ",grid.best_params_)
print("\n")
print("\n")

#get all pairs of C and gamma values
params=results['params']

C=np.log2([l['C'] for l in params])
gamma=np.log2([l['gamma'] for l in params])

#3D plot to visualize accuracy
fig = plt.figure()
ax = plt.axes(projection='3d')
ax.set_xlabel('log2 gamma')
ax.set_ylabel('log2 C')
ax.set_zlabel('Accuracy')

surf=ax.plot_trisurf(gamma,C, accuracy, cmap='viridis',edgecolor='none')
fig.colorbar(surf)
plt.title("Fine Grid Search")
#plt.savefig("grid2.png")
plt.show()
