"""
ECE 592 IOT ANALYTICS

Project 3 Task 2 and Task 5

Simple Moving Average Model (SMA)
"""

import pandas as pd 
from matplotlib import pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error
from math import sqrt
import os

#Read Data
dir_1="C:/Users/priya/Desktop/IOT/project3/"
filename="psdiwaka.csv"
path=os.path.join(dir_1,filename)
data = pd.read_table(path, header=None,sep=" ")
data.columns=["Values"]
train_size = int(len(data) * 0.75)
train, test = data[0:train_size], data[train_size:len(data)]

#Run the SMA model for different k values
Y_SMA=train.copy()
rmselist=[]
kvalues=np.asarray(range(2,290)).reshape(-1)
for i in kvalues:
    output="SMA"+str(i)
    Y_SMA[output] = Y_SMA["Values"].rolling(window=i).mean().shift()
    
    #RMSE
    rmse=sqrt(mean_squared_error(Y_SMA.dropna()["Values"],Y_SMA.dropna()[output]))
    #print("For k value = {} RMSE is : {}".format(i,round(rmse,4)))
    rmselist.append(rmse)
 
#Find k for which RMSE is minimum
minrmse=np.argmin(np.asarray(rmselist))
k=kvalues[minrmse]
print("\n\nThe best k value for the Simple Moving Average Model based on lowest RMSE of {} is : {}".format(round(rmselist[minrmse],4),k))

#Plot RMSE vs K
plt.plot(kvalues,rmselist)
plt.title("RMSE vs k")
plt.xlabel("k")
plt.ylabel("RMSE values")
#plt.savefig("rmseplot_sma.png")
plt.show()

#Fit model for selected k
Y_SMA["Final"] = Y_SMA["Values"].rolling(window=k).mean().shift()

#Plot Original (Train data) and Predicated values
out="SMA"+str(k)
imgname="SMA_"+str(k)+".png"
title='Simple Moving Average Model for k = '+str(k)
plt.plot(Y_SMA.Values, color='orange',label='Train Original')
plt.plot(Y_SMA.Final, color='blue', label='Simple Moving Average')
plt.legend(loc='best')
plt.title(title)
#plt.savefig(imgname)
plt.show()

#Task 5 Test data
Y_SMA_test = test.copy()
test=test.values.reshape(-1).tolist()

for t in range(len(test)):
    
    yhat  = Y_SMA["Values"].rolling(window=k).mean().iloc[-1]
    Y_SMA_test=Y_SMA_test.append({'Pred':yhat},ignore_index=True)
    obs = test[t]
    Y_SMA=Y_SMA.append({'Values':obs},ignore_index=True)
    
#RMSE for Test data
rmse_test=round(sqrt(mean_squared_error(Y_SMA_test["Values"].dropna(),Y_SMA_test["Pred"].dropna())),4)
print("\n\nThe RMSE for test set for Simple Moving Average Model with k = {} is {}".format(k,rmse_test))

#Plot Original (Test data) and Predicated values
imgname="SMA_"+str(k)+"_test.png"
plt.plot(Y_SMA_test.Values.dropna().values, color='orange',label='Test Original')
plt.plot(Y_SMA_test.Pred.dropna().values, color='blue', label='Simple Moving Average')
plt.legend(loc='best')
plt.title(title)
#plt.savefig(imgname)
plt.show()
