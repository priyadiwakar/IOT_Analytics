"""
ECE 592 IOT ANALYTICS

Project 3 Task 3 and Task 5

Exponential Smoothing Model
"""

import pandas as pd 
from matplotlib import pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error
from math import sqrt
from statsmodels.tsa.holtwinters import SimpleExpSmoothing
import os

#Read Data
dir_1="C:/Users/priya/Desktop/IOT/project3/"
filename="psdiwaka.csv"
path=os.path.join(dir_1,filename)
data = pd.read_table(path, header=None,sep=" ")
data.columns=["Values"]
train_size = int(len(data) * 0.75)
train, test = data[0:train_size], data[train_size:len(data)]
alpha=np.asarray(range(1,10))/10
Y_SES=train.copy()
rmselist=[]

#Run the model for different alpha values
for a in alpha:
    output="SES"+str(a)
    model = SimpleExpSmoothing(train.values)
    model_fit = model.fit(a,optimized=True)
    Y_SES[output] = model_fit.fittedvalues
    
    #RMSE
    rmse=sqrt(mean_squared_error(Y_SES["Values"],Y_SES[output]))
    rmselist.append(rmse)

#Determing alpha (smoothing factor) for which RMSE value is lowest    
minrmse=np.argmin(np.asarray(rmselist))
a=alpha[minrmse]
print("The best alpha value for the Exponential Smoothing Model based on lowest RMSE of {} is : {}".format(round(rmselist[minrmse],4),a))

#Plot RMSE vs K
plt.plot(alpha,rmselist)
plt.title("RMSE vs Alpha")
plt.xlabel("Alpha")
plt.ylabel("RMSE values")
#plt.savefig("rmseplot_ses.png")
plt.show()

#Fit model for selected alpha
model = SimpleExpSmoothing(train.values)
model_fit = model.fit(a,optimized=True)
Y_SES["Final"] = model_fit.fittedvalues

#Plot Original (Train data) and Predicated values
out="SES"+str(a)
imgname="SES_"+str(a)+".png"
title='Exponential Smoothing Model for alpha = '+str(a)
plt.plot(Y_SES.Values, color='orange',label='Train Original')
plt.plot(Y_SES.Final, color='blue', label='Exponential Smoothing')
plt.legend(loc='best')
plt.title(title)
#plt.savefig(imgname)
plt.show()

#Task 5 Test data
Y_SES_test = test.copy()
test=test.values.reshape(-1).tolist()
for t in range(len(test)):
    model = SimpleExpSmoothing(Y_SES.Values)
    model_fit = model.fit(a,optimized=True)

    yhat = model_fit.forecast()
    Y_SES_test=Y_SES_test.append({'Pred':yhat},ignore_index=True)
    obs = test[t]
    Y_SES=Y_SES.append({'Values':obs},ignore_index=True)
    
#RMSE for Test Data
rmse_test=round(sqrt(mean_squared_error(Y_SES_test["Values"].dropna(),Y_SES_test["Pred"].dropna())),4)
print("The RMSE for test set for Exponential Smoothing Model with alpha = {} is {}".format(a,rmse_test))

#Plot Original (Test data) and Predicated values
imgname="SES_"+str(a)+"_test.png"
plt.plot(Y_SES_test.Values.dropna().values, color='orange',label='Test Original')
plt.plot(Y_SES_test.Pred.dropna().values, color='blue', label='Exponential Smoothing')
plt.legend(loc='best')
plt.title(title)
#plt.savefig(imgname)
plt.show()