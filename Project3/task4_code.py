"""
ECE 592 IOT ANALYTICS

Project 3 Task 4 and Task 5

AR(p) Model
"""

import pandas as pd 
from matplotlib import pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error
from math import sqrt
from statsmodels.tsa.stattools import pacf
import statsmodels.api as smf
from statsmodels.tsa.arima_model import ARIMA
from scipy import stats
import os

#Read Data
dir_1="C:/Users/priya/Desktop/IOT/project3/"
filename="psdiwaka.csv"
path=os.path.join(dir_1,filename)
data = pd.read_table(path, header=None,sep=" ")
data.columns=["Values"]
train_size = int(len(data) * 0.75)
train, test = data[0:train_size], data[train_size:len(data)]

#Plot the PACF to find order p for the AR model
lag_pacf = pacf(train, nlags=20)
plt.plot(lag_pacf)
plt.xticks(list(range(21)))
plt.axhline(y=0,linestyle='--',color='gray')
plt.axhline(y=-1.96/np.sqrt(len(train)),linestyle='--',color='gray')
plt.axhline(y=1.96/np.sqrt(len(train)),linestyle='--',color='gray')
plt.title('Partial Autocorrelation Function')
#plt.savefig("PACF_AR.png")
plt.show()

#Fit the AR model
p=4
model = ARIMA(train, order=(p, 0, 0))  
results_AR = model.fit(disp=-1)  
Y_AR=results_AR.fittedvalues

#Parameters
par=results_AR.params
print("The parameters for the AR Model with order p = {} are \n{}".format(p,par))

#RMSE
rmse=sqrt(mean_squared_error(train.values,Y_AR))
print("\n\nThe RMSE for train set for AR Model with order p = {} is {:.3f}".format(p,rmse))

#Plot Original (Train data) and Predicated values
out="AR"+str(p)
imgname="AR_"+str(p)+".png"
title='AutoRegressive Model with order p = '+str(p)
label1="AR("+str(p)+")"
#Plot original against predicted values:
plt.plot(train.values, color='orange',label='Train Original')
plt.plot(Y_AR, color='blue', label=label1)
plt.legend(loc='best')
plt.title(title)
#plt.savefig(imgname)
plt.show()

#Residual Analysis
residuals=results_AR.resid.values
s=rmse

#Q-Q plot

fig=smf.qqplot(residuals,loc=0, scale=s,fit=True,line='45')
plt.title("Q-Q plot")
#plt.savefig("QQplotAR.png")
plt.show()

#Histogram of Residuals
Res_hist=plt.hist(residuals,30)
plt.title("Histogram Residuals")
#plt.savefig("Residual_hist.png")
plt.show()

#Chi square Test
_,pvalue=stats.normaltest(residuals)
print("The p value for the Chi Squared Test is :",pvalue)


#Scatter plot
plt.scatter(Y_AR,residuals)
plt.xlabel("Predictions")
plt.ylabel("Residuals")
#plt.savefig("Res_scatter.png")
plt.show()

#Task 5 Test data
test=test.values.reshape(-1).tolist()
history = [x for x in train["Values"].values]
predictions = list()
for t in range(len(test)):
	model = ARIMA(history, order=(p,0,0))
	model_fit = model.fit(disp=0)
	output = model_fit.forecast()
	yhat = output[0]
	predictions.append(yhat)
	obs = test[t]
	history.append(obs)
	
#RMSE
rmse_test=round(sqrt(mean_squared_error(test, predictions)),4)
print("\nThe RMSE for test set for AR Model with order p = {} is {}".format(p,rmse_test))

imgname="AR_"+str(p)+"_test.png"
plt.plot(test, color='orange',label='Test Original')
plt.plot(predictions, color='blue', label=label1)
plt.legend(loc='best')
plt.title(title)
#plt.savefig(imgname)
plt.show()
#