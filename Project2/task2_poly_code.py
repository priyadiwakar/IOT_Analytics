import statsmodels.api as smf
import csv
from matplotlib import pyplot as plt
import numpy as np
from scipy import stats

#read data from CSV file
with open('psdiwaka.csv') as csvfile:
    readCSV = csv.reader(csvfile, delimiter=',')
    X1 = []
    X2 = []
    X3 = []
    X4 = []
    X5 = []
    Y = []
    for row in readCSV:
       X1.append(float(row[0]))
       X2.append(float(row[1]))
       X3.append(float(row[2]))
       X4.append(float(row[3]))
       X5.append(float(row[4]))
       Y.append(float(row[5]))

X1=np.asarray(X1)
X2=np.asarray(X2)
X3=np.asarray(X3)
X4=np.asarray(X4)
X5=np.asarray(X5)
Y=np.asarray(Y)



X1=np.reshape(X1,(-1,1))
X2=np.reshape(X2,(-1,1))
X3=np.reshape(X3,(-1,1))
X4=np.reshape(X4,(-1,1))
X5=np.reshape(X5,(-1,1))
Y=np.reshape(Y,(-1,1))

a=np.append(X1,X2,axis=1)
b=np.append(a,X3,axis=1)
c=np.append(b,X4,axis=1)
data=np.append(c,X5,axis=1)

z = np.abs(stats.zscore(data))

data_T=np.append(data,Y,axis=1)
data_o = data_T[(z < 3).all(axis=1)]

#Linear Regression with only X1 and X1^2
X = np.reshape(data_o[:,0],(-1,1))
X = np.append(X,X**2,axis=1)
y = data_o[:,-1]
X = smf.add_constant(X)


model = smf.OLS(y, X).fit()
predictions=model.predict(X)
residuals = model.resid 
s=np.sqrt(model.mse_resid)

# Print out the statistics
print(model.summary())

f=open('output_task2poly.txt','w')
f.write(str(model.summary()))
f.write("\n\n\n")
f.write("The estimate for variance of residuals is:"+str(model.mse_resid))
f.close()

#Q-Q plot

fig=smf.qqplot(residuals,loc=0, scale=s,fit=True,line='45')
plt.title("Q-Q plot")
plt.savefig("QQplotX1poly.png")
plt.show()

#Histogram of Residuals
Res_hist=plt.hist(residuals,30)
plt.title("Histogram Residuals")
plt.savefig("Residual_hist.png")
plt.show()

#Chi square Test
_,pvalue=stats.normaltest(residuals)


#Scatter plot
plt.scatter(predictions,residuals)
plt.xlabel("Predictions")
plt.ylabel("Residuals")
plt.savefig("Res_scatter.png")
plt.show()

