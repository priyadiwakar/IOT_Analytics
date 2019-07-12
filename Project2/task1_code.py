import csv
from matplotlib import pyplot as plt
import numpy as np
from scipy import stats
import pandas as pd

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

#Histogram
X1_hist=plt.hist(X1,30)
plt.title("Histogram X1")
plt.savefig("X1_hist.png")
plt.show()
X2_hist=plt.hist(X2,30)
plt.title("Histogram X2")
plt.savefig("X2_hist.png")
plt.show()
X3_hist=plt.hist(X3,30)
plt.title("Histogram X3")
plt.savefig("X3_hist.png")
plt.show()
X4_hist=plt.hist(X4,30)
plt.title("Histogram X4")
plt.savefig("X4_hist.png")
plt.show()
X5_hist=plt.hist(X5,30)
plt.title("Histogram X5")
plt.savefig("X5_hist.png")
plt.show()

#Mean
X1_mean=np.mean(X1)
X2_mean=np.mean(X2)
X3_mean=np.mean(X3)
X4_mean=np.mean(X4)
X5_mean=np.mean(X5)

#variance
X1_var=np.var(X1)
X2_var=np.var(X2)
X3_var=np.var(X3)
X4_var=np.var(X4)
X5_var=np.var(X5)



#remove outliers using Z score
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
data_o = data_T[(z < 3).all(axis=1)]#8 outliers have been removed after this process


#Correlation Matrix
df = pd.DataFrame(data_o,columns=['X1','X2','X3','X4','X5','Y'])
Corr_Matrix=df.corr()
print("The correlation Matrix is \n",Corr_Matrix)
pd.plotting.scatter_matrix(df, figsize=(10, 10))
plt.savefig("Corr_Scatter.png")
plt.show()

plt.matshow(df.corr())
plt.xticks(range(len(df.columns)), df.columns)
plt.yticks(range(len(df.columns)), df.columns)
plt.colorbar()
plt.savefig("Corr_Map.png")
plt.show()

#write to file
f=open('output_task1.txt','w')
f.write("X1 mean: "+str(X1_mean)+"\n")
f.write("X2 mean: "+str(X2_mean)+"\n")
f.write("X3 mean: "+str(X3_mean)+"\n")
f.write("X4 mean: "+str(X4_mean)+"\n")
f.write("X5 mean: "+str(X5_mean)+"\n\n")
f.write("X1 variance: "+str(X1_var)+"\n")
f.write("X2 variance: "+str(X2_var)+"\n")
f.write("X3 variance: "+str(X3_var)+"\n")
f.write("X4 variance: "+str(X4_var)+"\n")
f.write("X5 variance: "+str(X5_var)+"\n\n")
f.write(str(Corr_Matrix))
f.close()

#Save Correlation matrix to CSV file
Corr_Matrix.to_csv("Corr.csv")