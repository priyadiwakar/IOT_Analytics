"""
ECE 592 IOT ANALYTICS
Project 3 Task 1

Check for stationarity

"""


import pandas as pd 
from matplotlib import pyplot as plt
import os

#Read Data
dir_1="C:/Users/priya/Desktop/IOT/project3/"
filename="psdiwaka.csv"
path=os.path.join(dir_1,filename)
data = pd.read_table(path, header=None,sep=" ")

#Plot Data
plt.plot(data)
plt.title("Plotting Time Series Data")
plt.ylabel("Data values")
#plt.savefig("dataplot.png")
plt.show()

#Split Data into train and test sets and Visualize
train_size = int(len(data) * 0.75)
train, test = data[0:train_size], data[train_size:len(data)]
print('Observations: %d' % (len(data)))
print('Training Observations: %d' % (len(train)))
print('Testing Observations: %d' % (len(test)))
plt.plot(range(train_size),train,label="Training set")
plt.plot(range(train_size,len(data)),test,label="Testing set")
plt.title("Train Test Data Split")
plt.legend(loc='best')
#plt.savefig("Train_Testplot.png")
plt.show()