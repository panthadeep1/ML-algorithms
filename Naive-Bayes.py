##Author: panthadeep_b; Time: 21.24 IST - 8.Aug.25
##WAP to implement Naive-Bayes classifier Source: tpointtech.com

import sys
import numpy as np
import pandas as pd
from array import *
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn import datasets
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
from sklearn.naive_bayes import GaussianNB

##Importing the dataset
data_set = pd.read_csv('magic.csv');
print("The loaded dataset is:\n",data_set);

#Assign the data (features) and target (house prices)
x = data_set.iloc[:,[0,1,2,3]];
y = data_set.iloc[:,[10]];

##***Convert data to suit the scikit-learn***
x = x.to_numpy().reshape(len(x),4);
y = y.to_numpy().reshape(len(y),1);


print("X:\n",x);
print("Y:\n",y);

# Splitting the dataset into the Training set and Test set  
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.25, random_state = 0)  


#Feature Scaling
std = StandardScaler()  
x_train = std.fit_transform(x_train)  
x_test = std.transform(x_test) 

#Fitting Naive Bayes to the Training set  
classifier = GaussianNB()  
classifier.fit(x_train, y_train)  


# Predicting the Test set results  
y_pred = classifier.predict(x_test)  

# Making the Confusion Matrix   
conf_mat = confusion_matrix(y_test, y_pred)  
print("The confusion matrix is:\n", conf_mat);
print("The classification report is:\n", classification_report(y_test, y_pred));








