#!/usr/bin/env python
# coding: utf-8

# In[98]:


##Author: panthadeep_b; Time: 20.48 IST - 13.Sept.25
##WAP to implement Random-Forest classifier Source: tpointtech.com

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from array import *
from sklearn.model_selection import train_test_split
from sklearn import metrics
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
from sklearn.ensemble import RandomForestClassifier

print("----Implementing Random-Forest classifier----");

##Load data-set
data_set = pd.read_csv("housing.csv");
print("The data-set is: ...\n");
print(data_set);


##Assign the data (features) and target
x = data_set.iloc[:,[0,1,2,3,4]];
y = data_set.iloc[:,[15]];

##Reshape the data-set to vector
x = x.to_numpy().reshape(len(x),5);
y = np.ravel(y);
#y = y.to_numpy().reshape(len(y),1);

print("X is: ...\n");
#print(x);
print("Y is: ...\n");
#print(y);

#Splitting the dataset into the Training set and Test set
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=0);


##Standard scale
#std = StandardScaler();
#x_train = std.fit_transform(x_train);
#x_test = std.transform(x_test);


#Fitting Random-Forest to the training set
classifier= RandomForestClassifier(n_estimators= 10, criterion="entropy");
classifier.fit(x_train,y_train);


##Predicting the model accuracy
y_pred = classifier.predict(x_test);

##Predicting the model accuracy
cm = confusion_matrix(y_test,y_pred);
print("Confusion-matrix:\n", cm);
print("Classification-report:\n", classification_report(y_test,y_pred));





# In[ ]:





# In[ ]:




