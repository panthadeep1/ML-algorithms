#!/usr/bin/env python
# coding: utf-8

# In[82]:


##Author: panthadeep_b; Time: 20.16 IST - 12.Sept.25
##WAP to implement Decision-Tree classifier Source: tpointtech.com

import sys
import numpy as np
import pandas as pd
from array import *
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import metrics
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
from sklearn.tree import DecisionTreeClassifier

print("----Implement the Decision-Tree ML classifier----");

##Loading the data-set
data_set = pd.read_csv('housing.csv');
print("The loaded dataset is:\n", data_set);

#Assign the data (features) and target (house prices)
x = data_set.iloc[:,[0,1,2,3]];
y = data_set.iloc[:,[15]];

##***Convert data to suit the scikit-learn***
x = x.to_numpy().reshape(len(x),4);
#x = np.ravel(x);

y = y.to_numpy().reshape(len(y),1);
#y = np.ravel(y);

print("X:\n",x);
print("Y:\n",y);

# Splitting the dataset into the Training set and Test set
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.25, random_state = 0);


#Feature Scaling
#std = StandardScaler();
#x_train = std.fit_transform(x_train);
#y_train = std.transform(x_test);

#Fitting Decision-Tree to the Training set
classifier = DecisionTreeClassifier(criterion='entropy', random_state=0)  
classifier.fit(x_train, y_train)  

#Predicting the test-set result
# In[84]:
#Predicting the test set result 
y_pred = classifier.predict(x_test);

#Evaluating the accuracy of the model
cm = confusion_matrix(y_test, y_pred);
print("The confusion matrix is:\n", cm);
print("The classification report is:", classification_report(y_test, y_pred));





