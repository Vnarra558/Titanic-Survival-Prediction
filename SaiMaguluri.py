#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#----------------------------------------------------------------------------
# Created By  : Sai Siddhardha Maguluri
# Created Date: 04/10/2023
# --------------------------------------------------------------------------

## Supervised Learning Project:

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

from algorithms.DecisionTree import DecisionTreeClassifier
from algorithms.RandomForest import RandomForestClassifier

#original training dataset
trainingDataSet = pd.read_csv(r'train.csv')

#Preprocess the training dataset

# we can neglect some of the features as they don't contribute much to our prediction rate like passengerId,Name etc.
processedDataset = trainingDataSet.drop(['PassengerId','Name','Ticket','Cabin'],axis=1)

# we map categorical values to real values
processedDataset['Sex'] = processedDataset['Sex'].map({'female': 0, 'male': 1})
processedDataset['Embarked'] = processedDataset['Embarked'].map({'S':0,'C':1,'Q':2})

# Some of the columns(like Age) are missing data,So we update them accordingly
processedDataset['Age'] = processedDataset['Age'].fillna(processedDataset['Age'].median())

processedDataset = processedDataset.dropna(subset=processedDataset.columns, how='any')

X = processedDataset.loc[:, processedDataset.columns != 'Survived']
y = processedDataset['Survived']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

X_train = np.array(X_train)
y_train = np.array(y_train)

X_test = np.array(X_test)
y_test = np.array(y_test)

# Decision Tree with entropy as criteria
clf = DecisionTreeClassifier(criterion='gini',max_depth=3,min_samples_split=2)
clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)

#calculating the metrics for Decision Trees
confusion_matrix_DT = confusion_matrix(y_pred, y_test)
print("The confusion matrix for Decision tree with gini as criteria: ", confusion_matrix_DT)

accuracy_DT = accuracy_score(y_test, y_pred)
precision_DT = precision_score(y_test, y_pred,average='weighted')
recall_DT = recall_score(y_test, y_pred,average='weighted')
f1score_DT = f1_score(y_test, y_pred,average='weighted')
print(f"The accuracy of Decision Trees with gini as criteria: {accuracy_DT}")
print(f"The precision of Decision Trees with gini as criteria: {precision_DT}")
print(f"The recall of Decision Trees with gini as criteria: {recall_DT}")
print(f"The f1score of Decision Trees with gini as criteria: {f1score_DT}")



# RandomForest 
clf_rf= RandomForestClassifier(ntrees=100)
clf_rf.fit(X_train,y_train)
y_pred = clf_rf.predict(X_test)

#calculating the metrics for Random Forest
confusion_matrix_RF = confusion_matrix(y_pred, y_test)
print()
print("The confusion matrix for Random Forest algorithm: ", confusion_matrix_RF)

accuracy_RF = accuracy_score(y_test, y_pred)
precision_RF = precision_score(y_test, y_pred,average='weighted')
recall_RF = recall_score(y_test, y_pred,average='weighted')
f1score_RF = f1_score(y_test, y_pred,average='weighted')
print(f"The accuracy of Random Forest algorithm: {accuracy_RF}")
print(f"The precision of Random Forest algorithm: {precision_RF}")
print(f"The recall of Random Forest algorithm: {recall_RF}")
print(f"The F1 Score of Random Forest algorithm: {f1score_RF}")