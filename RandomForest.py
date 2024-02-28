#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#----------------------------------------------------------------------------
# Created By  : Sai Siddhardha Maguluri
# --------------------------------------------------------------------------


import numpy as np

from algorithms.DecisionTree import DecisionTreeClassifier

class RandomForestClassifier:
    def __init__(self, ntrees=10,max_depth=3,min_samples_split=2):
        self.ntrees = ntrees
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.trees = []
    
    def fit(self, X, y):
        for _ in range(self.ntrees):
            tree = DecisionTreeClassifier()
            num_samples,num_features = X.shape
            idxs = np.random.choice(num_samples,size=num_samples, replace=True)
            X=X[idxs]
            y=y[idxs]
            tree = DecisionTreeClassifier(max_depth=self.max_depth,min_samples_split=self.min_samples_split)
            tree.fit(X, y)
            self.trees.append(tree)

    def predict(self, X):
        trees_predictions = np.array([tree.predict(X) for tree in self.trees])
        trees_predictions = np.swapaxes(trees_predictions, 0, 1)
        y_pred = [self.get_max_label(pred) for pred in trees_predictions]
        return np.array(y_pred)
    
    def get_max_label(self, y):
        unique_labels,label_counts= np.unique(y,return_counts=True)
        label_index = np.argmax(label_counts)
        return int(unique_labels[label_index])