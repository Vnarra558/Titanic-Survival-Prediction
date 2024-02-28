#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#----------------------------------------------------------------------------
# Created By  : Venkat Narra
# --------------------------------------------------------------------------
import numpy as np
from collections import Counter

class NaiveBayes:
    
    def __init__(self):
        self.prior_probabilities = {}
        self.conditional_probabilities = {}
        self.classes = []

    def fit(self, X, y):
        """
        Train the Gaussian Naive Bayes model on the given data.

        Parameters:
            X (numpy.ndarray): A numpy array of feature vectors.
            y (numpy.ndarray): A numpy array of target labels.

        Returns:
            None
        """
        n_samples, n_features = X.shape
        self.classes = np.unique(y)
        n_classes = len(self.classes)

        # calculate mean, variance, and prior for each class
        self.mean = np.zeros((n_classes, n_features), dtype=np.float64)
        self.variance = np.zeros((n_classes, n_features), dtype=np.float64)
        self.priors = np.zeros(n_classes, dtype=np.float64)

        for idx, c in enumerate(self.classes):
            X_c = []
            for i in range(X.shape[0]):
                if y[i] == c:
                    X_c.append(X[i])
            X_c = np.array(X_c)

            self.mean[idx, :] = X_c.mean(axis=0)
            self.variance[idx, :] = X_c.var(axis=0)


        # Calculates the prior probabilities for each class.
        for label in self.classes:
            self.prior_probabilities[label] = np.count_nonzero(y == label) / len(y)
        # Calculates the conditional probabilities for each class.
        self.calculate_conditional_probabilities(X, y)


    def calculate_conditional_probabilities(self, X, y):
        """
        Calculates the conditional probabilities for each feature value given each class.

        Parameters:
            X (numpy.ndarray): A numpy array of feature vectors.
            y (numpy.ndarray): A numpy array of target labels.

        Returns:
            None
        """
#         print(X.shape[1])
        n_features = X.shape[1]
        for label in self.classes:
            class_samples = X[y == label]
            class_probabilities = {}
            for i in range(n_features):
                feature_values = class_samples[:, i]
                feature_count = Counter(feature_values) # count the occurrences of each feature value in the list
                feature_probabilities = {}
                for value in feature_count:
                    feature_probabilities[value] = feature_count[value] / len(feature_values)
                class_probabilities[i] = feature_probabilities
            self.conditional_probabilities[label] = class_probabilities


    def predict(self, X):
        """
        Predict the target labels for the given data.

        Parameters:
            X (list): A list of feature vectors.

        Returns:
            y_pred (list): A list of predicted target labels.
        """
        y_pred = []
        for x in X:
            posteriors = []
            for idx, c in enumerate(self.classes):
                prior = np.log(self.prior_probabilities[c])
                numerator = np.exp(-((x - self.mean[idx]) ** 2) / (2 * self.variance[idx]))
                denominator = np.sqrt(2 * np.pi * self.variance[idx])
                posterior = np.sum(numerator / denominator)
                posterior = prior + posterior
                posteriors.append(posterior)
            y_pred.append(self.classes[np.argmax(posteriors)])
        return y_pred