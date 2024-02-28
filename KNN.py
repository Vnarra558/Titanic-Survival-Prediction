import numpy as np

class KNN:
    def __init__(self, K):
        self.K = K

    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    def predict(self, X_test):
        y_pred = np.zeros(X_test.shape[0])

        for i, x_test in enumerate(X_test):
            dists = np.sqrt(np.sum((self.X_train - x_test)**2, axis=1))
            idx = np.argsort(dists)[:self.K]
            k_labels = self.y_train[idx]
            counts = np.bincount(k_labels)
            y_pred[i] = np.argmax(counts)

        return y_pred