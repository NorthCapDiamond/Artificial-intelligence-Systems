import numpy as np
import pandas as pd


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


class LogisticRegression():
    def __init__(self, n_iter=1000, l_rate=0.001):
        self.n_iter = n_iter
        self.l_rate = l_rate
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        N, features = X.shape
        self.weights = np.zeros(features)
        self.bias = 0

        for i in range(self.n_iter):
            linear_predictions = np.dot(X, self.weights) + self.bias
            predictions = sigmoid(linear_predictions)

            dw = (1 / N) * 2 * np.dot(X.T, predictions - y)
            db = (1 / N) * 2 * sum(predictions - y)

            self.weights = self.weights - self.l_rate * dw
            self.bias = self.bias - self.l_rate * db

    def predict(self, X, percentile=0.5):
        linear_predictions = np.dot(X, self.weights) + self.bias
        y_predicted = sigmoid(linear_predictions)

        predict_class = [1 if y > percentile else 0 for y in y_predicted]
        return predict_class
