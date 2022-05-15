#!/usr/bin/env python
# coding: utf-8


def Model():
    #klasa modelu
    import numpy as np
    class Perceptron():

        def __init__(self, eta=0.01, n_iter=10):
            self.eta = eta
            self.n_iter = n_iter

        def fit(self, X, y):
            self.w_ = np.zeros(1 + X.shape[1])
            self.errors_ = []

            for _ in range(self.n_iter):
                errors = 0
                for xi, target in zip(X, y):
                    # print(xi, target)
                    update = self.eta * (target - self.predict(xi))
                    # print(update)
                    self.w_[1:] += update * xi
                    self.w_[0] += update
                    # print(self.w_)
                    errors += int(update != 0.0)
                self.errors_.append(errors)
            return self

        def net_input(self, X):
            return np.dot(X, self.w_[1:]) + self.w_[0]

        def predict(self, X):
            return np.where(self.net_input(X) >= 0.0, 1, -1)

    #przygotowanie danych do modelu
    import sklearn.datasets
    from sklearn.model_selection import train_test_split
    import pandas as pd
    X = sklearn.datasets.load_iris().data
    y = sklearn.datasets.load_iris().target
    X_train, X_test, y_train, y_test =  train_test_split(X, y, test_size=0.2, random_state=42)

    #trenowanie modelu
    model = Perceptron()
    model.fit(X_train,y_train)
    y_pred = model.predict(X_test)


    #macierz błędu
    import matplotlib.pyplot as plt
    from sklearn.datasets import make_classification
    from sklearn.metrics import ConfusionMatrixDisplay

    #zapisywanie macierzy
    ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
    plt.savefig('static/files/ConfusionMatrixPlot/ConfusionMatrix.png')
    return None
