#!/usr/bin/env python
# coding: utf-8


def Model():
    ##pobieram z sklearna model
    from sklearn.linear_model import Perceptron

    ##wczytuje dane
    from sklearn.model_selection import train_test_split
    import pandas as pd
    data = pd.read_csv("static//files//userload//data.csv")
    y = data.iloc[:, 4]
    X = data.drop(data.columns[4], axis=1)
    X_train, X_test, y_train, y_test =  train_test_split(X, y, test_size=0.2, random_state=42)

    #trenowanie modelu
    model = Perceptron()
    model.fit(X_train,y_train)
    y_pred = model.predict(X_test)


    #tworzenie macierzy błędu
    import matplotlib.pyplot as plt
    from sklearn.datasets import make_classification
    from sklearn.metrics import ConfusionMatrixDisplay


    #zapisywanie macierzy
    ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
    plt.savefig('static/files/ConfusionMatrixPlot/ConfusionMatrix.png')
    return None





