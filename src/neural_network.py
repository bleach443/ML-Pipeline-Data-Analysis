#!/usr/bin/env python
# coding: utf-8

# In[ ]:
import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from sklearn.linear_model import Perceptron
from predictor_func import predictor_func

#Single Perceptron
def neural_network(dataset, variables, target):
    X = dataset[list(variables)].to_numpy()
    y = dataset[target].to_numpy().ravel()
    scaler = StandardScaler() #instantiate scale function
    X = scaler.fit_transform(X) #normalizing the numerical values
    
    model, score, clf_report = predictor_func(X, y, Perceptron(max_iter=1000, eta0=0.5, tol=1e-3, random_state=90))
    return model, score, clf_report

