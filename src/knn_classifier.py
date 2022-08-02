#!/usr/bin/env python
# coding: utf-8

# In[ ]:
import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
from predictor_func import predictor_func


#K Nearest Neighbours Classifier
def knn_classifier(dataset, variables, target):
    X = dataset[list(variables)].to_numpy()
    y = dataset[target].to_numpy().ravel()
    scaler = StandardScaler() #instantiate scale function
    X = scaler.fit_transform(X) #normalizing the numerical values
    
    model, score, clf_report = predictor_func(X, y, KNeighborsClassifier(metric='euclidean'))
    return model, score, clf_report

