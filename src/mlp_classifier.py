#!/usr/bin/env python
# coding: utf-8

# In[ ]:
import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from sklearn.neural_network import MLPClassifier
from predictor_func import predictor_func

#multi-layer perceptrons
def mlp_classifier(dataset, variables, target):
    X = dataset[list(variables)].to_numpy()
    y = dataset[target].to_numpy().ravel()
    scaler = StandardScaler() #instantiate scale function
    X = scaler.fit_transform(X) #normalizing the numerical values
    
    model, score, clf_report = predictor_func(X, y, (MLPClassifier(hidden_layer_sizes=(10,5,2), max_iter=500,
                                                       activation = 'relu',solver='adam',random_state=1)))
    return model, score, clf_report

