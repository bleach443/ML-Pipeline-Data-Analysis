#!/usr/bin/env python
# coding: utf-8

# In[ ]:

import sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score

#Function to handle train/test split, model fit and report printing 
def predictor_func(X, y, ml_model):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=17) #train/test datasets
    model = ml_model #instantiate model instance
    classification = model.fit(X_train, y_train)
    print(classification)
    
    y_pred = model.predict(X_test)  # Prediction done on test data

    print('Target classes: ', classification.classes_) #no. of labels
        
    #Model Accuracy
    print('--------------------------------------------------------')
    score = accuracy_score(y_test, y_pred)
    print('Accuracy Score: ', score)
    print('--------------------------------------------------------')
    
    #Classification report for model evaluations
    clf_report = classification_report(y_test, y_pred)
    print(clf_report)
    
    return model, score, clf_report

