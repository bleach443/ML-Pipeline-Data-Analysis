#!/usr/bin/env python
# coding: utf-8

# In[ ]:
import pandas as pd
from pandas import DataFrame
import numpy as np
from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder

#Function to preprocess data
def preprocess_data(data):
    
    def remove_nan_values(data):
        if data.isnull().values.any() == True:
            data_drop = data.dropna() #drop NaN values
            data_drop_index = data_drop.reset_index(drop=True)
            return data_drop_index
        else:
            return data
    
    def drop_columns(data, column_names):
        for name in column_names:
            data = data.drop([name], axis=1) #drop specific columns using the column_name
        return data

    def positive_numbers(data, column_names):
        for name in column_names:
            data[name] = data[name].abs()
        return data
    
    def replace_elements(dataset, ele1, ele2, ele3):
        #replacing relevant element names
        dataset[ele1] = dataset[ele1].replace({'NO': 'No', 'YES': 'Yes'})
        dataset[ele2] = dataset[ele2].replace({'L': 'Low', 'N': 'Normal'})
        dataset[ele3] = dataset[ele3].replace({'No': '0', 'Yes': '1'})
        
        return dataset

    def variable_encoder(dataset, column_names):
        for name in column_names:
            label_encoder = LabelEncoder()
            outcome_categorical = dataset[name]
            outcome_encoded = label_encoder.fit_transform(outcome_categorical)
            col_names = outcome_categorical.unique()
            
            binary_encoder = OneHotEncoder(categories='auto')
            outcome_encoded_binary = binary_encoder.fit_transform(outcome_encoded.reshape(-1,1))
            outcome_encoded_binary_mat = outcome_encoded_binary.toarray()
            cols = name+str(":")+col_names 
            outcome_encoded_binary_DF = pd.DataFrame(outcome_encoded_binary_mat, columns = cols)
            dataset = dataset.drop([name],axis=1)
            #drop previous categorical variable column
            dataset = pd.concat([dataset,outcome_encoded_binary_DF], axis=1, verify_integrity=True)
        return dataset
    
    data_NaN_removed = remove_nan_values(data)
    data_dropped = drop_columns(data_NaN_removed, ['Favorite color','ID']) 
    data_positive_age = positive_numbers(data_dropped, ['Age'])
    data_replaced = replace_elements(data_positive_age, ['Smoke'], ['Ejection Fraction'], ['Survive'])
    data_preprocessed = variable_encoder(data_replaced,['Diabetes','Smoke','Ejection Fraction','Gender','Survive'])
    
    return data_preprocessed

