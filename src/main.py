#!/usr/bin/env python
# coding: utf-8

# In[10]:

import sys
from sql_to_pd import sql_to_pd
from preprocess_data import preprocess_data
from predictor_func import predictor_func
from gaussian_nb import gaussian_nb
from neural_network import neural_network
from knn_classifier import knn_classifier
from mlp_classifier import mlp_classifier
from save_txt_file import save_txt_file

def main():
    path = sys.argv[1] #obtain path of survive.db 
    df = sql_to_pd(path) #convert survive.db to dataframe
    df_new = preprocess_data(df)
    #remame columns after encoding
    df_preprocessed = df_new.rename(columns={'Diabetes:Normal': 'Diabetes:Diabetes', 'Diabetes:Pre-Diabetes': 'Diabetes:Normal', 'Diabetes:Diabetes': 'Diabetes:Pre-Diabetes'
                                     , 'Smoke:Yes':'Smoke:N','Smoke:No':'Smoke:Y','Ejection Fraction:Low':'Ejection Fraction:High','Ejection Fraction:Normal':"Ejection Fraction:Low",'Ejection Fraction:High':"Ejection Fraction:Normal"
                                     , 'Gender:Male':'Gender:Female','Gender:Female':'Gender:Male'})
    
    numerical_cols = ['Creatinine','Creatinine phosphokinase','Pletelets','Weight','Age','Sodium','Height','Hemoglobin','Blood Pressure']
    
    #sorry for the lazy formatting here, didn't have enough time :D
    model1, score1, clf_report1 = gaussian_nb(df_preprocessed, numerical_cols, ['Survive:1'])
    model2, score2, clf_report2 = neural_network(df_preprocessed, numerical_cols, ['Survive:1'])
    model3, score3, clf_report3 = knn_classifier(df_preprocessed, numerical_cols, ['Survive:1'])
    model4, score4, clf_report4 = mlp_classifier(df_preprocessed, numerical_cols, ['Survive:1'])
    results = str(model1) + clf_report1 + str(model2) + clf_report2 + str(model3) + clf_report3 + str(model4) + clf_report4
    txt = save_txt_file(results)
    
    return txt
    
if __name__ == "__main__":
    main()


# 

# In[ ]:




