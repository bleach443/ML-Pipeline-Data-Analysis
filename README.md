# ML-Pipeline-Data-Analysis
Machine Learning Pipeline that takes in a dataset to feed it into ML algorithms

# b. Overview
### src 
    relevant files found here contain
    1. gaussian_nb.py
    2. knn_classifier.py
    3. main.py
    4. mlp_classifier.py
    5. neural_network.py
    6. predictor_func.py
    7. preprocess_data.py
    8. save_txt_file.py
    9. sql_to_pd.py
### eda.ipynb
### README.md
### requirements.txt
### run.sh
    Please enter relative path of survive.db here since survive.db is not allowed in final submission


# c. Instructions for execution
   ### survive.db is not included in the submission as conveyed in the instructions, hence to execute run.sh please either provide the relative path of survive.db when prompted during execution or place survive.db in the same directory as run.sh and just enter "survive.db" when prompted during execution
    
    if the program doesn't run, one issue could be the name of the table was changed from "survive" to something else. Please go to sql_to_pd.py and change the string("SELECT * FROM (appropriate name of table)")
    
    Disclaimer: the instructions were alittle confusing as another the "Data" section  we were told to retrive the dataset using the relative path "data/survive.db". However, right below that, we were told not to submit "survive.db" in my final submission. Hence, I've interpreted that section accordingly and written run.sh as such to prompt the user for the path of survive.db
    

# d. Flow of pipeline
    main.py starts with obtaining the path of survive.db from the user input in shell and converts it to a dataframe. The dataframe is then preprocessed according to parameters defined in eda.ipynb. The columns then need to be renamed during the encoding process. The ML algorithms are then applied to the preprocessed dataset accordingly and the final classification reports are saved to a text file

# e. Key findings from EDA
    survive.db consists of both numerical and categorical variables. Categorical variables had to be encoded using a combination of LabelEncoder and OneHotEncoding. The usual preprocessing methods were employed to remove NaN values and irrelevant columns that do not participate in the ML algorithms were dropped. 
    Dimension reduction techniques such as Principal Component Analysis was used on the numerical variables of the dataset. With lesser noisy variables in the dataset, PCA can provide more accurate principal components as inputs for ML by comparing the correlations between the variables. We can consider the first 6 principal components. The heatmap is also provide to showcase the correlations between the components and variables. Regions of dark and light colours can also be consider as inputs
    Outlier detection can also be done using Elleptic Envelope where an envelope is used to cluster points within relative distance of each other. The scatterplot in the eda is done using only 2 of the numerical variables for visualisation purposes. The index values of the red outliers can be retrieved and removed accordingly from our dataset
    Finally, RandomForest regressor is used to determine the relative importance of the numerical variables by comparing them to the predictablity of the target variable. 

# f/g. Explanation/Evaluation of models used
    A fairly straight forward model such as the gaussian naive bayes model which assumes the numerical values have a guassian distribution was used at first. Returning with an accuracy score of only around 0.75, it was safe to assume the numerical values were most likely not guassian distributed. 
    A single NN perceptron works by using Stochastic Gradient Descent, where backprogation is used to zero in on the predicted value. However, the alogrithm returns with only a 0.76 accuracy score. Since we assumed here a linear loss function, we will try the KNN classifier next.
    We then move the KNN Classifier where data points are predicted using their relative distance with other points as a classifier. Our assumption that the data points have clusters rather than are linear seem to be right as reflected by an accuracy score of 0.98
    Furthermore, let us improve on using neural networks as they are very powerful when it comes to predictions. Using the Multi Layer Perceptron classifier, we add in more hidden layers and nodes. We also use a relu activation function (makes use of the tanh function) to classify binary as compared to linear ones and we try a adam optimizer instead of a SGD one. We obtain an accuracy score of 0.95. Increasing the number of hidden layers and iterations would surely improve the model's predictions (up to a certain point due to overfitting), however I've kept it small to make sure my program runs quickly. 

# other considerations
    It is worth noting that I have not applied the techniques we done on the dataset in the EDA process because some of the ML models already run accurately enough and other factor would be the lack of time :)

# THANK YOU FOR READING !


```python

```
