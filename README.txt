This is Weather Prediction model implemented by using Logistic Regression

Dataset from kaggle

Modules used in this project:
 
  NUMPY - This is a python library for numericals and array
  
  Pandas - This is a python library for data processing and pulling data from csv file

  scikit-learn - This is a machine learning library which cosists of most of the ML algorithms

Steps involved:

  1.Importing Libraries (Numpy,Pandas,sklearn)

  2.Importing dataset by using pandas library

  3.Cleaning dataset 

  4.Converting predictions to Numericals

  5.Spliting dataset for training and testing using sklearn.model_selection.train_test_split

  6.importing LogisticRegression model sklearn.linear_models.LogisticRegression

  7.Training the model with training dataset using LogisticRegression.fit(train_labels,train_target)

  8.Predicting the labels the testing dataset using LogisticRegression.predict(test_labels)

  9.Checking accuracy score of testing dataset with respect to model


**accurrcy score = 0.8412391434297971