import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression


#importing data using pandas

data = pd.read_csv("./dataset/weatherAUS.csv")
#data cleaning
data.drop(['Date', 'Location', 'WindGustDir', 'WindDir9am', 'WindDir3pm', 'RISK_MM'], axis=1, inplace=True)
data.fillna(data.mean(), inplace=True)
data.RainToday = [1 if each == 'Yes' else 0 for each in data.RainToday]
data.RainTomorrow = [1 if each == 'Yes' else 0 for each in data.RainTomorrow]

y = data.RainTomorrow.values
x_data = data.drop('RainTomorrow', axis=1)
#Normalization
#feature scaling
x = (x_data - np.min(x_data)) / (np.max(x_data) - np.min(x_data))

#using Logistic regression model from sklearn
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=75)

LR = LogisticRegression()
LR.fit(x_train,y_train)

pred = LR.predict(x_test)

#obtaining score for test dataset
print(LR.score(x_test,y_test))

