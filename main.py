import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split


#importing data using pandas

data = pd.read_csv("./dataset/weatherAUS.csv")
#data cleaning
data.drop(['Date', 'Location', 'WindGustDir', 'WindDir9am', 'WindDir3pm', 'RISK_MM'], axis=1, inplace=True)
data.fillna(data.mean(), inplace=True)
data.RainToday = [1 if each == 'Yes' else 0 for each in data.RainToday]
data.RainTomorrow = [1 if each == 'Yes' else 0 for each in data.RainTomorrow]

y = data.RainTomorrow.values
x_data = data.drop('RainTomorrow', axis=1)
#feature scaling
x = (x_data - np.min(x_data)) / (np.max(x_data) - np.min(x_data))

print(x)

