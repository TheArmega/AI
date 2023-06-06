'''
Regression Models with Keras
    - How to use Keras library to build a regression model
    - Download and clean dataset
    - Build a neural network
    - Train and test the network
'''

import pandas as pd
import numpy  as np
import keras

from keras.models import Sequential
from keras.layers import Dense

import requests
from IPython.display import display

path = 'https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/'\
    'CognitiveClass/DL0101EN/labs/data/concrete_data.csv'
#r = requests.get(path, allow_redirects=True)
#open('Deep-Learning/Artificial_Neural_Networks/Regression/Concrete_Data.csv',
#    'wb').write(r.content)

concrete_data = pd.read_csv('Deep-Learning/Artificial_Neural_Networks/'\
                            'Regression/Concrete_Data.csv')
display(concrete_data.head())
display(concrete_data.shape)
display(concrete_data.describe())
display(concrete_data.isnull().sum())

concrete_data_columns = concrete_data.columns

predictors = concrete_data[concrete_data_columns[
    concrete_data_columns != 'Strength']]
target = concrete_data['Strength']

# Normalize the data
predictors_norm = (predictors - predictors.mean()) / predictors.std()
display(predictors_norm.head())

n_cols = predictors_norm.shape[1]

def regression_model():
    # create model
    model = Sequential()
    model.add(Dense(50, activation='relu', input_shape=(n_cols,)))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(1))

    # compile model
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# build the model
model = regression_model()
# fit the model
model.fit(predictors_norm, target, validation_split=0.3, epochs=100, verbose=2)