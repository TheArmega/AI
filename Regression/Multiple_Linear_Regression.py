'''
Multiple Linear Regression
    - Use scikit-learn to implement Multiple Linear Regression
    - Create a model, train it, test it and use the model
'''

import matplotlib.pyplot as plt
import pandas as pd
import pylab as pl
import numpy as np

from sklearn import linear_model

import requests
from IPython.display import display
import warnings

path = 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/'\
    'IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%202/data/'\
    'FuelConsumptionCo2.csv'

# Download the datase in csv extension
#r = requests.get(path, allow_redirects=True)
#open('Regression/FuelConsumptionCO2.csv', 'wb').write(r.content)

df = pd.read_csv('Regression/FuelConsumptionCO2.csv')
#display(df.head())
#display(df.describe())

print('###### DATASET ######')
cdf = df[['ENGINESIZE', 'CYLINDERS', 'FUELCONSUMPTION_CITY', 
        'FUELCONSUMPTION_HWY', 'FUELCONSUMPTION_COMB', 'CO2EMISSIONS']]
display(cdf.head(9))

# PLOT THE ENGINESIZE AGAINST CO2EMISSIONS
plt.scatter(cdf.ENGINESIZE, cdf.CO2EMISSIONS, color='blue')
plt.title('Engine Size against CO2 Emissions')
plt.xlabel('ENGINESIZE')
plt.ylabel('CO2EMISSIONS')
plt.show()

# CREATE THE TRTAIN AND TEST SET
msk = np.random.rand(len(df)) < 0.8
train = cdf[msk]
test = cdf[~msk]
regr = linear_model.LinearRegression()

warnings.filterwarnings('ignore')

# MULTPLE LINEAR REGRESSION 
print('###### MULTIPLE LINEAR REGRESSION ######')
x = np.asanyarray(train[['ENGINESIZE', 'CYLINDERS', 'FUELCONSUMPTION_COMB']])
y = np.asanyarray(train[['CO2EMISSIONS']])
regr.fit(x, y)
print('Coefficients: ', regr.coef_)
print('Intercept: ', regr.intercept_)

# PREDICTION
print('###### PREDICITION ######')
y_hat = regr.predict(test[['ENGINESIZE', 'CYLINDERS', 'FUELCONSUMPTION_COMB']])
x = np.asanyarray(test[['ENGINESIZE', 'CYLINDERS', 'FUELCONSUMPTION_COMB']])
y = np.asanyarray(test[['CO2EMISSIONS']])
print('Mean Squared Error: %.2f' % np.mean((y_hat - y)**2))
print('Variance Score: %.2f' % regr.score(x, y))

################################################################################

# MULTIPLE LINEAR REGRESSION
print('###### MULTIPLE LINEAR REGRESSION ######')
x = np.asanyarray(train[['ENGINESIZE', 'CYLINDERS', 'FUELCONSUMPTION_CITY', 
                         'FUELCONSUMPTION_HWY']])
y = np.asanyarray(train[['CO2EMISSIONS']])
regr.fit(x, y)
print('Coefficients: ', regr.coef_)
print('Intercept', regr.intercept_)

# PREDICTION
print('###### PREDICTION ######')
y_hat = regr.predict(test[['ENGINESIZE', 'CYLINDERS', 'FUELCONSUMPTION_CITY',
                           'FUELCONSUMPTION_HWY']])
x = np.asanyarray(test[['ENGINESIZE', 'CYLINDERS', 'FUELCONSUMPTION_CITY',
                           'FUELCONSUMPTION_HWY']])
y = np.asanyarray(test[['CO2EMISSIONS']])
print('Mean Squared Error: %.2f' % np.mean((y_hat - y)**2))
print('Variance Score: %.2f' % regr.score(x, y))