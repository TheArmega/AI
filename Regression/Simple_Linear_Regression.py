'''
Simple Linear Regression:
    - We will use scikit-learn to implement simple linear regression
    - Also, we will creater a model, train it and use the model
'''

import matplotlib.pyplot as plt
import pandas as pd
import pylab as pl
import numpy as np

from sklearn import linear_model
from sklearn.metrics import r2_score

import requests
from IPython.display import display

path= 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud'\
        '/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%202'\
        '/data/FuelConsumptionCo2.csv'

# Download the dataset in csv extension
#r = requests.get(path, allow_redirects=True)
#open('Regression/FuelConsumptionCO2.csv', 'wb').write(r.content)

df = pd.read_csv('Regression/FuelConsumptionCO2.csv')
#display(df.head())
#display(df.describe())

cdf = df[['ENGINESIZE', 'CYLINDERS', 'FUELCONSUMPTION_COMB', 'CO2EMISSIONS']]
display(cdf.head(9))

# Plot the features written down here
viz = cdf[['ENGINESIZE', 'CYLINDERS', 'FUELCONSUMPTION_COMB', 'CO2EMISSIONS']]
viz.hist()
plt.show()

plt.scatter(cdf.ENGINESIZE, cdf.CO2EMISSIONS, color='blue')
plt.title('Engine Size against CO2 Emissions')
plt.xlabel('ENGINESIZE')
plt.ylabel('CO2EMISSIONS')
plt.show()

plt.scatter(cdf.CYLINDERS, cdf.CO2EMISSIONS, color='red')
plt.title('Cylinders against CO2 Emissions')
plt.xlabel('CYLINDERS')
plt.ylabel('CO2EMISSIONS')
plt.show()

plt.scatter(cdf.FUELCONSUMPTION_COMB, cdf.CO2EMISSIONS, color='green')
plt.title('Fuel Consumption against CO2 Emissions')
plt.xlabel('FUELCONSUMPTION_COMB')
plt.ylabel('CO2EMISSIONS')
plt.show()

# SETTING TRAIN AND TEST SETS

msk = np.random.rand(len(df)) < 0.8
train = cdf[msk]
test = cdf[~msk]

# LINEAR REGRESSION WITH ENGINESIZE AGAINST CO2EMISSIONS
print('###### SIMPLE LINEAR REGRESSION ######\nEngine Size against CO2' \
       'Emissions')
regr = linear_model.LinearRegression()
train_x = np.asanyarray(train[['ENGINESIZE']])
train_y = np.asanyarray(train[['CO2EMISSIONS']])
regr.fit(train_x, train_y)
print('Coefficients: ', regr.coef_)
print('Intercept: ', regr.intercept_)

plt.scatter(train.ENGINESIZE, train.CO2EMISSIONS, color='blue')
plt.title('Simple Linear Regression')
plt.plot(train_x, regr.intercept_[0] + regr.coef_[0][0]*train_x, '-r')
plt.xlabel('Engine Size')
plt.ylabel('CO2 Emission')
plt.show()

test_x = np.asanyarray(test[['ENGINESIZE']])
test_y = np.asanyarray(test[['CO2EMISSIONS']])
test_y_ = regr.predict(test_x)

print('Mean Absolute Error: %.2f' % np.mean(np.absolute(test_y_ - test_y)))
print('Mean Squared Error: %.2f' % np.mean((test_y_ - test_y)**2))
print('R2-score: %.2f' % r2_score(test_y, test_y_))

# LINEAR REGRESSION WITH FUELCONSUMPTION_COMB AGAINST CO2EMISSIONS
print('###### SIMPLE LINEAR REGRESSION ######\nFuel Consumption against CO2' \
       'Emissions')
train_x = np.asanyarray(train[['FUELCONSUMPTION_COMB']])
train_y = np.asanyarray(train[['CO2EMISSIONS']])
regr.fit(train_x, train_y)
print('Coefficients: ', regr.coef_)
print('Intercept: ', regr.intercept_)

plt.scatter(train.FUELCONSUMPTION_COMB, train.CO2EMISSIONS, color='green')
plt.title('Simple Linear Regression')
plt.plot(train_x, regr.intercept_[0] + regr.coef_[0][0]*train_x, '-r')
plt.xlabel('Fuel Consumption')
plt.ylabel('CO2 Emissions')
plt.show()

test_x = np.asanyarray(test[['FUELCONSUMPTION_COMB']])
test_y = np.asanyarray(test[['CO2EMISSIONS']])
test_y_ = regr.predict(test_x)

print('Mean Absolute Error: %.2f' % np.mean(np.absolute(test_y_ - test_y)))
print('Mean Squared Error: %.2f' % np.mean((test_y_ - test_y)**2))
print('R2-score: %.2f' % r2_score(test_y, test_y_))