'''
Regression Trees
    - Train a Regression Tree
    - Evaluate a Regression Trees Performance
'''
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split

import requests
from IPython.display import display

# DOWNLOAD THE DATASET
path = 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/'\
    'IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%203/data/'\
    'real_estate_data.csv'
#r = requests.get(path, allow_redirects=True)
#open('Regression/Real_State_Data.csv', 'wb').write(r.content)

print('###### DATASET INFO ######')
df = pd.read_csv('Regression/Real_State_Data.csv')
display(df.head(10))
display(df.columns)
display(df.describe())
display(df.shape)

# DATA PRE-PROCESSING
df.dropna(inplace=True)
# CHECKING MISSING VALUES
display(df.isna().sum())

X = df.drop(columns=['MEDV'])
y = df['MEDV']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                    random_state=1)

# CREATE REGRESSION TREE
regression_tree = DecisionTreeRegressor(criterion='squared_error')
regression_tree.fit(X_train, y_train)
display(regression_tree.score(X_test, y_test))

# EVALUATION
prediction = regression_tree.predict(X_test)
# mean absolute difference between the predicted value and the true value, and
# then multiply that value by 1000
print("$",(prediction - y_test).abs().mean()*1000)