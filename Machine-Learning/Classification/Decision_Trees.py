'''
Decision Trees
    - Develop a classification model using Decision Tree Algorithm
'''

import numpy as np
import pandas as pd
from sklearn import metrics
import matplotlib.pyplot as plt

from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import sklearn.tree as tree

import requests
from IPython.display import display
import warnings

warnings.filterwarnings('ignore')

# DOWNLOAD THE DATASET

path = 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/'\
    'IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%203/data/'\
    'drug200.csv'

#r = requests.get(path, allow_redirects=True)
#open('Classification/Drug200.csv', 'wb').write(r.content)

# SIZE AND FIRST ROWS OF THE DATASET
print('###### DATASET THAT WE ARE USING ######')
df = pd.read_csv('Classification/Drug200.csv')
display(df.columns)
display(df.describe())
display(df.shape)
display(df.head(5))

X = df[['Age', 'Sex', 'BP', 'Cholesterol', 'Na_to_K']].values
y = df[['Drug']].values

# AS SKLEARN DECISION TREE DOES NOT HANDLE CATEGORICAL VBARIABLES, WE WILL
# CONVERT THESE FEATURES TO NUMERICAL VALUES USING LABELENCODER TO CONVERT THE
# CATEGORICAL VARIABLE INTO NUMERICAL VARIABLES

# SEX FEATURE
le_sex = preprocessing.LabelEncoder()
le_sex.fit(['F', 'M'])
X[:,1] = le_sex.transform(X[:,1])

# BP FEATURE
le_BP = preprocessing.LabelEncoder()
le_BP.fit(['LOW', 'NORMAL', 'HIGH'])
X[:,2] = le_BP.transform(X[:,2])

# CHOLESTEROL FEATURE
le_Chol = preprocessing.LabelEncoder()
le_Chol.fit(['NORMAL', 'HIGH'])
X[:,3] = le_Chol.transform(X[:,3])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,
                                                    random_state=3)

# MODELING TREE
drugTree = DecisionTreeClassifier(criterion='entropy', max_depth=4)
drugTree.fit(X_train, y_train)

predTree = drugTree.predict(X_test)
print(predTree[0:5])
print(y_test[0:5])

# ACCURAACY OF THE MODEL
print('Decision Tree Accuracy: ', metrics.accuracy_score(y_test, predTree))

# PLOT THE TREE
tree.plot_tree(drugTree)
plt.show()