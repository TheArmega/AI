'''
K-Nearest Neighbors
    - Use K Nearest neighbors to classify data
'''

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import preprocessing

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics

import requests
from IPython.display import display
import warnings

warnings.filterwarnings('ignore')

path = 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/'\
    'IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%203/data/'\
    'teleCust1000t.csv'

# DOWNLOAD THE DATASET IN CSV EXTENSION
#r = requests.get(path, allow_redirects=True)
#open('Classification/TeleCust100t.csv', 'wb').write(r.content)

df = pd.read_csv('Classification/TeleCust100t.csv')
#display(df.head(9))
#display(df.describe())

# SEE HOW MANY CLASS IN OUR DATA SET
display(df['custcat'].value_counts())
df.hist(column='income', bins=50)
plt.show()

display(df.columns)

# DEFINE FEATURE SETS
X = df[['region', 'tenure', 'age', 'marital', 'address', 'income', 'ed',
        'employ', 'retire', 'gender', 'reside']].values
y = df[['custcat']].values

# NORMALIZE DATA
X = preprocessing.StandardScaler().fit(X).transform(X.astype(float))

# TRAIN TEST SPLIT
print('###### TRAIN TEST SPLIT ######')
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                    random_state=4)
print('Train set: ', X_train.shape, y_train.shape)
print('Test set: ', X_test.shape, y_test.shape)

# K-NEAREST NEIGHBOR (KNN) K = 4
print('###### K-NEAREST NEIGHBOR (KNN) ######\nWith K=4')
k = 4
neigh = KNeighborsClassifier(n_neighbors=k).fit(X_train, y_train)
yhat = neigh.predict(X_test)

print('Train set Accuracy: ', metrics.accuracy_score(y_train,
                                                     neigh.predict(X_train)))
print('Test set Accuracy: ', metrics.accuracy_score(y_test, yhat))

# K-NEAREST NEIGHBOR (KNN) K = 6
print('###### K-NEAREST NEIGHBOR (KNN) ######\nWith K=6')
k = 6
neigh = KNeighborsClassifier(n_neighbors=k).fit(X_train, y_train)
yhat = neigh.predict(X_test)

print('Train set Accuracy: ', metrics.accuracy_score(y_train,
                                                     neigh.predict(X_train)))
print('Test set Accuracy: ', metrics.accuracy_score(y_test, yhat))

Ks = 10
mean_acc = np.zeros((Ks-1))
std_acc = np.zeros((Ks-1))

for n in range(1,Ks):
    
    #Train Model and Predict  
    neigh = KNeighborsClassifier(n_neighbors = n).fit(X_train,y_train)
    yhat=neigh.predict(X_test)
    mean_acc[n-1] = metrics.accuracy_score(y_test, yhat)
    std_acc[n-1]=np.std(yhat==y_test)/np.sqrt(yhat.shape[0])

plt.plot(range(1,Ks),mean_acc,'g')
plt.fill_between(range(1,Ks),mean_acc - 1 * std_acc, mean_acc + 1 * std_acc,
                 alpha=0.10)
plt.fill_between(range(1,Ks),mean_acc - 3 * std_acc, mean_acc + 3 * std_acc,
                 alpha=0.10,color="green")
plt.legend(('Accuracy ', '+/- 1xstd','+/- 3xstd'))
plt.ylabel('Accuracy ')
plt.xlabel('Number of Neighbors (K)')
plt.tight_layout()
plt.show()

print( "The best accuracy was with", mean_acc.max(), "with k=",
      mean_acc.argmax()+1) 