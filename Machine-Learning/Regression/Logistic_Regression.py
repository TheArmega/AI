'''
Logistic Regression
    - Use scikit Logistic Regression to clasify
    - Understand confusion matrix
'''

import pandas as pd
import pylab as pl
import numpy as np
import scipy.optimize as opt
from sklearn import preprocessing
import matplotlib.pyplot as plt

from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import jaccard_score
from sklearn.metrics import classification_report, confusion_matrix
import itertools
from sklearn.metrics import log_loss

import requests
from IPython.display import display

# DOWNLOAD THE DATASET
path = 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/'\
    'IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%203/data/'\
    'ChurnData.csv'
#r = requests.get(path, allow_redirects=True)
#open('Regression/Churn_Data.csv', 'wb').write(r.content)

df = pd.read_csv('Regression/Churn_Data.csv')
display(df.head(5))
display(df.columns)

# TRAIN AND TEST DATASET
X = np.asanyarray(df[['tenure', 'age', 'address', 'income', 'ed', 'employ', 'equip',
       'callcard', 'wireless']])
y = np.asanyarray(df['churn'].astype('int'))

X = preprocessing.StandardScaler().fit(X).transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y , test_size=0.2,
                                                    random_state=4)
print('Train set: ', X_train.shape, y_train.shape)
print('Test set: ', X_test.shape, y_test.shape)

# MODELING
LR = LogisticRegression(C=0.01, solver='liblinear').fit(X_train, y_train)
prediction = LR.predict(X_test)
print('Test: ', y_test)
print('Prediction: ', prediction)
prediction_prob = LR.predict_proba(X_test)
print('Probability: ', prediction_prob)

# EVALUATION
# JACCARD INDEX
jaccardScore = jaccard_score(y_test, prediction, pos_label=0)
print('Jaccard Score: ', jaccardScore)

# CONFUSION MATRIX
def plot_confusion_matrix(cm, classes, normalize=False, 
                        title='Confusion matrix', cmap=plt.cm.Blues):
    '''
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting 'normalize=True'.
    '''
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print('Normalized confusion matrix')
    else:
        print('Confusion matrix, without normalization')
    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()
print(confusion_matrix(y_test, prediction, labels=[1,0]))

# Compute confusion matrix
cnf_matrix = confusion_matrix(y_test, prediction, labels=[1,0])
np.set_printoptions(precision=2)


# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=['churn=1', 'churn=0'],
                    normalize= False, title='Confusion matrix')

print (classification_report(y_test, prediction))
display(log_loss(y_test, prediction_prob))