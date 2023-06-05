'''
Support Vector Machine
    - Use scikit-learn to Support Vector Machine to classify.
    - I will build a model using human cell records, and classify cells to 
      whetherthe samples are benign or malignant.
'''

import pandas as pd
import pylab as pl
import numpy as np
import scipy.optimize as opt
import matplotlib.pyplot as plt

from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import classification_report, confusion_matrix
import itertools

import requests
from IPython.display import display

path = 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/'\
    'IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%203/data/'\
    'cell_samples.csv'
#r = requests.get(path, allow_redirects=True)
#open('Classification/Cell_Samples.csv', 'wb').write(r.content)

df = pd.read_csv('Classification/Cell_Samples.csv')
display(df.head(20))
display(df.columns)
display(df.dtypes)
display(df.shape)

# DISTRIBUTION OF THE CLASSES BASED ON CLUMP THICKNESS AND UNIFORMITY OF CELL 
# SIZE
ax = df[df['Class'] == 4][0:50].plot(kind='scatter', x='Clump', y='UnifSize', color='DarkBlue', label='malignant');
df[df['Class'] == 2][0:50].plot(kind='scatter', x='Clump', y='UnifSize', color='Yellow', label='benign', ax=ax);
#plt.show()

# BareNuc IS NOT NUMERICAL, WE CAN DROP THIS ROW
df = df[pd.to_numeric(df['BareNuc'], errors='coerce').notnull()]
df['BareNuc'] = df['BareNuc'].astype('int')
display(df.dtypes)

X = df[['Clump', 'UnifSize', 'UnifShape', 'MargAdh', 'SingEpiSize', 'BareNuc',
        'BlandChrom', 'NormNucl', 'Mit']]
X = np.asanyarray(X)

y = np.asanyarray(df['Class'])

# TRAIN/TEST DATASET
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                    random_state=4)

# MODELING
clf = svm.SVC(kernel='rbf') # Will use RBF (Radial Basis Function)
clf.fit(X_train, y_train)

yhat = clf.predict(X_test)

# EVALUATION
def plot_confusion_matrix(cm, classes, normalize=False, 
                        title='Confusion Matrix', cmap=plt.cm.Blues):
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
        plt.text(j, i, format(cm[i, j], fmt), horizontalalignment='center',
                 color='white' if cm[i, j] > thresh else 'black')
    
    plt.tight_layout()
    plt.xlabel('True label')
    plt.ylabel('Predicted label')
    plt.show()

# COMPUTE CONFUSION MATRIX
cnf_matrix = confusion_matrix(y_test, yhat, labels=[2,4])
np.set_printoptions(precision=2)

print(classification_report(y_test, yhat))

# PLOT COFUSION MATRIX
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=['Benign(2)', 'Malignant(4)'], 
                    normalize=False, title='Confusion matrix')