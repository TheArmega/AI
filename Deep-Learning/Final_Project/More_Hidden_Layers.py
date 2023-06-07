import keras
import pandas as pd
import numpy  as np

from sklearn.model_selection import train_test_split

from keras.models import Sequential
from keras.layers import Dense
from keras.utils  import to_categorical

from keras.datasets import mnist

from IPython.display import display

# read the data
concrete_data = pd.read_csv('https://cocl.us/concrete_data')
display(concrete_data.head())

# Split data into predictors and target
concrete_data_columns = concrete_data.columns
predictors = concrete_data[concrete_data_columns
                        [concrete_data_columns != 'Strength']]
target = concrete_data['Strength']

# Normalize data
predictors_norm = (predictors - predictors.mean()) - predictors.std()

n_cols = predictors_norm.shape[1]

def regression_model():

    # Create the model
    model = Sequential()
    model.add(Dense(10, activation='relu', input_shape=(n_cols,)))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(1))

    # Compile model
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Build the model
model = regression_model()

list_of_mean_squared_error = []

for i in range(50):
    # Split the data randomly into a training set of 70% and a test set of 30%
    X_train, X_test, y_train, y_test = train_test_split(predictors, target,
                                                        test_size=0.3)
    # Fit the model
    fit = model.fit (X_train, y_train, validation_data=(X_test, y_test),
                     epochs=50)
    # Finf mean aquared error as last value in history
    mean_squared_error = fit.history['val_loss'][-1]
    # Save mean squared error of all cycles
    list_of_mean_squared_error.append(mean_squared_error)
    print('Cycle {}: mean squared error {}'.format(i + 1, mean_squared_error))

print('Mean of mean squared errors: {}'.format(np.mean(
    list_of_mean_squared_error)))
print('Standar derivation of the mean squared errors: {}'.format(np.std(
    list_of_mean_squared_error)))

'''
SOLUTION:
The mean and the standard deviation of the mean squared errors in case D is less
than in case A, B and C. And it's the only case where error is not very big.
It means additional layers in neural network are more important than other
things.
'''