'''
Convolutional Neural Networks with Keras
    - How to use the Keras library to build convolutional neural networks.
    - Convolutional neural network with one convolutional and pooling layers.
    - Convolutional neural network with two convolutional and pooling layers.
'''

import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.utils  import to_categorical

from keras.layers.convolutional import Conv2D # to add convolutional layers
from keras.layers.convolutional import MaxPooling2D # to add pooling layers
from keras.layers import Flatten # to flatten data for fully connected layers

from keras.datasets import mnist

# CONVOLUTIONAL LAYER WITH ONE SET OF CONVOLUTIONAL AND POOLING LAYERS
# load data
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# reshape to be [samples][pixels][width][height]
X_train = X_train.reshape(X_train.shape[0], 28, 28, 1).astype('float32')
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1).astype('float32')

#  normalize the pixel values between 0 and 1
X_train = X_train / 255
X_test  = X_test / 255

# let's convert the target variable into binary categories
y_train = to_categorical(y_train)
y_test  = to_categorical(y_test)

num_classes = y_test.shape[1] # number of categories

def convolutional_model():
    
    # create model
    model = Sequential()
    model.add(Conv2D(16, (5, 5), strides=(1, 1), activation='relu',
                     input_shape=(28, 28, 1)))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    model.add(Flatten())
    model.add(Dense(100, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))

    # compile model
    model.compile(optimizer='adam', loss='categorical_crossentropy',
                   metrics=['accuracy'])
    return model

# build, fit and evaluate the model
model = convolutional_model()
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10,
          batch_size=200, verbose=2)

scores = model.evaluate(X_test, y_test, verbose=0)
print('Accuracy: {}\nError: {}'.format(scores[1], 100-scores[1]*100))

def convolutional_model_2():
    # create model
    model = Sequential()
    model.add(Conv2D(16, (5, 5), activation='relu', input_shape=(28, 28, 1)))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    
    model.add(Conv2D(8, (2, 2), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    
    model.add(Flatten())
    model.add(Dense(100, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    
    # Compile model
    model.compile(optimizer='adam', loss='categorical_crossentropy',  metrics=['accuracy'])
    return model

# build, fit and evaluate the model
model = convolutional_model_2()
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10,
          batch_size=200, verbose=2)

scores = model.evaluate(X_test, y_test, verbose=0)
print('Accuracy: {}\nError: {}'.format(scores[1], 100-scores[1]*100))