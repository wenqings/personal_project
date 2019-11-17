'''Trains a simple deep NN on the MNIST dataset.
Gets to 98.40% test accuracy after 20 epochs
(there is *a lot* of margin for parameter tuning).
2 seconds per epoch on a K520 GPU.
'''

from __future__ import print_function

import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout
import cv2
# The batch size defines the number of samples that will be propagated through the network.
# For large data, we can start at 32, then try 64, 128...
# If batch size = 1, DNN may jump into a local minimum instead of Global minimum for gradient decent
batch_size = 128
num_classes = 10
epochs = 20

# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()



cv2.imshow('',x_test[2])
cv2.waitKey(0)

# Change each digit picture into 784 points
x_train = x_train.reshape(60000, 784)
x_test = x_test.reshape(10000, 784)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
# Change each points between 0~1
x_train /= 255
x_test /= 255
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

# ---------- All you need focus is here ----------------
# What is activation function: After each loop training, change a node just to ON or OFF
# Like 四舍五入， or some other rule
# Available activations
# elu
# softmax
# selu
# softplus
# softsign
# relu (most common use)
# tanh
# sigmoid
# hard_sigmoid
# exponential
# linear

model = Sequential()
# 666 is I put here, which define how many nodes on the first layer
model.add(Dense(666, activation='relu', input_shape=(784,)))
# Dropout consists in randomly setting a fraction rate of input units to 0
# at each update during training time, which helps prevent overfitting.
model.add(Dropout(0.2))
# Second layer we set 888 nodes, change the node number can somehow effect the result
# You need to try and test
model.add(Dense(888, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(num_classes, activation='softmax'))
# ---------- All you need focus is here ----------------


model.summary()
# Available loss functions (I don't care which one to use)
# mean_squared_error
# mean_absolute_error
# mean_absolute_percentage_error
# mean_squared_logarithmic_error
# squared_hinge
# hinge
# categorical_hinge
# logcosh
# huber_loss
# categorical_crossentropy
# sparse_categorical_crossentropy
# binary_crossentropy
# kullback_leibler_divergence
# poisson
# cosine_proximity
# is_categorical_crossentropy

# Available optimizers (I don't know anything about it)
# SGD
# RMSprop
# Adagrad
# Adadelta
# Adam
# Adamax
# Nadam
model.compile(loss='mean_squared_error',
              optimizer=keras.optimizers.Nadam(),
              metrics=['accuracy'])

# verbose: Integer. 0, 1, or 2. Verbosity mode. 0 = silent, 1 = progress bar, 2 = one line per epoch.
# It just used for showing you the log
history = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_data=(x_test, y_test))
score = model.evaluate(x_test, y_test, verbose=0)
model.save('digit_classifier.h5')
print('Test loss:', score[0])
print('Test accuracy:', score[1])



