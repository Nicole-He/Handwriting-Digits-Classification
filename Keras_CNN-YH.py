# -*- coding: utf-8 -*-
"""
Created on Sun Oct 21 12:00:33 2018

@author: Yiling
"""

import keras
from keras.datasets import mnist
from keras.layers import Dense, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.models import Sequential
import matplotlib.pylab as plt

batch_size = 128
num_classes = 10
epochs = 10

img_x, img_y = 28, 28

(x_train, y_train), (x_test, y_test) = mnist.load_data()

f, axarr = plt.subplots(2, 2)
axarr[0, 0].imshow(x_train[0, :, :], cmap = 'gray')
axarr[0, 1].imshow(x_train[1, :, :], cmap = 'gray')
axarr[1, 0].imshow(x_train[2, :, :], cmap = 'gray')
axarr[1, 1].imshow(x_train[3, :, :], cmap = 'gray')

plt.show()

print("The data type of x_train is {0} and the size = {1}".format(type(x_train),x_train.shape))
print("The data type of y_train is {0} and the size = {1}".format(type(y_train),y_train.shape))
print("The data type of x_test is {0} and the size = {1}".format(type(x_test),x_test.shape))
print("The data type of y_test is {0} and the size = {1}".format(type(y_test),y_test.shape))

x_train = x_train.reshape(x_train.shape[0], img_x, img_y, 1)
x_test = x_test.reshape(x_test.shape[0], img_x, img_y, 1)
input_shape = (img_x, img_y, 1)

print("The data types of array elements of x_train and x_test are {0} and {1}, respectively".format(x_train.dtype, x_test.dtype))

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

x_train /= 255
x_test /= 255

print("x_train shape :", x_train.shape)
print(x_train.shape[0], "train samples")
print(x_test.shape[0], "test samples")

y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

model = Sequential()
model.add(Conv2D(32, kernel_size = (5, 5), strides = (1, 1), activation = 'relu', input_shape = input_shape))
model.add(MaxPooling2D(pool_size = (2, 2), strides = (2, 2)))
model.add(Conv2D(64, kernel_size = (5, 5), activation ='relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(Flatten())
model.add(Dense(1000, activation = 'relu'))
model.add(Dense(num_classes, activation = 'softmax'))

model.compile(loss = keras.losses.categorical_crossentropy, 
              optimizer = keras.optimizers.Adam(),
              metrics = ['accuracy'])

model.fit(x_train, y_train, batch_size = batch_size, epochs = epochs, verbose = 1,
          validation_data = (x_test, y_test))

score = model.evaluate(x_test, y_test, verbose = 0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

