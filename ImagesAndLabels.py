import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf
from tensorflow import keras


def rgb2gray(rgb):
    return np.dot(rgb[..., :3], [3, 6, 1]).astype('int16')


def brighten(regImages):
    for ind in np.arange(len(regImages)):
        regImages[ind] = np.clip(regImages[ind] + int(np.random.normal(scale=400)), 0, 2550)
    return regImages


def trainValTest(length, trainPer, valPer):
    indices = np.arange(length)
    np.random.shuffle(indices)
    trainIndices = indices[0:int(trainPer*length)]
    valIndices = indices[int(trainPer*length):int((trainPer + valPer)*length)]
    testIndices = indices[int((trainPer + valPer)*length):]
    return trainIndices, valIndices, testIndices


images = []
labels = []

path = r'asl_alphabet_train/asl_alphabet_train/'

types = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V',
         'W', 'X', 'Y', 'Z', 'space', 'del', 'nothing']

for i in range(len(types)):
    file_list = os.listdir(path + types[i])
    for file in file_list:
        img = mpimg.imread(path + types[i] + '/' + file)
        gray = rgb2gray(img)
        images.append(gray)
        label = np.zeros(29)
        label[i] = 1
        labels.append(label)

images = np.array(images)
labels = np.array(labels)

trainInds, valInds, testInds = trainValTest(len(labels), 0.8, 0.1)

XTrain = images[trainInds]
XTrain = brighten(XTrain)
yTrain = labels[trainInds]

XVal = images[valInds]
yVal = labels[valInds]

XTest = images[testInds]
yTest = images[testInds]

net = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(3, 5, padding='same', activation='relu', input_shape=(200, 200, 1)),
    tf.keras.layers.MaxPool2D((2, 2)),
    tf.keras.layers.Conv2D(5, 3, padding='same', activation='relu'),
    tf.keras.layers.MaxPool2D((2, 2)),
    tf.keras.layers.Conv2D(10, 3, padding='same', activation='relu'),
    tf.keras.layers.MaxPool2D((2, 2)),
    tf.keras.layers.Conv2D(10, 5, strides=5, padding='valid', activation='relu'),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(50, activation='relu'),
    tf.keras.layers.Dense(40, activation='relu'),
    tf.keras.layers.Dense(29, activation='softmax')])

net.summary()

net.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

net.fit(XTrain, yTrain, epochs=100, batch_size=300, validation_data=(XVal, yVal))
