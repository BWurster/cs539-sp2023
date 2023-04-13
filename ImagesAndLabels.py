import matplotlib.image as mpimg
import numpy as np
import os
import tensorflow as tf


# This converts an image to grayscale while keeping it as an int.
def rgb2gray(rgb):
    return np.dot(rgb[..., :3], [3, 6, 1]).astype('int16')


# This takes grayscale images and applies random brightening to them.
def brighten(regImages):
    for ind in np.arange(len(regImages)):
        regImages[ind] = np.clip(regImages[ind] + int(np.random.normal(scale=400)), 0, 2550)
    return regImages


# This gets a random list of indices for training, validating, and testing based on the percentage the user wants to be trained and validated.
def trainValTest(length, trainPer, valPer):
    indices = np.arange(length)
    np.random.shuffle(indices)
    trainIndices = indices[0:int(trainPer*length)]
    valIndices = indices[int(trainPer*length):int((trainPer + valPer)*length)]
    testIndices = indices[int((trainPer + valPer)*length):]
    return trainIndices, valIndices, testIndices


# This is done so that testing is possible after the fact since we can refind the testing data.
np.random.seed(0)

images = []
labels = []

path = r'asl_alphabet_train/asl_alphabet_train/'

types = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V',
         'W', 'X', 'Y', 'Z', 'space', 'del', 'nothing']

# This reads in all the images and gives them a one hot label.
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

# Test, val, train split with a 0.8, 0.1, 0.1 distribution. The training images are artificially brightened.
trainInds, valInds, testInds = trainValTest(len(labels), 0.8, 0.1)

XTrain = images[trainInds]
XTrain = brighten(XTrain)
yTrain = labels[trainInds]

XVal = images[valInds]
yVal = labels[valInds]

XTest = images[testInds]
yTest = labels[testInds]

# Structure of the neural network.
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

# Cross entropy is used as this is a categorical classifier.
net.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# This is used to save model weights.
checkpointFile = r'SavedModels/ChangingBatchSizes'

# This is used to save whichever model has the highest validation accuracy.
modelCheckpointCallback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpointFile,
    monitor='val_accuracy',
    verbose=1,
    mode='max',
    save_best_only=True,
    save_weights_only=True)

# Fitting is done in three stages with differing batch sizes to learn quicker at the beginning but more accurately at the end.
net.fit(XTrain, yTrain, epochs=30, batch_size=50, validation_data=(XVal, yVal), callbacks=[modelCheckpointCallback])

net.fit(XTrain, yTrain, epochs=30, batch_size=300, validation_data=(XVal, yVal), callbacks=[modelCheckpointCallback])

net.fit(XTrain, yTrain, epochs=60, batch_size=600, validation_data=(XVal, yVal), callbacks=[modelCheckpointCallback])
