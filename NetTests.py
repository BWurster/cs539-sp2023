# Everything in this file is identical to the training file up until the end when testing is done.
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf
from sklearn.metrics import confusion_matrix
import seaborn as sn


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


np.random.seed(0)

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
yTest = labels[testInds]

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

net.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

checkpointFile = r'SavedModels/ChangingBatchSizes'

# Instead of training, we load in the pre trained model weights.
net.load_weights(checkpointFile)

# This gives the validation set accuracy.
score = net.evaluate(XVal, yVal, verbose=0)
print("Validation loss:", format(score[0],".4f"))
print("Validation accuracy:", format(score[1],".5f"))

# This gives the test set accuracy.
score = net.evaluate(XTest, yTest, verbose=0)
print("Test loss:", format(score[0],".4f"))
print("Test accuracy:", format(score[1],".5f"))

# This is all code from class to produce confusion matrices.
yClassified = np.argmax(net.predict(XTest), axis=1)
yTrue = np.argmax(yTest, axis=1)
print("Confusion matrix: \n", confusion_matrix(yTrue, yClassified))


def plot_confusion_matrix(yClassified, yTrue):
    # Compute confusion matrix
    c_mat = np.zeros((yTest.shape[1],yTest.shape[1]))
    for i in range(len(yTrue)):
        c_mat[yClassified[i], yTrue[i] ] += 1

    group_counts = ["{0:0.0f}".format(value) for value in c_mat.flatten()]
    group_percentages = ["{0:.2%}".format(value) for value in c_mat.flatten()/np.sum(c_mat)]
    labels = [f"{v1}\n{v2}" for v1, v2 in zip(group_counts, group_percentages)]
    labels = np.asarray(labels).reshape(c_mat.shape[0], c_mat.shape[1])

    plt.figure(figsize=(12,10))
    sn.heatmap(c_mat, annot=labels, fmt='', cmap='rocket_r')
    plt.title("Confusion Matrix")
    plt.ylabel('Output Class')
    plt.xlabel('Target Class')
    plt.show()


plot_confusion_matrix(yClassified, yTrue)
