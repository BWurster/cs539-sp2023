import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf
from sklearn.metrics import confusion_matrix
import seaborn as sn


def trainValTest(length, trainPer, valPer):
    indices = np.arange(length)
    np.random.shuffle(indices)
    trainIndices = indices[0:int(trainPer*length)]
    valIndices = indices[int(trainPer*length):int((trainPer + valPer)*length)]
    testIndices = indices[int((trainPer + valPer)*length):]
    return trainIndices, valIndices, testIndices


np.random.seed(0)

TARGET_SIZE = (224, 224)

images = []
labels = []

path = r'../data/asl_alphabet_train/asl_alphabet_train/'

types = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V',
         'W', 'X', 'Y', 'Z', 'Space', 'Del', 'Nothing']

data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip('horizontal'),
    tf.keras.layers.RandomContrast(0.2),
    tf.keras.layers.RandomBrightness(0.2)
])

preprocess_input = tf.keras.applications.vgg16.preprocess_input

IMG_SHAPE = TARGET_SIZE + (3,)
base_model = tf.keras.applications.vgg16.VGG16(input_shape=IMG_SHAPE, include_top=False, weights='imagenet')

base_model.trainable = False

pooling_average_layer = tf.keras.layers.AveragePooling2D(pool_size=7)

flatten_layer = tf.keras.layers.Flatten()

for i in range(len(types)):
    file_list = os.listdir(path + types[i])
    for file in file_list:
        img = mpimg.imread(path + types[i] + '/' + file)
        img = tf.image.resize(img, TARGET_SIZE)
        img = tf.cast(img, dtype=tf.uint8).numpy()
        img = np.array([img])
        img = data_augmentation(img)
        img = preprocess_input(img)
        img = base_model(img)
        img = pooling_average_layer(img)
        img = flatten_layer(img).numpy()
        images.append(img)
        label = np.zeros(29)
        label[i] = 1
        labels.append(label)

images = np.array(images)
labels = np.array(labels)

trainInds, valInds, testInds = trainValTest(len(labels), 0.8, 0.1)

XTrain = images[trainInds]
yTrain = labels[trainInds]

XVal = images[valInds]
yVal = labels[valInds]

XTest = images[testInds]
yTest = labels[testInds]

prediction_layer = tf.keras.layers.Dense(29, activation='softmax')

BASE_OUTPUT_SIZE = (1, 512)

inputs = tf.keras.Input(shape=BASE_OUTPUT_SIZE)
x = flatten_layer(inputs)
outputs = prediction_layer(x)
model = tf.keras.Model(inputs, outputs)

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

checkpointFile = r'TransferVGG16'

model.load_weights(checkpointFile)

model.summary()

score = model.evaluate(XVal, yVal, verbose=0)
print("Validation loss:", format(score[0],".4f"))
print("Validation accuracy:", format(score[1],".5f"))

score = model.evaluate(XTest, yTest, verbose=0)
print("Test loss:", format(score[0],".4f"))
print("Test accuracy:", format(score[1],".5f"))

yClassified = np.argmax(model.predict(XTest), axis=1)
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
