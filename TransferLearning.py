import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf


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

path = r'data_Kyle/asl_alphabet_train/asl_alphabet_train/'

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

# This reads the data through all of the constant layers so that feed forward steps are quicker down the line.
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

# This is the top layer which actually makes predictions.
inputs = tf.keras.Input(shape=BASE_OUTPUT_SIZE)
x = flatten_layer(inputs)
outputs = prediction_layer(x)
model = tf.keras.Model(inputs, outputs)

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

checkpointFile = r'SavedModels/TransferVGG16'

modelCheckpointCallback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpointFile,
    monitor='val_accuracy',
    verbose=1,
    mode='max',
    save_best_only=True,
    save_weights_only=True)

model.summary()

model.fit(XTrain, yTrain, epochs=30, batch_size=50, validation_data=(XVal, yVal), callbacks=[modelCheckpointCallback])

model.fit(XTrain, yTrain, epochs=30, batch_size=300, validation_data=(XVal, yVal), callbacks=[modelCheckpointCallback])

model.fit(XTrain, yTrain, epochs=60, batch_size=600, validation_data=(XVal, yVal), callbacks=[modelCheckpointCallback])
