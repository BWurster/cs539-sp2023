import mediapipe as mp
import matplotlib.image as mpimg
import numpy
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import tensorflow as tf
import os


def trainValTest(length, trainPer, valPer):
    indices = np.arange(length)
    np.random.shuffle(indices)
    trainIndices = indices[0:int(trainPer*length)]
    valIndices = indices[int(trainPer*length):int((trainPer + valPer)*length)]
    testIndices = indices[int((trainPer + valPer)*length):]
    return trainIndices, valIndices, testIndices


base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
options = vision.HandLandmarkerOptions(base_options=base_options)
detector = vision.HandLandmarker.create_from_options(options)

# Repeatable testing set.
np.random.seed(0)

inputs = []
labels = []

path = r'data_Kyle/asl_alphabet_train/asl_alphabet_train/'

types = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V',
         'W', 'X', 'Y', 'Z', 'Space', 'Del', 'Nothing']

data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip('horizontal'),
    tf.keras.layers.RandomContrast(0.2),
    tf.keras.layers.RandomBrightness(0.2)
])

for i in range(len(types)):
    # Lists all the files with the desired letter.
    file_list = os.listdir(path + types[i])
    for file in file_list:
        # Image is read in as a numpy array.
        img = mpimg.imread(path + types[i] + '/' + file)
        img = np.array(img)
        # Data augmentation is performed.
        # Critical step is flipping horizontal so right and left hands are trained.
        numpy_image = np.array(data_augmentation(img)).astype(numpy.uint8)
        # Image is converted into format that can be used for hand detection.
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=numpy_image)
        # Detection is run on the object.
        detection_result = detector.detect(image)
        this_input = []
        # Nothing will just be characterized by no hand being detected.
        # The data format is different though, so it can't be trained.
        if len(detection_result.handedness) == 0:
            continue

        if detection_result.handedness[0][0].category_name == 'Left':
            this_input.append(-1.0)
        else:
            this_input.append(1.0)

        # There are 21 total joint locations that get recorded.
        for joint in range(21):
            this_input.append(detection_result.hand_landmarks[0][joint].x)
            this_input.append(detection_result.hand_landmarks[0][joint].y)
            this_input.append(detection_result.hand_landmarks[0][joint].z)

        # This stores the input.
        inputs.append(np.array(this_input))

        # This records the labels
        label = np.zeros(29)
        label[i] = 1
        labels.append(label)

inputs = np.array(inputs)
labels = np.array(labels)
print(len(labels))

trainInds, valInds, testInds = trainValTest(len(labels), 0.8, 0.1)

XTrain = inputs[trainInds]
yTrain = labels[trainInds]

XVal = inputs[valInds]
yVal = labels[valInds]

XTest = inputs[testInds]
yTest = labels[testInds]

net = tf.keras.models.Sequential([
    tf.keras.layers.Dense(70, activation='relu', input_dim=len(XTrain[0])),
    tf.keras.layers.Dense(50, activation='relu'),
    tf.keras.layers.Dense(29, activation='softmax')])

net.summary()

net.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

checkpointFile = r'SavedModels/JointDetection'

modelCheckpointCallback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpointFile,
    monitor='val_accuracy',
    verbose=1,
    mode='max',
    save_best_only=True,
    save_weights_only=True)

net.fit(XTrain, yTrain, epochs=30, batch_size=50, validation_data=(XVal, yVal), callbacks=[modelCheckpointCallback])

net.fit(XTrain, yTrain, epochs=30, batch_size=300, validation_data=(XVal, yVal), callbacks=[modelCheckpointCallback])

net.fit(XTrain, yTrain, epochs=60, batch_size=600, validation_data=(XVal, yVal), callbacks=[modelCheckpointCallback])

score = net.evaluate(XTest, yTest, verbose=0)
print("Test loss:", format(score[0],".4f"))
print("Test accuracy:", format(score[1],".5f"))
