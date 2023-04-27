import mediapipe as mp
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import os
from sklearn.metrics import confusion_matrix
import seaborn as sn
import cv2


def trainValTest(length, trainPer, valPer):
    indices = np.arange(length)
    np.random.shuffle(indices)
    trainIndices = indices[0:int(trainPer*length)]
    valIndices = indices[int(trainPer*length):int((trainPer + valPer)*length)]
    testIndices = indices[int((trainPer + valPer)*length):]
    return trainIndices, valIndices, testIndices


mpHands = mp.solutions.hands
hands = mpHands.Hands()

# Repeatable testing set.
np.random.seed(0)

inputs = []
labels = []
noHand = np.zeros(29)

path = r'../data/asl_alphabet_train/asl_alphabet_train/'

types = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V',
         'W', 'X', 'Y', 'Z', 'space', 'del', 'nothing']

for i in range(len(types)):
    # Lists all the files with the desired letter.
    file_list = os.listdir(path + types[i])
    for file in file_list:
        # Image is read in as a numpy array.
        img = cv2.imread(path + types[i] + '/' + file)

        imageRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(imageRGB)

        this_input = []
        # checking whether a hand is detected
        if results.multi_hand_landmarks:
            handedness = results.multi_handedness[0].classification[0].label
            if handedness == 'Left':
                this_input.append(-1.0)
            else:
                this_input.append(1.0)

            for id, lm in enumerate(results.multi_hand_landmarks[0].landmark):
                this_input.append(lm.x)
                this_input.append(lm.y)
                this_input.append(lm.z)
        else:
            continue

        # This stores the input.
        inputs.append(np.array(this_input))

        # This records the labels
        label = np.zeros(29)
        label[i] = 1
        labels.append(label)

inputs = np.array(inputs)
labels = np.array(labels)

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

net.load_weights(checkpointFile)

print('Objects without a detected hand (total, so 3,000 potentially for each):')
print(noHand)

score = net.evaluate(XVal, yVal, verbose=0)
print("Validation loss:", format(score[0],".4f"))
print("Validation accuracy:", format(score[1],".5f"))

score = net.evaluate(XTest, yTest, verbose=0)
print("Test loss:", format(score[0],".4f"))
print("Test accuracy:", format(score[1],".5f"))

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
