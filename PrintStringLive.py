# import the opencv library
import cv2
import matplotlib.pyplot as plt
import numpy as np
import time
import tensorflow as tf
from tensorflow import keras
import mediapipe as mp

types = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V',
         'W', 'X', 'Y', 'Z', 'Space', 'Del', 'Nothing']

# define a video capture object
vid = cv2.VideoCapture(0)

mpHands = mp.solutions.hands
hands = mpHands.Hands()

net = tf.keras.models.Sequential([
    tf.keras.layers.Dense(70, activation='relu', input_dim=64),
    tf.keras.layers.Dense(50, activation='relu'),
    tf.keras.layers.Dense(29, activation='softmax')])

net.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

checkpointFile = r'JointDetection'

net.load_weights(checkpointFile)

frame_number = 1

streak = 0
previous_result = ''
Full_string = ''

while frame_number > 0:

    # the 'q' button is set as the
    # quitting button you may use any
    # desired button of your choice
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    frame_number += 1

    if frame_number % 10 != 0:
        continue

    # Capture the video frame
    # by frame
    ret, frame = vid.read()

    # resize to be 200x200 without distortion
    # height = frame.shape[0]
    # width = frame.shape[1]
    # frame = frame[:, int(width / 2 - height / 2):int(width / 2 + height / 2)]

    # Display the resulting frame
    cv2.imshow('frame', frame)

    imageRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
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

    this_input = np.array(this_input)
    this_input = np.array([this_input])

    res = net.predict(this_input, verbose=0)
    maxi = np.argmax(res)
    result = types[maxi]
    if (previous_result == '') & (result != 'Nothing'):
        previous_result = result
    if result != 'Nothing':
        if result == previous_result:
            streak += 1
            if streak == 8:
                if result == 'Del':
                    if len(Full_string) >= 1:
                        Full_string = Full_string[:-1]
                elif result == 'Space':
                    Full_string = Full_string + ' '
                else:
                    Full_string = Full_string + result
                print(Full_string)
                streak = 0
        else:
            streak = 0
            previous_result = result

# After the loop release the cap object
vid.release()
# Destroy all the windows
cv2.destroyAllWindows()
