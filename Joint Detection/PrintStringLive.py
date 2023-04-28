# import the opencv library
import cv2
import matplotlib.pyplot as plt
import numpy as np
import time
import tensorflow as tf
from tensorflow import keras
import mediapipe as mp

types = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V',
         'W', 'X', 'Y', 'Z', 'space', 'del', 'nothing']

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

streak = 0
previous_result = ''
full_string = ''
active_result = ''
active_count = 0
CONFIDENCE_THRESHOLD = 10

while True:

    # the 'q' button is set as the
    # quitting button you may use any
    # desired button of your choice
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

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

        this_input = np.array(this_input)
        this_input = np.array([this_input])

        res = net.predict(this_input, verbose=0)
        maxi = np.argmax(res)
        result = types[maxi]
    else:
        result = "nothing"

    if(result == active_result and result != previous_result):
        if active_count > CONFIDENCE_THRESHOLD:
            previous_result = active_result

            if result != 'nothing':
                if result == 'del':
                    if len(full_string) >= 1:
                        full_string = full_string[:-1]
                elif result == 'space':
                    full_string = full_string + ' '
                else:
                    full_string = full_string + result
                print('\x1b[2K', end='')
                print(">" + full_string + "<", end='\r')
        else:
            active_count += 1
    else:
        active_result = result
        active_count = 0

# After the loop release the cap object
vid.release()
# Destroy all the windows
cv2.destroyAllWindows()
