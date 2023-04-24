# import the opencv library
import cv2
import matplotlib.pyplot as plt
import numpy as np
import time
import tensorflow as tf
from tensorflow import keras

types = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V',
         'W', 'X', 'Y', 'Z', 'space', 'del', 'nothing']
  
# define a video capture object
vid = cv2.VideoCapture(0)

TARGET_SIZE = (224, 224)
preprocess_input = tf.keras.applications.vgg16.preprocess_input
IMG_SHAPE = TARGET_SIZE + (3,)
base_model = tf.keras.applications.vgg16.VGG16(input_shape=IMG_SHAPE, include_top=False, weights='imagenet')
base_model.trainable = False
pooling_average_layer = tf.keras.layers.AveragePooling2D(pool_size=7)
flatten_layer = tf.keras.layers.Flatten()
prediction_layer = tf.keras.layers.Dense(29, activation='softmax')
BASE_OUTPUT_SIZE = 512
# This defines the top layers
top_input = tf.keras.Input(shape=BASE_OUTPUT_SIZE)
x = flatten_layer(top_input)
top_output = prediction_layer(x)
top_model = tf.keras.Model(top_input, top_output)
checkpointFile = r'TransferVGG16'
# This reads in the weights for the top layers.
top_model.load_weights(checkpointFile)
# This is the bottom layers (the predefined stuff)
bottom_input = tf.keras.Input(shape=(TARGET_SIZE + (3,)))
x = preprocess_input(bottom_input)
x = base_model(x)
x = pooling_average_layer(x)
bottom_output = flatten_layer(x)
bottom_model = tf.keras.Model(bottom_input, bottom_output)
  
while(True):
    # Capture the video frame
    # by frame
    ret, frame = vid.read()

    # resize to be 200x200 without distortion
    height = frame.shape[0]
    width = frame.shape[1]
    frame = frame[:, int(width/2-height/2):int(width/2+height/2)]
  
    # Display the resulting frame
    cv2.imshow('frame', frame)

    img = tf.image.resize(frame, TARGET_SIZE)
    img = tf.cast(img, dtype=tf.uint8).numpy()
    img = np.reshape(img, (1, 224, 224, 3))

    res = top_model.predict(bottom_model.predict(img, verbose=0), verbose=0)[0]
    maxi = np.argmax(res)
    result = types[maxi]
    print(result)
      
    # the 'q' button is set as the
    # quitting button you may use any
    # desired button of your choice
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
  
# After the loop release the cap object
vid.release()
# Destroy all the windows
cv2.destroyAllWindows()