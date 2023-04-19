import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf

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

checkpointFile = r'SavedModels/TransferVGG16'

# This reads in the weights for the top layers.
top_model.load_weights(checkpointFile)

# This is the bottom layers (the predefined stuff)
bottom_input = tf.keras.Input(shape=(TARGET_SIZE + (3,)))
x = preprocess_input(inputs)
x = base_model(x)
x = pooling_average_layer(x)
bottom_output = flatten_layer(x)
bottom_model = tf.keras.Model(bottom_input, bottom_output)

# This is a demo of running it on a picture
img = mpimg.imread('Atest.jpg')
img = tf.image.resize(img, TARGET_SIZE)
img = tf.cast(img, dtype=tf.uint8).numpy()
img = np.array([img])

prediction = top_model.predict(bottom_model.predict(img))[0]

print(np.where(prediction == np.max(prediction))[0])
