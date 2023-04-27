This is a transfer learning model that utilizes the position of joints from a
pre-trained Google Mediapipe hand feature detection model. This allowed us to
generate points for various hand positions, regardless of where they are in the
frame and train a neural network off of 64 hand features rather than a whole
200x200 image of pixel values. This signifiantly sped up training and resulted
in a highly generalizable model.

`PrintStringLive.py` is a great program to showcase the power of the model that
was developed. It allows for the typing of a string by the user with only hand
gestures. I highly recommend you check this out! (note that to run it you need
to install all necessary dependencies in the `requirements.txt` file in the root 
directory)