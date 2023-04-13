import cv2
import numpy as np
import time
import os

# constants for file generation
# make base name something unique for each run
# make type the classifier for the letter; this will be used to put things in
# the right folder.
DATA_PATH = os.path.join("data", "asl_alphabet_train", "asl_alphabet_train")
BASE_FILE_NAME = "Ben_4-13"
TYPE = 'A'

# set up video capture object
vid = cv2.VideoCapture(0)

LOOP_CONSTANT = 50
i = 0
image_count = 0

while(True):
    # get image from camera
    ret, frame = vid.read()

    # resize to be 200x200 without distortion
    height = frame.shape[0]
    width = frame.shape[1]
    frame = frame[:, int(width/2-height/2):int(width/2+height/2), :]
    frame=cv2.resize(frame, (200, 200), interpolation = cv2.INTER_AREA)

    # save image only some of the time
    i += 1
    if(i > LOOP_CONSTANT):
        cv2.imwrite(DATA_PATH + '/' + TYPE + '/' + BASE_FILE_NAME + '_' + str(image_count) + ".jpg", frame)
        image_count += 1

        frame *= 0 # black to signify capture
        i = 0

  
    # display frame
    cv2.imshow('frame', frame)
      
    # the 'q' button is set as the
    # quitting button you may use any
    # desired button of your choice
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
  
# After the loop release the cap object
vid.release()
# Destroy all the windows
cv2.destroyAllWindows()

