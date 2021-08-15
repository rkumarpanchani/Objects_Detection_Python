"""
Packages I used: 1) OpenCV
https://github.com/opencv/opencv/tree/master/data/haarcascades  (They provided us with the trained datasets)
https://github.com/anaustinbeing/haar-cascade-files
Control+ALt+L for auto-formatting code
Classifier or detector => CascadeClassifier uses haar cascade algorithm.
"""

import cv2  # Open source computer vision library for gray scale images
from random import randrange

# load pre-trained data 'haarcascade_frontalface_default.xml' from opencv (haar cascade algorithm)
trainedFaceData = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

webcam = cv2.VideoCapture('abc.mp4') # will read from this exactly video file from file path

while True:
    # to read the current frame from video
    successfulFrameRead, frame = webcam.read()

    # now, converting to gray color
    grayScaledImage = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # now detecting faces from gray color using training
    faceCoordinates = trainedFaceData.detectMultiScale(grayScaledImage)

    # drawing rectangle around the face
    for (x, y, w, h) in faceCoordinates:  # now, by doing loop will detect all faces inside the picture
        cv2.rectangle(frame, (x, y), (x + w, y + h), (randrange(256), randrange(256), randrange(256)), 5)

    # will display the video into colorful format
    cv2.imshow('Face_Detector', frame)

    # wait key just put little delay and if don't declared then won't work
    key = cv2.waitKey(1)

    #ASCII for Q key, to terminate the video
    if key==81 or key==113:
        break

webcam.release()
print("\nSuccessful 😎")