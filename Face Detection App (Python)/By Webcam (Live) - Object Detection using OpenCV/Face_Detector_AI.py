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

# will turn on the webcam and (0) means by default laptop webcam will be turned on
webcam = cv2.VideoCapture(0) # will open your webcam

while True:
    # to read the current frame from video
    successfulFrameRead, frame = webcam.read()

    # now, converting to gray color
    grayScaledImage = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # now detecting faces from gray color using training
    faceCoordinates = trainedFaceData.detectMultiScale(grayScaledImage)

    '''
    drawing rectangle around the face
    (x, y), (x + w, y + h) => x-axis, y-axis, width, height
    (randrange(256), randrange(256), randrange(256) rainbow box or (255,255,255) or (0,255,0)
    2 is thickness of the box, can modify it
    '''
    for (x, y, w, h) in faceCoordinates:  # now, by doing loop will detect all faces inside the picture
        cv2.rectangle(frame, (x, y), (x + w, y + h), (randrange(256), randrange(256), randrange(256)), 2)

    # will display the video into colorful format
    cv2.imshow('Face_Detector', frame)

    # don't auto-close (wait here in the code and listen for a key press)
    key = cv2.waitKey(1)

    #ASCII for Q key, to terminate the video
    if key==81 or key==113:
        break

webcam.release()
print("\nSuccessful 😎")