'''
Steps:
Import openCV for gray scale conversion, so that algo run fast
Import video file format
Used already pre-trained model from Haar cascaded classifier
Created car and pedestrian classifier
Video reader to read our file
Gray conversion
Detect cars and pedestrian
draw rectangle around cars and pedestrian
Will display into color format
Q button to terminate
'''
import cv2  # real-time computer vision library. Focuses on image processing, video capture and obj detections
from random import randrange  # for rainbow color (randrange(256), randrange(256), randrange(256)

# Car and pedestrian videos
# video = cv2.VideoCapture('car.mp4')
# video = cv2.VideoCapture('pedestrian.mp4')
video = cv2.VideoCapture('car and people.avi')

# Pre-trained car classifier
preTrainedCarClassifier = 'trained-cars-detection-haar-classifier-data.xml'
# Pre-trained pedestrian classifier
preTrainedPedestrianClassifier = 'haarcascade_fullbody.xml'

# Create car classifier
car_tracker = cv2.CascadeClassifier(preTrainedCarClassifier)
# Create pedestrian classifier
pedastrian_tracker = cv2.CascadeClassifier(preTrainedPedestrianClassifier)

# loop which will run and pressing Q will stop the execution
while True:
    # Create openCV image
    (read_successful, frame) = video.read()

    # Safe coding
    if read_successful:
        # Convert to gray scale - required for HAAR cascade classifier
        grayScaleFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        break

    # Detect cars and pedestrian
    cars = car_tracker.detectMultiScale(grayScaleFrame)
    print(cars)
    pedestrian = pedastrian_tracker.detectMultiScale(grayScaleFrame)
    print(pedestrian)

    # draw rectangle around cars and pedestrian
    for (x, y, w, h) in cars:  # now, by doing loop will detect all cars inside the picture
        cv2.rectangle(frame, (x, y), (x + w, y + h), (randrange(256), randrange(256), randrange(256)), 5)
    for (x, y, w, h) in pedestrian:  # now, by doing loop will detect all cars inside the picture
        cv2.rectangle(frame, (x, y), (x + w, y + h), (randrange(256), randrange(256), randrange(256)), 5)

    # will display the image into colorful format
    cv2.imshow('Cars_Detector', frame)

    # don't auto-close (wait here in the code and listen for a key press)
    key = cv2.waitKey(1)

    # press Q key to terminate video
    if key == 81 or key == 113:
        break

video.release()
print("\nSuccessful 😎")