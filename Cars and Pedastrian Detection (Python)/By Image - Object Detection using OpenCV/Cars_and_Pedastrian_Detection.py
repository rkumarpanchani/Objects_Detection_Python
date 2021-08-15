import cv2  # real-time computer vision library. Focuses on image processing, video capture and obj detections
from random import randrange # for rainbow color (randrange(256), randrange(256), randrange(256)

# Our image
image = 'cars-photo.jpg'

# Pre-trained car classifier
preTrainedClassifier = 'trained-cars-detection-haar-classifier-data.xml'

# Create openCV image
img = cv2.imread(image)

# Convert to gray scale - required for HAAR cascade classifier
grayScaleCoversion = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Create car classifier
car_tracker = cv2.CascadeClassifier(preTrainedClassifier)

# Detect cars
cars = car_tracker.detectMultiScale(grayScaleCoversion)
print(cars)

# draw rectangle around cars
for (x, y, w, h) in cars:  # now, by doing loop will detect all cars inside the picture
    cv2.rectangle(img, (x, y), (x + w, y + h), (randrange(256), randrange(256), randrange(256)), 5)

# will display the image into colorful format
cv2.imshow('Cars_Detector', img)

# don't autoclose (wait here in the code and listen for a key press)
key = cv2.waitKey()

print("\nSuccessful 😎")