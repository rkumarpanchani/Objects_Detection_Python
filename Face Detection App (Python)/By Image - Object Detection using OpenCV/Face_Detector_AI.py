"""
Packages I used: 1) OpenCV
https://github.com/opencv/opencv/tree/master/data/haarcascades (They provided us with the trained datasets)
https://github.com/anaustinbeing/haar-cascade-files
Control+ALt+L for auto-formatting code
Classifier or detector => CascadeClassifier uses haar cascade algorithm.
"""
import cv2 # Open source computer vision library for gray scale images
from random import randrange

# Step 1:
# load pre-trained data 'haarcascade_frontalface_default.xml' from opencv (haar cascade algorithm)
trainedFaceData = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Step 2:
# reading an image smile_photo.jpg
image = cv2.imread('multiple_faces_photo.jpg')

# now, I changed the image 'smile_photo' to gray color
grayScaledImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# now, detectMultiScale will take all sizes and shapes of images, help us to detect the image and it is doing all these from gray scaled image
faceCoordinates = trainedFaceData.detectMultiScale(grayScaledImage)
print(faceCoordinates)

# drawing rectangle around the face
for (x, y, w, h) in faceCoordinates: # now, by doing loop will detect all faces inside the picture
    cv2.rectangle(image, (x, y), (x + w, y + h), (randrange(256), randrange(256), randrange(256)), 2) # 255 is green and 2 is thickness

# will display the image
cv2.imshow('Face_Detector', image)

# wait key just put little delay before popping up the image
cv2.waitKey()

print("\nSuccessful 😎")