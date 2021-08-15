import cv2
from random import randrange

# Pre-trained face classifier
preTrainedFaceClassifier = 'haarcascade_frontalface_default.xml'
# Pre-trained smile classifier
preTrainedSmileClassifier = 'haarcascade_smile.xml'
# Pre-trained eye classifier
preTrainedEyeClassifier = 'haarcascade_eye.xml'

# Create face classifier
faceTracker = cv2.CascadeClassifier(preTrainedFaceClassifier)
# Create smile classifier
smileTracker = cv2.CascadeClassifier(preTrainedSmileClassifier)
# Create eye classifier
eyeTracker = cv2.CascadeClassifier(preTrainedEyeClassifier)

# accessing webcam, 0 means webcam
webcam = cv2.VideoCapture(0)
# webcam = cv2.VideoCapture(abc.mp4)

# loop which will run and pressing Q will stop the execution
while True:
    # Create openCV image
    (read_successful, frame) = webcam.read()

    # If there is an error then it should abort
    if not read_successful:
        break

    # Change to grayscale
    grayScaleFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = faceTracker.detectMultiScale(grayScaleFrame)
    print(faces)

    # run smile detection within each of faces
    for (x, y, w, h) in faces:  # now, by doing loop will detect all cars inside the picture
        cv2.rectangle(frame, (x, y), (x + w, y + h), (randrange(256), randrange(256), randrange(256)), thickness=3)
        # create face sub-image (opencv allows you to subindex like this). It's built on numpy, slice is a n-dimensional array
        faces = frame[y:y + h, x:x + w]
        # Grayscale the face
        faceGrayScale = cv2.cvtColor(faces, cv2.COLOR_BGR2GRAY)
        # Detect smiles in the face, scaleFactor deal with blur effect, minNeighbor boxes in rectangle so 20 means smile
        smile = smileTracker.detectMultiScale(faceGrayScale, scaleFactor=1.7, minNeighbors=20)
        # Detect eyes in the face
        eye = eyeTracker.detectMultiScale(faceGrayScale, scaleFactor=1.1, minNeighbors=10)

        for (x_, y_, w_, h_) in smile:
            cv2.rectangle(faces, (x_,y_), (x_, w_ + y_ + h_), (50, 50, 200), 4)

        for (x__, y__, w__, h__) in eye:
            cv2.rectangle(faces, (x__,y__), (x__, w__ + y__ + h__), (255, 255, 255), 2)

        if len(smile) > 0:  # 50 means it should not touch the rectangle box
            cv2.putText(frame, 'smiling', (x, y + h + 40), fontScale=3, fontFace=cv2.FONT_HERSHEY_PLAIN,
                        color=(255, 255, 255))

        if len(eye) > 0:  # 50 means it should not touch the rectangle box
            cv2.putText(frame, 'eyes', (x, y + h + 90), fontScale=3, fontFace=cv2.FONT_HERSHEY_PLAIN,
                        color=(255, 255, 255))

    # will display the image into colorful format
    cv2.imshow('Face_Detector', frame)

    # don't auto-close (wait here in the code and listen for a key press)
    key = cv2.waitKey(1)

    # press Q key to terminate video
    if key == 81 or key == 113:
        break

webcam.release()
print("\nSuccessful 😎")
