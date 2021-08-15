import cv2
from random import randrange

# Pre-trained face classifier
preTrainedFacClassifier = 'haarcascade_frontalface_default.xml'
# Pre-trained smile classifier
preTrainedSmileClassifier = 'haarcascade_smile.xml'

# Create face classifier
faceTracker = cv2.CascadeClassifier(preTrainedFacClassifier)
# Create smile classifier
smileTracker = cv2.CascadeClassifier(preTrainedSmileClassifier)

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
        cv2.rectangle(frame, (x, y), (x + w, y + h), (randrange(256), randrange(256), randrange(256)), thickness=2)
        # create face sub-image (opencv allows you to subindex like this). It's built on numpy, slice is a n-dimensional array
        faces = frame[y:y + h, x:x + w]
        # Grayscale the face
        faceGrayScale = cv2.cvtColor(faces, cv2.COLOR_BGR2RGB)
        # Detect smiles in the face, scaleFactor deal with blur effect, minNeighbor boxes in rectangle so 20 means smile
        smile = smileTracker.detectMultiScale(faceGrayScale, scaleFactor=1.7, minNeighbors=20)

        if len(smile) > 0:  # 50 means it should not touch the rectangle box
            cv2.putText(frame, 'smiling', (x, y + h + 50), fontScale=3, fontFace=cv2.FONT_HERSHEY_PLAIN, color=(255, 255, 255))

    # will display the image into colorful format
    cv2.imshow('Face_Detector', frame)

    # don't auto-close (wait here in the code and listen for a key press)
    key = cv2.waitKey(1)

    # press Q key to terminate video
    if key == 81 or key == 113:
        break

webcam.release()
print("\nSuccessful 😎")