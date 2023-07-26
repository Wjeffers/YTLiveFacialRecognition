import cv2
import sys
import pafy
from vidgear.gears import CamGear


# importing pretrained OpenCV classifiers to handle face and eye detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

url = ''

# using vidgear to capture live youtube stream and run the face tracker on it
cap = CamGear(source = url, stream_mode= True, logging= True).start()

# read each frame and draw boxes around detected faces and eyes
while True:
    frame = cap.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    faces = face_cascade.detectMultiScale(gray, 
                                          scaleFactor = 1.1, 
                                          minNeighbors = 5,
                                          minSize=(30,30))
    
    eyes  = eye_cascade.detectMultiScale(gray, 1.1, 4)
    
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    for (x, y, w, h) in eyes:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)

    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xff == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()