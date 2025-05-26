import cv2
import numpy as np

# Load Haar Cascade
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Load ONNX model
net = cv2.dnn.readNetFromONNX('emotion-ferplus.onnx')

# Emotion labels
emotions = ['Neutral', 'Happy', 'Surprise', 'Sad', 'Angry', 'Disgust', 'Fear', 'Contempt']

# Start webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        face_img = gray[y:y+h, x:x+w]
        face_img = cv2.resize(face_img, (64, 64))

        # Create blob
        blob = cv2.dnn.blobFromImage(face_img, scalefactor=1/255.0, size=(64, 64))

        net.setInput(blob)
        output = net.forward()

        # Find max value manually
        max_score = -1
        max_index = 0
        for i in range(len(output[0])):
            if output[0][i] > max_score:
                max_score = output[0][i]
                max_index = i

        emotion_text = emotions[max_index]

        # Draw box and label
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, emotion_text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 2)

    if len(faces) == 0:
        cv2.putText(frame, 'No Face Found', (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow('Emotion Detector', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

