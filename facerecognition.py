import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from keras.preprocessing.image import img_to_array
import datetime
import matplotlib.pyplot as plt

# Ensure OpenCV is properly installed
try:
    cv2.imshow  # Check if OpenCV GUI functions are available
except AttributeError:
    print("Error: OpenCV is missing GUI support! Install full OpenCV with:")
    print("pip uninstall opencv-python-headless -y && pip install opencv-python")
    exit()

# Load the emotion detection model
model_path = r"C:\Users\singh\OneDrive\Documents\Projects\my_model.keras"

if not os.path.exists(model_path):
    print(f"Error: Model file not found at {model_path}. Please check the file path!")
    exit()

emotion_model = load_model(model_path)

# Define emotions list
emotions = ['anger', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

# Load face detection model
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Open webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Function to log emotions with timestamp
def log_emotion(predicted_emotion):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open("emotion_log.txt", "a", encoding="utf-8") as log_file:
        log_file.write(f"{timestamp} - Detected Emotion: {predicted_emotion}\n")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture image.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        face = frame[y:y+h, x:x+w]
        face_gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        face_resized = cv2.resize(face_gray, (48, 48))
        face_normalized = face_resized / 255.0
        face_array = img_to_array(face_normalized)
        face_array = np.expand_dims(face_array, axis=0)

        # Predict emotion
        emotion_probabilities = emotion_model.predict(face_array)
        max_index = np.argmax(emotion_probabilities)
        predicted_emotion = emotions[max_index]

        # Log emotion
        log_emotion(predicted_emotion)

        # Draw rectangle & label
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.putText(frame, predicted_emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # Display using OpenCV (with fix)
    cv2.imshow('Real-Time Emotion Detection', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()