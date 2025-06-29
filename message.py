import numpy as np
from keras.models import load_model
import cv2
import sys
import io
import os
import gdown  # Added for downloading model from Google Drive

# Redirecting standard output to use UTF-8 encoding
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# ---------------------- MODEL DOWNLOAD CONFIG ---------------------- #
MODEL_PATH = "my_model.keras"
MODEL_URL = "https://drive.google.com/file/d/1AJrrEXhiYs6Swv9HkqwRcUyMT4OyfnMJ/view?usp=sharing"

# Download model if not exists
if not os.path.exists(MODEL_PATH):
    print("Downloading model from Google Drive...")
    gdown.download(MODEL_URL, MODEL_PATH, quiet=False)
    print("Model downloaded successfully.")
# ------------------------------------------------------------------- #

# Load the emotion model
emotion_model = load_model(MODEL_PATH)
emotion_model.summary()

# Define the emotions list (ensure it matches your model outputs)
emotions = ['anger', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

# Function to process the input image
def process_image(image_path):
    image_path = os.path.abspath(image_path)
    print("Trying to load image from:", image_path)

    if not os.path.exists(image_path):
        print(f"File not found: {image_path}")
        return None

    face_image = cv2.imread(image_path)
    if face_image is None:
        print("OpenCV failed to read the image.")
        return None

    gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (48, 48))
    face_array = resized.astype('float32') / 255.0
    face_array = np.expand_dims(face_array, axis=0)  # Add batch dimension
    face_array = np.expand_dims(face_array, axis=-1)  # Add channel dimension if needed
    return face_array

# Load and process the image
if len(sys.argv) > 1:
    image_path = sys.argv[1]
else:
    image_path = 'static/happy3.jpg'  # Use forward slashes for compatibility

face_array = process_image(image_path)

if face_array is None:
    print("Exiting due to missing or unreadable image.")
    sys.exit()

# Predict emotion
emotion_probabilities = emotion_model.predict(face_array)
print("Raw prediction probabilities:", emotion_probabilities)

max_index = np.argmax(emotion_probabilities)
predicted_emotion = emotions[max_index]

# Remove non-ASCII characters if any
predicted_emotion_clean = ''.join([c for c in predicted_emotion if ord(c) < 128])

# Log the prediction
with open("emotion_log.txt", "w", encoding="utf-8") as log_file:
    log_file.write(predicted_emotion_clean)

print(f"Predicted Emotion: {predicted_emotion_clean}")

# Example of handling different emotions
if predicted_emotion_clean == "happy":
    print("User seems happy!")
elif predicted_emotion_clean == "sad":
    print("User seems sad!")
elif predicted_emotion_clean == "anger":
    print("User seems angry!")
elif predicted_emotion_clean == "fear":
    print("User seems fearful!")
elif predicted_emotion_clean == "surprise":
    print("User seems surprised!")
elif predicted_emotion_clean == "disgust":
    print("User seems disgusted!")
elif predicted_emotion_clean == "neutral":
    print("User seems neutral!")
