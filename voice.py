import numpy as np
from keras.models import load_model
import cv2
import pyttsx3  # Offline Text-to-Speech (TTS)
import sys
sys.stdout.reconfigure(encoding='utf-8')


# Load the pre-trained emotion detection model
emotion_model = load_model('my_model.keras')  # Replace with your model's actual path

# Define emotion labels corresponding to model outputs
emotions = ['anger', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

# Initialize the Text-to-Speech engine
def speak(text):
    engine = pyttsx3.init()
    engine.setProperty('rate', 150)  # Adjust speech speed
    engine.setProperty('volume', 1)  # Set volume level (0.0 to 1.0)
    
    try:
        text = text.encode("utf-8").decode("utf-8")  # Ensure UTF-8 encoding
        engine.say(text)
        engine.runAndWait()
    except UnicodeEncodeError:
        print("Error: Could not process some characters in speech output.")

# Function to preprocess the input image
def process_image(image_path):
    face_image = cv2.imread(image_path)  # Read the image
    if face_image is None:
        print("Error: Could not load image. Check the path.")
        return None
    
    gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    resized = cv2.resize(gray, (48, 48))  # Resize to match model input size
    face_array = resized.astype('float32') / 255.0  # Normalize pixel values
    face_array = np.expand_dims(face_array, axis=0)  # Add batch dimension
    face_array = np.expand_dims(face_array, axis=-1)  # Add channel dimension
    
    print(f"Processed Image Shape: {face_array.shape}")  # Debugging
    return face_array

# Specify the input image path
image_path = 'WhatsApp Image 2025-02-02 at 16.32.11_fff57ad6.jpg'  # Replace with your actual image path

# Process the image
face_array = process_image(image_path)
if face_array is None:
    exit()  # Stop execution if image processing fails

# Predict emotion
emotion_probabilities = emotion_model.predict(face_array)
max_index = np.argmax(emotion_probabilities)
predicted_emotion = emotions[max_index]

# Print and announce the detected emotion
print(f"Predicted Emotion: {predicted_emotion}")
speak(f"Detected emotion is {predicted_emotion}")

# Handle different emotions (Custom actions can be added)
emotion_responses = {
    "happy": "You seem happy! Keep smiling!",
    "sad": "You seem sad. Stay strong, everything will be okay.",
    "anger": "You seem angry. Try to relax and take deep breaths.",
    "fear": "You look afraid. Don't worry, you're safe.",
    "surprise": "You look surprised! What happened?",
    "disgust": "You seem disgusted. Hope things get better.",
    "neutral": "You look neutral. Have a great day!"
}

# Speak the corresponding response
speak(emotion_responses.get(predicted_emotion, "Emotion detected."))

# Save the detected emotion to a log file
with open("emotion_log.txt", "a", encoding="utf-8") as log_file:
    log_file.write(f"Predicted Emotion: {predicted_emotion}\n")

# Show final message
print(f"Final Predicted Emotion: {predicted_emotion}")
