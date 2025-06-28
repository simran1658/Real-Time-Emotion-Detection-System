import numpy as np
from keras.models import load_model
import cv2
import sys
import io
import os

# Redirecting standard output to use UTF-8 encoding
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Load the emotion model
emotion_model = load_model('my_model.keras')  # Replace with the actual model path

# Print model summary to confirm final layer output
emotion_model.summary()

# Print model class indices if available (useful if model has built-in class indices attribute)
# Uncomment the below if you used a Keras ImageDataGenerator for training and saved class_indices as part of the model
# print(emotion_model.class_indices)

# Define the emotions list (ensure it matches the output classes of your model)
emotions = ['anger', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

# Function to process the input image (face_array)
def process_image(image_path):
    # Convert to absolute path for robustness
    image_path = os.path.abspath(image_path)
    print("Trying to load image from:", image_path)

    # Check if the image exists before reading
    if not os.path.exists(image_path):
        print(f"File not found: {image_path}")
        return None

    # Assuming you use OpenCV to read the image, and you need to preprocess it for the model
    face_image = cv2.imread(image_path)

    # Check if image read was successful
    if face_image is None:
        print("OpenCV failed to read the image.")
        return None

    # Convert image to grayscale if needed
    gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
    # Resize the image to the size the model expects, for example, 48x48 pixels (check model requirements)
    resized = cv2.resize(gray, (48, 48))
    # Normalize the image
    face_array = resized.astype('float32') / 255.0
    face_array = np.expand_dims(face_array, axis=0)  # Add batch dimension
    face_array = np.expand_dims(face_array, axis=-1)  # Add channel dimension if needed
    return face_array

# Load and process the image
# image_path = 'static\\angry.png'  # Use double backslash for Windows path consistency
# Check if image path is passed via command line, else use default
if len(sys.argv) > 1:
    image_path = sys.argv[1]
else:
    image_path = 'static\\happy3.jpg'
face_array = process_image(image_path)

# If image processing failed, exit gracefully
if face_array is None:
    print("Exiting due to missing or unreadable image.")
    sys.exit()

# Predict emotion
emotion_probabilities = emotion_model.predict(face_array)

# Print raw prediction probabilities for debug
print("Raw prediction probabilities:", emotion_probabilities)

# Find the emotion with the highest probability
max_index = np.argmax(emotion_probabilities)
predicted_emotion = emotions[max_index]

# Print the predicted emotion directly
print(f"Predicted Emotion: {predicted_emotion}")

# Check for non-ASCII characters and handle them
if any(ord(c) > 127 for c in predicted_emotion):
    print("Non-ASCII character detected:", predicted_emotion)
    # Optionally remove non-ASCII characters
    predicted_emotion = ''.join([c for c in predicted_emotion if ord(c) < 128])
    print(f"Cleaned Predicted Emotion: {predicted_emotion}")
else:
    print("ASCII character only:", predicted_emotion)

# Continue with further actions based on the emotion
# Example: log the predicted emotion into a text file
# Filter non-ASCII characters in predicted_emotion
predicted_emotion = ''.join([c for c in predicted_emotion if ord(c) < 128])

# Now try writing it to the file
with open("emotion_log.txt", "w", encoding="utf-8") as log_file:
    log_file.write(predicted_emotion)

# Optionally, display the cleaned predicted emotion on the screen
# Print the predicted emotion directly to check if there are non-ASCII characters
print(f"Predicted Emotion: {predicted_emotion}")

# Handle non-ASCII characters before logging
predicted_emotion = ''.join([c for c in predicted_emotion if ord(c) < 128])

# Example of handling different emotions
if predicted_emotion == "happy":
    print("User seems happy!")
elif predicted_emotion == "sad":
    print("User seems sad!")
elif predicted_emotion == "anger":
    print("User seems angry!")
elif predicted_emotion == "fear":
    print("User seems fearful!")
elif predicted_emotion == "surprise":
    print("User seems surprised!")
elif predicted_emotion == "disgust":
    print("User seems disgusted!")
elif predicted_emotion == "neutral":
    print("User seems neutral!")
