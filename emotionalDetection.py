import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
import sys
import io

# Set default encoding to utf-8
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

label_map = {
    'angry': 0,
    'disgust':1,
    'fear':2,
    'happy': 3,
    'neutral': 4,
    'sad': 5,
    'surprise': 6
}

# Function to load images from folder
def load_images_from_folder(folder):
    images = []
    labels = []
    for label_folder in os.listdir(folder):
        label_path = os.path.join(folder, label_folder)
        if os.path.isdir(label_path):
            for filename in os.listdir(label_path):
                img_path = os.path.join(label_path, filename)
                img = Image.open(img_path).convert('L')  # Convert to grayscale
                img = img.resize((48, 48))  # Ensure size matches dataset format
                images.append(np.array(img))
                labels.append(label_map[label_folder])  # Folder name as label
    return np.array(images), np.array(labels)

train_path = r"C:\Users\singh\OneDrive\Documents\Projects\train"
test_path = r"C:\Users\singh\OneDrive\Documents\Projects\test"

# Check if the paths exist
if not os.path.exists(train_path):
    print(f"Train path not found: {train_path}")
else:
    print(f"Train path found: {train_path}")

if not os.path.exists(test_path):
    print(f"Test path not found: {test_path}")
else:
    print(f"Test path found: {test_path}")

# Load the dataset
X_train, y_train = load_images_from_folder(train_path)
X_test, y_test = load_images_from_folder(test_path)

# Check if data is loaded properly
if X_train.size == 0 or X_test.size == 0:
    print("Error: No data loaded. Check dataset path and structure.")
else:
    print(f"Training data shape: {X_train.shape}")
    print(f"Test data shape: {X_test.shape}")

# Normalize the data
X_train = X_train / 255.0
X_test = X_test / 255.0

# Add channel dimension (for grayscale image input)
X_train = X_train[..., np.newaxis]
X_test = X_test[..., np.newaxis]

# Convert labels to categorical (one-hot encoding)
y_train = to_categorical(y_train, num_classes=7)  # 7 emotion classes
y_test = to_categorical(y_test, num_classes=7)

# Define the model
model = Sequential()

# Convolutional Layer 1
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 1)))
model.add(MaxPooling2D(pool_size=(2, 2)))

# Convolutional Layer 2
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# Flatten the output for the fully connected layers
model.add(Flatten())

# Fully connected Layer 1
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))

# Output Layer
model.add(Dense(7, activation='softmax'))  # 7 output classes for emotions

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy', 'categorical_accuracy'])

# Train the model
history = model.fit(X_train, y_train, epochs=10, batch_size=64, validation_data=(X_test, y_test), verbose=1)

# Save the trained model
model.save('my_model.keras')

# Evaluate the model
results = model.evaluate(X_test, y_test, verbose=2)
test_loss, test_acc, test_categorical_acc = results

# Sanitize the results
test_loss = str(test_loss).encode('ascii', 'ignore').decode('ascii')
test_acc = str(test_acc).encode('ascii', 'ignore').decode('ascii')
test_categorical_acc = str(test_categorical_acc).encode('ascii', 'ignore').decode('ascii')

# Print sanitized results
print(f'Test loss: {test_loss}')
print(f'Test accuracy: {test_acc}')
print(f'Test categorical accuracy: {test_categorical_acc}')

# Check and sanitize history keys
sanitized_keys = [key.encode('ascii', 'ignore').decode('ascii') for key in history.history.keys()]
print("Sanitized Keys in history:", sanitized_keys)

# Plot the training history with sanitized labels
# plt.plot(history.history['accuracy'], label='accuracy')
# plt.plot(history.history['val_accuracy'], label='val_accuracy')
# plt.xlabel('Epochs')
# plt.ylabel('Accuracy')

# # Ensure the legend is sanitized
# plt.legend([label.encode('ascii', 'ignore').decode('ascii') for label in plt.legend().get_texts()])
# plt.show()

# Plot the training history
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')

# Sanitize the labels
handles, labels = plt.gca().get_legend_handles_labels()
labels = [label.encode('ascii', 'ignore').decode('ascii') for label in labels]  # Sanitize the labels
plt.legend(handles, labels)  # Set the sanitized labels

plt.show()


