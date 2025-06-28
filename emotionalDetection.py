import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import sys
import io
from sklearn.utils.class_weight import compute_class_weight

# Set default encoding to utf-8
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Label map
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
                img = img.resize((48, 48))  # Resize to model input size
                images.append(np.array(img))
                labels.append(label_map[label_folder])
    return np.array(images), np.array(labels)

# Paths
train_path = r"C:\Users\singh\OneDrive\Documents\Projects\Emotional detection\train"
test_path = r"C:\Users\singh\OneDrive\Documents\Projects\Emotional detection\test"

# Check paths
if not os.path.exists(train_path):
    print(f"Train path not found: {train_path}")
else:
    print(f"Train path found: {train_path}")

if not os.path.exists(test_path):
    print(f"Test path not found: {test_path}")
else:
    print(f"Test path found: {test_path}")

# Load dataset
X_train, y_train = load_images_from_folder(train_path)
X_test, y_test = load_images_from_folder(test_path)

# Check data loading
if X_train.size == 0 or X_test.size == 0:
    print("Error: No data loaded. Check dataset path and structure.")
    sys.exit()
else:
    print(f"Training data shape: {X_train.shape}")
    print(f"Test data shape: {X_test.shape}")

# Normalize
X_train = X_train / 255.0
X_test = X_test / 255.0

# Add channel dimension
X_train = X_train[..., np.newaxis]
X_test = X_test[..., np.newaxis]

# Convert labels to categorical
y_train = to_categorical(y_train, num_classes=7)
y_test = to_categorical(y_test, num_classes=7)

# 🔴 Check dataset class distribution
y_train_labels = np.argmax(y_train, axis=1)
unique, counts = np.unique(y_train_labels, return_counts=True)
print("Class distribution in training set:", dict(zip(unique, counts)))

# 🔴 Compute class weights for imbalance handling
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train_labels),
    y=y_train_labels
)
class_weights_dict = dict(zip(np.unique(y_train_labels), class_weights))
print("Class weights:", class_weights_dict)

# Define improved deeper model
model = Sequential()
model.add(Conv2D(64, (3,3), activation='relu', input_shape=(48,48,1)))
model.add(MaxPooling2D(2,2))
model.add(Conv2D(128, (3,3), activation='relu'))
model.add(MaxPooling2D(2,2))
model.add(Conv2D(256, (3,3), activation='relu'))
model.add(MaxPooling2D(2,2))
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(7, activation='softmax'))

# Compile with reduced learning rate for stability
model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss='categorical_crossentropy',
    metrics=['accuracy', 'categorical_accuracy']
)

# 🔴 Add EarlyStopping to prevent overfitting
early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Train with increased epochs (50) and class weights
history = model.fit(
    X_train,
    y_train,
    epochs=50,
    batch_size=64,
    validation_data=(X_test, y_test),
    class_weight=class_weights_dict,
    callbacks=[early_stop],
    verbose=1
)

# Save model
model.save('my_model.keras')

# Evaluate model
results = model.evaluate(X_test, y_test, verbose=2)
test_loss, test_acc, test_categorical_acc = results

# Sanitize results
test_loss = str(test_loss).encode('ascii', 'ignore').decode('ascii')
test_acc = str(test_acc).encode('ascii', 'ignore').decode('ascii')
test_categorical_acc = str(test_categorical_acc).encode('ascii', 'ignore').decode('ascii')

print(f'Test loss: {test_loss}')
print(f'Test accuracy: {test_acc}')
print(f'Test categorical accuracy: {test_categorical_acc}')

# Plot training history
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')

# Sanitize legend labels
handles, labels = plt.gca().get_legend_handles_labels()
labels = [label.encode('ascii', 'ignore').decode('ascii') for label in labels]
plt.legend(handles, labels)
plt.show()
