!pip install tensorflow keras numpy matplotlib scikit-learn

import os
import cv2
import zipfile
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from google.colab import drive
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten, Dropout

drive.mount('/content/drive')
DATASET_PATH = "/content/drive/MyDrive/GitHub dataset /dataset.zip"
extract_path = "/content/dataset"
with zipfile.ZipFile(DATASET_PATH, 'r') as zip_ref:
    zip_ref.extractall(extract_path)
print("Extracted files:", os.listdir(extract_path))

!pip uninstall -y tensorflow keras
!pip install tensorflow keras

# Define constants
image_size = (224, 224)  # ViT uses 224x224 images
batch_size = 32
num_classes = 4  # glioma, meningioma, pituitary, no tumor

# Define dataset path
train_path = "/content/dataset/Training"

# Load images & labels
images = []
labels = []

for label in os.listdir(train_path):
    label_path = os.path.join(train_path, label)

    if not os.path.isdir(label_path):
        continue  # Ensure it's a directory

    for img_name in os.listdir(label_path):
        img_path = os.path.join(label_path, img_name)
        if os.path.isdir(img_path):
            continue  # Skip directories

        try:
            img = load_img(img_path, target_size=image_size)  # Load image
            img_array = img_to_array(img) / 255.0  # Normalize
            images.append(img_array)
            labels.append(label)  # Store class
        except Exception as e:
            print(f"Skipping {img_path}: {e}")

# Convert lists to NumPy arrays
images = np.array(images)
labels = np.array(labels)

print(f"Total images loaded: {len(images)}")
print(f"Total labels loaded: {len(labels)}")

# Encode labels into integers
label_encoder = LabelEncoder()
labels_encoded = label_encoder.fit_transform(labels)
labels_categorical = to_categorical(labels_encoded, num_classes)

# Split dataset into Train (80%) & Validation (20%)
X_train, X_val, y_train, y_val = train_test_split(images, labels_categorical, test_size=0.2, random_state=42)

# Load Pretrained ViT Model
base_model = ResNet50(weights="imagenet", include_top=False, input_shape=(224, 224, 3))

# Freeze base model layers (optional for faster training)
base_model.trainable = False

# Custom Classification Head
x = Flatten()(base_model.output)
x = Dense(512, activation="relu")(x)
x = Dropout(0.5)(x)
x = Dense(num_classes, activation="softmax")(x)  # Final layer

# Create the final model
model = Model(inputs=base_model.input, outputs=x)

# Compile Model
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# Train Model
history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=5, batch_size=32)
#history = model.fit(X_train[:100], y_train[:100], validation_data=(X_val[:20], y_val[:20]), epochs=2, batch_size=16)
# Evaluate Model
val_loss, val_acc = model.evaluate(X_val, y_val)
print(f"Validation Accuracy: {val_acc * 100:.2f}%")





