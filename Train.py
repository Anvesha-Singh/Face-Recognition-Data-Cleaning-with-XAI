import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.applications import ResNet50
from sklearn.preprocessing import LabelEncoder
from sklearn import svm
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
import numpy as np
import pickle
import random
import cv2
import matplotlib.pyplot as plt  # Importing matplotlib for plotting

# Dataset paths
train_images = r"C:\Users\gravy\Downloads\FaceRecognitionDataCleaningWithXAI-main\FaceRecognitionDataCleaningWithXAI-main\Sampletrain"
validation_images = r"C:\Users\gravy\Downloads\images\images"

# Custom random erasing function
def random_erasing(image, p=0.5, area_range=(0.02, 0.15), min_aspect_ratio=0.3):
    if random.uniform(0, 1) > p:
        return image
    h, w, _ = image.shape
    mask_area = random.uniform(*area_range) * h * w
    aspect_ratio = random.uniform(min_aspect_ratio, 1 / min_aspect_ratio)
    mask_w = int(np.sqrt(mask_area * aspect_ratio))
    mask_h = int(np.sqrt(mask_area / aspect_ratio))

    top_x = random.randint(0, w - mask_w)
    top_y = random.randint(0, h - mask_h)
    image[top_y:top_y + mask_h, top_x:top_x + mask_w, :] = np.random.randint(0, 256, (mask_h, mask_w, 3))
    return image

# Custom preprocessing function for random erasing
def custom_preprocess(image):
    image = random_erasing(image)
    return image

# Image preprocessing with added augmentation techniques
train_gen = ImageDataGenerator(
    shear_range=0.2,
    zoom_range=0.3,
    horizontal_flip=True,
    rotation_range=40,
    width_shift_range=0.3,
    height_shift_range=0.3,
    brightness_range=[0.5, 1.5],
    channel_shift_range=0.3,
    preprocessing_function=custom_preprocess,
    fill_mode='nearest'
)

# Validation generator (no augmentation for validation data)
test_gen = ImageDataGenerator()

# Generating training data
training_data = train_gen.flow_from_directory(
    train_images,
    target_size=(100, 100),
    batch_size=16,
    class_mode='categorical'
)

# Generating validation data
validation_data = test_gen.flow_from_directory(
    validation_images,
    target_size=(100, 100),
    batch_size=8,
    class_mode='categorical'
)

# Printing class labels
Train_class = training_data.class_indices
Result_class = {value: key for key, value in Train_class.items()}

# Save the class mapping using pickle
with open(r'C:/Users/gravy/OneDrive/Documents/AWS/ResultMap.pkl', 'wb') as f:
    pickle.dump(Result_class, f)

# Define number of output neurons based on the number of classes
Output_Neurons = len(Result_class)

# Label Smoothing
label_smoothing = 0.1  # Smoothing factor
loss_fn = tf.keras.losses.CategoricalCrossentropy(label_smoothing=label_smoothing)

# Load ResNet50 without the top layers for feature extraction
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(100, 100, 3))
for layer in base_model.layers[-4:]:  # Unfreeze last few layers
    layer.trainable = True

# Initialize the model with custom layers on top of ResNet50
Model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(Output_Neurons, activation='softmax')
])

# Set a custom learning rate
learning_rate = 0.0001
optimizer = Adam(learning_rate=learning_rate)

# Compile the model with label smoothing loss
Model.compile(loss=loss_fn, optimizer=optimizer, metrics=['accuracy'])

# Define the learning rate scheduler
lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=0.00001)

# Training the model
history = Model.fit(
    training_data,
    epochs=10,
    validation_data=validation_data,
    callbacks=[lr_scheduler]
)

# Print final training and validation accuracy
train_accuracy = history.history['accuracy'][-1]
val_accuracy = history.history['val_accuracy'][-1]

print(f"Final Training Accuracy: {train_accuracy*100:.2f}%")
print(f"Final Validation Accuracy: {val_accuracy*100:.2f}%")

# Save the trained model
Model.save(r'C:/Users/gravy/OneDrive/Documents/AWS/face_recognition_model.keras')
print("Model saved to 'face_recognition_model.keras'")

# Plotting the training and validation accuracy and loss graphs
# Plot accuracy
plt.figure(figsize=(12, 6))

# Training accuracy
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training vs Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

# Training loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training vs Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

# Show plots
plt.tight_layout()
plt.show()
