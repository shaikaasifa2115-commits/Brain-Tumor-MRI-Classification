import os
import matplotlib.pyplot as plt
import tensorflow as tf
from PIL import Image
import numpy as np

# Path to the training images
image_folder = "dataset/train/glioma"

# Find the first MRI image
images = [
    file for file in os.listdir(image_folder)
    if file.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))
]

image_path = os.path.join(image_folder, images[0])

# Load and resize the image
image = Image.open(image_path).convert("RGB")
image = image.resize((224, 224))

# Convert the image into a numerical array
image_array = np.array(image, dtype=np.float32) / 255.0

# Add a batch dimension for TensorFlow
image_tensor = tf.expand_dims(image_array, axis=0)

# Define data augmentation transformations
augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomRotation(0.05),
    tf.keras.layers.RandomZoom(0.10),
    tf.keras.layers.RandomTranslation(0.05, 0.05)
])

# Generate augmented images
augmented_batch = augmentation(image_tensor, training=True)

# Display the original and augmented images
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.imshow(image_array)
plt.title("Original MRI")
plt.axis("off")

for i in range(2):
    augmented_image = augmented_batch[i % 1].numpy()

    plt.subplot(1, 3, i + 2)
    plt.imshow(np.clip(augmented_image, 0, 1))
    plt.title(f"Augmented MRI {i + 1}")
    plt.axis("off")

plt.tight_layout()
plt.show()