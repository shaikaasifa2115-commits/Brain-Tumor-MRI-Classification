import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Step 1: Select an MRI image
image_folder = "dataset/train/glioma"

# Find the first image in the folder
images = [
    file for file in os.listdir(image_folder)
    if file.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))
]

image_path = os.path.join(image_folder, images[0])

# Step 2: Open the original image
original_image = Image.open(image_path).convert("RGB")

# Step 3: Resize the image
resized_image = original_image.resize((224, 224))

# Step 4: Convert the image into a NumPy array
image_array = np.array(resized_image)

# Step 5: Normalize pixel values between 0 and 1
normalized_image = image_array / 255.0

# Step 6: Display the results
print("Original image size:", original_image.size)
print("Resized image size:", resized_image.size)
print("Array shape:", image_array.shape)
print("Minimum normalized pixel:", normalized_image.min())
print("Maximum normalized pixel:", normalized_image.max())

# Display original and resized images
plt.figure(figsize=(8, 4))

plt.subplot(1, 2, 1)
plt.imshow(original_image)
plt.title("Original MRI Image")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(normalized_image)
plt.title("Preprocessed MRI Image")
plt.axis("off")

plt.tight_layout()
plt.show()