import os
import matplotlib.pyplot as plt
from PIL import Image

# Path to the training dataset
dataset_path = "dataset/train"

# Our four categories
classes = ["glioma", "meningioma", "no_tumor", "pituitary"]

# Create a figure with four sections
plt.figure(figsize=(12, 5))

for i, category in enumerate(classes):
    # Get the folder path
    folder_path = os.path.join(dataset_path, category)

    # Find image files
    images = [
        file for file in os.listdir(folder_path)
        if file.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))
    ]

    # Select the first image
    image_path = os.path.join(folder_path, images[0])

    # Open the image
    image = Image.open(image_path)

    # Display the image
    plt.subplot(1, 4, i + 1)
    plt.imshow(image)
    plt.title(category)
    plt.axis("off")

plt.suptitle("Sample Brain MRI Images")
plt.tight_layout()
plt.show()