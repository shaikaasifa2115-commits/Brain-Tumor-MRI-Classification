import os
import matplotlib.pyplot as plt

# Path to the training dataset
train_path = "dataset/train"

# Categories in our dataset
classes = ["glioma", "meningioma", "no_tumor", "pituitary"]

# Store the number of images in each category
image_counts = []

for category in classes:
    category_path = os.path.join(train_path, category)

    images = [
        file for file in os.listdir(category_path)
        if file.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))
    ]

    image_counts.append(len(images))

# Display the image counts
print("Training Dataset Image Counts")
for category, count in zip(classes, image_counts):
    print(f"{category}: {count}")

# Create a bar chart
plt.figure(figsize=(8, 5))
plt.bar(classes, image_counts)

plt.title("Distribution of Training MRI Images")
plt.xlabel("Tumor Category")
plt.ylabel("Number of Images")

plt.tight_layout()
plt.show()