import os

# Path to our dataset
dataset_path = "dataset"

# Dataset folders
folders = ["train", "test", "valid"]

# Tumor categories
classes = ["glioma", "meningioma", "no_tumor", "pituitary"]

print("BRAIN TUMOR MRI DATASET ANALYSIS")
print("-" * 40)

for folder in folders:
    print(f"\n{folder.upper()} DATASET")

    for category in classes:
        category_path = os.path.join(
            dataset_path, folder, category
        )

        if os.path.exists(category_path):
            images = [
                file for file in os.listdir(category_path)
                if file.lower().endswith(
                    (".jpg", ".jpeg", ".png", ".bmp")
                )
            ]

            print(f"{category}: {len(images)} images")

        else:
            print(f"{category}: Folder not found")