import tkinter as tk
from tkinter import filedialog, messagebox
import numpy as np
import tensorflow as tf
from PIL import Image, ImageTk

# ============================================
# BRAIN TUMOR MRI CLASSIFICATION
# Model: MobileNetV2
# ============================================

MODEL_PATH = "best_mobilenetv2.keras"
IMAGE_SIZE = (224, 224)

CLASS_NAMES = [
    "glioma",
    "meningioma",
    "no_tumor",
    "pituitary"
]

# ============================================
# LOAD TRAINED MODEL
# ============================================

print("Loading MobileNetV2 model...")

try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print("Model loaded successfully!")

except Exception as error:
    print("Error loading model:", error)
    raise SystemExit(
        "Model not found. Check best_mobilenetv2.keras "
        "in your project folder."
    )


# ============================================
# IMAGE PREDICTION
# ============================================

def predict_image(image_path):

    # Open the image
    image = Image.open(image_path).convert("RGB")

    # Resize to the model's expected input size
    image = image.resize(IMAGE_SIZE)

    # Convert to NumPy array
    image_array = np.array(image, dtype=np.float32)

    
    
    # Apply the same preprocessing used during MobileNetV2 training
    image_array = tf.keras.applications.mobilenet_v2.preprocess_input(
        image_array
    )    

    # Add batch dimension
    image_array = np.expand_dims(image_array, axis=0)

    # Generate predictions
    predictions = model.predict(image_array, verbose=0)[0]

    # Find the predicted class
    predicted_index = int(np.argmax(predictions))
    predicted_class = CLASS_NAMES[predicted_index]

    # Highest model output score
    confidence = float(predictions[predicted_index]) * 100

    return image, predicted_class, confidence, predictions


# ============================================
# SELECT AND DISPLAY MRI IMAGE
# ============================================

def choose_image():

    file_path = filedialog.askopenfilename(
        title="Select a Brain MRI Image",
        filetypes=[
            ("MRI Image Files", "*.jpg *.jpeg *.png *.bmp"),
            ("All Files", "*.*")
        ]
    )

    if not file_path:
        return

    try:
        image, predicted_class, confidence, predictions = (
            predict_image(file_path)
        )

        # Display selected image
        display_image = image.copy()
        display_image.thumbnail((350, 350))

        photo = ImageTk.PhotoImage(display_image)

        image_label.config(image=photo)
        image_label.image = photo

        # Display predicted class
        result_label.config(
            text=(
                f"Predicted Class: "
                f"{predicted_class.replace('_', ' ').title()}\n"
                f"Model Confidence Score: {confidence:.2f}%"
            ),
            fg="#17365d"
        )

        # Display scores for all classes
        probability_text = "\n".join(
            f"{name.replace('_', ' ').title()}: "
            f"{float(score) * 100:.2f}%"
            for name, score in zip(CLASS_NAMES, predictions)
        )

        probabilities_label.config(text=probability_text)

    except Exception as error:
        messagebox.showerror(
            "Prediction Error",
            f"Unable to process the image.\n\n{error}"
        )


# ============================================
# APPLICATION WINDOW
# ============================================

root = tk.Tk()

root.title("Brain Tumor MRI Classification")

root.geometry("600x700")

root.configure(bg="#f0f4f8")

# Heading
title_label = tk.Label(
    root,
    text="Brain Tumor MRI Classification",
    font=("Arial", 18, "bold"),
    bg="#f0f4f8",
    fg="#17365d"
)

title_label.pack(pady=20)

# Description
description_label = tk.Label(
    root,
    text="Select an MRI image for experimental classification",
    font=("Arial", 10),
    bg="#f0f4f8"
)

description_label.pack(pady=5)

# Select image button
select_button = tk.Button(
    root,
    text="Select MRI Image",
    command=choose_image,
    font=("Arial", 12, "bold"),
    bg="#2563eb",
    fg="white",
    padx=20,
    pady=10,
    cursor="hand2"
)

select_button.pack(pady=20)

# Image display area
image_label = tk.Label(
    root,
    bg="#f0f4f8"
)

image_label.pack(pady=10)

# Prediction result
result_label = tk.Label(
    root,
    text="Your prediction will appear here",
    font=("Arial", 13, "bold"),
    bg="#f0f4f8",
    fg="#17365d",
    wraplength=550
)

result_label.pack(pady=15)

# Class scores
probabilities_label = tk.Label(
    root,
    text="",
    font=("Arial", 11),
    bg="#f0f4f8",
    justify="left"
)

probabilities_label.pack(pady=10)

# Medical disclaimer
disclaimer_label = tk.Label(
    root,
    text=(
        "Educational demonstration only. "
        "Not intended for medical diagnosis."
    ),
    font=("Arial", 9),
    bg="#f0f4f8",
    fg="#b91c1c",
    wraplength=550
)

disclaimer_label.pack(pady=15)

# Start application
root.mainloop()