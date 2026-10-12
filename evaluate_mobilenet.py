import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

# ==========================================
# 1. CONFIGURATION
# ==========================================

TEST_PATH = "dataset/test"
MODEL_PATH = "best_mobilenetv2.keras"

IMG_SIZE = (224, 224)
BATCH_SIZE = 32

CLASS_NAMES = [
    "glioma",
    "meningioma",
    "no_tumor",
    "pituitary"
]

print("=" * 50)
print("MOBILENETV2 MODEL EVALUATION")
print("=" * 50)


# ==========================================
# 2. LOAD THE BEST MODEL
# ==========================================

print("\nLoading trained MobileNetV2 model...")

model = tf.keras.models.load_model(MODEL_PATH)

print("MobileNetV2 loaded successfully!")


# ==========================================
# 3. LOAD THE TEST DATASET
# ==========================================

print("\nLoading test dataset...")

test_dataset = tf.keras.utils.image_dataset_from_directory(
    TEST_PATH,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="int",
    shuffle=False
)

if test_dataset.class_names != CLASS_NAMES:
    raise ValueError(
        f"Unexpected class order: {test_dataset.class_names}"
    )


# ==========================================
# 4. PREPROCESS TEST IMAGES
# ==========================================

def preprocess_images(images, labels):
    images = tf.cast(images, tf.float32)

    images = tf.keras.applications.mobilenet_v2.preprocess_input(
        images
    )

    return images, labels


test_dataset = test_dataset.map(
    preprocess_images,
    num_parallel_calls=tf.data.AUTOTUNE
)

test_dataset = test_dataset.prefetch(
    tf.data.AUTOTUNE
)


# ==========================================
# 5. EVALUATE TEST ACCURACY AND LOSS
# ==========================================

print("\nEvaluating MobileNetV2...")

test_loss, test_accuracy = model.evaluate(
    test_dataset,
    verbose=1
)

print("\nTEST RESULTS")
print("-" * 35)

print(f"Test Accuracy: {test_accuracy * 100:.2f}%")
print(f"Test Loss: {test_loss:.4f}")


# ==========================================
# 6. GENERATE PREDICTIONS
# ==========================================

print("\nGenerating predictions...")

y_true = np.concatenate([
    labels.numpy()
    for images, labels in test_dataset
])

predictions = model.predict(
    test_dataset,
    verbose=1
)

y_pred = np.argmax(
    predictions,
    axis=1
)


# ==========================================
# 7. CLASSIFICATION REPORT
# ==========================================

print("\nCLASSIFICATION REPORT")
print("-" * 50)

report = classification_report(
    y_true,
    y_pred,
    labels=[0, 1, 2, 3],
    target_names=CLASS_NAMES,
    digits=4,
    zero_division=0
)

print(report)

with open(
    "mobilenetv2_classification_report.txt",
    "w"
) as file:
    file.write(report)

print("\nClassification report saved!")


# ==========================================
# 8. CONFUSION MATRIX
# ==========================================

cm = confusion_matrix(
    y_true,
    y_pred,
    labels=[0, 1, 2, 3]
)

print("\nCONFUSION MATRIX")
print(cm)

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=CLASS_NAMES
)

fig, ax = plt.subplots(figsize=(8, 6))

display.plot(
    ax=ax,
    cmap="Blues",
    values_format="d"
)

plt.title("MobileNetV2 Test Confusion Matrix")
plt.tight_layout()

plt.savefig(
    "mobilenetv2_confusion_matrix.png",
    dpi=300
)

plt.show()

print("\nSaved: mobilenetv2_confusion_matrix.png")
print("\nMobileNetV2 evaluation completed!")