import tensorflow as tf
from tensorflow.keras import layers, models

# ==========================================
# 1. PROJECT CONFIGURATION
# ==========================================

IMG_HEIGHT = 224
IMG_WIDTH = 224
BATCH_SIZE = 32
NUM_CLASSES = 4
SEED = 42

TRAIN_PATH = "dataset/train"
VALID_PATH = "dataset/valid"

print("=" * 50)
print("BRAIN TUMOR MRI CLASSIFICATION")
print("=" * 50)


# ==========================================
# 2. LOAD TRAINING DATASET
# ==========================================

print("\nLoading training images...")

train_dataset = tf.keras.utils.image_dataset_from_directory(
    TRAIN_PATH,
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    label_mode="int",
    shuffle=True,
    seed=SEED
)


# ==========================================
# 3. LOAD VALIDATION DATASET
# ==========================================

print("\nLoading validation images...")

validation_dataset = tf.keras.utils.image_dataset_from_directory(
    VALID_PATH,
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    label_mode="int",
    shuffle=False
)


# ==========================================
# 4. CHECK CLASS NAMES
# ==========================================

class_names = train_dataset.class_names

print("\nDetected classes:", class_names)
print("Number of classes:", len(class_names))

if len(class_names) != NUM_CLASSES:
    raise ValueError(
        f"Expected {NUM_CLASSES} classes, "
        f"but found {len(class_names)}."
    )

if class_names != validation_dataset.class_names:
    raise ValueError(
        "Training and validation class orders do not match."
    )


# ==========================================
# 5. NORMALIZATION
# ==========================================

normalization_layer = layers.Rescaling(1.0 / 255)

train_dataset = train_dataset.map(
    lambda images, labels: (
        normalization_layer(images),
        labels
    )
)

validation_dataset = validation_dataset.map(
    lambda images, labels: (
        normalization_layer(images),
        labels
    )
)


# ==========================================
# 6. DATA AUGMENTATION
# ==========================================

data_augmentation = models.Sequential([
    layers.RandomRotation(0.03),
    layers.RandomZoom(0.05),
    layers.RandomTranslation(0.03, 0.03)
], name="data_augmentation")


# Apply augmentation only to training images
train_dataset = train_dataset.map(
    lambda images, labels: (
        data_augmentation(images, training=True),
        labels
    )
)


# ==========================================
# 7. OPTIMIZE DATASET PIPELINE
# ==========================================

AUTOTUNE = tf.data.AUTOTUNE

train_dataset = train_dataset.prefetch(
    buffer_size=AUTOTUNE
)

validation_dataset = validation_dataset.prefetch(
    buffer_size=AUTOTUNE
)


# ==========================================
# 8. BUILD CUSTOM CNN
# ==========================================

print("\nBuilding custom CNN model...")

model = models.Sequential([
    layers.Input(
        shape=(IMG_HEIGHT, IMG_WIDTH, 3)
    ),

    # First convolution block
    layers.Conv2D(
        32, (3, 3),
        activation="relu",
        padding="same"
    ),
    layers.MaxPooling2D((2, 2)),

    # Second convolution block
    layers.Conv2D(
        64, (3, 3),
        activation="relu",
        padding="same"
    ),
    layers.MaxPooling2D((2, 2)),

    # Third convolution block
    layers.Conv2D(
        128, (3, 3),
        activation="relu",
        padding="same"
    ),
    layers.MaxPooling2D((2, 2)),

    # Feature extraction
    layers.GlobalAveragePooling2D(),

    layers.BatchNormalization(),

    layers.Dropout(0.4),

    # Classification layer
    layers.Dense(
        64,
        activation="relu"
    ),

    layers.Dropout(0.3),

    # Four output categories
    layers.Dense(
        NUM_CLASSES,
        activation="softmax"
    )
])


# ==========================================
# 9. COMPILE THE MODEL
# ==========================================

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.001
    ),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)


# ==========================================
# 10. DISPLAY MODEL SUMMARY
# ==========================================

model.summary()


# ==========================================
# 11. VERIFY DATASET BATCHES
# ==========================================

print("\nChecking one training batch...")

for images, labels in train_dataset.take(1):
    print("Training image batch shape:", images.shape)
    print("Training label batch shape:", labels.shape)
    print(
        "Normalized pixel range:",
        float(tf.reduce_min(images)),
        "to",
        float(tf.reduce_max(images))
    )

print("\nChecking one validation batch...")

for images, labels in validation_dataset.take(1):
    print("Validation image batch shape:", images.shape)
    print("Validation label batch shape:", labels.shape)


# ==========================================
# 12. COMPLETION MESSAGE
# ==========================================

print("\n" + "=" * 50)
print("CNN MODEL CREATED SUCCESSFULLY!")
print("Training dataset loaded successfully!")
print("Validation dataset loaded successfully!")
print("Model training has NOT started yet.")
print("=" * 50)

# ==========================================
# 13. TRAIN THE CNN MODEL
# ==========================================

print("\nStarting CNN training...")

# Save the best model during training
checkpoint = tf.keras.callbacks.ModelCheckpoint(
    "best_cnn_model.keras",
    monitor="val_loss",
    save_best_only=True,
    verbose=1
)

# Stop training if validation loss stops improving
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True,
    verbose=1
)

# Train the model
history = model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=20,
    callbacks=[checkpoint, early_stopping]
)

# Save the final training history
import pickle

with open("cnn_history.pkl", "wb") as file:
    pickle.dump(history.history, file)

print("\nCNN training completed!")
print("Best model saved as best_cnn_model.keras")
print("Training history saved as cnn_history.pkl")