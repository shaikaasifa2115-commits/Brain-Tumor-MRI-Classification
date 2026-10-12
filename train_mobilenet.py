import tensorflow as tf
import matplotlib.pyplot as plt

# ==========================================
# 1. CONFIGURATION
# ==========================================

TRAIN_PATH = "dataset/train"
VALID_PATH = "dataset/valid"

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 15

CLASS_NAMES = [
    "glioma",
    "meningioma",
    "no_tumor",
    "pituitary"
]

print("=" * 50)
print("MOBILENETV2 TRANSFER LEARNING")
print("=" * 50)


# ==========================================
# 2. LOAD TRAINING DATA
# ==========================================

print("\nLoading training dataset...")

train_dataset = tf.keras.utils.image_dataset_from_directory(
    TRAIN_PATH,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="int",
    shuffle=True,
    seed=42
)


# ==========================================
# 3. LOAD VALIDATION DATA
# ==========================================

print("\nLoading validation dataset...")

validation_dataset = tf.keras.utils.image_dataset_from_directory(
    VALID_PATH,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="int",
    shuffle=False
)


# ==========================================
# 4. VERIFY CLASS ORDER
# ==========================================

print("\nDetected classes:", train_dataset.class_names)

if train_dataset.class_names != CLASS_NAMES:
    raise ValueError(
        f"Unexpected class order: {train_dataset.class_names}"
    )

if validation_dataset.class_names != CLASS_NAMES:
    raise ValueError(
        f"Unexpected validation class order: "
        f"{validation_dataset.class_names}"
    )


# ==========================================
# 5. PREPROCESS THE IMAGES
# ==========================================

# MobileNetV2 expects pixel values scaled to [-1, 1].
# The original image pixels range from 0 to 255.

def preprocess_images(images, labels):
    images = tf.cast(images, tf.float32)
    images = tf.keras.applications.mobilenet_v2.preprocess_input(
        images
    )
    return images, labels


train_dataset = train_dataset.map(
    preprocess_images,
    num_parallel_calls=tf.data.AUTOTUNE
)

validation_dataset = validation_dataset.map(
    preprocess_images,
    num_parallel_calls=tf.data.AUTOTUNE
)

train_dataset = train_dataset.prefetch(
    tf.data.AUTOTUNE
)

validation_dataset = validation_dataset.prefetch(
    tf.data.AUTOTUNE
)

print("Image preprocessing configured successfully!")


# ==========================================
# 6. LOAD PRETRAINED MOBILENETV2
# ==========================================

print("\nLoading pretrained MobileNetV2...")

base_model = tf.keras.applications.MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights="imagenet"
)

# Freeze pretrained layers initially.
base_model.trainable = False

print("MobileNetV2 loaded successfully!")


# ==========================================
# 7. BUILD THE CLASSIFICATION MODEL
# ==========================================

inputs = tf.keras.Input(
    shape=(224, 224, 3)
)

features = base_model(
    inputs,
    training=False
)

features = tf.keras.layers.GlobalAveragePooling2D()(
    features
)

features = tf.keras.layers.Dropout(0.3)(
    features
)

outputs = tf.keras.layers.Dense(
    4,
    activation="softmax"
)(features)

model = tf.keras.Model(
    inputs=inputs,
    outputs=outputs
)


# ==========================================
# 8. COMPILE THE MODEL
# ==========================================

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.001
    ),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

print("\nModel architecture:")
model.summary()


# ==========================================
# 9. SAVE THE BEST MODEL
# ==========================================

checkpoint = tf.keras.callbacks.ModelCheckpoint(
    "best_mobilenetv2.keras",
    monitor="val_loss",
    save_best_only=True,
    verbose=1
)


# ==========================================
# 10. EARLY STOPPING
# ==========================================

early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=4,
    restore_best_weights=True,
    verbose=1
)


# ==========================================
# 11. TRAIN THE MODEL
# ==========================================

print("\nStarting MobileNetV2 training...")

history = model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=EPOCHS,
    callbacks=[
        checkpoint,
        early_stopping
    ]
)


# ==========================================
# 12. SAVE THE FINAL TRAINED MODEL
# ==========================================

model.save("mobilenetv2_final.keras")


# ==========================================
# 13. SAVE TRAINING HISTORY
# ==========================================

import pickle

with open(
    "mobilenetv2_history.pkl",
    "wb"
) as file:
    pickle.dump(history.history, file)


# ==========================================
# 14. DISPLAY FINAL RESULTS
# ==========================================

best_val_accuracy = max(
    history.history["val_accuracy"]
)

print("\n" + "=" * 50)
print("MOBILENETV2 TRAINING COMPLETED!")
print("=" * 50)

print(
    f"Best validation accuracy: "
    f"{best_val_accuracy * 100:.2f}%"
)

print("Best model: best_mobilenetv2.keras")
print("Final model: mobilenetv2_final.keras")
print("Training history: mobilenetv2_history.pkl")