import pickle
import matplotlib.pyplot as plt

# ==========================================
# 1. LOAD TRAINING HISTORY
# ==========================================

with open("mobilenetv2_history.pkl", "rb") as file:
    history = pickle.load(file)

train_accuracy = history["accuracy"]
val_accuracy = history["val_accuracy"]

train_loss = history["loss"]
val_loss = history["val_loss"]

epochs = range(1, len(train_accuracy) + 1)


# ==========================================
# 2. ACCURACY GRAPH
# ==========================================

plt.figure(figsize=(8, 5))

plt.plot(
    epochs,
    train_accuracy,
    marker="o",
    label="Training Accuracy"
)

plt.plot(
    epochs,
    val_accuracy,
    marker="o",
    label="Validation Accuracy"
)

plt.title("MobileNetV2 Training and Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.savefig("mobilenetv2_accuracy.png", dpi=300)

plt.show()


# ==========================================
# 3. LOSS GRAPH
# ==========================================

plt.figure(figsize=(8, 5))

plt.plot(
    epochs,
    train_loss,
    marker="o",
    label="Training Loss"
)

plt.plot(
    epochs,
    val_loss,
    marker="o",
    label="Validation Loss"
)

plt.title("MobileNetV2 Training and Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.savefig("mobilenetv2_loss.png", dpi=300)

plt.show()


# ==========================================
# 4. COMPLETION MESSAGE
# ==========================================

print("\nMobileNetV2 graphs created successfully!")

print("Saved: mobilenetv2_accuracy.png")
print("Saved: mobilenetv2_loss.png")