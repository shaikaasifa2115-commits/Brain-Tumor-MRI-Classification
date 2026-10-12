import pickle
import matplotlib.pyplot as plt

# Load the saved training history
with open("cnn_history.pkl", "rb") as file:
    history = pickle.load(file)

# Get accuracy and loss values
train_accuracy = history["accuracy"]
val_accuracy = history["val_accuracy"]

train_loss = history["loss"]
val_loss = history["val_loss"]

# Epoch numbers
epochs = range(1, len(train_accuracy) + 1)

# ==========================================
# GRAPH 1: ACCURACY
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

plt.title("CNN Training and Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.savefig("cnn_accuracy.png", dpi=300)
plt.show()

# ==========================================
# GRAPH 2: LOSS
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

plt.title("CNN Training and Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.savefig("cnn_loss.png", dpi=300)
plt.show()

print("Training graphs created successfully!")
print("Saved: cnn_accuracy.png")
print("Saved: cnn_loss.png")