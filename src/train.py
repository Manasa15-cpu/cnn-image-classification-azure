from pathlib import Path

import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models


# ---------------------------------------------------------
# 1. Project paths
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TRAIN_DIR = PROJECT_ROOT / "data" / "seg_train"
TEST_DIR = PROJECT_ROOT / "data" / "seg_test"
MODEL_DIR = PROJECT_ROOT / "models"
IMAGES_DIR = PROJECT_ROOT / "images"

MODEL_DIR.mkdir(exist_ok=True)
IMAGES_DIR.mkdir(exist_ok=True)


# Some Kaggle datasets contain one extra nested folder.
if (TRAIN_DIR / "seg_train").exists():
    TRAIN_DIR = TRAIN_DIR / "seg_train"

if (TEST_DIR / "seg_test").exists():
    TEST_DIR = TEST_DIR / "seg_test"


# ---------------------------------------------------------
# 2. Project settings
# ---------------------------------------------------------

IMAGE_SIZE = (150, 150)
BATCH_SIZE = 32
EPOCHS = 15
SEED = 42


# ---------------------------------------------------------
# 3. Check that the dataset exists
# ---------------------------------------------------------

if not TRAIN_DIR.exists():
    raise FileNotFoundError(
        f"Training folder not found: {TRAIN_DIR}"
    )

if not TEST_DIR.exists():
    raise FileNotFoundError(
        f"Testing folder not found: {TEST_DIR}"
    )

print(f"Training folder: {TRAIN_DIR}")
print(f"Testing folder: {TEST_DIR}")


# ---------------------------------------------------------
# 4. Load the training and testing images
# ---------------------------------------------------------

train_dataset = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="int",
    shuffle=True,
    seed=SEED,
)

test_dataset = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="int",
    shuffle=False,
)

class_names = train_dataset.class_names
number_of_classes = len(class_names)

print("\nImage classes:")
for index, class_name in enumerate(class_names):
    print(f"{index}: {class_name}")


# ---------------------------------------------------------
# 5. Improve dataset loading performance
# ---------------------------------------------------------

AUTOTUNE = tf.data.AUTOTUNE

train_dataset = train_dataset.cache().shuffle(1000).prefetch(
    buffer_size=AUTOTUNE
)

test_dataset = test_dataset.cache().prefetch(
    buffer_size=AUTOTUNE
)


# ---------------------------------------------------------
# 6. Data augmentation
# ---------------------------------------------------------

data_augmentation = tf.keras.Sequential(
    [
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1),
    ],
    name="data_augmentation",
)


# ---------------------------------------------------------
# 7. Build the CNN model
# ---------------------------------------------------------

model = models.Sequential(
    [
        layers.Input(shape=(150, 150, 3)),

        data_augmentation,

        layers.Rescaling(1.0 / 255),

        layers.Conv2D(
            filters=32,
            kernel_size=(3, 3),
            activation="relu",
            padding="same",
        ),
        layers.MaxPooling2D(pool_size=(2, 2)),

        layers.Conv2D(
            filters=64,
            kernel_size=(3, 3),
            activation="relu",
            padding="same",
        ),
        layers.MaxPooling2D(pool_size=(2, 2)),

        layers.Conv2D(
            filters=128,
            kernel_size=(3, 3),
            activation="relu",
            padding="same",
        ),
        layers.MaxPooling2D(pool_size=(2, 2)),

        layers.Dropout(0.3),

        layers.Flatten(),

        layers.Dense(
            units=128,
            activation="relu",
        ),

        layers.Dropout(0.4),

        layers.Dense(
            units=number_of_classes,
            activation="softmax",
        ),
    ]
)


# ---------------------------------------------------------
# 8. Compile the model
# ---------------------------------------------------------

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

model.summary()


# ---------------------------------------------------------
# 9. Training callbacks
# ---------------------------------------------------------

best_model_path = MODEL_DIR / "best_model.keras"

callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=3,
        restore_best_weights=True,
    ),

    tf.keras.callbacks.ModelCheckpoint(
        filepath=best_model_path,
        monitor="val_accuracy",
        save_best_only=True,
    ),

    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=2,
        min_lr=0.000001,
    ),
]


# ---------------------------------------------------------
# 10. Train the model
# ---------------------------------------------------------

history = model.fit(
    train_dataset,
    validation_data=test_dataset,
    epochs=EPOCHS,
    callbacks=callbacks,
)


# ---------------------------------------------------------
# 11. Evaluate the model
# ---------------------------------------------------------

test_loss, test_accuracy = model.evaluate(
    test_dataset,
    verbose=1,
)

print(f"\nTest loss: {test_loss:.4f}")
print(f"Test accuracy: {test_accuracy:.4f}")
print(f"Test accuracy percentage: {test_accuracy * 100:.2f}%")


# ---------------------------------------------------------
# 12. Save the final model
# ---------------------------------------------------------

final_model_path = MODEL_DIR / "intel_image_classifier.keras"

model.save(final_model_path)

print(f"\nFinal model saved to: {final_model_path}")
print(f"Best model saved to: {best_model_path}")


# ---------------------------------------------------------
# 13. Plot training results
# ---------------------------------------------------------

training_accuracy = history.history["accuracy"]
validation_accuracy = history.history["val_accuracy"]

training_loss = history.history["loss"]
validation_loss = history.history["val_loss"]

epochs_completed = range(1, len(training_accuracy) + 1)


plt.figure(figsize=(8, 5))

plt.plot(
    epochs_completed,
    training_accuracy,
    label="Training Accuracy",
)

plt.plot(
    epochs_completed,
    validation_accuracy,
    label="Validation Accuracy",
)

plt.title("Training and Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.tight_layout()

accuracy_plot_path = IMAGES_DIR / "accuracy_plot.png"
plt.savefig(accuracy_plot_path)
plt.close()


plt.figure(figsize=(8, 5))

plt.plot(
    epochs_completed,
    training_loss,
    label="Training Loss",
)

plt.plot(
    epochs_completed,
    validation_loss,
    label="Validation Loss",
)

plt.title("Training and Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.tight_layout()

loss_plot_path = IMAGES_DIR / "loss_plot.png"
plt.savefig(loss_plot_path)
plt.close()


print(f"Accuracy graph saved to: {accuracy_plot_path}")
print(f"Loss graph saved to: {loss_plot_path}")
print("\nTraining completed successfully.")