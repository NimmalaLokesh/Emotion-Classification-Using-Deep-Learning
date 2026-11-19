import os
import argparse
from pathlib import Path
import numpy as np
import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import (
    Input,
    Rescaling,
    RandomFlip,
    RandomRotation,
    Conv2D,
    MaxPooling2D,
    BatchNormalization,
    Flatten,
    Dense,
    Dropout,
)
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.metrics import classification_report

# Keep this order identical in video_prediction.py
EMOTIONS = [
    "angry",
    "disgust",
    "fear",
    "happy",
    "neutral",
    "sad",
    "surprise",
]

IMAGE_SIZE = (48, 48)
BATCH_SIZE = 64
EPOCHS = 45


def build_model(num_classes: int) -> tf.keras.Model:
    model = Sequential([
        Input(shape=(48, 48, 1)),
        Rescaling(1.0 / 255),
        RandomFlip("horizontal"),
        RandomRotation(0.08),

        Conv2D(64, (3, 3), activation="relu", padding="same"),
        BatchNormalization(),
        Conv2D(64, (3, 3), activation="relu", padding="same"),
        BatchNormalization(),
        MaxPooling2D(2),
        Dropout(0.25),

        Conv2D(128, (3, 3), activation="relu", padding="same"),
        BatchNormalization(),
        Conv2D(128, (3, 3), activation="relu", padding="same"),
        BatchNormalization(),
        MaxPooling2D(2),
        Dropout(0.25),

        Conv2D(256, (3, 3), activation="relu", padding="same"),
        BatchNormalization(),
        MaxPooling2D(2),
        Dropout(0.25),

        Flatten(),
        Dense(256, activation="relu"),
        BatchNormalization(),
        Dropout(0.5),
        Dense(num_classes, activation="softmax"),
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/images", help="Path containing train/ and test/ folders")
    args = parser.parse_args()

    train_dir = Path(args.data) / "train"
    test_dir = Path(args.data) / "test"

    if not train_dir.exists() or not test_dir.exists():
        raise SystemExit(f"Train or test folder not found inside '{args.data}'.")

    for emotion in EMOTIONS:
        if not (train_dir / emotion).exists() or not (test_dir / emotion).exists():
            raise SystemExit(f"Missing emotion subfolder: {emotion}")

    train_dataset = tf.keras.utils.image_dataset_from_directory(
        str(train_dir),
        labels="inferred",
        label_mode="categorical",
        class_names=EMOTIONS,
        color_mode="grayscale",
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=True,
        seed=42,
        validation_split=0.1,
        subset="training",
    )

    validation_dataset = tf.keras.utils.image_dataset_from_directory(
        str(train_dir),
        labels="inferred",
        label_mode="categorical",
        class_names=EMOTIONS,
        color_mode="grayscale",
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=False,
        seed=42,
        validation_split=0.1,
        subset="validation",
    )

    test_dataset = tf.keras.utils.image_dataset_from_directory(
        str(test_dir),
        labels="inferred",
        label_mode="categorical",
        class_names=EMOTIONS,
        color_mode="grayscale",
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    autotune = tf.data.AUTOTUNE
    train_dataset = train_dataset.prefetch(autotune)
    validation_dataset = validation_dataset.prefetch(autotune)
    test_dataset = test_dataset.prefetch(autotune)

    os.makedirs("models", exist_ok=True)
    model_path = os.path.join("models", "video_model.keras")

    model = build_model(len(EMOTIONS))
    model.summary()

    callbacks = [
        ModelCheckpoint(model_path, monitor="val_accuracy", save_best_only=True, verbose=1),
        EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-5),
    ]

    model.fit(
        train_dataset,
        validation_data=validation_dataset,
        epochs=EPOCHS,
        callbacks=callbacks,
    )

    test_loss, test_accuracy = model.evaluate(test_dataset, verbose=1)
    print(f"\nTest Loss: {test_loss:.4f} | Test Accuracy: {test_accuracy * 100:.2f}%")

    y_true = np.concatenate([np.argmax(labels.numpy(), axis=1) for _, labels in test_dataset])
    y_pred = np.argmax(model.predict(test_dataset, verbose=0), axis=1)
    print("\nClassification Report:\n")
    print(classification_report(y_true, y_pred, target_names=EMOTIONS, zero_division=0))

    model.save(model_path)
    print(f"Saved model to {model_path}")


if __name__ == "__main__":
    main()