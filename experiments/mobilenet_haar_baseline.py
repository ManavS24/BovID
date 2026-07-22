"""Reference-only baseline: MobileNetV3Small classifier with Haar-cascade cropping.

Consolidated from the original Colab notebook (notebooks/original_colab_notebook.ipynb, cell 0).
This was the earliest approach explored -- it predates the ResNet50 baseline
(experiments/resnet_baseline.py) and the deployed YOLOv8s-cls pipeline (src/train.py) -- and was
abandoned in favor of them. NOT used for the deployed model in sih2025/best_saved_model/.

Known limitation carried over from the original: no real cattle/buffalo Haar cascade was ever
sourced, so this uses OpenCV's bundled cat-face cascade as a placeholder detector. Detection
against cattle images with it is unreliable, which is the main reason this approach was dropped.

Needs TensorFlow, which is NOT in the main requirements.txt (no TF wheel exists for this repo's
Python version at time of writing) -- see experiments/requirements-mobilenet.txt and install it
in a separate venv with a TensorFlow-compatible Python version.
"""

import os

import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV3Small
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tqdm import tqdm

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE_PATH = os.path.join(PROJECT_ROOT, "sih2025", "data", "indianbovine1")
IMG_SIZE = (640, 640)  # MobileNet standard
NUM_EPOCHS = 20
BATCH_SIZE = 32

# Placeholder cascade -- see module docstring.
CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalcatface.xml"
animal_cascade = cv2.CascadeClassifier(CASCADE_PATH)


def detect_and_crop(img_path):
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    animals = animal_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    crops = []
    for (x, y, w, h) in animals:
        crop = img[y:y + h, x:x + w]
        crop = cv2.resize(crop, IMG_SIZE)
        crop = crop.astype("float32") / 255.0
        crops.append(crop)

    if not crops:
        img_resized = cv2.resize(img, IMG_SIZE)
        crops.append(img_resized.astype("float32") / 255.0)

    return np.array(crops)


def load_labels(label_file):
    if label_file.endswith(".xlsx"):
        return pd.read_excel(label_file, engine="openpyxl")
    if label_file.endswith(".csv"):
        return pd.read_csv(label_file)
    raise ValueError("Unsupported label format: " + label_file)


def load_dataset(img_dir, label_file):
    df = load_labels(label_file)
    class_names = df.columns[1:].tolist()
    images, labels = [], []

    print(f"Loading {len(df)} images from {img_dir} ...")
    for _, row in tqdm(df.iterrows(), total=len(df)):
        img_path = os.path.join(img_dir, row["filename"])
        if not os.path.exists(img_path):
            continue

        crops = detect_and_crop(img_path)
        images.append(crops[0])  # assume one animal per image
        labels.append(np.argmax(row[1:].values))

    print(f"Finished loading {len(images)} images")
    return np.array(images), np.array(labels), class_names


def build_model(num_classes):
    base_model = MobileNetV3Small(input_shape=(*IMG_SIZE, 3), include_top=False, weights="imagenet")
    base_model.trainable = False

    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation="softmax"),
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-4),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def predict_breed(model, class_names, img_path, topk=3):
    crops = detect_and_crop(img_path)
    preds = model.predict(crops)
    final_pred = np.mean(preds, axis=0)  # average across crops if multiple detections
    top_idx = final_pred.argsort()[-topk:][::-1]
    return [(class_names[i], float(final_pred[i])) for i in top_idx]


def main():
    train_dir = os.path.join(BASE_PATH, "train")
    valid_dir = os.path.join(BASE_PATH, "valid")
    test_dir = os.path.join(BASE_PATH, "test")

    X_train, y_train, class_names = load_dataset(train_dir, os.path.join(train_dir, "_classes.csv"))
    X_valid, y_valid, _ = load_dataset(valid_dir, os.path.join(valid_dir, "_classes.csv"))
    X_test, y_test, _ = load_dataset(test_dir, os.path.join(test_dir, "_classes.csv"))
    print("Train:", X_train.shape, "Valid:", X_valid.shape, "Test:", X_test.shape)
    print("Classes:", class_names)

    model = build_model(len(class_names))
    model.summary()

    callbacks = [
        EarlyStopping(patience=5, restore_best_weights=True),
        ReduceLROnPlateau(factor=0.5, patience=3),
    ]
    model.fit(
        X_train, y_train,
        validation_data=(X_valid, y_valid),
        epochs=NUM_EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
    )

    test_loss, test_acc = model.evaluate(X_test, y_test)
    print("Test accuracy:", test_acc)

    sample_img = os.path.join(test_dir, os.listdir(test_dir)[0])
    print("Sample prediction:", predict_breed(model, class_names, sample_img))


if __name__ == "__main__":
    main()
