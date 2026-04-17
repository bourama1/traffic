import os
import sys

import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow import keras  # ty:ignore[unresolved-import]

EPOCHS = 10
IMG_WIDTH = 30
IMG_HEIGHT = 30
NUM_CATEGORIES = 43
TEST_SIZE = 0.4


def main():

    # Check command-line arguments
    if len(sys.argv) not in [2, 3]:
        sys.exit("Usage: python traffic.py data_directory [model.h5]")

    # Get image arrays and labels for all image files
    images, labels = load_data(sys.argv[1])

    # Split data into training and testing sets
    labels = keras.utils.to_categorical(labels)
    x_train, x_test, y_train, y_test = train_test_split(
        np.array(images), np.array(labels), test_size=TEST_SIZE
    )

    # Get a compiled neural network
    model = get_model()

    # Fit model on training data
    model.fit(x_train, y_train, epochs=EPOCHS)

    # Evaluate neural network performance
    model.evaluate(x_test, y_test, verbose=2)

    # Save model to file
    if len(sys.argv) == 3:
        filename = sys.argv[2]
        model.save(filename)
        print(f"Model saved to {filename}.")


def load_data(data_dir):
    """
    Load image data from directory `data_dir`.

    Assume `data_dir` has one directory named after each category, numbered
    0 through NUM_CATEGORIES - 1. Inside each category directory will be some
    number of image files.

    Return tuple `(images, labels)`. `images` should be a list of all
    of the images in the data directory, where each image is formatted as a
    numpy ndarray with dimensions IMG_WIDTH x IMG_HEIGHT x 3. `labels` should
    be a list of integer labels, representing the categories for each of the
    corresponding `images`.
    """
    images = []
    labels = []

    for category in range(NUM_CATEGORIES):
        category_path = os.path.join(data_dir, str(category))

        if not os.path.isdir(category_path):
            continue

        for filename in os.listdir(category_path):
            filepath = os.path.join(category_path, filename)
            img = cv2.imread(filepath)

            if img is None:
                continue

            img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
            img = img / 255.0  # normalize pixel values

            images.append(img)
            labels.append(category)

    return (images, labels)


# --- Model variants ---
def _model_final():
    """Final model: two conv blocks (32->64), dense 256, dropout 0.4. ~98.7% test acc."""
    return keras.models.Sequential(
        [
            keras.layers.Conv2D(
                32, (3, 3), activation="relu", input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)
            ),
            keras.layers.MaxPooling2D(pool_size=(2, 2)),
            keras.layers.Conv2D(64, (3, 3), activation="relu"),
            keras.layers.MaxPooling2D(pool_size=(2, 2)),
            keras.layers.Flatten(),
            keras.layers.Dense(256, activation="relu"),
            keras.layers.Dropout(0.4),
            keras.layers.Dense(NUM_CATEGORIES, activation="softmax"),
        ]
    )


def _model_single_conv():
    """Baseline: one conv block only. Expected ~85-88% test accuracy."""
    return keras.models.Sequential(
        [
            keras.layers.Conv2D(
                32, (3, 3), activation="relu", input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)
            ),
            keras.layers.MaxPooling2D(pool_size=(2, 2)),
            keras.layers.Flatten(),
            keras.layers.Dense(256, activation="relu"),
            keras.layers.Dropout(0.4),
            keras.layers.Dense(NUM_CATEGORIES, activation="softmax"),
        ]
    )


def _model_large_filters():
    """Larger filter counts (64->128). Similar accuracy to final but slower per epoch."""
    return keras.models.Sequential(
        [
            keras.layers.Conv2D(
                64, (3, 3), activation="relu", input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)
            ),
            keras.layers.MaxPooling2D(pool_size=(2, 2)),
            keras.layers.Conv2D(128, (3, 3), activation="relu"),
            keras.layers.MaxPooling2D(pool_size=(2, 2)),
            keras.layers.Flatten(),
            keras.layers.Dense(256, activation="relu"),
            keras.layers.Dropout(0.4),
            keras.layers.Dense(NUM_CATEGORIES, activation="softmax"),
        ]
    )


def _model_dense_512():
    """Dense 512 with weak dropout 0.25. Overfitting expected by epoch 7-8."""
    return keras.models.Sequential(
        [
            keras.layers.Conv2D(
                32, (3, 3), activation="relu", input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)
            ),
            keras.layers.MaxPooling2D(pool_size=(2, 2)),
            keras.layers.Conv2D(64, (3, 3), activation="relu"),
            keras.layers.MaxPooling2D(pool_size=(2, 2)),
            keras.layers.Flatten(),
            keras.layers.Dense(512, activation="relu"),
            keras.layers.Dropout(0.25),
            keras.layers.Dense(NUM_CATEGORIES, activation="softmax"),
        ]
    )


def _model_three_conv():
    """Three conv blocks. Third conv sees ~5x5 feature maps, limited benefit."""
    return keras.models.Sequential(
        [
            keras.layers.Conv2D(
                32, (3, 3), activation="relu", input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)
            ),
            keras.layers.MaxPooling2D(pool_size=(2, 2)),
            keras.layers.Conv2D(64, (3, 3), activation="relu"),
            keras.layers.MaxPooling2D(pool_size=(2, 2)),
            keras.layers.Conv2D(64, (3, 3), activation="relu"),
            keras.layers.Flatten(),
            keras.layers.Dense(256, activation="relu"),
            keras.layers.Dropout(0.4),
            keras.layers.Dense(NUM_CATEGORIES, activation="softmax"),
        ]
    )


def get_model():
    """
    Returns a compiled convolutional neural network model. Assume that the
    `input_shape` of the first layer is `(IMG_WIDTH, IMG_HEIGHT, 3)`.
    The output layer should have `NUM_CATEGORIES` units, one for each category.
    """
    model = _model_final()

    model.compile(
        optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
    )

    return model


def compare_models():
    """
    Runs all model variants on the same train/test split and prints a comparison table.
    Usage: python traffic.py data_directory --compare
    """
    if len(sys.argv) < 2:
        sys.exit("Usage: python traffic.py data_directory --compare")

    print("Loading data...")
    images, labels = load_data(sys.argv[1])
    labels_cat = keras.utils.to_categorical(labels)
    x_train, x_test, y_train, y_test = train_test_split(
        np.array(images), np.array(labels_cat), test_size=TEST_SIZE
    )
    print(f"Data loaded. Train: {len(x_train)}, Test: {len(x_test)}\n")

    variants = [
        ("Final (32->64, dense 256, dropout 0.4)", _model_final),
        ("Single conv block", _model_single_conv),
        ("Large filters (64->128)", _model_large_filters),
        ("Dense 512 + weak dropout 0.25", _model_dense_512),
        ("Three conv blocks", _model_three_conv),
    ]

    results = []

    for name, model_fn in variants:
        print("=" * 60)
        print(f"Model: {name}")
        print("=" * 60)

        model = model_fn()
        model.compile(
            optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
        )
        history = model.fit(x_train, y_train, epochs=EPOCHS)
        loss, accuracy = model.evaluate(x_test, y_test, verbose=2)

        train_acc = history.history["accuracy"]
        results.append((name, train_acc, loss, accuracy))
        print()

    print("\n" + "=" * 60)
    print("COMPARISON SUMMARY")
    print("=" * 60)
    print(
        f"{'Model':<40} {'Ep1 Acc':>8} {'Ep10 Acc':>9} {'Test Loss':>10} {'Test Acc':>9}"
    )
    print("-" * 80)
    for name, train_acc, loss, accuracy in results:
        print(
            f"{name:<40} {train_acc[0]:>7.2%} {train_acc[-1]:>8.2%}"
            f" {loss:>10.4f} {accuracy:>8.2%}"
        )
    print("=" * 80)


if __name__ == "__main__":
    if "--compare" in sys.argv:
        sys.argv.remove("--compare")
        compare_models()
    else:
        main()
