# Dataset Documentation

## Dataset

This project uses the MNIST handwritten-digit dataset for ten-class classification (digits 0–9).

## Source

The dataset file is reused from Project 8:

- Project 8 repository: https://github.com/nahom-mersha/neural-network-from-scratch
- Dataset file: [`data/raw/mnist.npz`](https://github.com/nahom-mersha/neural-network-from-scratch/blob/main/data/raw/mnist.npz)

The file was downloaded from the repository's raw file URL and stored locally at `data/raw/mnist.npz`.

## File contents

The NumPy archive contains four arrays:

- `x_train`: 60,000 training images
- `y_train`: 60,000 training labels
- `x_test`: 10,000 test images
- `y_test`: 10,000 test labels

Images have dimensions 28 × 28 pixels and are initially stored as `uint8` arrays.

## Preprocessing

The data pipeline:

1. Preserves all ten digit classes (0–9).
2. Splits the original training set into 54,000 training examples and 6,000 validation examples.
3. Uses random seed 42 for the train/validation split.
4. Flattens each image from 28 × 28 pixels into 784 features.
5. Converts image values to `float32` and normalizes them by dividing by 255.
6. Preserves the 10,000-example test set separately.

Labels are represented as `int64` values.

## Reproducibility

The dataset file is kept locally and is not committed to Git. To obtain it, download the file from the source URL above and save it as `data/raw/mnist.npz`.

The preparation code is located in `src/pytorch_digit_classifier/data.py`.
