
# Handwritten Character Recognition Using CNN

## Overview

A deep learning project that uses a Convolutional Neural Network (CNN) to recognize handwritten letters and numbers.

The project uses the EMNIST ByClass dataset, which contains 814,255 handwritten character images across 62 classes.

Classes:
- Numbers: 0–9
- Uppercase letters: A–Z
- Lowercase letters: a–z

## Technologies

- Python 3
- TensorFlow / Keras
- TensorFlow Datasets
- NumPy
- Matplotlib
- Scikit-learn

## Project Structure

```text
handwritten-recognition/
│
├── data/
│   └── emnist/
│
├── models/
│   └── cnn_model.keras
│
├── src/
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── visualize.py
│   ├── model.py
│   └── train.py
│
├── evaluate.py
├── predict.py
├── requirements.txt
└── README.md
```

## Installation

Create a virtual environment:

```bash
python3 -m venv cnn-env
source cnn-env/bin/activate
```

Install dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Dataset

Dataset: EMNIST ByClass

Source:
https://www.nist.gov/itl/products-and-services/emnist-dataset

The dataset can be loaded through TensorFlow Datasets.

```python
import tensorflow_datasets as tfds

train, test = tfds.load(
    "emnist/byclass",
    split=["train", "test"],
    as_supervised=True,
    data_dir="./data"
)
```

## Model Architecture

The project uses a Convolutional Neural Network.

Planned architecture:

1. Input layer
2. Convolutional layers
3. ReLU activation
4. Max pooling
5. Flatten layer
6. Dense layers
7. Softmax output layer (62 classes)

## Development Roadmap

- [ ] Download EMNIST dataset
- [ ] Explore and visualize dataset
- [ ] Preprocess image data
- [ ] Implement CNN architecture
- [ ] Train the model
- [ ] Evaluate model accuracy
- [ ] Experiment with hyperparameters
- [ ] Test predictions on custom handwriting
- [ ] Build a character recognition interface

## Model Evaluation

The model will be evaluated using:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion matrix

## Goals

- Understand CNN architecture and feature extraction.
- Experiment with deep learning and image classification.
- Evaluate model performance on unseen data.
- Develop a working handwritten character recognition system.

## License

Educational and personal project.
