
import cv2
import numpy as np


def preprocess_image(image_path):

    # Load image
    image = cv2.imread(image_path)

    if image is None:
        raise ValueError("Could not load image")

    # Convert to grayscale
    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    # Remove small noise
    blurred = cv2.GaussianBlur(
        gray,
        (3, 3),
        0
    )

    # Convert handwriting to white pixels
    # and background to black pixels
    _, binary = cv2.threshold(
        blurred,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    return binary
