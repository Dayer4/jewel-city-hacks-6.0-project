import cv2
import numpy as np


def preprocess(image_path):

    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if image is None:
        raise ValueError(f"Image not found: {image_path}")

    _, binary = cv2.threshold(
        image, 0, 255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    coords = cv2.findNonZero(binary)

    if coords is None:
        raise ValueError("No handwriting detected")

    x, y, w, h = cv2.boundingRect(coords)
    cropped = binary[y:y+h, x:x+w]

    scale = 20 / max(w, h)

    new_w = max(1, round(w * scale))
    new_h = max(1, round(h * scale))

    resized = cv2.resize(
        cropped,
        (new_w, new_h),
        interpolation=cv2.INTER_AREA
    )

    output = np.zeros((28, 28), dtype=np.uint8)

    x_offset = (28 - new_w) // 2
    y_offset = (28 - new_h) // 2

    output[
        y_offset:y_offset+new_h,
        x_offset:x_offset+new_w
    ] = resized

    return output.astype(np.float32) / 255.0