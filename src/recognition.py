
import cv2
import numpy as np
import torch


# EMNIST ByClass mapping
CLASS_MAPPING = (
    list("0123456789")
    + list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    + list("abcdefghijklmnopqrstuvwxyz")
)


def preprocess_character(image):

    # Convert grayscale character to binary
    _, binary = cv2.threshold(
        image,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    coords = cv2.findNonZero(binary)

    if coords is None:
        raise ValueError("No character detected")

    x, y, w, h = cv2.boundingRect(coords)

    cropped = binary[y:y+h, x:x+w]

    # Preserve aspect ratio
    scale = 20 / max(w, h)

    new_w = max(1, round(w * scale))
    new_h = max(1, round(h * scale))

    resized = cv2.resize(
        cropped,
        (new_w, new_h),
        interpolation=cv2.INTER_AREA
    )

    # Center inside 28x28 canvas
    output = np.zeros((28, 28), dtype=np.uint8)

    x_offset = (28 - new_w) // 2
    y_offset = (28 - new_h) // 2

    output[
        y_offset:y_offset+new_h,
        x_offset:x_offset+new_w
    ] = resized

    return output.astype(np.float32) / 255.0


def recognize_character(image, model, device):

    processed = preprocess_character(image)

    # Convert to PyTorch tensor
    tensor = torch.from_numpy(processed)

    tensor = tensor.unsqueeze(0).unsqueeze(0)
    tensor = tensor.to(device)

    model.eval()

    with torch.no_grad():

        logits = model(tensor)

        probabilities = torch.softmax(logits, dim=1)

        confidence, prediction = torch.max(
            probabilities,
            dim=1
        )

    predicted_index = prediction.item()

    return {
        "character": CLASS_MAPPING[predicted_index],
        "confidence": confidence.item()
    }


def recognize_lines(segmented_lines, model, device):

    results = []

    for line in segmented_lines:

        recognized_line = []

        for segment in line:

            prediction = recognize_character(
                segment["image"],
                model,
                device
            )

            recognized_line.append({
                "text": prediction["character"],
                "confidence": prediction["confidence"],
                "space_before": segment["space_before"]
            })

        results.append(recognized_line)

    return results
