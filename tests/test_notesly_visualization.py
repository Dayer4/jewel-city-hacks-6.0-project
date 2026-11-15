
import os
import sys

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)

import cv2
import torch
import matplotlib.pyplot as plt

from src.model import CharacterCNN
from src.segmentation import segment_image
from src.recognition import recognize_character


# Device
device = torch.device(
    "mps" if torch.backends.mps.is_available() else "cpu"
)

# Load trained model
model = CharacterCNN(num_classes=62)

model.load_state_dict(
    torch.load(
        "checkpoints/best_model.pth",
        map_location=device,
        weights_only=True
    )
)

model.to(device)
model.eval()


# Input image
image_path = "data/raw/real/ryan.png"

original = cv2.imread(image_path)

if original is None:
    raise ValueError(f"Could not load {image_path}")

original = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)


# Segment image
lines = segment_image(image_path)


# Recognize every character
fig, ax = plt.subplots(figsize=(14, 8))

ax.imshow(original)

recognized_text = []

for line in lines:

    line_text = ""

    for character in line:

        # Bounding box
        x, y, w, h = character["bbox"]

        # CNN prediction
        result = recognize_character(
            character["image"],
            model,
            device
        )

        prediction = result["character"]
        confidence = result["confidence"]

        line_text += prediction

        # Draw bounding box
        rectangle = plt.Rectangle(
            (x, y),
            w,
            h,
            fill=False,
            edgecolor="lime",
            linewidth=2
        )

        ax.add_patch(rectangle)

        # Display prediction above character
        ax.text(
            x,
            max(y - 5, 0),
            f"{prediction} ({confidence:.0%})",
            color="red",
            fontsize=10,
            fontweight="bold",
            bbox=dict(facecolor="white", alpha=0.8)
        )

    recognized_text.append(line_text)


ax.set_title("Notesly - Character Recognition")
ax.axis("off")

plt.tight_layout()
plt.show()


# Display reconstructed text
print("\nRecognized Text:")

for line in recognized_text:
    print(line)
