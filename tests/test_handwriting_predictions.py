
import torch
import cv2
import matplotlib.pyplot as plt

from pathlib import Path
from src.model import CharacterCNN
from src.segmentation import segment_image
from src.recognition import recognize_character

# Directories
LETTER_DIR = Path("data/raw/real/letters")
RAW_DIR = Path("data/raw")

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}

# Collect individual letter images
image_paths = [
    path for path in LETTER_DIR.iterdir()
    if path.suffix.lower() in IMAGE_EXTENSIONS
]

# Collect full-word images directly inside data/raw
image_paths += [
    path for path in RAW_DIR.iterdir()
    if path.suffix.lower() in IMAGE_EXTENSIONS
]

if not image_paths:
    raise ValueError("No handwriting images found.")

# Select device
device = torch.device(
    "mps" if torch.backends.mps.is_available() else "cpu"
)

# Load trained model
model = CharacterCNN(num_classes=62)

checkpoint = torch.load(
    "checkpoints/best_model.pth",
    map_location=device,
    weights_only=True
)

model.load_state_dict(checkpoint)
model.to(device)
model.eval()

# Collect predictions
results = []

for image_path in image_paths:

    print(f"\nProcessing: {image_path}")

    lines = segment_image(str(image_path))

    for line_index, line in enumerate(lines):

        for segment in line:

            image = segment["image"]

            prediction = recognize_character(
                image,
                model,
                device
            )

            results.append({
                "filename": image_path.name,
                "image": image,
                "prediction": prediction["character"],
                "confidence": prediction["confidence"],
                "line": line_index
            })

# Display predictions
if not results:
    raise ValueError("No characters detected.")

columns = 5
rows = (len(results) + columns - 1) // columns

fig, axes = plt.subplots(
    rows,
    columns,
    figsize=(15, 3 * rows),
    squeeze=False
)

for i, ax in enumerate(axes.flat):

    if i >= len(results):
        ax.axis("off")
        continue

    result = results[i]

    ax.imshow(result["image"], cmap="gray")

    ax.set_title(
        f"{result['filename']}\n"
        f"Predicted: {result['prediction']}\n"
        f"Confidence: {result['confidence']:.1%}"
    )

    ax.axis("off")

plt.suptitle("Notesly - Handwriting Recognition")
plt.tight_layout()
plt.show()

# Print results
print("\n========== PREDICTIONS ==========")

for result in results:
    print(
        f"{result['filename']}: "
        f"{result['prediction']} "
        f"({result['confidence']:.1%})"
    )
