
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import cv2
import matplotlib.pyplot as plt

from src.segmentation import segment_image

# Input image
image_path = "data/raw/real/ryan.png"

# Load original image
original = cv2.imread(image_path)

if original is None:
    raise ValueError(f"Could not load {image_path}")

original = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)

# Run segmentation
lines = segment_image(image_path)

# Display original image and bounding boxes
fig, ax = plt.subplots(figsize=(12, 8))

ax.imshow(original)

for line in lines:
    for character in line:
        x, y, w, h = character["bbox"]

        rectangle = plt.Rectangle(
            (x, y),
            w,
            h,
            fill=False,
            edgecolor="red",
            linewidth=1.5
        )

        ax.add_patch(rectangle)

ax.set_title("Character Segmentation")
ax.axis("off")

plt.tight_layout()
plt.show()

# Display individual segmented characters
for line_index, line in enumerate(lines):

    count = len(line)

    fig, axes = plt.subplots(
        1,
        count,
        figsize=(max(count * 1.5, 4), 3)
    )

    if count == 1:
        axes = [axes]

    for i, character in enumerate(line):

        axes[i].imshow(character["image"], cmap="gray")
        axes[i].set_title(
            f"Char {i+1}\nSpace: {character['space_before']}"
        )
        axes[i].axis("off")

    fig.suptitle(f"Segmented Line {line_index + 1}")
    plt.tight_layout()
    plt.show()

print(f"Total lines: {len(lines)}")

for i, line in enumerate(lines):
    print(f"Line {i+1}: {len(line)} characters")
