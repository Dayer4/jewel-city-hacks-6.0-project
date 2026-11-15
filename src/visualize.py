
import cv2
import matplotlib.pyplot as plt


def show_preprocessing(original_path, processed):

    original = cv2.imread(original_path)

    if original is None:
        raise ValueError("Could not load original image")

    original = cv2.cvtColor(
        original,
        cv2.COLOR_BGR2RGB
    )

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    axes[0].imshow(original)
    axes[0].set_title("Original Image")
    axes[0].axis("off")

    axes[1].imshow(processed, cmap="gray")
    axes[1].set_title("28x28 Processed Image")
    axes[1].axis("off")

    plt.tight_layout()
    plt.show()
