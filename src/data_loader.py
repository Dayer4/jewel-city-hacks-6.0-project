from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import EMNIST

# Build the data path from this file's location so it works from any folder.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"

BATCH_SIZE = 128


def fix_orientation(image_tensor: torch.Tensor) -> torch.Tensor:
    """EMNIST images are stored transposed (rotated and mirrored).

    The tensor shape is (channels, height, width). Swapping height and
    width (dims 1 and 2) turns each image upright.
    """
    return image_tensor.transpose(1, 2)


transform = transforms.Compose([
    transforms.ToTensor(),                 # PIL image -> tensor, values 0-1
    transforms.Lambda(fix_orientation),    # make characters upright
])

# Load EMNIST ByClass training dataset
train_dataset = EMNIST(
    root=str(DATA_DIR),
    split="byclass",
    train=True,
    download=True,
    transform=transform,
)

# Load EMNIST ByClass testing dataset
test_dataset = EMNIST(
    root=str(DATA_DIR),
    split="byclass",
    train=False,
    download=True,
    transform=transform,
)

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)


if __name__ == "__main__":
    # Quick sanity check, only runs with: python3 -m src.data_loader
    images, labels = next(iter(train_loader))
    print("Image batch:", images.shape)
    print("Label batch:", labels.shape)
    print("Number of training images:", len(train_dataset))
    print("Number of testing images:", len(test_dataset))
