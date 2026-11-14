
import torch
from torchvision.datasets import EMNIST
from torchvision import transforms
from torch.utils.data import DataLoader

# Convert images into tensors
transform = transforms.Compose([
    transforms.ToTensor()
])

# Load EMNIST ByClass training dataset
train_dataset = EMNIST(
    root="./data",
    split="byclass",
    train=True,
    download=True,
    transform=transform
)

# Load EMNIST ByClass testing dataset
test_dataset = EMNIST(
    root="./data",
    split="byclass",
    train=False,
    download=True,
    transform=transform
)

# Create DataLoaders
train_loader = DataLoader(
    train_dataset,
    batch_size=128,
    shuffle=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=128,
    shuffle=False
)

# Test the DataLoaders
images, labels = next(iter(train_loader))

print("Image batch:", images.shape)
print("Label batch:", labels.shape)
print("Number of training images:", len(train_dataset))
print("Number of testing images:", len(test_dataset))
