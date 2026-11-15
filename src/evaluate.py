from pathlib import Path

import torch

from src.data_loader import test_loader
from src.model import CharacterCNN

CHECKPOINT = Path(__file__).resolve().parent.parent / "checkpoints" / "best_model.pth"

CLASS_MAPPING = (
    list("0123456789")
    + list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    + list("abcdefghijklmnopqrstuvwxyz")
)


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def main():
    device = get_device()
    model = CharacterCNN(num_classes=62).to(device)
    model.load_state_dict(torch.load(CHECKPOINT, map_location=device, weights_only=True))
    model.eval()  # turns off dropout so results are consistent

    correct = torch.zeros(62)
    total = torch.zeros(62)

    with torch.no_grad():  # we're only measuring, not learning
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            predictions = model(images).argmax(dim=1)
            for label, pred in zip(labels.cpu(), predictions.cpu()):
                total[label] += 1
                correct[label] += int(label == pred)

    print(f"Overall test accuracy: {correct.sum() / total.sum():.2%}\n")
    print("Per-class accuracy:")
    for i, char in enumerate(CLASS_MAPPING):
        if total[i] > 0:
            print(f"  {char}: {correct[i] / total[i]:.1%}  ({int(total[i])} samples)")


if __name__ == "__main__":
    main()
