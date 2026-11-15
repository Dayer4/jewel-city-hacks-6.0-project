from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim

from src.data_loader import train_loader, test_loader
from src.model import CharacterCNN

EPOCHS = 10
LEARNING_RATE = 0.001

CHECKPOINT_DIR = Path(__file__).resolve().parent.parent / "checkpoints"
CHECKPOINT_PATH = CHECKPOINT_DIR / "best_model.pth"


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def evaluate(model, loader, device) -> float:
    """Return accuracy (0-1) of the model on a data loader."""
    model.eval()  # turn off dropout
    correct = 0
    total = 0
    with torch.no_grad():  # no learning, just measuring
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            predictions = model(images).argmax(dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
    return correct / total


def main():
    device = get_device()
    print(f"Training on: {device}")

    model = CharacterCNN(num_classes=62).to(device)
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    CHECKPOINT_DIR.mkdir(exist_ok=True)
    best_accuracy = 0.0

    for epoch in range(1, EPOCHS + 1):
        model.train()  # turn dropout back on
        running_loss = 0.0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()                      # clear old gradients
            loss = loss_function(model(images), labels)
            loss.backward()                            # compute how to improve
            optimizer.step()                           # apply the improvement

            running_loss += loss.item()

        test_accuracy = evaluate(model, test_loader, device)
        avg_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch}/{EPOCHS}  loss: {avg_loss:.4f}  test accuracy: {test_accuracy:.2%}")

        # Only keep the checkpoint if it's the best so far.
        if test_accuracy > best_accuracy:
            best_accuracy = test_accuracy
            torch.save(model.state_dict(), CHECKPOINT_PATH)
            print(f"  saved new best model ({best_accuracy:.2%})")


if __name__ == "__main__":
    main()