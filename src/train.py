
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from tqdm import tqdm

from data_loader import train_loader, test_loader
from model import CharacterCNN


# -------------------------
# Configuration
# -------------------------

EPOCHS = 10
LEARNING_RATE = 0.001
NUM_CLASSES = 62

# Automatically select Apple GPU or CPU
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print(f"Training device: {device}")

# -------------------------
# Initialize model
# -------------------------

model = CharacterCNN(num_classes=NUM_CLASSES).to(device)

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)

# -------------------------
# Checkpoint directory
# -------------------------

checkpoint_dir = Path("checkpoints")
checkpoint_dir.mkdir(exist_ok=True)

best_accuracy = 0.0


# -------------------------
# Evaluation function
# -------------------------

def evaluate(model, loader):

    model.eval()

    correct = 0
    total = 0
    total_loss = 0

    with torch.no_grad():

        for images, labels in loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            loss = criterion(outputs, labels)

            total_loss += loss.item()

            predictions = outputs.argmax(dim=1)

            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    accuracy = 100 * correct / total
    average_loss = total_loss / len(loader)

    return accuracy, average_loss


# -------------------------
# Training loop
# -------------------------

for epoch in range(EPOCHS):

    model.train()

    running_loss = 0
    correct = 0
    total = 0

    for batch_idx, (images, labels) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch + 1}/{EPOCHS}")):

        images = images.to(device)
        labels = labels.to(device)

        # Reset gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(images)

        loss = criterion(outputs, labels)

        # Backpropagation
        loss.backward()

        # Update weights
        optimizer.step()

        running_loss += loss.item()

        predictions = outputs.argmax(dim=1)

        correct += (predictions == labels).sum().item()
        total += labels.size(0)

    train_accuracy = 100 * correct / total
    train_loss = running_loss / len(train_loader)

    # Evaluate
    test_accuracy, test_loss = evaluate(model, test_loader)

    print(f"\nEpoch [{epoch + 1}/{EPOCHS}]")
    print(f"Training Loss: {train_loss:.4f}")
    print(f"Training Accuracy: {train_accuracy:.2f}%")
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy:.2f}%")

    # Save best model
    if test_accuracy > best_accuracy:

        best_accuracy = test_accuracy

        torch.save(
            model.state_dict(),
            checkpoint_dir / "best_model.pth"
        )

        print("New best model saved!")

print("\nTraining complete.")
print(f"Best test accuracy: {best_accuracy:.2f}%")
