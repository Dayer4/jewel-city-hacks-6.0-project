
import argparse
from pathlib import Path

import torch

from src.model import CharacterCNN
from src.segmentation import segment_image
from src.recognition import recognize_lines
from src.reconstruction import reconstruct_text


CHECKPOINT_PATH = Path("checkpoints/best_model.pth")
OUTPUT_PATH = Path("uploads/notes.txt")


def predict(image_path):

    device = torch.device(
        "mps" if torch.backends.mps.is_available()
        else "cpu"
    )

    # Load CNN
    model = CharacterCNN(num_classes=62)

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=device,
        weights_only=True
    )

    model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()

    # 1. Segment image
    print("\n[1/4] Segmenting handwriting...")

    segmented = segment_image(image_path)

    total_characters = sum(len(line) for line in segmented)

    print(f"Detected {len(segmented)} lines")
    print(f"Detected {total_characters} character components")

    # 2. Recognize characters
    print("\n[2/4] Recognizing characters...")

    recognized = recognize_lines(
        segmented,
        model,
        device
    )

    # Display individual predictions
    for line_index, line in enumerate(recognized):

        print(f"\nLine {line_index + 1}:")

        for character in line:
            print(
                f"{character['text']} "
                f"({character['confidence']:.2%})",
                end=" | "
            )

        print()

    # 3. Reconstruct text
    print("\n[3/4] Reconstructing text...")

    text = reconstruct_text(recognized)

    print("\nRecognized Text:")
    print("----------------")
    print(text)

    # 4. Save text
    print("\n[4/4] Saving notes...")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    OUTPUT_PATH.write_text(text + "\n", encoding="utf-8")

    print(f"Saved to: {OUTPUT_PATH.resolve()}")

    return text


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Notesly handwriting recognition"
    )

    parser.add_argument(
        "image",
        help="Path to handwritten image"
    )

    args = parser.parse_args()

    predict(args.image)
