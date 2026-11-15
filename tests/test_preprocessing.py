
import os

from src.preprocessing import preprocess
from src.visualize import show_preprocessing


image_folder = "data/raw/real/letters"

images = [
    file for file in os.listdir(image_folder)
    if file.lower().endswith((".png", ".jpg", ".jpeg"))
]

if not images:
    raise ValueError("No images found in folder")

for filename in sorted(images):

    image_path = os.path.join(image_folder, filename)

    result = preprocess(image_path)

    print(
        filename,
        "Shape:", result.shape,
        "Min:", result.min(),
        "Max:", result.max()
    )

    show_preprocessing(image_path, result)
