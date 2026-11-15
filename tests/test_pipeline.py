import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="The prediction pipeline requires PyTorch")
cv2 = pytest.importorskip("cv2", reason="The prediction pipeline requires OpenCV")

import predict


class FixedClassifier(torch.nn.Module):
    def __init__(self, num_classes=62):
        super().__init__()
        self.num_classes = num_classes

    def forward(self, batch):
        logits = torch.zeros((batch.shape[0], self.num_classes), device=batch.device)
        logits[:, 10] = 10.0
        return logits


def test_predict_runs_segmentation_recognition_and_reconstruction(tmp_path, monkeypatch):
    image = np.full((90, 120), 255, dtype=np.uint8)
    cv2.rectangle(image, (5, 10), (12, 29), 0, thickness=-1)
    cv2.rectangle(image, (40, 10), (47, 29), 0, thickness=-1)
    cv2.rectangle(image, (8, 55), (15, 74), 0, thickness=-1)
    image_path = tmp_path / "handwriting.png"
    assert cv2.imwrite(str(image_path), image)

    monkeypatch.setattr(predict.torch.backends.mps, "is_available", lambda: False)
    monkeypatch.setattr(predict, "CharacterCNN", FixedClassifier)
    monkeypatch.setattr(predict.torch, "load", lambda *args, **kwargs: {})

    assert predict.predict(str(image_path)) == "A A\nA"
