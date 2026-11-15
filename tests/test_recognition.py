import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="Recognition requires the project's PyTorch runtime")
cv2 = pytest.importorskip("cv2", reason="Recognition requires OpenCV")

from src.recognition import (
    CLASS_MAPPING,
    preprocess_character,
    recognize_character,
    recognize_lines,
)


class FixedLogitModel:
    def __init__(self, predicted_index):
        self.predicted_index = predicted_index
        self.eval_called = False

    def eval(self):
        self.eval_called = True
        return self

    def __call__(self, batch):
        logits = torch.zeros((batch.shape[0], len(CLASS_MAPPING)), device=batch.device)
        logits[:, self.predicted_index] = 8.0
        return logits


def _character_image():
    image = np.full((40, 32), 255, dtype=np.uint8)
    cv2.line(image, (8, 5), (23, 34), 0, thickness=4)
    return image


def test_preprocess_character_returns_normalized_28_by_28_image():
    result = preprocess_character(_character_image())

    assert result.shape == (28, 28)
    assert result.dtype == np.float32
    assert np.isfinite(result).all()
    assert result.min() >= 0.0
    assert result.max() <= 1.0
    assert np.any(result > 0.0)


def test_preprocess_character_raises_for_blank_image():
    blank = np.full((32, 32), 255, dtype=np.uint8)

    with pytest.raises(ValueError, match="No character detected"):
        preprocess_character(blank)


def test_recognize_character_returns_mapped_class_and_confidence():
    model = FixedLogitModel(predicted_index=10)

    result = recognize_character(_character_image(), model, torch.device("cpu"))

    assert result["character"] == "A"
    assert result["character"] == CLASS_MAPPING[10]
    assert 0.0 <= result["confidence"] <= 1.0
    assert result["confidence"] > 0.9
    assert model.eval_called


def test_recognize_lines_preserves_space_flags():
    model = FixedLogitModel(predicted_index=11)
    segmented_lines = [[
        {"image": _character_image(), "space_before": False},
        {"image": _character_image(), "space_before": True},
    ]]

    result = recognize_lines(segmented_lines, model, torch.device("cpu"))

    assert len(result) == 1
    assert [character["text"] for character in result[0]] == ["B", "B"]
    assert [character["space_before"] for character in result[0]] == [False, True]
    assert all(0.0 <= character["confidence"] <= 1.0 for character in result[0])
