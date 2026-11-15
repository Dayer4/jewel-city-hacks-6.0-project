import cv2
import numpy as np
import pytest

from src.preprocessing import preprocess


def _write_handwriting_image(path):
    image = np.full((64, 64), 255, dtype=np.uint8)
    cv2.line(image, (20, 14), (42, 49), 0, thickness=4)
    cv2.line(image, (42, 14), (20, 49), 0, thickness=4)
    assert cv2.imwrite(str(path), image)
    return path


def test_preprocess_returns_normalized_28_by_28_image(tmp_path):
    image_path = _write_handwriting_image(tmp_path / "handwriting.png")

    result = preprocess(str(image_path))

    assert result.shape == (28, 28)
    assert result.dtype == np.float32
    assert np.isfinite(result).all()
    assert result.min() >= 0.0
    assert result.max() <= 1.0
    assert np.any(result > 0.0)


def test_preprocess_raises_for_missing_image(tmp_path):
    with pytest.raises(ValueError, match="Image not found"):
        preprocess(str(tmp_path / "missing.png"))


def test_preprocess_raises_when_image_has_no_handwriting(tmp_path):
    blank_path = tmp_path / "blank.png"
    assert cv2.imwrite(str(blank_path), np.full((32, 32), 255, dtype=np.uint8))

    with pytest.raises(ValueError, match="No handwriting detected"):
        preprocess(str(blank_path))
