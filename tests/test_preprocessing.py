import numpy as np
import torch
from PIL import Image

from spurbreast_repro.data import build_transform


def test_evaluation_transform_shape_and_range() -> None:
    image = Image.fromarray(np.full((40, 60), 128, dtype=np.uint8), mode="L").convert("RGB")
    tensor = build_transform("validation", image_size=224, normalize=False)(image)
    assert tensor.shape == (3, 224, 224)
    assert tensor.dtype == torch.float32
    assert 0.0 <= float(tensor.min()) <= float(tensor.max()) <= 1.0


def test_training_transform_outputs_requested_shape() -> None:
    image = Image.new("RGB", (80, 60), color=(20, 40, 60))
    tensor = build_transform("training", image_size=32, resize_size=40)(image)
    assert tensor.shape == (3, 32, 32)


def test_normalization_is_an_explicit_sensitivity() -> None:
    image = Image.new("RGB", (20, 20), color=(128, 128, 128))
    plain = build_transform("validation", image_size=16, normalize=False)(image)
    normalized = build_transform("validation", image_size=16, normalize=True)(image)
    assert not torch.allclose(plain, normalized)
