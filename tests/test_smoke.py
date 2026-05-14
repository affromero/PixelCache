"""Smoke tests: every public symbol is importable and constructible."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np
    import torch
    from PIL import Image


def test_imports_all_public_names() -> None:
    import pixelcache

    expected = (
        "BoundingBox",
        "HashableDict",
        "HashableImage",
        "HashableList",
        "ImageCrop",
        "ImageSize",
        "Points",
        "display_string",
        "get_logger",
        "pseudo_hash",
        "seed_everything",
    )
    for name in expected:
        assert hasattr(pixelcache, name), f"missing public symbol: {name}"


def test_construct_from_numpy(
    mid_rgb_np: UInt8[np.ndarray, "256 256 3"],
) -> None:
    from pixelcache import HashableImage

    img = HashableImage(mid_rgb_np.copy())
    # Lazy ctor must not materialize a temp file.
    assert img._image_str is None
    assert img.mode == "numpy"


def test_construct_from_pil(mid_rgb_pil: Image.Image) -> None:
    from pixelcache import HashableImage

    img = HashableImage(mid_rgb_pil)
    assert img._image_str is None
    assert img.mode == "pil"


def test_construct_from_tensor(
    mid_rgb_tensor: Float[torch.Tensor, "1 3 256 256"],
) -> None:
    from pixelcache import HashableImage

    img = HashableImage(mid_rgb_tensor)
    assert img._image_str is None
    assert img.mode == "torch"


def test_construct_from_path(mid_rgb_path: str) -> None:
    from pixelcache import HashableImage

    img = HashableImage(mid_rgb_path)
    # Path inputs DO record the source string.
    assert img._image_str == mid_rgb_path


def test_pixelcache_main_module_is_gone() -> None:
    import importlib

    import pytest

    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("pixelcache.main")
