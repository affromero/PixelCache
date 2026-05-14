"""Behavioral tests for resize / to_rgb / to_gray / to_binary / crop."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np

from pixelcache import HashableImage, ImageSize


def test_resize_changes_size(mid_rgb_np: np.ndarray) -> None:
    img = HashableImage(mid_rgb_np.copy())
    resized = img.resize(ImageSize(height=128, width=128))
    assert resized.size() == ImageSize(height=128, width=128)


def test_resize_same_size_returns_same_object(
    mid_rgb_np: np.ndarray,
) -> None:
    img = HashableImage(mid_rgb_np.copy())
    same = img.resize(img.size())
    assert same is img


def test_to_gray_l_mode(mid_rgb_np: np.ndarray) -> None:
    img = HashableImage(mid_rgb_np.copy())
    gray = img.to_gray()
    assert gray.dtype() == "L"
    # Grayscale shape should be (h, w) — not raise as it used to for L mode.
    assert gray.shape == (mid_rgb_np.shape[0], mid_rgb_np.shape[1])


def test_to_binary(mid_rgb_np: np.ndarray) -> None:
    img = HashableImage(mid_rgb_np.copy())
    bin_img = img.to_binary(0.5)
    assert bin_img.is_binary()
    assert bin_img.dtype() == "1"


def test_to_rgb_idempotent_on_rgb(mid_rgb_np: np.ndarray) -> None:
    img = HashableImage(mid_rgb_np.copy())
    rgb = img.to_rgb()
    assert rgb.is_rgb()


def test_to_rgb_promotes_grayscale(mid_rgb_np: np.ndarray) -> None:
    gray = HashableImage(mid_rgb_np.copy()).to_gray()
    rgb = gray.to_rgb()
    assert rgb.is_rgb()
    assert rgb.size() == gray.size()


def test_downsample(mid_rgb_np: np.ndarray) -> None:
    img = HashableImage(mid_rgb_np.copy())
    down = img.downsample(2)
    expected = ImageSize(
        height=mid_rgb_np.shape[0] // 2,
        width=mid_rgb_np.shape[1] // 2,
    )
    assert down.size() == expected


def test_save_then_load_round_trip(
    mid_rgb_np: np.ndarray,
    tmp_png_path: str,
) -> None:
    src = HashableImage(mid_rgb_np.copy())
    src.save(tmp_png_path)
    loaded = HashableImage(tmp_png_path)
    # Round-trip preserves shape (PNG is lossless for uint8).
    assert loaded.size() == src.size()
    # Values match exactly through PNG.
    import numpy as np

    assert np.array_equal(loaded.numpy(), src.numpy())
