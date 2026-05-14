"""numpy() copy-by-default + numpy_view() aliasing semantics."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np

from pixelcache import HashableImage


def test_numpy_returns_copy(mid_rgb_np: np.ndarray) -> None:
    img = HashableImage(mid_rgb_np.copy())
    out = img.numpy()
    out[0, 0, 0] = 0
    # Mutating the returned array must NOT mutate the HashableImage.
    assert img.numpy()[0, 0, 0] != 0 or mid_rgb_np[0, 0, 0] == 0


def test_numpy_view_aliases(mid_rgb_np: np.ndarray) -> None:
    img = HashableImage(mid_rgb_np.copy())
    view = img.numpy_view()
    original_value = view[0, 0, 0]
    new_value = (original_value + 1) % 256
    view[0, 0, 0] = new_value
    # numpy_view aliases the internal buffer — change propagates.
    assert img.numpy_view()[0, 0, 0] == new_value
