"""numpy() copy-by-default + numpy_view() aliasing semantics."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np
    from jaxtyping import UInt8

from pixelcache import HashableImage


def test_numpy_returns_copy(
    mid_rgb_np: UInt8[np.ndarray, "256 256 3"],
) -> None:
    img = HashableImage(mid_rgb_np.copy())
    out = img.numpy()
    out[0, 0, 0] = 0
    # Mutating the returned array must NOT mutate the HashableImage.
    assert img.numpy()[0, 0, 0] != 0 or mid_rgb_np[0, 0, 0] == 0


def test_numpy_view_is_read_only(
    mid_rgb_np: UInt8[np.ndarray, "256 256 3"],
) -> None:
    """`numpy_view()` returns an array with `writeable=False` so the
    HashableImage's cached hash can't be invalidated by accidental
    in-place mutation through the zero-copy escape hatch.
    """
    import pytest

    img = HashableImage(mid_rgb_np.copy())
    view = img.numpy_view()
    assert view.flags.writeable is False
    with pytest.raises(ValueError, match="read-only|writeable"):
        view[0, 0, 0] = 0


def test_tensor_returns_independent_clone(
    mid_rgb_np: UInt8[np.ndarray, "256 256 3"],
) -> None:
    """`tensor()` clones in torch mode so in-place ops on the result
    don't leak into the HashableImage's internal tensor.
    """
    import torch

    src = torch.from_numpy(mid_rgb_np.transpose(2, 0, 1)).unsqueeze(0)
    src = src.float() / 255.0
    img = HashableImage(src)
    t = img.tensor()
    t.zero_()
    assert img.tensor().abs().sum().item() > 0


def test_pil_returns_independent_copy(
    mid_rgb_np: UInt8[np.ndarray, "256 256 3"],
) -> None:
    """`pil()` returns a fresh PIL image in pil mode so mutating it
    doesn't leak into the HashableImage's internal PIL object.
    """
    from PIL import Image

    src = Image.fromarray(mid_rgb_np)
    img = HashableImage(src)
    p = img.pil()
    p.putpixel((0, 0), (99, 99, 99))
    assert img.pil().getpixel((0, 0)) != (99, 99, 99)
