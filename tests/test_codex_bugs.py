"""Regression tests for bugs surfaced by `codex exec review` against main.

Each test pins down a specific defect the adversarial review caught.
If any of these regress, a future change failed silently in a way
that's easy to introduce again.
"""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from pixelcache import (
    HashableDict,
    HashableImage,
    HashableList,
    ImageCrop,
    ImageSize,
    Points,
)

# ----------------------------- HashableList -----------------------------


def test_hashable_list_order_matters() -> None:
    """[1, 2] and [2, 1] must not hash or compare equal."""
    a = HashableList([1, 2])
    b = HashableList([2, 1])
    assert hash(a) != hash(b)
    assert a != b


def test_hashable_list_multiplicity_matters() -> None:
    """[1, 2] and [1, 1, 2] must not hash or compare equal."""
    a = HashableList([1, 2])
    b = HashableList([1, 1, 2])
    assert hash(a) != hash(b)
    assert a != b


def test_hashable_list_insert_wraps_raw_dict() -> None:
    """`HashableList.insert(0, {})` must wrap so the next hash() works."""
    hl: HashableList[object] = HashableList([])
    hl.insert(0, {"a": 1})
    # Without wrapping, hash() would raise on the unhashable raw dict.
    hash(hl)
    item = hl[0]
    assert isinstance(item, HashableDict)


# ----------------------------- HashableDict -----------------------------


def test_hashable_dict_setitem_wraps_raw_dict() -> None:
    """`hd['x'] = {'a': 1}` must wrap so the next hash() works."""
    hd: HashableDict[str, object] = HashableDict({})
    hd["x"] = {"a": 1}
    hash(hd)
    assert isinstance(hd["x"], HashableDict)


def test_hashable_dict_eq_handles_ndarray_values() -> None:
    """`HashableDict.__eq__` with ndarray values must not raise the
    'truth value of an array is ambiguous' error.
    """
    arr = np.array([1, 2, 3])
    a: HashableDict[str, object] = HashableDict({"k": arr})
    b: HashableDict[str, object] = HashableDict({"k": arr.copy()})
    assert a == b


# ----------------------------- ImageCrop --------------------------------


def test_image_crop_unnormalized_dimensions() -> None:
    """Unnormalized `ImageCrop(left=10, top=20, right=50, bottom=50)`
    must produce a (height=30, width=40) result, not bottom/right.
    """
    img = HashableImage(np.zeros((128, 256, 3), dtype=np.uint8))
    ic = ImageCrop(left=10, top=20, right=50, bottom=50)
    cropped = ic(img)
    assert cropped.size() == ImageSize(height=30, width=40)


def test_image_crop_normalized_dimensions() -> None:
    """Normalized crop on a 128x256 image must scale h/w independently."""
    img = HashableImage(np.zeros((128, 256, 3), dtype=np.uint8))
    ic = ImageCrop(left=0.5, top=0.25, right=1.0, bottom=0.75)
    cropped = ic(img)
    # top=0.25*128=32 → 32, bottom=0.75*128=96 → height=64
    # left=0.5*256=128, right=1.0*256=256 → width=128
    assert cropped.size() == ImageSize(height=64, width=128)


# ----------------------------- Points -----------------------------------


def test_points_xy_non_square_image() -> None:
    """Normalized (0.5, 0.25) on a 100h x 200w image must be (100, 25)
    pixel space — x along width, y along height.
    """
    pts = Points(
        points=np.array([[0.5, 0.25]]),
        is_normalized=True,
        image_size=ImageSize(height=100, width=200),
    )
    xy = pts.xy
    assert xy[0, 0] == 100.0
    assert xy[0, 1] == 25.0


def test_points_xyn_non_square_image() -> None:
    """Pixel (100, 25) on a 100h x 200w image must normalize to (0.5, 0.25)."""
    pts = Points(
        points=np.array([[100.0, 25.0]]),
        is_normalized=False,
        image_size=ImageSize(height=100, width=200),
    )
    xyn = pts.xyn
    assert xyn[0, 0] == 0.5
    assert xyn[0, 1] == 0.25


# ----------------------------- HashableImage (PIL) ----------------------


def test_is_rgb_on_pil_image() -> None:
    """`is_rgb()` must not crash on PIL images (no `.shape` attribute)."""
    img_rgb = HashableImage(Image.new("RGB", (16, 16), color=(255, 0, 0)))
    img_gray = HashableImage(Image.new("L", (16, 16), color=128))
    assert img_rgb.is_rgb() is True
    assert img_gray.is_rgb() is False


def test_hash_and_eq_on_pil_image() -> None:
    """Hashing and equality on PIL-mode images must work end-to-end."""
    a = HashableImage(Image.new("RGB", (16, 16), color=(255, 0, 0)))
    b = HashableImage(Image.new("RGB", (16, 16), color=(255, 0, 0)))
    c = HashableImage(Image.new("RGB", (16, 16), color=(0, 255, 0)))
    assert hash(a) == hash(b)
    assert a == b
    assert hash(a) != hash(c)


def test_shape_on_pil_grayscale() -> None:
    """PIL L-mode must report (h, w) shape, not crash."""
    img = HashableImage(Image.new("L", (32, 16), color=128))
    assert img.shape == (16, 32)


# ----------------------------- __setitem__ ------------------------------


def test_setitem_does_not_mutate_torch_tensor_in_place() -> None:
    """`HashableImage.__setitem__` (masked assign) must not mutate the
    source tensor or invalidate the parent HashableImage's hash.
    """
    import torch

    src = torch.zeros(1, 3, 16, 16)
    img = HashableImage(src)
    original_hash = hash(img)
    mask_arr = np.zeros((16, 16), dtype=bool)
    mask_arr[0:4, 0:4] = True
    mask = HashableImage(mask_arr)
    new_img = img.__setitem__(mask, 0.5)
    # Source tensor and source HashableImage's hash should be unchanged.
    assert torch.equal(src, torch.zeros(1, 3, 16, 16))
    assert hash(img) == original_hash
    # New image differs.
    assert hash(new_img) != original_hash


# ----------------------------- numpy_view read-only --------------------


def test_numpy_view_is_read_only_raises() -> None:
    """`numpy_view()` returns a writeable=False array; mutation raises."""
    img = HashableImage(np.zeros((16, 16, 3), dtype=np.uint8))
    view = img.numpy_view()
    with pytest.raises(ValueError, match="read-only|writeable"):
        view[0, 0, 0] = 99
