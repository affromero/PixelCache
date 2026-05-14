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


def test_hashable_list_constructor_wraps_raw_dict() -> None:
    """`HashableList([{...}])` wraps nested raw dict at construction."""
    hl = HashableList([{"a": 1}])
    hash(hl)  # would raise pre-fix on an unhashable raw dict
    assert isinstance(hl[0], HashableDict)


def test_hashable_list_is_immutable() -> None:
    """`HashableList` has no mutation API — `insert`, `append`, etc.
    are not part of the `Sequence` protocol it implements.
    """
    hl = HashableList([1, 2, 3])
    assert not hasattr(hl, "insert")
    assert not hasattr(hl, "append")
    # `__setitem__` only exists at the class level if `MutableSequence`
    # is the base — `Sequence` doesn't define it.
    assert not hasattr(hl, "__setitem__")


# ----------------------------- HashableDict -----------------------------


def test_hashable_dict_constructor_wraps_raw_dict() -> None:
    """`HashableDict({"x": {"a": 1}})` wraps the nested raw dict."""
    hd = HashableDict({"x": {"a": 1}})
    hash(hd)
    assert isinstance(hd["x"], HashableDict)


def test_hashable_dict_is_immutable() -> None:
    """`HashableDict` has no mutation API — it implements the
    read-only `Mapping` protocol, not `MutableMapping`.
    """
    hd = HashableDict({"x": 1})
    assert not hasattr(hd, "__setitem__")
    assert not hasattr(hd, "__delitem__")


def test_hashable_dict_eq_handles_ndarray_values() -> None:
    """`HashableDict.__eq__` with ndarray values must not raise the
    'truth value of an array is ambiguous' error.
    """
    arr = np.array([1, 2, 3])
    a: HashableDict[str, object] = HashableDict({"k": arr})
    b: HashableDict[str, object] = HashableDict({"k": arr.copy()})
    assert a == b


def test_hashable_dict_eq_pil_compares_by_content() -> None:
    """PIL.Image's own __eq__ is identity. HashableDict equality must
    compare PIL values by mode + size + bytes content.
    """
    a: HashableDict[str, object] = HashableDict(
        {"img": Image.new("RGB", (8, 8), color=(10, 20, 30))}
    )
    b: HashableDict[str, object] = HashableDict(
        {"img": Image.new("RGB", (8, 8), color=(10, 20, 30))}
    )
    c: HashableDict[str, object] = HashableDict(
        {"img": Image.new("RGB", (8, 8), color=(99, 0, 0))}
    )
    assert a == b
    assert a != c


def test_hashable_dict_construction_isolates_from_source() -> None:
    """Mutating the source ndarray/tensor/PIL after construction must
    NOT change the HashableDict's cached hash.
    """
    arr = np.array([1, 2, 3])
    hd: HashableDict[str, object] = HashableDict({"k": arr})
    h1 = hash(hd)
    arr[0] = 99
    assert hash(hd) == h1
    assert hd["k"][0] != 99 if isinstance(hd["k"], np.ndarray) else True


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


def test_mask2bbox_returns_immutable_hashable_list() -> None:
    """`HashableImage.mask2bbox` previously built its result via
    `HashableList([]).append(...)`. After collections became immutable
    that broke at runtime. Verify it now returns a populated list.
    """
    arr = np.zeros((64, 64, 3), dtype=np.uint8)
    arr[10:30, 10:30] = 255
    arr[40:50, 40:50] = 255
    img = HashableImage(arr).to_binary(0.5)
    bboxes = img.mask2bbox(margin=0.0)
    assert len(bboxes) == 2


# ----------------------------- PIL mode normalization ------------------


def test_construct_from_pil_rgba_normalizes_to_rgb() -> None:
    """RGBA PIL images get normalized to RGB so `dtype()` honors its
    `Literal["L", "RGB", "1"]` contract and downstream hash/eq don't
    crash.
    """
    rgba = Image.new("RGBA", (32, 32), color=(100, 200, 50, 128))
    img = HashableImage(rgba)
    assert img.dtype() == "RGB"
    # And hash/eq are usable.
    h = hash(img)
    assert h == hash(HashableImage(rgba.copy()))


def test_construct_from_pil_palette_normalizes_to_rgb() -> None:
    """P (palette) PIL images get normalized to RGB."""
    p_img = Image.new("P", (32, 32), color=5)
    img = HashableImage(p_img)
    assert img.dtype() == "RGB"


def test_numpy_view_is_read_only_raises() -> None:
    """`numpy_view()` returns a writeable=False array; mutation raises."""
    img = HashableImage(np.zeros((16, 16, 3), dtype=np.uint8))
    view = img.numpy_view()
    with pytest.raises(ValueError, match="read-only|writeable"):
        view[0, 0, 0] = 99


# ----------------------------- Constructor isolation -------------------


def test_ctor_isolates_numpy_source_mutation() -> None:
    """Mutating the source ndarray AFTER construction must NOT change
    the HashableImage's cached hash. Constructor must `.copy()`.
    """
    arr = np.zeros((8, 8, 3), dtype=np.uint8)
    img = HashableImage(arr)
    h_before = hash(img)
    arr[0, 0, 0] = 99  # mutate the original
    assert hash(img) == h_before


def test_ctor_isolates_tensor_source_mutation() -> None:
    """Mutating the source tensor AFTER construction must NOT change
    the HashableImage's cached hash. Constructor must `.clone()`.
    """
    import torch

    src = torch.zeros(1, 3, 16, 16)
    img = HashableImage(src)
    h_before = hash(img)
    src.add_(1.0)  # in-place mutation of the source
    assert hash(img) == h_before


def test_ctor_isolates_pil_source_mutation() -> None:
    """Mutating the source PIL image AFTER construction must NOT
    change the HashableImage's cached hash. Constructor must
    `.copy()`.
    """
    src = Image.new("RGB", (16, 16), color=(10, 20, 30))
    img = HashableImage(src)
    h_before = hash(img)
    src.putpixel((0, 0), (99, 99, 99))
    assert hash(img) == h_before


# ----------------------------- raw() safety ----------------------------


def test_raw_returns_independent_copy() -> None:
    """`raw()` must not return the internal buffer for any storage
    mode. Mutating the result must not affect the HashableImage.
    """
    img = HashableImage(np.zeros((16, 16, 3), dtype=np.uint8))
    r = img.raw()
    r[0, 0, 0] = 99
    assert img.raw()[0, 0, 0] != 99
    assert img.numpy()[0, 0, 0] != 99


def test_raw_view_aliases_internal_buffer() -> None:
    """`raw_view()` returns the internal reference (caller acknowledges
    the mutation risk).
    """
    img = HashableImage(np.zeros((16, 16, 3), dtype=np.uint8))
    assert img.raw_view() is img._image


# ----------------------------- Bytes constructor isolation -------------


def test_bytes_ctor_eagerly_loads() -> None:
    """`HashableImage(bytes)` must eagerly decode and not retain a
    reference to the source `BytesIO` buffer. We exercise this by
    constructing from a bytes payload that we then mutate (we can't
    mutate `bytes` itself, so we verify equality stability across
    two independent reads of the same bytes).
    """
    import io as _io

    buf = _io.BytesIO()
    Image.new("RGB", (16, 16), color=(100, 200, 50)).save(buf, format="PNG")
    raw = buf.getvalue()
    a = HashableImage(raw)
    b = HashableImage(raw)
    assert a == b
    assert hash(a) == hash(b)


# ----------------------------- PIL eq mode/size guard ------------------


def test_get_filename_detects_overwritten_source() -> None:
    """If the source path is overwritten under us, `get_filename()`
    must materialize a fresh temp instead of returning a path whose
    bytes no longer match the image we hold in memory.
    """
    import os
    import tempfile

    fd, path = tempfile.mkstemp(suffix=".png")
    os.close(fd)
    Image.new("RGB", (16, 16), color=(50, 50, 50)).save(path)
    try:
        img = HashableImage(path)
        first = img.get_filename()
        assert first == path
        # Overwrite the source with different pixels.
        Image.new("RGB", (16, 16), color=(99, 99, 99)).save(path)
        second = img.get_filename()
        assert second != path
    finally:
        for p in (path, second):
            if os.path.exists(p):
                os.unlink(p)


def test_hashable_dict_getitem_returns_read_only_ndarray() -> None:
    """`hd[k]` for ndarray values must come back as a read-only view."""
    arr = np.array([1, 2, 3])
    hd: HashableDict[str, object] = HashableDict({"k": arr})
    got = hd["k"]
    assert isinstance(got, np.ndarray)
    with pytest.raises(ValueError, match="read-only|writeable"):
        got[0] = 99


def test_hashable_dict_to_dict_returns_read_only_ndarray() -> None:
    """`hd.to_dict()` must protect ndarray leaves the same way."""
    arr = np.array([1, 2, 3])
    hd: HashableDict[str, np.ndarray] = HashableDict({"k": arr})
    unpacked: dict[str, np.ndarray] = hd.to_dict()
    with pytest.raises(ValueError, match="read-only|writeable"):
        unpacked["k"][0] = 99


def test_read_image_supports_bmp() -> None:
    """`read_image` must accept non-JPEG/PNG formats via PIL fallback."""
    import os
    import tempfile

    fd, path = tempfile.mkstemp(suffix=".bmp")
    os.close(fd)
    try:
        Image.new("RGB", (16, 16), color=(50, 50, 50)).save(path)
        img = HashableImage(path)
        assert img.size().height == 16
        assert img.size().width == 16
    finally:
        if os.path.exists(path):
            os.unlink(path)


def test_bounding_box_eq_compares_fields_not_just_hash() -> None:
    """BoundingBox equality must compare all coordinates + image_size,
    not just rely on hash equality (which could collide).
    """
    from pixelcache import BoundingBox

    a = BoundingBox(
        xmin=0,
        ymin=0,
        xmax=10,
        ymax=10,
        image_size=ImageSize(height=20, width=20),
    )
    b = BoundingBox(
        xmin=0,
        ymin=0,
        xmax=10,
        ymax=10,
        image_size=ImageSize(height=20, width=20),
    )
    c = BoundingBox(
        xmin=0,
        ymin=0,
        xmax=10,
        ymax=11,
        image_size=ImageSize(height=20, width=20),
    )
    assert a == b
    assert a != c


def test_points_eq_compares_fields_not_just_hash() -> None:
    """Points equality must compare the points array, is_normalized,
    and image_size — not just hash.
    """
    pts_a = Points(
        points=np.array([[1.0, 2.0]]),
        is_normalized=False,
        image_size=ImageSize(height=10, width=10),
    )
    pts_b = Points(
        points=np.array([[1.0, 2.0]]),
        is_normalized=False,
        image_size=ImageSize(height=10, width=10),
    )
    pts_c = Points(
        points=np.array([[1.0, 3.0]]),
        is_normalized=False,
        image_size=ImageSize(height=10, width=10),
    )
    assert pts_a == pts_b
    assert pts_a != pts_c


def test_set_filename_refreshes_fingerprint(tmp_png_path: str) -> None:
    """`set_filename(path)` must update `_src_fingerprint` so the next
    `get_filename()` returns the same path instead of materializing a
    new temp.
    """
    arr = np.zeros((16, 16, 3), dtype=np.uint8)
    img = HashableImage(arr)
    # Save the image to a known path.
    img.save(tmp_png_path)
    img.set_filename(tmp_png_path)
    # First call should return the path we just set.
    assert img.get_filename() == tmp_png_path
    # Second call must also return the same path (no temp leak).
    assert img.get_filename() == tmp_png_path


def test_bgr2rgb_torch_actually_swaps_channels() -> None:
    """`bgr2rgb()` on a torch-mode image must swap channels 0 and 2,
    not return identity. Pre-fix the torch path indexed `[0, 1, 2]`.
    """
    import torch

    # Build a (1, 3, 1, 1) tensor with distinct channel values so we
    # can see the swap.
    src = torch.tensor([[[[10.0]], [[20.0]], [[30.0]]]])
    img = HashableImage(src)
    swapped = img.bgr2rgb()
    out = swapped.tensor()
    # Channel order should now be [30, 20, 10].
    assert out[0, 0, 0, 0].item() == 30
    assert out[0, 1, 0, 0].item() == 20
    assert out[0, 2, 0, 0].item() == 10


def test_pil_eq_compares_mode_size_bytes() -> None:
    """PIL equality must include explicit mode + size guards (not rely
    on hash non-collision for correctness).
    """
    a = HashableImage(Image.new("RGB", (16, 16), color=(50, 50, 50)))
    b = HashableImage(Image.new("RGB", (16, 16), color=(50, 50, 50)))
    c = HashableImage(Image.new("RGB", (16, 16), color=(60, 60, 60)))
    d = HashableImage(Image.new("RGB", (32, 8), color=(50, 50, 50)))
    assert a == b
    assert a != c
    assert a != d  # different size, same total pixels
