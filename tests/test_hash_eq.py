"""Hash stability, cache idempotence, eq short-circuit."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np
    import torch
    from jaxtyping import Float, UInt8

from pixelcache import HashableDict, HashableImage, HashableList


def test_same_content_same_hash(
    mid_rgb_np: UInt8[np.ndarray, "256 256 3"],
) -> None:
    a = HashableImage(mid_rgb_np.copy())
    b = HashableImage(mid_rgb_np.copy())
    assert hash(a) == hash(b)
    assert a == b


def test_different_content_different_hash(
    mid_rgb_np: UInt8[np.ndarray, "256 256 3"],
) -> None:
    a = HashableImage(mid_rgb_np.copy())
    other = mid_rgb_np.copy()
    other[0, 0, 0] ^= 0xFF
    b = HashableImage(other)
    assert hash(a) != hash(b)
    assert a != b


def test_different_shape_different_hash(
    mid_rgb_np: UInt8[np.ndarray, "256 256 3"],
) -> None:
    a = HashableImage(mid_rgb_np.copy())
    # Trim to 128x128 to vary shape but keep dtype/mode.
    b = HashableImage(mid_rgb_np[:128, :128].copy())
    assert hash(a) != hash(b)


def test_hash_is_cached(mid_rgb_np: UInt8[np.ndarray, "256 256 3"]) -> None:
    img = HashableImage(mid_rgb_np.copy())
    assert img._cached_hash is None
    h1 = hash(img)
    assert img._cached_hash == h1
    h2 = hash(img)
    assert h1 == h2


def test_eq_identity_shortcut(
    mid_rgb_np: UInt8[np.ndarray, "256 256 3"],
) -> None:
    img = HashableImage(mid_rgb_np.copy())
    # `img == img` triggers PLR0124; call __eq__ directly to exercise
    # the identity short-circuit without disabling the lint.
    assert img.__eq__(img) is True


def test_eq_returns_not_implemented_for_non_hashable_image(
    mid_rgb_np: UInt8[np.ndarray, "256 256 3"],
) -> None:
    img = HashableImage(mid_rgb_np.copy())
    # eq with a foreign type should be False (Python's fallback when
    # NotImplemented is returned on both sides).
    assert img != 42
    assert img != "not an image"


def test_torch_mode_hash_works(
    mid_rgb_tensor: Float[torch.Tensor, "1 3 256 256"],
) -> None:
    img = HashableImage(mid_rgb_tensor)
    h1 = hash(img)
    h2 = hash(img)
    assert h1 == h2


def test_hashable_dict_hash_cache(
    mid_rgb_np: UInt8[np.ndarray, "256 256 3"],
) -> None:
    img = HashableImage(mid_rgb_np.copy())
    hd = HashableDict({"a": img, "b": 42})
    assert hd._cached_hash is None
    h1 = hash(hd)
    assert hd._cached_hash == h1
    assert hash(hd) == h1


def test_hashable_dict_constructed_with_extra_key_differs(
    mid_rgb_np: UInt8[np.ndarray, "256 256 3"],
) -> None:
    """HashableDict is immutable, so 'add a key' = build a new dict.
    The new dict must hash to a different value.
    """
    img = HashableImage(mid_rgb_np.copy())
    hd = HashableDict({"a": img, "b": 42})
    merged = HashableDict({**hd.to_dict(), "c": "new"})
    assert hash(hd) != hash(merged)


def test_hashable_list_hash_cache(
    mid_rgb_np: UInt8[np.ndarray, "256 256 3"],
) -> None:
    img = HashableImage(mid_rgb_np.copy())
    hl = HashableList([img, img.to_gray()])
    h1 = hash(hl)
    assert hl._cached_hash == h1
    assert hash(hl) == h1


def test_hashable_list_constructed_with_extra_item_differs(
    mid_rgb_np: UInt8[np.ndarray, "256 256 3"],
) -> None:
    """HashableList is immutable, so 'append' = build a new list.
    The new list must hash to a different value.
    """
    img = HashableImage(mid_rgb_np.copy())
    hl = HashableList([img])
    extended = HashableList([*hl.to_list(), img.to_gray()])
    assert hash(hl) != hash(extended)
