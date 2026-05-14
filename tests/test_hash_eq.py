"""Hash stability, cache idempotence, eq short-circuit."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np
    import torch

from pixelcache import HashableDict, HashableImage, HashableList


def test_same_content_same_hash(mid_rgb_np: np.ndarray) -> None:
    a = HashableImage(mid_rgb_np.copy())
    b = HashableImage(mid_rgb_np.copy())
    assert hash(a) == hash(b)
    assert a == b


def test_different_content_different_hash(mid_rgb_np: np.ndarray) -> None:
    a = HashableImage(mid_rgb_np.copy())
    other = mid_rgb_np.copy()
    other[0, 0, 0] ^= 0xFF
    b = HashableImage(other)
    assert hash(a) != hash(b)
    assert a != b


def test_different_shape_different_hash(mid_rgb_np: np.ndarray) -> None:
    a = HashableImage(mid_rgb_np.copy())
    # Trim to 128x128 to vary shape but keep dtype/mode.
    b = HashableImage(mid_rgb_np[:128, :128].copy())
    assert hash(a) != hash(b)


def test_hash_is_cached(mid_rgb_np: np.ndarray) -> None:
    img = HashableImage(mid_rgb_np.copy())
    assert img._cached_hash is None
    h1 = hash(img)
    assert img._cached_hash == h1
    h2 = hash(img)
    assert h1 == h2


def test_eq_identity_shortcut(mid_rgb_np: np.ndarray) -> None:
    img = HashableImage(mid_rgb_np.copy())
    assert img == img  # same object  # noqa: PLR0124


def test_eq_returns_not_implemented_for_non_hashable_image(
    mid_rgb_np: np.ndarray,
) -> None:
    img = HashableImage(mid_rgb_np.copy())
    # eq with a foreign type should be False (Python's fallback when
    # NotImplemented is returned on both sides).
    assert img != 42
    assert img != "not an image"


def test_torch_mode_hash_works(mid_rgb_tensor: torch.Tensor) -> None:
    img = HashableImage(mid_rgb_tensor)
    h1 = hash(img)
    h2 = hash(img)
    assert h1 == h2


def test_hashable_dict_hash_cache(mid_rgb_np: np.ndarray) -> None:
    img = HashableImage(mid_rgb_np.copy())
    hd = HashableDict({"a": img, "b": 42})
    assert hd._cached_hash is None
    h1 = hash(hd)
    assert hd._cached_hash == h1
    assert hash(hd) == h1


def test_hashable_dict_mutation_invalidates(mid_rgb_np: np.ndarray) -> None:
    img = HashableImage(mid_rgb_np.copy())
    hd = HashableDict({"a": img, "b": 42})
    h1 = hash(hd)
    hd["c"] = "new"
    h2 = hash(hd)
    assert h1 != h2


def test_hashable_list_hash_cache(mid_rgb_np: np.ndarray) -> None:
    img = HashableImage(mid_rgb_np.copy())
    hl = HashableList([img, img.to_gray()])
    h1 = hash(hl)
    assert hl._cached_hash == h1
    assert hash(hl) == h1


def test_hashable_list_mutation_invalidates(mid_rgb_np: np.ndarray) -> None:
    img = HashableImage(mid_rgb_np.copy())
    hl = HashableList([img])
    h1 = hash(hl)
    hl.append(img.to_gray())
    h2 = hash(hl)
    assert h1 != h2
