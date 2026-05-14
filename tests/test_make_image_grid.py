"""Regression tests for `HashableImage.make_image_grid`.

Pinpoint the immutable-collection input path: the function must accept a
`HashableDict[str, HashableList[HashableImage]]`, preserve insertion order
(the grid layout depends on it), and never mutate the caller's input.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pixelcache import (
    HashableDict,
    HashableImage,
    HashableList,
    ImageSize,
)

if TYPE_CHECKING:
    import numpy as np
    from jaxtyping import UInt8


def _cell(
    color: tuple[int, int, int],
    base: UInt8[np.ndarray, "16 16 3"],
) -> HashableImage:
    arr = base.copy()
    arr[:] = color
    return HashableImage(arr)


def test_make_image_grid_accepts_hashable_dict(
    tiny_rgb_np: UInt8[np.ndarray, "16 16 3"],
) -> None:
    """Grid accepts an immutable `HashableDict` input without raising."""
    images = HashableDict(
        {
            "zzz": HashableList([_cell((255, 0, 0), tiny_rgb_np)]),
            "aaa": HashableList([_cell((0, 255, 0), tiny_rgb_np)]),
            "mmm": HashableList([_cell((0, 0, 255), tiny_rgb_np)]),
        }
    )
    grid = HashableImage.make_image_grid(
        images, orientation="vertical", with_text=False
    )
    cell = ImageSize(height=16, width=16)
    assert grid.size().height >= cell.height
    assert grid.size().width >= cell.width * 3


def test_make_image_grid_preserves_key_order(
    tiny_rgb_np: UInt8[np.ndarray, "16 16 3"],
) -> None:
    """Non-alphabetical key order must survive into the laid-out grid.

    For ``orientation="vertical"`` the dict-iteration order becomes the
    left-to-right column order; we read the centre column of each input
    cell's column band and assert the colours appear in the input order.
    """
    red = _cell((255, 0, 0), tiny_rgb_np)
    green = _cell((0, 255, 0), tiny_rgb_np)
    blue = _cell((0, 0, 255), tiny_rgb_np)

    images = HashableDict(
        {
            "zzz": HashableList([red]),
            "aaa": HashableList([green]),
            "mmm": HashableList([blue]),
        }
    )
    grid = HashableImage.make_image_grid(
        images, orientation="vertical", with_text=False
    )
    grid_np = grid.numpy()
    # 3 cells of width 16 → centre columns at x=8, 24, 40
    centres = [grid_np[8, x] for x in (8, 24, 40)]
    assert tuple(centres[0]) == (255, 0, 0)
    assert tuple(centres[1]) == (0, 255, 0)
    assert tuple(centres[2]) == (0, 0, 255)


def test_make_image_grid_does_not_mutate_input(
    tiny_rgb_np: UInt8[np.ndarray, "16 16 3"],
) -> None:
    """The immutable input must be unchanged after the call.

    Stable hashes for the input ``HashableDict`` and inner ``HashableList``
    before/after the call prove no leaf, key, or ordering changed.
    """
    images = HashableDict(
        {
            "first": HashableList([_cell((10, 20, 30), tiny_rgb_np)]),
            "second": HashableList(
                [
                    _cell((40, 50, 60), tiny_rgb_np),
                    _cell((70, 80, 90), tiny_rgb_np),
                ]
            ),
        }
    )
    before_dict_hash = hash(images)
    before_list_hashes = [hash(images[k]) for k in images]
    before_keys = list(images.keys())

    HashableImage.make_image_grid(
        images, orientation="vertical", with_text=False
    )

    assert hash(images) == before_dict_hash
    assert [hash(images[k]) for k in images] == before_list_hashes
    assert list(images.keys()) == before_keys


def test_make_image_grid_pads_unequal_list_lengths(
    tiny_rgb_np: UInt8[np.ndarray, "16 16 3"],
) -> None:
    """Lists of differing lengths must pad with zeros_like without raising."""
    images = HashableDict(
        {
            "one": HashableList([_cell((255, 0, 0), tiny_rgb_np)]),
            "two_three": HashableList(
                [
                    _cell((0, 255, 0), tiny_rgb_np),
                    _cell((0, 0, 255), tiny_rgb_np),
                ]
            ),
        }
    )
    grid = HashableImage.make_image_grid(
        images, orientation="vertical", with_text=False
    )
    # 2 columns x 2 rows of 16x16 cells -> at least 32x32
    assert grid.size().height >= 32
    assert grid.size().width >= 32
