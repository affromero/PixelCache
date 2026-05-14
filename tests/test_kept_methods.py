"""Regression tests for methods kept after the Phase 3 cull.

`morphologyEx` and `logical_or_reduce` have live Hax-CV callers; this
file pins down their happy-path behavior so future cleanups can't
silently break them.
"""

from __future__ import annotations

import numpy as np

from pixelcache import HashableImage


def test_morphology_ex_dilate(mid_rgb_np: np.ndarray) -> None:
    mask = HashableImage(mid_rgb_np.copy()).to_binary(0.5)
    kernel = np.ones((3, 3), dtype=np.float32)
    dilated = mask.morphologyEx("dilate", kernel)
    # Dilation can only add pixels — sum should be >= original.
    assert dilated.numpy().sum() >= mask.numpy().sum()


def test_logical_or_reduce_idempotent(mid_rgb_np: np.ndarray) -> None:
    a = HashableImage(mid_rgb_np.copy()).to_binary(0.5)
    out = a.logical_or_reduce([])
    assert out == a


def test_logical_or_reduce_union(mid_rgb_np: np.ndarray) -> None:
    a = HashableImage(mid_rgb_np.copy()).to_binary(0.3)
    b = HashableImage(mid_rgb_np.copy()).to_binary(0.7)
    merged = a.logical_or_reduce([b])
    # Union is a superset of either operand.
    assert merged.numpy().sum() >= a.numpy().sum()
    assert merged.numpy().sum() >= b.numpy().sum()
