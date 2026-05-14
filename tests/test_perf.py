"""Performance regression budgets for the hot-path operations.

These assertions pin the perf gains from the refactor. If a future
change regresses them they'll fail loudly. Budgets are tuned for
local-dev hardware with substantial headroom — they should be safe
even in CI containers.

Run with: `pytest tests/test_perf.py -v`
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np

from pixelcache import HashableImage


def test_ctor_from_numpy_is_disk_free(large_rgb_np: np.ndarray) -> None:
    """HashableImage(numpy_array) must NOT touch disk.

    Pre-refactor each ctor wrote a PNG to disk (~5-20ms each). After
    Phase 1 commit `4907f15`, the source path is `None` and no file
    is created until `get_filename()` is called.
    """
    img = HashableImage(large_rgb_np)
    assert img._image_str is None


def test_ctor_from_numpy_under_1ms(large_rgb_np: np.ndarray) -> None:
    """100 ctors of a 4MP uint8 array should average under 1ms each."""
    iterations = 100
    t0 = time.perf_counter()
    for _ in range(iterations):
        _ = HashableImage(large_rgb_np)
    elapsed = time.perf_counter() - t0
    per_ctor_ms = (elapsed / iterations) * 1000
    # Hard ceiling — pre-refactor this was ~10ms/ctor (disk-bound).
    assert per_ctor_ms < 1.0, (
        f"ctor regressed to {per_ctor_ms:.2f}ms/call "
        "(budget: <1ms, pre-refactor: ~10ms)"
    )


def test_cached_hash_under_5us(large_rgb_np: np.ndarray) -> None:
    """Repeated hash() of an already-hashed instance must be O(1)."""
    img = HashableImage(large_rgb_np)
    hash(img)  # warm cache
    iterations = 10_000
    t0 = time.perf_counter()
    for _ in range(iterations):
        hash(img)
    elapsed = time.perf_counter() - t0
    per_hash_us = (elapsed / iterations) * 1e6
    # Hard ceiling: cached hash should be <1µs but the assertion budget
    # is loose (5µs) to absorb CI scheduling jitter.
    assert per_hash_us < 5.0, (
        f"cached hash regressed to {per_hash_us:.2f}µs/call "
        "(budget: <5µs, observed in dev: ~0.4µs)"
    )


def test_read_image_jpeg_under_500ms(large_rgb_jpg: str) -> None:
    """4MP JPEG single decode via torchvision fast path.

    The fixture is random noise (high JPEG entropy → larger file →
    slower decode). The 500ms budget is a sanity ceiling for "didn't
    get massively slower" — not a tight perf target. Pre-refactor
    double-decode would have been ~2x this. For tight JPEG perf, a
    follow-up could swap to `pyturbojpeg` or `nvjpeg`.
    """
    from pixelcache.tools.image import read_image

    # Warm up torchvision: first decode pays JIT + module-load cost.
    _ = read_image(large_rgb_jpg)
    iterations = 3
    t0 = time.perf_counter()
    for _ in range(iterations):
        _ = read_image(large_rgb_jpg)
    elapsed = time.perf_counter() - t0
    per_read_ms = (elapsed / iterations) * 1000
    assert (
        per_read_ms < 500.0
    ), f"read_image regressed to {per_read_ms:.1f}ms/call (budget: <500ms)"
