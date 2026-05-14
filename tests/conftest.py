"""Shared fixtures for the pixelcache test suite.

Provides deterministic uint8 arrays in three sizes (tiny / mid /
large) and their tensor / PIL / on-disk equivalents. Session-scoped
where the data is immutable.

Types use `jaxtyping` shape/dtype annotations to match the rest of
the project — no raw `np.ndarray` / `torch.Tensor` in signatures.
"""

from __future__ import annotations

import os
import tempfile
from typing import TYPE_CHECKING

import numpy as np
import pytest
import torch
from PIL import Image

if TYPE_CHECKING:
    from collections.abc import Iterator

    from jaxtyping import Float, UInt8


@pytest.fixture(scope="session")
def tiny_rgb_np() -> UInt8[np.ndarray, "16 16 3"]:
    """16x16x3 uint8 deterministic RGB image."""
    rng = np.random.default_rng(seed=0)
    return rng.integers(0, 256, (16, 16, 3), dtype=np.uint8)


@pytest.fixture(scope="session")
def mid_rgb_np() -> UInt8[np.ndarray, "256 256 3"]:
    """256x256x3 uint8 deterministic RGB image."""
    rng = np.random.default_rng(seed=1)
    return rng.integers(0, 256, (256, 256, 3), dtype=np.uint8)


@pytest.fixture(scope="session")
def large_rgb_np() -> UInt8[np.ndarray, "2048 2048 3"]:
    """2048x2048x3 uint8 deterministic RGB image (~12.6MB).

    Used for perf benchmarks. Lazy-allocated and shared across tests.
    """
    rng = np.random.default_rng(seed=2)
    return rng.integers(0, 256, (2048, 2048, 3), dtype=np.uint8)


@pytest.fixture(scope="session")
def mid_rgb_pil(
    mid_rgb_np: UInt8[np.ndarray, "256 256 3"],
) -> Image.Image:
    """`mid_rgb_np` as a PIL Image."""
    return Image.fromarray(mid_rgb_np)


@pytest.fixture(scope="session")
def mid_rgb_tensor(
    mid_rgb_np: UInt8[np.ndarray, "256 256 3"],
) -> Float[torch.Tensor, "1 3 256 256"]:
    """`mid_rgb_np` as a `1 3 256 256` float tensor in `[0, 1]`."""
    arr = mid_rgb_np.transpose(2, 0, 1)
    return torch.from_numpy(arr).unsqueeze(0).float() / 255.0


@pytest.fixture(scope="session")
def mid_rgb_path(
    mid_rgb_np: UInt8[np.ndarray, "256 256 3"],
) -> Iterator[str]:
    """`mid_rgb_np` saved to a temp PNG. Cleaned up after the session."""
    fd, path = tempfile.mkstemp(suffix=".png", prefix="pixelcache_test_")
    os.close(fd)
    Image.fromarray(mid_rgb_np).save(path)
    yield path
    if os.path.exists(path):
        os.unlink(path)


@pytest.fixture(scope="session")
def large_rgb_jpg(
    large_rgb_np: UInt8[np.ndarray, "2048 2048 3"],
) -> Iterator[str]:
    """`large_rgb_np` saved as a JPEG (quality 95) for read-image perf tests."""
    fd, path = tempfile.mkstemp(suffix=".jpg", prefix="pixelcache_test_")
    os.close(fd)
    Image.fromarray(large_rgb_np).save(path, quality=95)
    yield path
    if os.path.exists(path):
        os.unlink(path)


@pytest.fixture
def tmp_png_path() -> Iterator[str]:
    """Fresh empty `.png` temp path per test. The test owns writing it."""
    fd, path = tempfile.mkstemp(suffix=".png", prefix="pixelcache_test_")
    os.close(fd)
    os.unlink(path)  # caller writes
    yield path
    if os.path.exists(path):
        os.unlink(path)
