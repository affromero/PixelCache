"""Regenerate the visual examples used in `README.md`.

Produces two PNGs under ``pixelcache/assets/``:

* ``transformations.png`` — color / threshold / palette / geometric ops
  arranged as a 4x2 grid driven entirely by ``HashableImage`` methods.
* ``mask_workflow.png`` — original / mask / blend / crop strip showing the
  mask-driven workflow advertised in the README.

Run from the repo root::

    uv run python -m pixelcache.examples.visuals
"""

from __future__ import annotations

import numpy as np
from difflogtest import get_logger

from pixelcache import HashableImage, ImageSize

logger = get_logger()

ASSETS = "pixelcache/assets"
SOURCE = f"{ASSETS}/pixel_cache.png"


def _transformations_grid(out_path: str) -> None:
    src = HashableImage(SOURCE).resize(ImageSize(height=192, width=192))

    row1 = {
        "original": [src],
        "to_gray()": [src.to_gray().to_rgb()],
        "to_binary(0.5)": [src.to_gray().to_binary(0.5).to_rgb()],
        "invert_binary()": [
            src.to_gray().to_binary(0.5).invert_binary().to_rgb()
        ],
    }
    row2 = {
        "apply_palette('viridis')": [src.to_gray().apply_palette("viridis")],
        "equalize_hist()": [src.equalize_hist().to_rgb()],
        "rotate(45)": [
            src.rotate(45.0).resize(ImageSize(height=192, width=192))
        ],
        "downsample(2)": [
            src.downsample(2).resize(ImageSize(height=192, width=192))
        ],
    }
    top = HashableImage.make_image_grid(
        row1, orientation="vertical", with_text=True
    ).numpy()
    bot = HashableImage.make_image_grid(
        row2, orientation="vertical", with_text=True
    ).numpy()

    width = max(top.shape[1], bot.shape[1])

    def _center_pad(a: np.ndarray) -> np.ndarray:
        if a.shape[1] == width:
            return a
        gap = width - a.shape[1]
        left = gap // 2
        right = gap - left
        return np.pad(a, ((0, 0), (left, right), (0, 0)))

    stacked = np.vstack([_center_pad(top), _center_pad(bot)])
    HashableImage(stacked).save(out_path)
    logger.success(f"Wrote {out_path}")


def _mask_workflow(out_path: str) -> None:
    photo = HashableImage(SOURCE).resize(ImageSize(height=256, width=256))
    mask_np = np.zeros((256, 256), dtype=np.uint8)
    mask_np[64:192, 64:192] = 255
    mask = HashableImage(mask_np).to_binary(0.5)

    blended = photo.blend(mask.to_rgb(), alpha=0.45, with_bbox=False)
    cropped = photo.crop_from_mask(mask).resize(
        ImageSize(height=256, width=256)
    )

    HashableImage.make_image_grid(
        {
            "1. original": [photo],
            "2. mask": [mask.to_rgb()],
            "3. blend(0.45)": [blended],
            "4. crop_from_mask": [cropped],
        },
        orientation="vertical",
        with_text=True,
    ).save(out_path)
    logger.success(f"Wrote {out_path}")


def main() -> None:
    """Regenerate every visual asset referenced from `README.md`."""
    _transformations_grid(f"{ASSETS}/transformations.png")
    _mask_workflow(f"{ASSETS}/mask_workflow.png")


if __name__ == "__main__":
    main()
