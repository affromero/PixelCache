# PixelCache

<img src="pixelcache/assets/pixel_cache.png" width="100" height="100"/>

**Sometimes you do not care whether the image processing is done using NumPy or PyTorch or Pillow, and transfering between these libraries can be cumbersome. PixelCache provides a simple interface to perform image processing and transformation using these libraries, allowing you to focus on the task at hand.**

PixelCache is a Python library designed for versatile image processing and transformation, integrating the power of Pillow, NumPy, and PyTorch while supporting LRU caching.

PixelCache also supports LRU caching, which can be useful when you need to reuse the results of previously computed operations. This can be particularly helpful when working with large images or when you need to apply the same operation multiple times.

## Features

- **Image Manipulation:** Transform and process images using a simple interface that supports Pillow, NumPy, and PyTorch.
- **Smart Caching:** Utilize LRU caching to enhance performance by reusing previously computed results. It can be easily disable for whatever reason using the environment variable `DISABLE_LRU_CACHE=True`.
- **Versatile:** PixelCache supports a wide range of image processing operations, including resizing, cropping, blending, transform to RGB/Binary, flipping, bounding box extraction from binary masks, several operations suitable for binary masks, and more.

## Installation

You can install PixelCache using pip:

```bash
pip install pixelcache
```

Or poetry:

```bash
poetry add pixelcache
```

Or from source:

```bash
poetry add git+ssh://git@github.com:affromero/pixelcache.git
```

## Basic Usage

HashableImage receives as input several types of inputs, such as URLs, paths, or Pillow images, Numpy arrays, or PyTorch tensors in the following form or shape:

### Valdid inputs:

- Paths: `str` | `Path`
- Pillow: `Image.Image`
- Numpy Arrays: `UInt8[np.ndarray, "h w 3"]` | `UInt8[np.ndarray, "h w"]` | `Bool[np.ndarray, "h w"]`
- Torch Tensors: `Float[torch.Tensor, "1 c h w"]` | `Bool[torch.Tensor, "1 1 h w"]`

Example:

```python
from pixelcache import HashableImage
import torch
image = HashableImage(torch.rand(1, 3, 256, 256).float())
image_pil = image.pil()
image_numpy_bool = image.to_binary(0.5).numpy()
```

### Basic Transformations:

At all times, it is possible to transform between them using the following methods:

- `pil()`: Returns a Pillow Image object
- `numpy()`: Returns a Numpy Array object
- `tensor()`: Returns a PyTorch Tensor object

Additionally, there is a method to convert the image to 3 channels RGB or binary:

- `to_rgb()`: Convert to RGB, which returns another HashableImage
- `to_binary(threshold: float)`: Convert to Boolean, which returns another HashableImage

Finally, there is a method to save the image to disk:

- `save(path: str | Path)`: Save the image to disk

## Usage Example 1

Blending two images using PixelCache:

```python
from pathlib import Path

from pixelcache import HashableDict, HashableImage, HashableList
from pixelcache.tools.logger import get_logger

logger = get_logger()

image0 = "https://images.pexels.com/photos/28811907/pexels-photo-28811907/free-photo-of-majestic-elk-standing-in-forest-clearing.jpeg"
image1 = Path("pixelcache") / "assets" / "pixel_cache.png"
images_hash = [HashableImage(image) for image in [image0, image1]]
for image in images_hash:
    logger.info(f"Image: {image} - Hash: {hash(image)}")
logger.info(f"Hash for list of images: {hash(HashableList(images_hash))}")
image_size = images_hash[1].size()
logger.info(f"Image size: {image_size} - Resizing all to this size")
resized_images = [image.resize(image_size) for image in images_hash]
# blend images
blended_image = resized_images[0].blend(
    resized_images[1], alpha=0.5, with_bbox=False
)
# binarize second image
blended_image_binarized = resized_images[0].blend(
    resized_images[1].to_binary(0.5).invert_binary(),
    alpha=0.2,
    with_bbox=True,
)
output_debug = HashableDict(
    {
        "image base": HashableList([resized_images[0]]),
        "image reference": HashableList([resized_images[1]]),
        "blended_image": HashableList([blended_image]),
        "blended_image_binarized": HashableList([blended_image_binarized]),
    }
)
output = image1.parent / (str(image1.stem) + "_demo_blend.jpg")
HashableImage.make_image_grid(
    output_debug, orientation="horizontal", with_text=True
).save(output)
logger.success(f"Output saved to: {output}")

````

![Output](pixelcache/assets/pixel_cache_demo_blend.jpg)

## Usage Example 2

Extracting bounding boxes for cropping / unpadding from binary masks using PixelCache:

```python
from pathlib import Path

from pixelcache import HashableDict, HashableImage, HashableList, ImageSize
from pixelcache.tools.logger import get_logger

logger = get_logger()

image0 = "https://images.pexels.com/photos/18624700/pexels-photo-18624700/free-photo-of-a-vintage-typewriter.jpeg"
image1 = Path("pixelcache") / "assets" / "pixel_cache.png"
images_hash = [HashableImage(image) for image in [image0, image1]]
for image in images_hash:
    logger.info(f"Image: {image} - Hash: {hash(image)}")
logger.info(f"Hash for list of images: {hash(HashableList(images_hash))}")
image_size = images_hash[1].size()
logger.info(f"Image size: {image_size} - Resizing all to this size")
resized_images = [image.resize(image_size) for image in images_hash]
# crop images
increased_size_pad = ImageSize(
    width=image_size.width + 1000, height=image_size.height + 1000
)
mask = (
    images_hash[1]
    .center_pad(increased_size_pad, fill=255)
    .resize(image_size)
    .to_space_color("HSV", getchannel="S")
    .to_binary(0.3)
)
cropped = resized_images[1].crop_from_mask(mask)
uncropped = cropped.uncrop_from_bbox(
    base=resized_images[0], bboxes=mask.mask2bbox(margin=0.0), resize=True
)
output_debug = HashableDict(
    {
        "image base": HashableList([resized_images[0]]),
        "image reference": HashableList([resized_images[1]]),
        "cropped_image": HashableList([cropped.resize(image_size)]),
        "uncropped_image": HashableList([uncropped]),
    }
)
output = image1.parent / (str(image1.stem) + "_demo_cropUncrop.jpg")
HashableImage.make_image_grid(
    output_debug, orientation="horizontal", with_text=True
).save(output)
logger.success(f"Output saved to: {output}")

```

![Output](pixelcache/assets/pixel_cache_demo_cropUncrop.jpg)

### Both examples can be found in the `examples` folder.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request.

## License

This project is licensed under the MIT License
