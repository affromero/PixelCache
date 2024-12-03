from importlib.metadata import version

from pixelcache.main import (
    MAX_IMG_CACHE,
    BoundingBox,
    HashableDict,
    HashableImage,
    HashableList,
    ImageCrop,
    Points,
)
from pixelcache.tools.image import ImageSize

__all__ = [
    "MAX_IMG_CACHE",
    "BoundingBox",
    "HashableDict",
    "HashableImage",
    "HashableList",
    "ImageCrop",
    "ImageSize",
    "Points",
]

__version__ = version("pixelcache")
