import base64
import io
import random
import string
import tempfile
from copy import deepcopy
from numbers import Number
from pathlib import Path
from typing import (
    Any,
    Literal,
    ParamSpec,
    TypeVar,
    cast,
)

import cv2
import numpy as np
import torch
import xxhash
from beartype import beartype
from difflogtest import get_logger
from difflogtest.utils.path import path_basename, path_stat
from jaxtyping import Bool, Float, UInt8
from matplotlib import colormaps
from PIL import Image, ImageOps
from torchvision.transforms import functional as TF

from pixelcache._collections import HashableList
from pixelcache._types import BoundingBox, Points
from pixelcache.data.palette import color_palette
from pixelcache.tools.bbox import crop_from_bbox, uncrop_from_bbox
from pixelcache.tools.cache import jaxtyped
from pixelcache.tools.image import (
    ImageSize,
    center_pad,
    make_image_grid,
    numpy2tensor,
    pil2tensor,
    read_image,
    save_image,
    tensor2numpy,
    tensor2pil,
    to_binary,
)
from pixelcache.tools.mask import (
    crop_from_mask,
    mask2bbox,
    mask2squaremask,
    mask_blend,
    morphologyEx,
    remove_small_regions,
)
from pixelcache.tools.text import create_text

PCropArgs = ParamSpec("PCropArgs")
PSquareMaskArgs = ParamSpec("PSquareMaskArgs")
P_PathStr = TypeVar("P_PathStr", bound=Path | str)

VALID_IMAGES = Literal["pil", "numpy", "torch"]
PALETTE_DEFAULT = color_palette()

logger = get_logger()


@jaxtyped(typechecker=beartype)
def pseudo_hash(idx: int, length: int = 6) -> str:
    """Generate a pseudo-random hash based on the given index and length.

    Args:
        idx (int): The index used to seed the random number generator.
        length (int, optional): The length of the hash to be generated.
            Defaults to 6.

    Returns:
        str: A string representing the pseudo-random hash generated based on
            the given index and length.

    """
    random.seed(idx)
    return "".join(random.choice(string.ascii_letters) for _ in range(length))  # noqa: S311


# PIL modes pixelcache stores natively: RGB (3-channel uint8), L
# (grayscale uint8), 1 (binary). Anything else (RGBA, P, CMYK, YCbCr,
# LAB, etc.) is normalized to one of these on construction.
_PIL_NATIVE_MODES = frozenset({"RGB", "L", "1"})


def _path_fingerprint(path: str) -> tuple[int, int] | None:
    """Return `(mtime_ns, size)` for `path`, or `None` if it doesn't exist.

    Used by `HashableImage.get_filename()` to detect when a path
    constructed from a local file has been overwritten since
    construction. If the fingerprint changed, the cached path no
    longer represents this HashableImage's pixels and we must
    materialize a fresh temp file from `self._image` instead.
    """
    try:
        st = path_stat(path)
    except OSError:
        return None
    return (st.st_mtime_ns, st.st_size)


def _normalize_pil_mode(img: Image.Image) -> Image.Image:
    """Convert PIL images outside the supported mode set to RGB.

    Pixelcache's `dtype()` claims a `Literal["L", "RGB", "1"]` return
    type — beartype/jaxtyping enforce that at runtime. PIL inputs in
    other modes (RGBA from transparent PNGs, P from palette images,
    etc.) violate the contract and crash downstream `hash()` / `eq` /
    `dtype()` calls. Normalize on the way in so the contract holds.

    RGBA → RGB drops alpha (the most common case — transparent PNGs);
    callers needing alpha should handle it explicitly before passing
    in.
    """
    if img.mode in _PIL_NATIVE_MODES:
        return img
    return img.convert("RGB")


class HashableImage:
    """Hashable image class."""

    @jaxtyped(typechecker=beartype)
    def __init__(
        self,
        image: (
            str
            | Path
            | bytes
            | Image.Image
            | UInt8[np.ndarray, "h w 3"]
            | UInt8[np.ndarray, "h w"]
            | Bool[np.ndarray, "h w"]
            | Float[torch.Tensor, "1 c h w"]
            | Bool[torch.Tensor, "1 1 h w"]
        ),
    ) -> None:
        """Initialize a HashableImage from a path, URL, bytes, PIL, numpy, or tensor.

        Instances are immutable: `_image` is set only here and never mutated.
        Construction does not touch disk; a temp file is materialized lazily
        on the first `get_filename()` call.

        Args:
            image: Source data. `str` / `Path` is loaded via `read_image`
                (file path or HTTP URL). `bytes` is decoded as image data.
                `Image.Image`, `np.ndarray`, and `torch.Tensor` are stored
                in-memory without writing to disk.

        """
        # Deep-copy mutable inputs so external mutation of the source
        # can't silently invalidate the cached hash. `read_image` and
        # `Image.open` already return fresh objects, so they don't
        # need an extra copy.
        if isinstance(image, torch.Tensor):
            self._image = image.detach().cpu().clone()
        elif isinstance(image, str | Path):
            try:
                self._image = read_image(image)
            except Exception as e:
                msg = f"Error reading image {image}: {e}"
                raise RuntimeError(msg) from e
        elif isinstance(image, Image.Image):
            self._image = _normalize_pil_mode(image).copy()
        elif isinstance(image, bytes):
            # `Image.open` on BytesIO is lazy — the PIL object keeps a
            # reference to the buffer. `.copy()` materializes the pixel
            # data eagerly so the buffer can be GC'd and no later
            # mutation of `image` (if it were aliased somewhere) can
            # corrupt the cached hash.
            self._image = _normalize_pil_mode(
                Image.open(io.BytesIO(image)),
            ).copy()
        elif isinstance(image, np.ndarray):
            self._image = image.copy()
        else:
            self._image = image

        # Source string: a real path/URL if we were constructed from one,
        # otherwise None until get_filename() materializes a temp file.
        self._image_str: str | None = (
            str(image) if isinstance(image, str | Path) else None
        )
        # Source fingerprint at construction time (mtime_ns, size).
        # Used by `get_filename` to detect "source file changed under us"
        # and re-materialize a temp from `self._image` instead of
        # handing back a stale path that no longer matches our pixels.
        # Only meaningful for local-file construction.
        self._src_fingerprint: tuple[int, int] | None = (
            _path_fingerprint(self._image_str)
            if self._image_str is not None
            and not self._image_str.startswith(("http://", "https://"))
            else None
        )
        # Lazy content-fingerprint cache for __hash__. Safe because _image
        # is set only in __init__ and never mutated.
        self._cached_hash: int | None = None

    def _create_tmp_file(self) -> str:
        """Materialize this image to a fresh temp PNG and cache the path.

        Updates both `_image_str` and `_src_fingerprint` so subsequent
        `get_filename()` calls hit the cached path instead of writing
        a new temp every time. Without the fingerprint refresh, an
        in-memory HashableImage (whose `_src_fingerprint` starts at
        `None`) would leak a new file on every call.
        """
        self._image_str = tempfile.NamedTemporaryFile(
            prefix="pixelcache_", suffix=".png", delete=False
        ).name
        self.save(self._image_str)
        self._src_fingerprint = _path_fingerprint(self._image_str)
        return self._image_str

    @staticmethod
    def from_base64(base64_str: str) -> "HashableImage":
        """Create a HashableImage from a base64 string."""
        return HashableImage(base64.b64decode(base64_str))

    @staticmethod
    def decode_bytes(
        data: bytes,
        *,
        unchanged: bool = False,
    ) -> UInt8[np.ndarray, "h w 3"] | UInt8[np.ndarray, "h w"]:
        """Decode image bytes to a uint8 numpy array without temp-file I/O.

        For high-throughput use cases (e.g. sensor frame loops)
        where raw JPEG/PNG bytes need efficient decoding without
        creating a HashableImage instance or temp files.

        Args:
            data: Raw image bytes (JPEG, PNG, etc.).
            unchanged: If True, preserve original channel layout
                (single-channel grayscale shape `h w`). If False,
                decode as BGR then convert to RGB → shape `h w 3`.

        Returns:
            Decoded image as a `uint8` numpy array — `h w 3` for RGB,
            `h w` for grayscale (`unchanged=True`).

        Raises:
            ValueError: If decoding fails.

        """
        buf = np.frombuffer(data, dtype=np.uint8)
        flag = cv2.IMREAD_UNCHANGED if unchanged else cv2.IMREAD_COLOR
        img = cv2.imdecode(buf, flag)
        if img is None:
            msg = "Failed to decode image bytes"
            raise ValueError(msg)
        if not unchanged and img.ndim == 3 and img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    @property
    def _mode(self) -> VALID_IMAGES:
        if isinstance(self._image, torch.Tensor):
            return "torch"
        if isinstance(self._image, np.ndarray):
            return "numpy"
        if isinstance(self._image, Image.Image):
            return "pil"
        msg = "Invalid image type"
        raise ValueError(msg)

    def get_filename(self) -> str:
        """Return a local filename whose bytes match this image's pixels.

        Returns the source path if the HashableImage was constructed
        from one AND that path still has the same `(mtime_ns, size)`
        fingerprint we captured at construction. If the source file
        was overwritten, deleted, or replaced under us — or if
        construction was from an in-memory source / URL / temp that's
        since been GC'd — materialize a fresh temp file from
        `self._image` and cache its path.

        This is the v0.1.0 fidelity contract: the path we return is
        guaranteed to decode to the same image we hold in memory.
        """
        if (
            self._image_str is not None
            and self._src_fingerprint is not None
            and _path_fingerprint(self._image_str) == self._src_fingerprint
        ):
            return self._image_str
        # Either we were constructed from a non-path source, or the
        # path has changed under us. Materialize a fresh temp.
        return self._create_tmp_file()

    def get_local_filename(self) -> str:
        """Retrieve the local filename of the HashableImage object.

        If the original filename starts with 'http', this method saves the
            image to
        a temporary file and returns the path of the temporary file.
            Otherwise, it
        simply returns the original filename.

        Returns:
            str: The local filename of the HashableImage object.

        """
        _filename = self.get_filename()
        if _filename.startswith("http"):
            return self._create_tmp_file()
        return _filename

    def set_filename(self, filename: str) -> None:
        """Set the canonical local filename for this HashableImage.

        Used when an existing image was just written to a known path
        (e.g. by an inpainting pass) and we want subsequent
        `get_filename()` calls to return that path instead of
        re-materializing a new temp.

        Refreshes `_src_fingerprint` to match the new path so the next
        `get_filename()` recognises the file as authoritative and
        doesn't trigger the "source changed under us" temp-materialize
        branch.
        """
        self._image_str = filename
        self._src_fingerprint = _path_fingerprint(filename)

    def save(
        self,
        path: Path | str,
        transparency: Literal["white", "black"] | None = None,
    ) -> None:
        """Save the image represented by the HashableImage object to a.

            specified file path.

        This method uses the image data stored in the HashableImage object
            and writes it to a file at the given path. The image format is
            determined by the file extension in the path.

        Args:
            path (str): The file path where the image will be saved. This
                should include the filename and the extension.

        Returns:
            None: This method doesn't return any value. It writes the image
                data to a file.

        """
        if transparency is not None:
            image = self.to_rgb().pil()
            image_np = self.to_rgb().numpy()
            if transparency == "white":
                mask = (image_np != 255).all(axis=-1)
            elif transparency == "black":
                mask = (image_np != 0).all(axis=-1)
            else:
                msg = f"Invalid transparency: {transparency}"
                raise ValueError(msg)
            # convert to rgba
            image_rgba = Image.new("RGBA", image.size)
            image_rgba.paste(image, mask=Image.fromarray(mask))
            image_rgba.save(path)
        else:
            save_image(self._image, path=str(path), normalize=False)

    def show(self) -> None:
        """Display the image represented by the HashableImage object.

        This method displays the image data stored in the HashableImage
            object.

        Args:
            self (HashableImage): The HashableImage object to be displayed.

        Returns:
            None: This method doesn't return any value. It displays the image
                data.

        """
        self.pil().show()

    def downsample(self, factor: int) -> "HashableImage":
        """Downsample the given image by a specified factor.

        Args:
            factor (int): The factor by which the image should be
                downsampled. This must be an integer greater than 0.

        Returns:
            HashableImage: A new HashableImage object that is a downscaled
                version of the original image.

        """
        new_size = ImageSize(
            height=round(self.size().height // factor),
            width=round(self.size().width // factor),
        )
        return self.resize(new_size)

    @jaxtyped(typechecker=beartype)
    def resize(
        self,
        size: ImageSize,
        mode: Literal["bilinear", "lanczos", "nearest"] = "bilinear",
    ) -> "HashableImage":
        """Resize the image to a specified size using different interpolation.

            methods based on the image mode.

        Args:
            size (ImageSize): An object containing the desired image height
                and width.
            mode (Literal["bilinear", "lanczos", "nearest"], optional): The
                interpolation method to use. Defaults to "bilinear".

        Returns:
            HashableImage: A new HashableImage object with the resized image
                if the size is different from the current image size.
                Otherwise, it returns the original HashableImage object.

        """
        is_binary = self.is_binary()
        image = self.to_rgb()._image
        height = int(size.height)
        width = int(size.width)
        if size != self.size():
            if self._mode == "torch":
                _kwargs: dict[str, Any] = {}
                if mode == "nearest":
                    _kwargs["mode"] = "nearest-exact"
                else:
                    _kwargs["mode"] = mode
                if mode == "bilinear":
                    _kwargs["align_corners"] = False
                _image = torch.nn.functional.interpolate(
                    image,
                    size=(height, width),
                    **_kwargs,
                )
            elif self._mode == "pil":
                if mode == "nearest":
                    _mode = Image.Resampling.NEAREST
                elif mode == "bilinear":
                    _mode = Image.Resampling.BILINEAR
                elif mode == "lanczos":
                    _mode = Image.Resampling.LANCZOS
                else:
                    msg = f"Invalid mode: {mode}"
                    raise ValueError(msg)
                _image = image.resize((width, height), _mode)
            else:
                if mode == "nearest":
                    _mode = cv2.INTER_NEAREST
                elif mode == "bilinear":
                    _mode = cv2.INTER_LINEAR
                elif mode == "lanczos":
                    _mode = cv2.INTER_LANCZOS4
                else:
                    msg = f"Invalid mode: {mode}"
                    raise ValueError(msg)
                _image = cv2.resize(
                    cast("np.ndarray", image),
                    (width, height),
                    interpolation=_mode,
                )
            output = HashableImage(_image)
            if is_binary:
                output = output.to_binary()
            return output
        return self

    @jaxtyped(typechecker=beartype)
    def resize_min_size(
        self, min_size: int, modulo: int = 16
    ) -> "HashableImage":
        """Resize the image to a specified minimum size.

        This method resizes the image to the specified minimum size while
            maintaining the aspect ratio.

        Args:
            self (HashableImage): The HashableImage object to be resized.
            min_size (int): The minimum size to which the image should be
                resized.
            modulo (int, optional): The value to which the image dimensions
                should be divisible. Defaults to 16.

        Returns:
            HashableImage: A new HashableImage object with the resized image
                based on the minimum size.

        """
        image_size = self.size()
        height = image_size.height
        width = image_size.width
        if height < width:
            new_h = min_size
            new_w = round(width * (new_h / height))
        else:
            new_w = min_size
            new_h = round(height * (new_w / width))
        new_h = new_h - (new_h % modulo)
        new_w = new_w - (new_w % modulo)
        return self.resize(ImageSize(height=new_h, width=new_w))

    def rotate(
        self,
        rotation: float,
        mode: TF.InterpolationMode = TF.InterpolationMode.BILINEAR,
        *,
        expand: bool = True,
    ) -> "HashableImage":
        """Rotate the image by a given angle."""
        image_pt = self.tensor()
        image_pt = TF.rotate(
            image_pt, rotation, interpolation=mode, expand=expand
        )
        return HashableImage(image_pt)

    @jaxtyped(typechecker=beartype)
    def to_gray(self) -> "HashableImage":
        """Converts the current image to grayscale.

        This method does not take any arguments. It processes the current
            image object and returns a new HashableImage object that
            represents the grayscale version of the original image.

        Returns:
            HashableImage: A new image object that is the grayscale version
                of the original image.

        """
        image = self.to_rgb()
        if self._mode == "torch":
            if image._image.shape[1] == 3:
                return HashableImage(
                    image._image.mean(1, keepdim=True).float(),
                )
            return self
        if self._mode == "numpy":
            if len(image._image.shape) == 3 and image._image.shape[2] == 3:
                return HashableImage(
                    cv2.cvtColor(image._image, cv2.COLOR_RGB2GRAY),
                )
            return self
        return HashableImage(image._image.convert("L"))

    @jaxtyped(typechecker=beartype)
    def apply_palette(
        self, _palette: UInt8[np.ndarray, "256 3"] | str = PALETTE_DEFAULT, /
    ) -> "HashableImage":
        """Apply a color palette to the HashableImage object.

        This method applies a color palette to the HashableImage object.

        Args:
            self (HashableImage): The HashableImage object to which the
                color palette will be applied.
            _palette (np.ndarray, optional): The color palette to be applied
                to the HashableImage object. Defaults to PALETTE_DEFAULT.
                Can be a string, in which case it will be converted to a
                matplotlib colormap.

        Returns:
            HashableImage: A new HashableImage object with the color palette
                applied.

        """
        rgb = self.to_rgb().numpy()
        # make sure all three channels are the same
        if not np.all(rgb[:, :, 0] == rgb[:, :, 1]) or not np.all(
            rgb[:, :, 0] == rgb[:, :, 2]
        ):
            msg = "To apply a palette, the image must be grayscale."
            raise ValueError(msg)
        # apply the palette
        image_np = rgb[:, :, 0]
        # replace the values with the palette
        unique_values = np.unique(image_np)
        new_image = np.zeros_like(rgb)
        if isinstance(_palette, str) and _palette in colormaps:
            _palette = colormaps.get_cmap(_palette)(range(256))[:, :3]
            _palette = (_palette * 255).astype(np.uint8)
        elif isinstance(_palette, str):
            msg = f"Invalid palette: {_palette}. Valid colormaps are: {list(colormaps.keys())}"
            raise ValueError(msg)
        for value in unique_values:
            new_image[image_np == value] = _palette[value]
        return HashableImage(new_image)

    @jaxtyped(typechecker=beartype)
    def to_rgb(self) -> "HashableImage":
        """Convert an image to RGB format.

        This method transforms the current mode of a HashableImage object to
            an RGB format.

        Args:
            self ('HashableImage'): The HashableImage object to be converted
                to RGB.

        Returns:
            HashableImage: The HashableImage object converted to RGB format.

        """
        if self._mode == "torch":
            if self._image.shape[1] == 1:
                return HashableImage(self._image.repeat(1, 3, 1, 1).float())
            return self
        if self._mode == "numpy":
            if len(self._image.shape) == 2:
                if self._image.dtype == bool:
                    return HashableImage(
                        cv2.cvtColor(
                            (self._image * 255).astype(np.uint8),
                            cv2.COLOR_GRAY2RGB,
                        ),
                    )
                return HashableImage(
                    cv2.cvtColor(self._image, cv2.COLOR_GRAY2RGB),
                )
            return self
        return HashableImage(self._image.convert("RGB"))

    @jaxtyped(typechecker=beartype)
    def to_binary(
        self,
        threshold: float = 0.0,
        area_min: float = 0,
        connectivity: int = 8,
    ) -> "HashableImage":
        """Convert an image to binary format.

        Args:
            threshold (float): The threshold for converting the image to binary.
            area_min (float): The minimum area for removing disconnected
                regions. Area is in percentage of the image area.

        Returns:
            HashableImage: A HashableImage object representing the converted
                image in binary format.

        """
        # check if it is bool already
        if (
            (self._mode == "torch" and self._image.dtype == torch.bool)
            or (self._mode == "numpy" and self._image.dtype == bool)
            or (self._mode == "pil" and self._image.mode == "1")
        ):
            return self
        mask = to_binary(self.numpy(), threshold=threshold)
        if area_min > 0:
            mask = remove_small_regions(
                mask, area_min, mode="holes", connectivity=connectivity
            )[0]
            mask = remove_small_regions(
                mask, area_min, mode="islands", connectivity=connectivity
            )[0]
        return HashableImage(mask)

    @jaxtyped(typechecker=beartype)
    def unique_values(self) -> tuple[list[float], torch.Tensor, list[int]]:
        """Get the unique values in the image.

        This method does not take any arguments. It processes the image data
            stored in the HashableImage object and returns the unique values
            in the image.

        Returns:
            tuple: A tuple containing the unique values in the image, the
                indices of the unique values, and the count of each unique
                value.

        """
        output: tuple[torch.Tensor, torch.Tensor, torch.Tensor] = (
            self.tensor().unique(return_counts=True, return_inverse=True, sorted=True)
        )
        _unique = output[0].tolist()
        _indices = output[1]
        _count = output[2].tolist()
        return _unique, _indices, _count

    @jaxtyped(typechecker=beartype)
    def invert_binary(self) -> "HashableImage":
        """Invert the binary representation of the image data in a.

            HashableImage object.

        This method checks the mode of the image data and returns a new
            HashableImage object
        with the inverted binary data.

        Args:
            self (HashableImage): The HashableImage object on which the
                method is called.

        Returns:
            HashableImage: A new HashableImage object with the inverted
                binary data based on
            the mode of the original image data.

        """
        if self._mode == "torch":
            return HashableImage(~self.to_binary().tensor())
        return HashableImage(~self.to_binary().numpy())

    @jaxtyped(typechecker=beartype)
    def invert_rgb(self) -> "HashableImage":
        """Invert the RGB values of the HashableImage object.

        This method checks the mode of the HashableImage object and performs
            the inversion accordingly.

        Args:
            self ('HashableImage'): The HashableImage object on which the
                method is called.

        Returns:
            'HashableImage': A new HashableImage object with inverted RGB
                values. If the mode of the image is 'torch', it returns the
                inverted tensor values. If the mode is not 'torch', it
                returns the inverted numpy values.

        """
        if self._mode == "torch":
            return HashableImage(1 - self.tensor())
        return HashableImage(255 - self.numpy())

    @staticmethod
    @jaxtyped(typechecker=beartype)
    def zeros_from_size(size: ImageSize) -> "HashableImage":
        """Create a HashableImage object with all elements initialized to zero.

        This static method generates a HashableImage object of the specified
            size with all pixel values set to zero.

        Args:
            size (ImageSize): An object representing the height and width of
                the image in pixels.

        Returns:
            HashableImage: A HashableImage object with all pixel values
                initialized to zero. The size of the image is determined by
                the input argument.

        """
        return HashableImage(
            torch.zeros((1, 3, int(size.height), int(size.width))),
        )

    @jaxtyped(typechecker=beartype)
    def zeros_like(self) -> "HashableImage":
        """Create a new HashableImage object with all elements set to zero.

        This method generates a new HashableImage object, with the same
            shape and type as the original image, but with all its elements
            set to zero.

        Args:
            self ('HashableImage'): The HashableImage object calling the
                method.

        Returns:
            'HashableImage': A new HashableImage object with all elements
                set to zero, maintaining the shape and type of the original
                image.

        """
        if self._mode == "torch":
            return HashableImage(torch.zeros_like(self._image))
        if self._mode == "numpy":
            return HashableImage(np.zeros_like(self._image))
        return HashableImage(
            Image.new(self._image.mode, self._image.size, 0),
        )

    @jaxtyped(typechecker=beartype)
    def ones_like(self) -> "HashableImage":
        """Create a new HashableImage object filled with ones.

        This method generates a new HashableImage object, maintaining the
            dimensions of the original image,
        but replacing all pixel values with ones.

        Args:
            self ('HashableImage'): The HashableImage object on which the
                ones_like method is called.

        Returns:
            'HashableImage': A new HashableImage object with the same
                dimensions as the original image but filled with ones.

        """
        if self._mode == "torch":
            return HashableImage(torch.ones_like(self._image))
        if self._mode == "numpy":
            return HashableImage(np.ones_like(self._image))
        return HashableImage(
            Image.new(
                self._image.mode,
                self._image.size,
                255 if self._image.mode != "RGB" else (255, 255, 255),
            ),
        )

    @jaxtyped(typechecker=beartype)
    def rgb2bgr(self) -> "HashableImage":
        """Convert the image from RGB to BGR color space in a HashableImage.

            object.

        This method takes a HashableImage object that contains an image in
            RGB color space and converts it to BGR color space.

        Args:
            self (HashableImage): The HashableImage object that contains the
                image to be converted.

        Returns:
            HashableImage: A new HashableImage object with the image
                converted to BGR color space.

        """
        if self._mode == "numpy":
            return HashableImage(cv2.cvtColor(self._image, cv2.COLOR_RGB2BGR))
        if self._mode == "pil":
            return HashableImage(
                Image.fromarray(
                    cv2.cvtColor(np.asarray(self._image), cv2.COLOR_RGB2BGR),
                ),
            )
        return HashableImage(self._image[:, [2, 1, 0], :, :])

    @jaxtyped(typechecker=beartype)
    def bgr2rgb(self) -> "HashableImage":
        """Convert the image from BGR to RGB color space.

        For torch-mode images the channel axis is dim=1
        (`1 c h w`); swap channels 0 and 2 to flip B↔R. The pre-fix
        version indexed `[0, 1, 2]` which is a no-op (identity).
        """
        if self._mode == "numpy":
            return HashableImage(cv2.cvtColor(self._image, cv2.COLOR_BGR2RGB))
        if self._mode == "pil":
            return HashableImage(
                Image.fromarray(
                    cv2.cvtColor(np.asarray(self._image), cv2.COLOR_BGR2RGB),
                ),
            )
        return HashableImage(self._image[:, [2, 1, 0], :, :])

    @jaxtyped(typechecker=beartype)
    def equalize_hist(self) -> "HashableImage":
        """Equalizes the histogram of the image stored in the HashableImage.

            object.

        This method adjusts the intensity values of the image to improve
            contrast and enhance details.

        Args:
            self (HashableImage): The HashableImage object containing the
                image to be processed.

        Returns:
            HashableImage: A new HashableImage object with the histogram
                equalized image.

        """
        if self._mode == "pil":
            return HashableImage(ImageOps.equalize(self._image))
        return HashableImage(cv2.equalizeHist(self.to_gray().numpy()))

    @jaxtyped(typechecker=beartype)
    def __add__(self, other: object) -> "HashableImage":
        """Add a HashableImage object to another HashableImage or Number.

            object.

        This method takes a HashableImage object and another object (either
            a HashableImage or a Number)
        and returns a new HashableImage object that results from the
            addition of the two input objects.

        Args:
            self (HashableImage): The HashableImage object to be added.
            other (HashableImage | Number): The other object (either a
                HashableImage or a Number) to be added to the HashableImage
                object.

        Returns:
            HashableImage: A new HashableImage object that is the result of
                adding the two input objects.

        """
        if not isinstance(other, HashableImage | Number):
            return NotImplemented
        if self._mode == "torch":
            other_value = (
                other if isinstance(other, Number) else other.tensor()
            )
            return HashableImage((self.tensor() + other_value).clamp(0, 1))
        other_value = other if isinstance(other, Number) else other.numpy()
        return HashableImage((self.numpy() + other_value).clip(0, 255))

    @jaxtyped(typechecker=beartype)
    def __sub__(self, other: object) -> "HashableImage":
        """Subtract pixel values of a HashableImage object or a number from.

            this HashableImage object.

        This method takes either another HashableImage object or a number as
            an argument. If it's another HashableImage object,
        it subtracts the pixel values of the second image from the pixel
            values of the first image. If it's a number, it subtracts
        this number from every pixel value of the first image.

        Args:
            self (HashableImage): The HashableImage object from which the
                pixel values are subtracted.
            other (Union[HashableImage, Number]): The object to subtract
                from the HashableImage object. It can be either another
            HashableImage object or a number.

        Returns:
            HashableImage: A new HashableImage object with pixel values
                subtracted based on the type of 'other' object.

        """
        if not isinstance(other, HashableImage | Number):
            return NotImplemented
        if self._mode == "torch":
            other_value = (
                other if isinstance(other, Number) else other.tensor()
            )
            return HashableImage((self.tensor() - other_value).clamp(0, 1))
        other_value = other if isinstance(other, Number) else other.numpy()
        return HashableImage((self.numpy() - other_value).clip(0, 255))

    @jaxtyped(typechecker=beartype)
    def __mul__(self, other: object) -> "HashableImage":
        """Performs element-wise multiplication between two HashableImage.

            objects or a HashableImage object and a Number.

        This method multiplies the pixel data of the HashableImage object on
            which it is called with the pixel data of another HashableImage
            object or a Number. The multiplication is performed element-
            wise, and a new HashableImage object is returned with the
            resulting pixel data.

        Args:
            self (HashableImage): The HashableImage object on which the
                method is called.
            other (HashableImage | Number): The object to be multiplied with
                the HashableImage object. It can be another HashableImage
                object or a Number.

        Returns:
            HashableImage: A new HashableImage object containing the result
                of the element-wise multiplication of the two input objects.

        """
        if not isinstance(other, HashableImage | Number):
            return NotImplemented
        if self._mode == "torch":
            self_value = self.tensor()
            other_value = (
                other if isinstance(other, Number) else other.tensor()
            )
            is_bool = self_value.dtype == torch.bool
            output = (self_value * other_value).clamp(0, 1)
            return HashableImage(output.bool() if is_bool else output.float())
        other_value_np: Number | np.ndarray = (
            other if isinstance(other, Number) else other.numpy()
        )
        # in case self is hxwx3 and other hxw, then broadcast
        # helpful for multiplication with binary masks
        self_value = self.numpy()
        is_bool = self_value.dtype == bool
        if (
            isinstance(other_value_np, np.ndarray)
            and len(self_value.shape) == 3
            and len(other_value_np.shape) == 2
        ):
            other_value_np = np.expand_dims(other_value_np, axis=2)
        output = (self_value * other_value_np).clip(0, 255)
        return HashableImage(
            output.astype(bool) if is_bool else output.astype(np.uint8),
        )

    @jaxtyped(typechecker=beartype)
    def __truediv__(self, other: object) -> "HashableImage":
        """Divide the HashableImage object by another object.

        This method is used to divide the current HashableImage object by
            another object. It checks if the other object is an instance of
            HashableImage or a Number. If it is, it performs the division
            operation and returns a new HashableImage object with the
            result.

        Args:
            self (HashableImage): The HashableImage object on which the
                division operation is performed.
            other (HashableImage or Number): The object by which the
                HashableImage object is divided.

        Returns:
            HashableImage: A new HashableImage object resulting from the
                division operation.

        """
        if not isinstance(other, HashableImage | Number):
            return NotImplemented
        if self._mode == "torch":
            other_value = (
                other if isinstance(other, Number) else other.tensor()
            )
            return HashableImage((self.tensor() / other_value).clamp(0, 1))
        other_value = other if isinstance(other, Number) else other.numpy()
        return HashableImage(
            (self.numpy() / other_value).clip(0, 255).astype(np.uint8),
        )

    @jaxtyped(typechecker=beartype)
    def size(self) -> ImageSize:
        """Calculate the size of the HashableImage object.

        This method calculates and returns the size of the HashableImage
            object
        as an ImageSize object. The size is determined based on the
            dimensions
        of the image stored in the HashableImage object.

        Args:
            self (HashableImage): The HashableImage object for which the
                size needs to be determined.

        Returns:
            ImageSize: An ImageSize object representing the size (width and
                height) of the HashableImage object.

        """
        return ImageSize.from_image(self._image)

    @jaxtyped(typechecker=beartype)
    def copy(self) -> "HashableImage":
        """Create a copy of a HashableImage object.

        Args:
            self (HashableImage): The HashableImage object to be copied.

        Returns:
            HashableImage: A new HashableImage object that is a copy of the
                original HashableImage object.

        """
        if self._mode == "torch":
            image = HashableImage(self._image.clone())
        else:
            image = HashableImage(self._image.copy())
        image.set_filename(self.get_filename())
        return image

    @jaxtyped(typechecker=beartype)
    def mean(self) -> float:
        """Calculate the mean value of the image data stored in the.

            HashableImage object.

        This method does not accept any arguments.

        Returns:
            float: The mean value of the image data, rounded to two decimal
                places.

        """
        if self._mode == "torch":
            value = self._image.float().mean().item()
        else:
            value = np.mean(self._image)
        # two decimal places
        return round(value, 5)

    @jaxtyped(typechecker=beartype)
    def std(self) -> float:
        """Calculate the standard deviation of the image data.

        This method operates on the image data stored in the HashableImage
            object.

        Returns:
            float: The standard deviation of the image data, rounded to two
                decimal places.

        """
        if self._mode == "torch":
            value = self._image.float().std().item()
        else:
            value = np.std(self._image)
        return round(value, 5)

    @jaxtyped(typechecker=beartype)
    def min(self) -> float:
        """Calculate and return the minimum value in the HashableImage object.

        This method analyzes the HashableImage object and returns the
            smallest value found within it. The returned value is rounded to
            two decimal places for precision.

        Args:
            None
        Returns:
            float: The minimum value in the HashableImage object, rounded to
                2 decimal places.

        """
        if self._mode == "torch":
            value = self._image.float().min().item()
        else:
            value = float(np.min(self._image))
        return round(value, 5)

    @jaxtyped(typechecker=beartype)
    def max(self) -> float:
        """Calculate and return the maximum value in the HashableImage object.

        This method does not require any arguments. It traverses through the
            HashableImage object,
        finds the maximum value, and then rounds it to two decimal places
            before returning.

        Returns:
            float: The maximum value in the HashableImage object, rounded to
                two decimal places.

        """
        if self._mode == "torch":
            value = self._image.float().max().item()
        else:
            value = float(np.max(self._image))
        return round(value, 5)

    @jaxtyped(typechecker=beartype)
    def sum(self) -> float:
        """Calculate the sum of all elements in the HashableImage object.

        This method iterates over all elements in the HashableImage object
            and sums them up.

        Args:
            None
        Returns:
            float: The sum of all elements in the HashableImage object. The
                sum is rounded to two decimal places for precision.

        """
        if self._mode == "torch":
            value = self._image.float().sum().item()
        else:
            value = float(np.sum(self._image))
        return round(value, 5)

    @jaxtyped(typechecker=beartype)
    def dtype(self) -> Literal["L", "RGB", "1"]:
        """Get the dtype of the image.

        This method returns the dtype of the image based on the mode of the
            image.

        Returns:
            str: The dtype of the image.

        """
        if self._mode == "pil":
            return self._image.mode
        if self._mode == "numpy":
            if self._image.ndim == 2 and self._image.dtype == np.uint8:
                return "L"
            if self._image.ndim == 3 and self._image.dtype == np.uint8:
                return "RGB"
            if self._image.ndim == 2 and self._image.dtype == bool:
                return "1"
            msg = "Invalid numpy image type"
            raise ValueError(msg)
        if self._mode == "torch":
            if self._image.size(1) == 3 or (
                self._image.size(1) == 1 and self._image.dtype == torch.float32
            ):
                return "RGB"
            if self._image.size(1) == 1 and self._image.dtype == torch.bool:
                return "1"
            msg = "Invalid torch image type"
            raise ValueError(msg)
        return None

    def __repr__(self) -> str:
        """Return a cheap repr — mode, dtype, size, source basename.

        Does NOT call `mean/std/min/max` or `get_filename()` (which
        previously made `repr()` an O(image) operation that hit disk
        for in-memory images). For the verbose form including
        statistics, call `summary()` explicitly.
        """
        src = (
            path_basename(self._image_str)
            if self._image_str is not None
            else "mem"
        )
        return (
            f"HashableImage(mode={self._mode}, dtype={self.dtype()}, "
            f"size={self.size()}, src={src})"
        )

    def summary(self) -> str:
        """Return a verbose summary including content statistics.

        Materializes mean, std, min, max and the local filename — O(image)
        for the stats, plus a temp-file write for in-memory inputs whose
        `get_filename()` hasn't been called yet. Prefer `__repr__` for
        cheap debugging output.
        """
        return (
            f"HashableImage(mode={self._mode}, dtype={self.dtype()}, "
            f"size={self.size()}, mean={self.mean()}, std={self.std()}, "
            f"min={self.min()}, max={self.max()}, "
            f"filename={self.get_filename()})"
        )

    @jaxtyped(typechecker=beartype)
    def pil(self) -> Image.Image:
        """Return image data as a PIL Image (independent copy).

        Always returns a fresh image — callers can safely mutate it
        without affecting the HashableImage's internal state. For
        zero-copy access when you promise not to mutate, use
        `pil_view()`.
        """
        if self._mode == "torch":
            return tensor2pil(self._image)
        if self._mode == "numpy":
            return Image.fromarray(self._image)
        return self._image.copy()

    @jaxtyped(typechecker=beartype)
    def pil_view(self) -> Image.Image:
        """Return image data as a PIL Image, **no copy** in PIL mode.

        Zero-copy alternative to `pil()` for read-only consumers in
        perf-critical loops. **Mutating the returned image mutates the
        HashableImage's internal state and silently invalidates the
        cached hash** — only use when you control the buffer lifetime.
        """
        if self._mode == "torch":
            return tensor2pil(self._image)
        if self._mode == "numpy":
            return Image.fromarray(self._image)
        return self._image

    @jaxtyped(typechecker=beartype)
    def numpy(
        self,
    ) -> (
        UInt8[np.ndarray, "h w 3"]
        | UInt8[np.ndarray, "h w"]
        | Bool[np.ndarray, "h w 3"]
        | Bool[np.ndarray, "h w"]
    ):
        """Return image data as a NumPy array (independent copy).

        Always returns a fresh array — callers can safely mutate it
        without affecting the HashableImage's internal state. For
        zero-copy access when you promise not to mutate, use
        `numpy_view()`.

        Returns:
            Independent `np.ndarray` with shape `h w 3` / `h w` and
            dtype `uint8` (or `bool` for binary images).

        """
        if self._mode == "torch":
            return tensor2numpy(
                self._image,
                output_type=(
                    bool if self._image.dtype == torch.bool else np.uint8
                ),
            )
        if self._mode == "numpy":
            return self._image.copy()
        return np.array(self._image)

    @jaxtyped(typechecker=beartype)
    def numpy_view(
        self,
    ) -> (
        UInt8[np.ndarray, "h w 3"]
        | UInt8[np.ndarray, "h w"]
        | Bool[np.ndarray, "h w 3"]
        | Bool[np.ndarray, "h w"]
    ):
        """Return image data as a read-only NumPy view (no copy in numpy mode).

        Zero-copy alternative to `numpy()` for read-only consumers in
        perf-critical loops. The returned array is marked
        `writeable=False`: accidental mutation raises `ValueError`
        instead of silently invalidating the cached hash.

        For torch and PIL modes the conversion path inherently
        allocates a new array, but the same read-only flag is applied
        for API consistency.
        """
        if self._mode == "torch":
            arr = tensor2numpy(
                self._image,
                output_type=(
                    bool if self._image.dtype == torch.bool else np.uint8
                ),
            )
        elif self._mode == "numpy":
            arr = self._image.view()
        else:
            arr = np.asarray(self._image)
        arr.setflags(write=False)
        return arr

    @jaxtyped(typechecker=beartype)
    def tensor(
        self,
    ) -> Float[torch.Tensor, "1 c h w"] | Bool[torch.Tensor, "1 c h w"]:
        """Return image data as a `1 c h w` tensor (independent copy).

        Always returns a fresh tensor — callers can safely mutate it
        (in-place ops, masked assignment, etc.) without affecting the
        HashableImage's internal state. For zero-copy access when you
        promise not to mutate, use `tensor_view()`.
        """
        if self._mode == "torch":
            return self._image.clone()
        if self._mode == "numpy":
            return numpy2tensor(self._image)
        return pil2tensor(self._image)

    @jaxtyped(typechecker=beartype)
    def tensor_view(
        self,
    ) -> Float[torch.Tensor, "1 c h w"] | Bool[torch.Tensor, "1 c h w"]:
        """Return image data as a `1 c h w` tensor, **no copy** in torch mode.

        Zero-copy alternative to `tensor()` for read-only consumers in
        perf-critical loops. **Mutating the returned tensor mutates the
        HashableImage's internal state and silently invalidates the
        cached hash** — only use when you control the buffer lifetime.
        """
        if self._mode == "torch":
            return self._image
        if self._mode == "numpy":
            return numpy2tensor(self._image)
        return pil2tensor(self._image)

    def bytes(self) -> bytes:
        """Convert the image data to a bytes object.

        This method converts the image data stored in the HashableImage
            object into a bytes object.

        Returns:
            bytes: The image data as a bytes object.

        """
        pil_image = self.pil()
        # BytesIO is a file-like buffer stored in memory
        img_bytes = io.BytesIO()
        # image.save expects a file-like as a argument
        pil_image.save(img_bytes, format="PNG")
        # Turn the BytesIO object back into a bytes object
        return img_bytes.getvalue()

    def b64(self, *, open_rb: bool = False) -> str:
        """Convert the image data to a base64 string.

        This method converts the image data stored in the HashableImage
            object into a base64 string.

        Returns:
            str: The image data as a base64 string.

        """
        if open_rb:
            with Path(self.get_filename()).open("rb") as f:
                return base64.b64encode(f.read()).decode("utf-8")
        return base64.b64encode(self.bytes()).decode("utf-8")

    @property
    @jaxtyped(typechecker=beartype)
    def mode(self) -> Literal["pil", "numpy", "torch"]:
        """Retrieve the mode of the HashableImage object.

        This method returns the mode of the HashableImage object. The mode
            can be one of three values: 'pil', 'numpy', or 'torch', each
            representing a different image format.

        Args:
            None
        Returns:
            str: The mode of the HashableImage object. This is a string
                indicating whether the image is in 'pil', 'numpy', or
                'torch' format.

        """
        return self._mode

    @jaxtyped(typechecker=beartype)
    def is_binary(self) -> bool:
        """Check if the image data in the HashableImage object is binary.

        This method evaluates whether the image data contained within the
        HashableImage object is binary or not.

        Args:
            self (HashableImage): The HashableImage object containing image
                data.

        Returns:
            bool: Returns True if the image data is binary, False otherwise.

        """
        if self._mode == "torch":
            return self._image.dtype == torch.bool
        if self._mode == "numpy":
            return self._image.dtype == bool
        if self._mode == "pil":
            return self._image.mode == "1"
        msg = "Invalid image mode"
        raise ValueError(msg)

    @jaxtyped(typechecker=beartype)
    def is_rgb(self) -> bool:
        """Return True if this image has 3 RGB channels.

        Each backend exposes channel count differently:
        - torch: `(1, c, h, w)` → `shape[1] == 3`.
        - numpy: `(h, w, c)` for 3-channel, `(h, w)` for 1-channel.
        - PIL: no `.shape`; check `Image.mode == "RGB"`.
        """
        if self._mode == "torch":
            return self._image.shape[1] == 3
        if self._mode == "pil":
            return self._image.mode == "RGB"
        return len(self._image.shape) == 3 and self._image.shape[2] == 3

    @property
    def shape(self) -> tuple[int, int] | tuple[int, int, int]:
        """Return the shape of the HashableImage object.

        This method determines and returns the shape of the HashableImage
            object. For binary images, the shape is a tuple of two integers
            representing height and width. For RGB images, the shape is a
            tuple of three integers representing height, width, and
            channels.

        Args:
            self (HashableImage): The HashableImage object for which the
                shape needs to be determined.

        Returns:
            Tuple[int, int]: If the image is binary, returns a tuple
                representing (height, width).
            Tuple[int, int, int]: If the image is RGB, returns a tuple
                representing (height, width, 3).

        Raises:
            ValueError: If the image is neither binary nor RGB.

        """
        h = int(self.size().height)
        w = int(self.size().width)
        if self.is_rgb():
            return (h, w, 3)
        # Binary ("1") and grayscale ("L") are both single-channel —
        # dtype() distinguishes them where needed.
        return (h, w)

    @jaxtyped(typechecker=beartype)
    def concat(
        self,
        other: list["HashableImage"],
        mode: Literal["horizontal", "vertical"],
    ) -> "HashableImage":
        """Concatenate multiple images either horizontally or vertically.

        Args:
            self (HashableImage): The instance of the HashableImage class.
            other (List[HashableImage]): A list of HashableImage objects to
                be concatenated with the self image.
            mode (str): A string specifying the concatenation mode. It can
                be 'horizontal' or 'vertical'.

        Returns:
            HashableImage: A new HashableImage object that represents the
                concatenated image based on the specified mode.

        """
        if self._mode == "torch":
            other_value = [img.tensor() for img in other]
            if mode == "horizontal":
                return HashableImage(
                    torch.cat([self.tensor(), *other_value], dim=3),
                )
            return HashableImage(
                torch.cat([self.tensor(), *other_value], dim=2),
            )
        other_value = [img.numpy() for img in other]
        if mode == "horizontal":
            return HashableImage(
                np.concatenate([self.numpy(), *other_value], axis=1),
            )
        return HashableImage(
            np.concatenate([self.numpy(), *other_value], axis=0),
        )

    @jaxtyped(typechecker=beartype)
    def raw(
        self,
    ) -> (
        Image.Image
        | UInt8[np.ndarray, "h w 3"]
        | UInt8[np.ndarray, "h w"]
        | Bool[np.ndarray, "h w"]
        | Float[torch.Tensor, "1 c h w"]
        | Bool[torch.Tensor, "1 1 h w"]
    ):
        """Return the underlying image as its native type (independent copy).

        For callers who don't know or care about the storage mode,
        this returns whatever native type the image was stored as
        (PIL.Image, np.ndarray, or torch.Tensor) — but always a
        fresh, mutation-safe copy. For zero-copy access use
        `raw_view()`.
        """
        if isinstance(self._image, torch.Tensor):
            return self._image.clone()
        if isinstance(self._image, np.ndarray):
            return self._image.copy()
        return self._image.copy()

    @jaxtyped(typechecker=beartype)
    def raw_view(
        self,
    ) -> (
        Image.Image
        | UInt8[np.ndarray, "h w 3"]
        | UInt8[np.ndarray, "h w"]
        | Bool[np.ndarray, "h w"]
        | Float[torch.Tensor, "1 c h w"]
        | Bool[torch.Tensor, "1 1 h w"]
    ):
        """Return the underlying image as its native type, **no copy**.

        Zero-copy alternative to `raw()` for read-only consumers in
        perf-critical loops. **Mutating the returned object mutates
        the HashableImage's internal state and silently invalidates
        the cached hash** — only use when you control the buffer
        lifetime.
        """
        return self._image

    @jaxtyped(typechecker=beartype)
    def logical_or_reduce(
        self,
        other: list["HashableImage"],
    ) -> "HashableImage":
        """Perform a logical OR operation on binary representations of.

            HashableImage objects.

        This method takes a list of HashableImage objects, converts them
            into binary representations and performs a logical OR operation
            on them. The result of this operation is used to create a new
            HashableImage object which is returned.

        Args:
            self (HashableImage): The HashableImage object on which the
                logical OR operation is performed.
            other (List[HashableImage]): A list of HashableImage objects to
                be combined using logical OR operation.

        Returns:
            HashableImage: A new HashableImage object representing the
                result of the logical OR operation on the input
                HashableImage objects.

        """
        if self._mode == "torch":
            other_value = self.to_binary().tensor()
            for img in other:
                other_value = torch.logical_or(
                    other_value,
                    img.to_binary().tensor(),
                )
            return HashableImage(other_value)
        return HashableImage(
            np.logical_or.reduce(
                [self.to_binary().numpy()]
                + [img.to_binary().numpy() for img in other],
            ),
        )

    def __hash__(self) -> int:
        """Return a cached content-and-shape hash for this image.

        The first call materializes the image bytes and hashes them with
        `xxhash.xxh3_64`; subsequent calls return the cached int. The
        cache is safe because instances are immutable (see `__init__`).

        Buffer extraction by mode:
        - `torch`: `.contiguous().cpu().numpy().tobytes()`.
        - `numpy`: `np.ascontiguousarray(...).tobytes()`.
        - `pil`: `np.asarray(...).tobytes()` (PIL is not
          buffer-protocol-compatible; one copy is unavoidable).

        Final hash combines `(mode, dtype, shape, content)` so two
        images with different shapes never collide on content alone.
        """
        if self._cached_hash is not None:
            return self._cached_hash
        mode = self._mode
        if mode == "torch":
            buf = self._image.contiguous().cpu().numpy().tobytes()
        elif mode == "numpy":
            buf = np.ascontiguousarray(self._image).tobytes()
        else:
            buf = np.asarray(self._image).tobytes()
        content = xxhash.xxh3_64_intdigest(buf)
        self._cached_hash = hash((mode, self.dtype(), self.shape, content))
        return self._cached_hash

    def __eq__(self, other: object) -> bool:
        """Compare by identity, then cached hash, then mode-aware content.

        Short-circuits in this order:
        1. `self is other` — same object.
        2. `not isinstance(other, HashableImage)` — `NotImplemented`.
        3. `hash(self) != hash(other)` — fast-path rejection. Cached
           hashes make this O(1) after the first call.
        4. Mode mismatch → False.
        5. Mode-aware content comparison. The full check is correct
           even if hashes collide (PIL adds an explicit mode + size
           guard; numpy uses `np.array_equal` which handles
           dtype/shape; torch uses `torch.equal`).
        """
        if self is other:
            return True
        if not isinstance(other, HashableImage):
            return NotImplemented
        if hash(self) != hash(other):
            return False
        if self._mode != other._mode:
            return False
        if self._mode == "torch":
            return bool(torch.equal(self._image, other._image))
        if self._mode == "numpy":
            return bool(np.array_equal(self._image, other._image))
        # PIL: explicit mode + size guard so collisions can't bypass
        # them via just-bytes comparison.
        return (
            self._image.mode == other._image.mode
            and self._image.size == other._image.size
            and self._image.tobytes() == other._image.tobytes()
        )

    @jaxtyped(typechecker=beartype)
    def crop_from_mask(
        self,
        mask: "HashableImage",
        *args: PCropArgs.args,
        **kwargs: PCropArgs.kwargs,
    ) -> "HashableImage":
        """Crop an image based on a provided mask image.

        Args:
            mask (HashableImage): The mask image used for cropping. It
                should be of the same size as the input image.
            **kwargs: Additional keyword arguments that can be passed to the
                cropping function. These could include parameters like
                'border' for additional padding or 'interpolation' for
                resizing method.

        Returns:
            HashableImage: A new HashableImage object that is the result of
                cropping the original image based on the provided mask. It
                will have the same dimensions as the mask image.

        """
        kwargs.setdefault("verbose", False)
        return HashableImage(
            crop_from_mask(
                self.to_rgb().numpy(),
                mask.to_binary().numpy(),
                *args,
                **kwargs,
            )
        )

    @jaxtyped(typechecker=beartype)
    def crop_from_points(
        self,
        points: "Points",
        *args: PCropArgs.args,
        **kwargs: PCropArgs.kwargs,
    ) -> "HashableImage":
        """Crop an image based on the provided points."""
        # convert points to mask first
        mask = points.to_mask()
        return self.crop_from_mask(mask, *args, **kwargs)

    @jaxtyped(typechecker=beartype)
    def crop_from_bbox(
        self,
        bboxes: "HashableList[BoundingBox]",
    ) -> "HashableImage":
        """Crop an image based on the provided bounding boxes.

        This method takes a list of bounding boxes and uses them to crop the
            instance of the HashableImage class. Each bounding box in the
            list should define a region in the image that will be included
            in the cropped image. The order of the bounding boxes in the
            list does not affect the result.

        Args:
            self (HashableImage): The instance of the HashableImage class to
                be cropped.
            bboxes (HashableList[_BBOX_TYPE]): A list of bounding boxes.
                Each bounding box is a tuple of four integers (x, y, width,
                height), where (x, y) is the top-left corner of the bounding
                box, and width and height are the dimensions of the bounding
                box.

        Returns:
            HashableImage: A new HashableImage object that is cropped based
                on the provided bounding boxes. The cropped image will
                include all regions defined by the bounding boxes and
                exclude everything else.

        """
        # set bbox to the size of the image in case it is bigger, for both float and int
        return HashableImage(
            crop_from_bbox(
                self.to_rgb().numpy(),
                [bbox.xyxyn for bbox in bboxes.to_list()],
                is_normalized=True,
            )
        )

    @jaxtyped(typechecker=beartype)
    def uncrop_from_bbox(
        self,
        base: "HashableImage",
        bboxes: "HashableList[BoundingBox]",
        *,
        resize: bool = False,
        blend_width: int = 10,
    ) -> "HashableImage":
        """Uncrop an image from a specified list of bounding boxes using a.

            Least Recently Used (LRU) cache.

        This method in the HashableImage class uncrops an image from regions
            specified by a list of bounding boxes.
        It returns the uncropped image as a HashableImage object.

        Args:
            self ('HashableImage'): The HashableImage object on which the
                method is called.
            base ('HashableImage'): The base HashableImage from which to
                uncrop the image.
            bboxes ('HashableList'): A HashableList of bounding boxes
                specifying the regions to uncrop.
            resize (bool): A boolean flag indicating whether to resize the
                uncropped image. Defaults to False.

        Returns:
            HashableImage: A HashableImage object representing the uncropped
                image.

        """
        is_normalized = True
        _bboxes = [bbox.xyxyn for bbox in bboxes.to_list()]
        return HashableImage(
            uncrop_from_bbox(
                base.to_rgb().numpy(),
                self.to_rgb().numpy(),
                _bboxes,
                resize=resize,
                is_normalized=is_normalized,
                blend_width=blend_width,
            )
        )

    @jaxtyped(typechecker=beartype)
    def mask2bbox(
        self,
        margin: float,
        *,
        normalized: bool = False,
        merge: bool = False,
        verbose: bool = True,
        closing: tuple[int, int] = (0, 0),
        opening: tuple[int, int] = (0, 0),
        area_threshold: float = 0.0,
        number_of_objects: int = -1,
    ) -> "HashableList[BoundingBox]":
        """Convert a mask image to a bounding box in HashableList format.

        This method takes an instance of HashableImage class and additional
            keyword arguments,
        and applies the mask2bbox function to convert a mask image into a
            bounding box.
        The bounding box coordinates are then returned in a HashableList
            format.

        Args:
            self (HashableImage): An instance of the HashableImage class
                representing the mask image.
            margin (float): The margin to be added to the bounding box
                coordinates.
            normalized (bool, optional): A boolean flag indicating whether
                the bounding box coordinates should be normalized. Defaults
                to False.
            merge (bool, optional): A boolean flag indicating whether to
                merge bounding boxes. Defaults to False.
            verbose (bool, optional): A boolean flag indicating whether to
                display verbose output. Defaults to True.
            closing (Tuple[int, int], optional): A tuple of two integers
                representing the kernel size for morphological closing.
                Defaults to (0, 0).
            opening (Tuple[int, int], optional): A tuple of two integers
                representing the kernel size for morphological opening.
                Defaults to (0, 0).
            area_threshold (float, optional): A float value representing the
                area threshold for filtering bounding boxes. Defaults to 0.0.
            number_of_objects (int, optional): An integer representing the
                maximum number of objects to detect. Defaults to -1. If set
                to -1, all objects will be detected.

        Returns:
            HashableList: A list containing the bounding box coordinates
                generated from the mask image.

        """
        _bbox = mask2bbox(
            self.to_binary().numpy(),
            margin=margin,
            normalized=normalized,
            merge=merge,
            verbose=verbose,
            closing=closing,
            opening=opening,
            area_threshold=area_threshold,
            number_of_objects=number_of_objects,
        )
        size = self.size()
        return HashableList(
            [
                BoundingBox(
                    xmin=box[0],
                    ymin=box[1],
                    xmax=box[2],
                    ymax=box[3],
                    image_size=size,
                )
                for box in _bbox
            ],
        )

    @jaxtyped(typechecker=beartype)
    def mask2squaremask(
        self,
        *args: PSquareMaskArgs.args,
        **kwargs: PSquareMaskArgs.kwargs,
    ) -> "HashableImage":
        """Convert the mask of a HashableImage object to a square mask.

        This method uses the mask2squaremask function from the image_tools
            module to convert the mask of the HashableImage object to a
            square mask.

        Args:
            self (HashableImage): The HashableImage object for which the
                mask needs to be converted to a square mask.
            **kwargs: Additional keyword arguments that can be passed to the
                mask2squaremask function from the image_tools module.

        Returns:
            HashableImage: A new HashableImage object with the square mask
                generated from the original mask.

        """
        return HashableImage(
            mask2squaremask(self.to_binary().numpy(), *args, **kwargs)
        )

    @jaxtyped(typechecker=beartype)
    def blend(
        self,
        mask: "HashableImage",
        alpha: float,
        *,
        with_bbox: bool,
        merge_bbox: bool = True,
    ) -> "HashableImage":
        """Blend the current HashableImage object with another using a mask,.

            alpha value, and other parameters.

        Args:
            mask (HashableImage): The HashableImage object representing the
                mask used for blending.
            alpha (float): The transparency level of the blending operation
                (0.0 - 1.0). (0.0 is fully transparent, 1.0 is fully visible)
            with_bbox (bool): Whether to include bounding box information in
                the blending operation.
            merge_bbox (bool, optional): Whether to merge bounding boxes
                during blending. Defaults to True.

        Returns:
            HashableImage: The HashableImage object resulting from the
                blending operation.

        """
        if mask.sum() == 0:
            with_bbox = False
        return HashableImage(
            mask_blend(
                self.to_rgb().numpy(),
                mask.numpy(),
                alpha,
                with_bbox=with_bbox,
                merge_bbox=merge_bbox,
            )
        )

    @jaxtyped(typechecker=beartype)
    def morphologyEx(  # noqa: N802
        self,
        operation: Literal["erode", "dilate", "open", "close"],
        kernel: Float[np.ndarray, "k k"],
    ) -> "HashableImage":
        """Perform morphological operations on an image.

        This function applies a specified morphological operation to the
            image using a given kernel.

        Args:
            operation (str): A string representing the morphological
                operation to be performed. It can be one of the following:
                'erode', 'dilate', 'open', or 'close'.
            kernel (np.array): A NumPy array representing the structuring
                element for the operation.

        Returns:
            HashableImage: A new instance of HashableImage with the
                morphological operation applied to the image.

        """
        # https://docs.opencv.org/4.x/d9/d61/tutorial_py_morphological_ops.html
        _operation = getattr(cv2, f"MORPH_{operation.upper()}")
        return HashableImage(
            morphologyEx(self.to_binary().numpy(), _operation, kernel),
        )

    @jaxtyped(typechecker=beartype)
    def center_pad(
        self,
        image_size: ImageSize,
        fill: int = 0,
    ) -> "HashableImage":
        """Center pad an image to a specified size with a specified fill value.

        This method in the HashableImage class is used to center pad an
            image to the given size, using the provided fill value.

        Args:
            image_size (Tuple[int, int]): A tuple representing the desired
                size of the image after center padding.
            fill (int): An integer value representing the fill value to be
                used for padding. Defaults to 0.

        Returns:
            HashableImage: A new HashableImage object with the image center
                padded according to the specified image_size and fill value.

        """
        return HashableImage(
            center_pad(self.to_rgb().numpy(), image_size, fill)
        )

    @staticmethod
    @jaxtyped(typechecker=beartype)
    def make_image_grid(
        images: dict[str, list["HashableImage"]],
        *,
        orientation: Literal["horizontal", "vertical"] = "horizontal",
        with_text: bool = False,
        verbose: bool = False,
        output: str | None = None,
    ) -> "HashableImage":
        """Arrange a dictionary of images into a grid either horizontally or.

            vertically.

        This static method in the 'HashableImage' class takes a dictionary
            of images and arranges them in a grid.
        Images are padded with black to match the maximum height and width.
            An optional text label can be included on the grid.

        Args:
            images (HashableDict[HashableList[HashableImage]]): A dictionary
                containing lists of HashableImage objects.
            orientation (Literal['horizontal', 'vertical']): Specifies the
                orientation of the grid. It can be either 'horizontal' or
                'vertical'.
            with_text (bool): Indicates whether to include text labels on
                the grid. Defaults to False.

        Returns:
            HashableImage: A HashableImage object representing the grid of
                images with optional text labels.

        """
        image_as_list = deepcopy(images)
        max_images = max([len(imgs) for imgs in image_as_list.values()])
        for key, imgs in image_as_list.items():
            if len(imgs) < max_images:
                black_images = [imgs[0].zeros_like()] * (
                    max_images - len(imgs)
                )
                image_as_list[key] += black_images
        # all images should have the same size, otherwise pad them with zeros to the max size
        max_height = max(
            [
                img.size().height
                for imgs in image_as_list.values()
                for img in imgs
            ],
        )
        max_width = max(
            [
                img.size().width
                for imgs in image_as_list.values()
                for img in imgs
            ],
        )
        new_size = ImageSize(height=max_height, width=max_width)
        for key, imgs in image_as_list.items():
            for idx, img in enumerate(imgs):
                if img.size() != new_size:
                    image_as_list[key][idx] = img.center_pad(new_size)

        # each index in the list is a different row
        all_images = []
        if orientation == "horizontal":
            # For horizontal orientation, stack images column-wise
            # Each column contains all images from one key
            nrows = len(image_as_list)
            ncols = max_images
            # all the first images from each key, then the second images from each key, etc.
            for imgs in image_as_list.values():
                all_images.extend([img.pil() for img in imgs])
        else:
            # For vertical orientation, stack images row-wise
            # Each row contains all images from one key
            nrows = max_images
            ncols = len(image_as_list)
            for idx in range(nrows):
                for key in image_as_list:
                    all_images.append(image_as_list[key][idx].pil())

        grid = make_image_grid(all_images, rows=nrows, cols=ncols)
        if with_text:
            grid = Image.fromarray(
                create_text(
                    np.asarray(grid),
                    texts=list(image_as_list.keys()),
                    orientation=orientation,
                ),
            )

        image_grid = HashableImage(grid)
        if output is not None:
            image_grid.save(output)
        if verbose:
            logger.log(f"{image_grid=} grid saved to {output}", stack_offset=3)
        return image_grid

    @jaxtyped(typechecker=beartype)
    def set_minmax(self, _min: float, _max: float, /) -> "HashableImage":
        """Set the minimum and maximum values of the image.

        This method sets the minimum and maximum values of the image to the
            specified values.

        Args:
            min (float): The minimum value to set for the image.
            max (float): The maximum value to set for the image.

        Returns:
            None

        """
        data = self.tensor()
        data = (data - data.min()) / (data.max() - data.min())
        data = data * (_max - _min) + _min
        return HashableImage(data)

    @jaxtyped(typechecker=beartype)
    def __setitem__(
        self, mask: "HashableImage", value: float, /
    ) -> "HashableImage":
        """Set the pixel values of the image based on a mask.

        This method sets the pixel values of the image to a specified value
            based on a mask.

        Args:
            mask (HashableImage): The mask image used to set the pixel values
                of the image.
            value (float): The value to set the pixel values to.

        Returns:
            HashableImage: A new HashableImage object with the pixel values
                set based on the mask.

        """
        if value < 0 or value > 1:
            msg = "Value must be between 0 and 1"
            raise ValueError(msg)
        # `self.tensor()` is now safe-by-default (returns an independent
        # clone), so in-place masked assignment cannot leak back into
        # `self._image`.
        image_pt = self.tensor()
        mask_pt = mask.to_binary().tensor()
        image_pt[mask_pt.expand_as(image_pt)] = value
        return HashableImage(image_pt)
