from __future__ import annotations

import base64
import io
import random
import string
import tempfile
from collections.abc import Iterable, Iterator, MutableMapping, MutableSequence
from numbers import Number
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Literal,
    SupportsIndex,
    TypeVar,
    cast,
    overload,
)

import cv2
import numpy as np
import torch
from beartype import beartype
from PIL import Image, ImageOps
from pydantic import ConfigDict
from pydantic.dataclasses import dataclass
from torchvision.transforms import functional as TF

from pixelcache.tools.bbox import crop_from_bbox, uncrop_from_bbox
from pixelcache.tools.cache import jaxtyped, lru_cache
from pixelcache.tools.image import (
    ImageSize,
    center_pad,
    compress_image,
    convert_to_space_color,
    get_canny_edge,
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
    bbox2mask,
    crop_from_mask,
    differential_mask,
    group_regions_from_binary,
    mask2bbox,
    mask2points,
    mask2squaremask,
    mask_blend,
    morphologyEx,
    polygon_to_mask,
    remove_small_regions,
)
from pixelcache.tools.text import create_text, draw_text
from pixelcache.tools.utils import color_palette

if TYPE_CHECKING:
    from jaxtyping import Bool, Float, UInt8

_T = TypeVar("_T")
_KT = TypeVar("_KT")
_VT = TypeVar("_VT")

MAX_IMG_CACHE = 5
VALID_IMAGES = Literal["pil", "numpy", "torch"]
PALETTE_DEFAULT = color_palette()


@jaxtyped(typechecker=beartype)
def pseudo_hash(idx: int, length: int = 6) -> str:
    """Generate a pseudo-random hash based on the given index and length.

    Arguments:
        idx (int): The index used to seed the random number generator.
        length (int, optional): The length of the hash to be generated.
            Defaults to 6.

    Returns:
        str: A string representing the pseudo-random hash generated based on
            the given index and length.

    Example:
        >>> generate_hash(10, 6)

    Note:
        The hash generated is pseudo-random, meaning it will generate the
            same result if the same index and length are provided.

    """
    random.seed(idx)
    return "".join(random.choice(string.ascii_letters) for _ in range(length))  # noqa: S311


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
        """Initialize an instance of the HashableImage class.

        This method sets the image data and mode based on the provided input
            type. If the input is not a file path, it saves the image data
            to a temporary file.

        Arguments:
            image (Union[str, Path, bytes, Image, np.ndarray, torch.Tensor,
                np.bool_]): The input image data. This can be a string file
                path, Path object, bytes, PIL Image object, numpy array, torch
                tensor, or boolean array.

        Returns:
            None
        Example:
            >>> img = HashableImage(image_data)

        Note:
            The temporary file created when the input is not a file path
                will be deleted when the instance is garbage collected.

        """
        # pytorch is hashable
        if isinstance(image, torch.Tensor):
            self._image = image.detach().cpu()
        elif isinstance(image, str | Path):
            self._image = read_image(image)
        elif isinstance(image, Image.Image):
            self._image = image
        elif isinstance(image, bytes):
            self._image = Image.open(io.BytesIO(image))
        else:
            self._image = image

        if isinstance(image, str | Path):
            self._image_str = str(image)
        else:
            self._create_tmp_file()

    def _create_tmp_file(self) -> None:
        """Create a temporary file."""
        self._image_str = tempfile.NamedTemporaryFile(suffix=".png").name
        self.save(self._image_str)

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
        """Retrieve the filename of the HashableImage object.

        This method does not require any arguments.

        Returns:
            str: A string representing the filename of the HashableImage
                object.

        Example:
            >>> hashable_image.get_filename()

        Note:
            This method is typically used when you need to access the file
                name of the image for further processing.

        """
        if HashableImage(self._image_str) != self:
            # update the filename
            self._create_tmp_file()
        return self._image_str

    def get_local_filename(self) -> str:
        """Retrieve the local filename of the HashableImage object.

        If the original filename starts with 'http', this method saves the
            image to
        a temporary file and returns the path of the temporary file.
            Otherwise, it
        simply returns the original filename.

        Returns:
            str: The local filename of the HashableImage object.

        Example:
            >>> image = HashableImage("http://example.com/image.jpg")
            >>> image.get_local_filename()
            '/tmp/tmp123.jpg'
        Note:
            This method is part of the HashableImage class and requires no
                arguments.

        """
        _filename = self.get_filename()
        if _filename.startswith("http"):
            # write it as a temp file
            temp_file = Path(tempfile.NamedTemporaryFile(suffix=".png").name)
            self.save(temp_file)
            return str(temp_file)
        return _filename

    def set_filename(self, filename: str) -> None:
        """Set the filename of the HashableImage object.

        This method in the 'HashableImage' class assigns a filename to the
            image object.

        Arguments:
            filename (str): A string representing the filename of the image.

        Returns:
            None
        Example:
            >>> image = HashableImage()
            >>> image.set_filename("image1.jpg")

        Note:
            The filename is used for saving and retrieving the image from
                storage.

        """
        # in case the image has been modified during inpainting, but the filename is still the same
        self._image_str = filename

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

        Arguments:
            path (str): The file path where the image will be saved. This
                should include the filename and the extension.

        Returns:
            None: This method doesn't return any value. It writes the image
                data to a file.

        Example:
            >>> image_object.save_image("/path/to/save/image.jpg")

        Note:
            Make sure the path exists and you have write permissions. If the
                file already exists, it will be overwritten.

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

        Arguments:
            self (HashableImage): The HashableImage object to be displayed.

        Returns:
            None: This method doesn't return any value. It displays the image
                data.

        Example:
            >>> image_object.display_image()

        Note:
            The method uses the default image viewer on your system to
                display the image.

        """
        self.pil().show()

    def downsample(self, factor: int) -> HashableImage:
        """Downsample the given image by a specified factor.

        Arguments:
            factor (int): The factor by which the image should be
                downsampled. This must be an integer greater than 0.

        Returns:
            HashableImage: A new HashableImage object that is a downscaled
                version of the original image.

        Example:
            >>> downsample_image(2)

        Note:
            The downsampling process may result in loss of image detail.

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
    ) -> HashableImage:
        """Resize the image to a specified size using different interpolation.

            methods based on the image mode.

        Arguments:
            size (ImageSize): An object containing the desired image height
                and width.
            mode (Literal["bilinear", "lanczos", "nearest"], optional): The
                interpolation method to use. Defaults to "bilinear".

        Returns:
            HashableImage: A new HashableImage object with the resized image
                if the size is different from the current image size.
                Otherwise, it returns the original HashableImage object.

        Example:
            >>> image = HashableImage(...)
            >>> new_size = ImageSize(200, 200)
            >>> resized_image = image.resize(new_size)

        Note:
            The interpolation method used for resizing depends on the mode
                of the image.

        """
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
                    self._image,
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
                _image = self._image.resize((width, height), _mode)
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
                    cast(np.ndarray, self._image),
                    (width, height),
                    interpolation=_mode,
                )
            return HashableImage(_image)
        return self

    @jaxtyped(typechecker=beartype)
    def resize_min_size(
        self, min_size: int, modulo: int = 16
    ) -> HashableImage:
        """Resize the image to a specified minimum size.

        This method resizes the image to the specified minimum size while
            maintaining the aspect ratio.

        Arguments:
            self (HashableImage): The HashableImage object to be resized.
            min_size (int): The minimum size to which the image should be
                resized.
            modulo (int, optional): The value to which the image dimensions
                should be divisible. Defaults to 16.

        Returns:
            HashableImage: A new HashableImage object with the resized image
                based on the minimum size.

        Example:
            >>> image = HashableImage(...)
            >>> new_image = image.resize_min_size(200)

        Note:
            The aspect ratio of the image is maintained during resizing.

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

    @jaxtyped(typechecker=beartype)
    def is_empty(self) -> bool:
        """Check if the HashableImage object is empty.

        This method determines if the HashableImage object is empty by
            summing up the values of the image array and comparing it to
            zero.

        Arguments:
            self (HashableImage): The HashableImage object to be checked for
                emptiness.

        Returns:
            bool: A boolean value indicating whether the HashableImage
                object is empty (True) or not (False).

        Example:
            >>> image = HashableImage(...)
            >>> image.is_empty()

        """
        if self._mode == "torch":
            return torch.sum(self._image).item() == 0
        if self._mode == "numpy":
            return np.sum(cast(np.ndarray, self._image)).item() == 0
        return np.sum(np.asarray(self._image)).item() == 0

    @lru_cache(maxsize=MAX_IMG_CACHE)
    @jaxtyped(typechecker=beartype)
    def to_gray(self) -> HashableImage:
        """Converts the current image to grayscale.

        This method does not take any arguments. It processes the current
            image object and returns a new HashableImage object that
            represents the grayscale version of the original image.

        Returns:
            HashableImage: A new image object that is the grayscale version
                of the original image.

        Example:
            >>> image_object.convert_to_grayscale()

        Note:
            The original image object remains unchanged. A new image object
                is created and returned.

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

    @lru_cache(maxsize=MAX_IMG_CACHE)
    @jaxtyped(typechecker=beartype)
    def flip_lr(self) -> HashableImage:
        """Flip the image horizontally.

        This method in the 'HashableImage' class takes the instance of the
            class as an argument and returns a new instance of
            'HashableImage' with the horizontally flipped image.

        Arguments:
            self ('HashableImage'): The instance of the 'HashableImage'
                class.

        Returns:
            'HashableImage': A new instance of 'HashableImage' with the
                horizontally flipped image.

        Example:
            >>> image = HashableImage()
            >>> flipped_image = image.flip_lr()

        Note:
            The original 'HashableImage' instance is not modified; a new
                instance is returned.

        """
        if self._mode == "torch":
            return HashableImage(torch.flip(self._image, [3]))
        return HashableImage(cv2.flip(self._image, 1))

    @jaxtyped(typechecker=beartype)
    def extract_binary_from_value(self, value: int) -> HashableImage:
        """Extract the binary mask from the value in the HashableImage object.

        This method extracts the binary mask from the value in the
            HashableImage object.

        Arguments:
            value (int): The value from which the binary mask will be
                extracted.

        Returns:
            HashableImage: A new HashableImage object with the binary mask
                extracted from the value.

        Example:
            >>> image = HashableImage(...)
            >>> binary_mask = HashableImage.extract_binary_from_value(10)

        Note:
            The binary mask is extracted based on the mode of the image data.

        """
        image_np = self.to_rgb().numpy()
        new_image: np.ndarray = np.zeros_like(image_np)
        new_image[image_np == value] = 255
        return HashableImage(new_image.astype(bool)[..., 0])

    @jaxtyped(typechecker=beartype)
    def convert_binary_to_value(self, value: int) -> HashableImage:
        """Convert the binary mask to the value in the HashableImage object.

        This method converts the binary mask to the value in the HashableImage
            object.

        Arguments:
            value (int): The value to which the binary mask will be converted.

        Returns:
            HashableImage: A new HashableImage object with the binary mask
                converted to the value.

        Example:
            >>> image = HashableImage(...)
            >>> new_image = image.convert_binary_to_value(10)

        Note:
            The binary mask is converted to the value based on the mode of the
                image data.

        """
        image_np = self.to_rgb().numpy()
        new_image: np.ndarray = np.zeros_like(image_np)
        new_image[image_np.astype(bool)] = value
        return HashableImage(new_image)

    @jaxtyped(typechecker=beartype)
    def apply_palette(
        self, _palette: UInt8[np.ndarray, "256 3"] = PALETTE_DEFAULT, /
    ) -> HashableImage:
        """Apply a color palette to the HashableImage object.

        This method applies a color palette to the HashableImage object.

        Arguments:
            self (HashableImage): The HashableImage object to which the
                color palette will be applied.
            _palette (np.ndarray, optional): The color palette to be applied
                to the HashableImage object. Defaults to PALETTE_DEFAULT.

        Returns:
            HashableImage: A new HashableImage object with the color palette
                applied.

        Example:
            >>> image = HashableImage(...)
            >>> new_image = image.apply_palette()

        Note:
            The color palette is applied to the HashableImage object based
                on the mode of the image data.

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
        for i, value in enumerate(unique_values):
            new_image[image_np == value] = _palette[i]
        return HashableImage(new_image)

    @lru_cache(maxsize=MAX_IMG_CACHE)
    @jaxtyped(typechecker=beartype)
    def to_rgb(self) -> HashableImage:
        """Convert an image to RGB format.

        This method transforms the current mode of a HashableImage object to
            an RGB format.

        Arguments:
            self ('HashableImage'): The HashableImage object to be converted
                to RGB.

        Returns:
            HashableImage: The HashableImage object converted to RGB format.

        Example:
            >>> img = HashableImage("path/to/image")
            >>> rgb_img = img.convert_to_rgb()

        Note:
            The original HashableImage object is not modified, a new object
                is returned.

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

    @lru_cache(maxsize=MAX_IMG_CACHE)
    @jaxtyped(typechecker=beartype)
    def to_binary(
        self, threshold: float = 0.0, area_min: int = 0, connectivity: int = 8
    ) -> HashableImage:
        """Convert an image to binary format.

        Arguments:
            threshold (float): The threshold for converting the image to binary.
            area_min (int): The minimum area for removing disconnected
                regions.

        Returns:
            HashableImage: A HashableImage object representing the converted
                image in binary format.

        Example:
            >>> convert_image_to_binary()

        Note:
            This function relies on the global state and does not take any
                parameters.

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

    @lru_cache(maxsize=MAX_IMG_CACHE)
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

        Example:
            >>> image = HashableImage(...)
            >>> unique_values = image.unique_values()

        Note:
            The unique values in the image are determined based on the mode
                of the image data.

        """
        output: tuple[torch.Tensor, torch.Tensor, torch.Tensor] = (
            self.tensor().unique(return_counts=True, return_inverse=True, sorted=True)
        )
        _unique = output[0].tolist()
        _indices = output[1]
        _count = output[2].tolist()
        return _unique, _indices, _count

    @lru_cache(maxsize=MAX_IMG_CACHE)
    @jaxtyped(typechecker=beartype)
    def split_masks(
        self,
        closing: tuple[int, int] = (0, 0),
        margin: float = 0.0,
        area_threshold: float = 0.0,
    ) -> HashableList[HashableImage]:
        """Split masks in a HashableImage object into multiple HashableImage.

            objects.

        This function processes a HashableImage object and splits the masks
            into multiple
        HashableImage objects based on the closing operation parameters,
            margin, and area threshold.

        Arguments:
            closing (tuple): A pair of integers specifying the closing
                operation parameters.
            margin (float): The margin for splitting masks. Masks that are
                closer than this margin will be split.
            area_threshold (float): The area threshold for splitting masks.
                Masks smaller than this area will not be split.

        Returns:
            HashableList: A list of HashableImage objects resulting from the
                mask splitting operation.

        Example:
            >>> split_masks((5, 5), 0.1, 200)

        Note:
            The closing operation is a morphological operation that is used
                to remove small holes in the foreground.
            This function assumes that the input is a HashableImage object
                that contains masks to be split.

        """
        return HashableList(
            [
                HashableImage(i)
                for i in group_regions_from_binary(
                    self.to_binary().numpy(),
                    closing=closing,
                    margin=margin,
                    area_threshold=area_threshold,
                )
            ],
        )

    @lru_cache(maxsize=MAX_IMG_CACHE)
    @jaxtyped(typechecker=beartype)
    def invert_binary(self) -> HashableImage:
        """Invert the binary representation of the image data in a.

            HashableImage object.

        This method checks the mode of the image data and returns a new
            HashableImage object
        with the inverted binary data.

        Arguments:
            self (HashableImage): The HashableImage object on which the
                method is called.

        Returns:
            HashableImage: A new HashableImage object with the inverted
                binary data based on
            the mode of the original image data.

        Example:
            >>> image = HashableImage(data)
            >>> inverted_image = image.invert()

        Note:
            The inversion of the binary data depends on the mode of the
                original image data.

        """
        if self._mode == "torch":
            return HashableImage(~self.to_binary().tensor())
        return HashableImage(~self.to_binary().numpy())

    @lru_cache(maxsize=MAX_IMG_CACHE)
    @jaxtyped(typechecker=beartype)
    def invert_rgb(self) -> HashableImage:
        """Invert the RGB values of the HashableImage object.

        This method checks the mode of the HashableImage object and performs
            the inversion accordingly.

        Arguments:
            self ('HashableImage'): The HashableImage object on which the
                method is called.

        Returns:
            'HashableImage': A new HashableImage object with inverted RGB
                values. If the mode of the image is 'torch', it returns the
                inverted tensor values. If the mode is not 'torch', it
                returns the inverted numpy values.

        Example:
            >>> image = HashableImage(...)
            >>> inverted_image = image.invert_image()

        Note:
            The inversion is performed based on the mode of the image. Two
                modes are supported: 'torch' and others. If the mode is
                'torch', tensor values are inverted. Otherwise, numpy values
                are inverted.

        """
        if self._mode == "torch":
            return HashableImage(1 - self.tensor())
        return HashableImage(255 - self.numpy())

    @staticmethod
    @jaxtyped(typechecker=beartype)
    def zeros_from_size(size: ImageSize) -> HashableImage:
        """Create a HashableImage object with all elements initialized to zero.

        This static method generates a HashableImage object of the specified
            size with all pixel values set to zero.

        Arguments:
            size (ImageSize): An object representing the height and width of
                the image in pixels.

        Returns:
            HashableImage: A HashableImage object with all pixel values
                initialized to zero. The size of the image is determined by
                the input argument.

        Example:
            >>> create_zero_image(ImageSize(800, 600))

        Note:
            The ImageSize object should contain positive integer values for
                both height and width.

        """
        return HashableImage(
            torch.zeros((1, 3, int(size.height), int(size.width))),
        )

    @jaxtyped(typechecker=beartype)
    def zeros_like(self) -> HashableImage:
        """Create a new HashableImage object with all elements set to zero.

        This method generates a new HashableImage object, with the same
            shape and type as the original image, but with all its elements
            set to zero.

        Arguments:
            self ('HashableImage'): The HashableImage object calling the
                method.

        Returns:
            'HashableImage': A new HashableImage object with all elements
                set to zero, maintaining the shape and type of the original
                image.

        Example:
            >>> image = HashableImage(...)
            >>> zeroed_image = image.zero_image()

        Note:
            The new HashableImage object does not alter the original image,
                it is a separate instance.

        """
        if self._mode == "torch":
            return HashableImage(torch.zeros_like(self._image))
        if self._mode == "numpy":
            return HashableImage(np.zeros_like(self._image))
        return HashableImage(
            Image.new(self._image.mode, self._image.size, 0),
        )

    @jaxtyped(typechecker=beartype)
    def ones_like(self) -> HashableImage:
        """Create a new HashableImage object filled with ones.

        This method generates a new HashableImage object, maintaining the
            dimensions of the original image,
        but replacing all pixel values with ones.

        Arguments:
            self ('HashableImage'): The HashableImage object on which the
                ones_like method is called.

        Returns:
            'HashableImage': A new HashableImage object with the same
                dimensions as the original image but filled with ones.

        Example:
            >>> image = HashableImage(...)
            >>> ones_image = image.ones_like()

        Note:
            The generated HashableImage object has the same dimensions as
                the original, but all pixel values are set to one.

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
    def rgb2bgr(self) -> HashableImage:
        """Convert the image from RGB to BGR color space in a HashableImage.

            object.

        This method takes a HashableImage object that contains an image in
            RGB color space and converts it to BGR color space.

        Arguments:
            self (HashableImage): The HashableImage object that contains the
                image to be converted.

        Returns:
            HashableImage: A new HashableImage object with the image
                converted to BGR color space.

        Example:
            >>> img = HashableImage("image_in_rgb.jpg")
            >>> img_bgr = img.rgb_to_bgr()

        Note:
            The input image must be in RGB color space. The output image
                will be in BGR color space.

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
    def equalize_hist(self) -> HashableImage:
        """Equalizes the histogram of the image stored in the HashableImage.

            object.

        This method adjusts the intensity values of the image to improve
            contrast and enhance details.

        Arguments:
            self (HashableImage): The HashableImage object containing the
                image to be processed.

        Returns:
            HashableImage: A new HashableImage object with the histogram
                equalized image.

        Example:
            >>> image = HashableImage("image.jpg")
            >>> equalized_image = image.equalize_hist()

        Note:
            Histogram equalization can improve the contrast of an image, but
                may also amplify noise.

        """
        if self._mode == "pil":
            return HashableImage(ImageOps.equalize(self._image))
        return HashableImage(cv2.equalizeHist(self.to_gray().numpy()))

    @jaxtyped(typechecker=beartype)
    def to_space_color(
        self, color_space: str, getchannel: str | None = None
    ) -> HashableImage:
        """Convert the image to a specified color space.

        This method converts the image stored in the HashableImage object to
            the specified color space.

        Arguments:
            self (HashableImage): The HashableImage object containing the
                image to be converted.
            color_space (str): The color space to which the image should be
                converted. This can be 'RGB', 'BGR', 'HSV', 'LAB', 'YUV',
                'XYZ', 'YCrCb', 'HLS', 'LUV', 'YCbCr', 'YIQ', 'YPbPr', or
                'YDbDr'.
            getchannel (str, optional): The channel to extract from the
                converted image. This can be 'R', 'G', 'B', 'H', 'S', 'V',

        Returns:
            HashableImage: A new HashableImage object with the image
                converted to the specified color space.

        Example:
            >>> image = HashableImage("image.jpg")
            >>> converted_image = image.to_space_color("HSV")

        Note:
            The color space must be one of the supported color spaces.

        """
        return HashableImage(
            convert_to_space_color(
                self._image, color_space, getchannel=getchannel
            )
        )

    @jaxtyped(typechecker=beartype)
    def compress_image(
        self,
        *,
        temp_dir: str | Path | None = None,
        jpeg_quality: int,
    ) -> str:
        """Compress the image stored in the HashableImage object.

        This method compresses the image stored in the HashableImage object
            using the JPEG format.

        Arguments:
            self (HashableImage): The HashableImage object containing the
                image to be compressed.
            temp_dir (str | Path, optional): The directory where the
                compressed image will be stored. If None, a temporary
                directory will be created. Defaults to None.
            jpeg_quality (int): The quality of the compressed image. This
                should be an integer between 0 and 100.

        Returns:
            str: The path to the compressed image file.

        """
        return compress_image(
            self.to_rgb().pil(),
            temp_dir=temp_dir,
            jpeg_quality=jpeg_quality,
        )

    @jaxtyped(typechecker=beartype)
    def __add__(self, other: object) -> HashableImage:
        """Add a HashableImage object to another HashableImage or Number.

            object.

        This method takes a HashableImage object and another object (either
            a HashableImage or a Number)
        and returns a new HashableImage object that results from the
            addition of the two input objects.

        Arguments:
            self (HashableImage): The HashableImage object to be added.
            other (HashableImage | Number): The other object (either a
                HashableImage or a Number) to be added to the HashableImage
                object.

        Returns:
            HashableImage: A new HashableImage object that is the result of
                adding the two input objects.

        Example:
            >>> img1 = HashableImage(...)
            >>> img2 = HashableImage(...)
            >>> new_img = img1.add(img2)

        Note:
            If 'other' is a Number, it is added to every pixel of the
                HashableImage object.

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
    def __sub__(self, other: object) -> HashableImage:
        """Subtract pixel values of a HashableImage object or a number from.

            this HashableImage object.

        This method takes either another HashableImage object or a number as
            an argument. If it's another HashableImage object,
        it subtracts the pixel values of the second image from the pixel
            values of the first image. If it's a number, it subtracts
        this number from every pixel value of the first image.

        Arguments:
            self (HashableImage): The HashableImage object from which the
                pixel values are subtracted.
            other (Union[HashableImage, Number]): The object to subtract
                from the HashableImage object. It can be either another
            HashableImage object or a number.

        Returns:
            HashableImage: A new HashableImage object with pixel values
                subtracted based on the type of 'other' object.

        Example:
            >>> img1.subtract(img2)
            or
            >>> img1.subtract(5)

        Note:
            The method does not modify the original HashableImage objects,
                it returns a new HashableImage object.

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
    def __mul__(self, other: object) -> HashableImage:
        """Performs element-wise multiplication between two HashableImage.

            objects or a HashableImage object and a Number.

        This method multiplies the pixel data of the HashableImage object on
            which it is called with the pixel data of another HashableImage
            object or a Number. The multiplication is performed element-
            wise, and a new HashableImage object is returned with the
            resulting pixel data.

        Arguments:
            self (HashableImage): The HashableImage object on which the
                method is called.
            other (HashableImage | Number): The object to be multiplied with
                the HashableImage object. It can be another HashableImage
                object or a Number.

        Returns:
            HashableImage: A new HashableImage object containing the result
                of the element-wise multiplication of the two input objects.

        Example:
            >>> img1.multiply(img2)
            or
            >>> img1.multiply(2)

        Note:
            If 'other' is a HashableImage, it should have the same
                dimensions as 'self'. If it is a number, it will be
                multiplied with each pixel of 'self'.

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
    def __truediv__(self, other: object) -> HashableImage:
        """Divide the HashableImage object by another object.

        This method is used to divide the current HashableImage object by
            another object. It checks if the other object is an instance of
            HashableImage or a Number. If it is, it performs the division
            operation and returns a new HashableImage object with the
            result.

        Arguments:
            self (HashableImage): The HashableImage object on which the
                division operation is performed.
            other (HashableImage or Number): The object by which the
                HashableImage object is divided.

        Returns:
            HashableImage: A new HashableImage object resulting from the
                division operation.

        Example:
            >>> img1 = HashableImage(...)
            >>> img2 = HashableImage(...)
            >>> result = img1 / img2
        Note:
            If the other object is neither a HashableImage nor a Number, a
                TypeError will be raised.

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

        Arguments:
            self (HashableImage): The HashableImage object for which the
                size needs to be determined.

        Returns:
            ImageSize: An ImageSize object representing the size (width and
                height) of the HashableImage object.

        Example:
            >>> hash_img = HashableImage("image.jpg")
            >>> size = hash_img.get_size()

        """
        return ImageSize.from_image(self._image)

    @jaxtyped(typechecker=beartype)
    def copy(self) -> HashableImage:
        """Create a copy of a HashableImage object.

        Arguments:
            self (HashableImage): The HashableImage object to be copied.

        Returns:
            HashableImage: A new HashableImage object that is a copy of the
                original HashableImage object.

        Example:
            >>> image = HashableImage()
            >>> copy = image.clone()

        Note:
            This method uses the copy module's deepcopy function to ensure a
                complete copy of the original object.

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

        Example:
            >>> hashable_image.calculate_mean()

        Note:
            The HashableImage object should already contain image data.

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
        Arguments: None
        Returns:
            float: The standard deviation of the image data, rounded to two
                decimal places.

        Example:
            >>> calculate_standard_deviation()

        Note:
            The image data must be stored in the HashableImage object before
                calling this method.

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

        Arguments:
            None
        Returns:
            float: The minimum value in the HashableImage object, rounded to
                2 decimal places.

        Example:
            >>> find_min_value()

        Note:
            The HashableImage object should be initialized before calling
                this method.

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

        Example:
            >>> get_max_value()

        Note:
            The HashableImage object must be initialized before calling this
                method.

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

        Arguments:
            None
        Returns:
            float: The sum of all elements in the HashableImage object. The
                sum is rounded to two decimal places for precision.

        Example:
            >>> hashable_image.calculate_sum()

        Note:
            The HashableImage object is assumed to contain numerical values
                only.

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

    @jaxtyped(typechecker=beartype)
    def __repr__(self) -> str:
        """Generate a string representation of the HashableImage object.

        This method constructs a string that includes the mode, dtype, size,
            mean, std, min, max, and filename of the HashableImage object,
            providing a comprehensive summary of the object's properties.

        Arguments:
            self (HashableImage): The instance of the HashableImage object.

        Returns:
            str: A string representation of the HashableImage object,
                including its mode, dtype, size, mean, std, min, max, and
                filename.

        Example:
            >>> image = HashableImage("example.jpg")
            >>> print(image)
            'mode: RGB, dtype: uint8, size: (1920, 1080), mean: 127.5, std:
                20.8, min: 0, max: 255, filename: example.jpg'
        Note:
            The returned string can be used for debugging or logging
                purposes.

        """
        _filename = (
            f" {self.get_filename()}"
            if "/tmp" not in self.get_filename()  # noqa: S108
            else ""
        )
        return f"HashableImage: {self._mode} {self.dtype()} {self.size()} - mean: {self.mean()} std: {self.std()} min {self.min()} max {self.max()}{_filename}"

    @jaxtyped(typechecker=beartype)
    def pil(self) -> Image.Image:
        """Convert the image data to a PIL Image object.

        This method in the 'HashableImage' class transforms the image data
            stored in the instance into a PIL (Python Imaging Library) Image
            object.

        Arguments:
            self (HashableImage): The instance of the 'HashableImage' class.

        Returns:
            Image.Image: A PIL Image object that represents the image data
                stored in the instance.

        Example:
            >>> image = HashableImage()
            >>> pil_image = image.pil()

        Note:
            The PIL Image object returned can be used for further image
                processing or visualization.

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
        """Retrieve the image data as a NumPy array.

        This function does not take any arguments.

        Returns:
            np.array: The image data returned as a NumPy array with the
                specified data type and shape.

        Example:
            >>> get_image_data()

        Note:
            The data type and shape of the returned NumPy array depend on
                the image data.

        """
        if self._mode == "torch":
            return tensor2numpy(
                self._image,
                output_type=(
                    bool if self._image.dtype == torch.bool else np.uint8
                ),
            )
        if self._mode == "numpy":
            return self._image
        return np.asarray(self._image)

    @lru_cache(maxsize=MAX_IMG_CACHE)
    @jaxtyped(typechecker=beartype)
    def tensor(
        self,
    ) -> Float[torch.Tensor, "1 c h w"] | Bool[torch.Tensor, "1 c h w"]:
        """Convert the image data to a torch tensor format.

        This method in the 'HashableImage' class converts the image data
            stored in the object into a torch tensor format.
        It checks the mode of the image data (torch, numpy, or pil) and
            converts it accordingly.

        Arguments:
            self (HashableImage): The instance of the HashableImage class.

        Returns:
            torch.Tensor | bool: A torch tensor representing the image data
                in the format '1 c h w' where c is the number of channels, h
                is the height, and w is the width. It can also return a
                boolean value indicating if the conversion was successful.

        Note:
            - If the mode of the image data is 'torch', the method returns
                the image data as is.
            - If the mode is 'numpy', it converts the numpy array to a torch
                tensor using the numpy2tensor function.
            - If the mode is 'pil', it converts the PIL image to a torch
                tensor using the pil2tensor function.

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

        Example:
            >>> image = HashableImage(...)
            >>> image_bytes = image.bytes()

        Note:
            The bytes object can be used for further processing or
                serialization.

        """
        pil_image = self.pil()
        # BytesIO is a file-like buffer stored in memory
        img_bytes = io.BytesIO()
        # image.save expects a file-like as a argument
        pil_image.save(img_bytes, format="PNG")
        # Turn the BytesIO object back into a bytes object
        return img_bytes.getvalue()

    def b64(self) -> str:
        """Convert the image data to a base64 string.

        This method converts the image data stored in the HashableImage
            object into a base64 string.

        Returns:
            str: The image data as a base64 string.

        Example:
            >>> image = HashableImage(...)
            >>> image_b64 = image.b64()

        Note:
            The base64 string can be used for further processing or
                serialization.

        """
        return base64.b64encode(self.bytes()).decode("utf-8")

    @property
    @jaxtyped(typechecker=beartype)
    def mode(self) -> Literal["pil", "numpy", "torch"]:
        """Retrieve the mode of the HashableImage object.

        This method returns the mode of the HashableImage object. The mode
            can be one of three values: 'pil', 'numpy', or 'torch', each
            representing a different image format.

        Arguments:
            None
        Returns:
            str: The mode of the HashableImage object. This is a string
                indicating whether the image is in 'pil', 'numpy', or
                'torch' format.

        Example:
            >>> image = HashableImage(...)
            >>> image.get_mode()
            'numpy'
        Note:
            This method does not take any arguments.

        """
        return self._mode

    @jaxtyped(typechecker=beartype)
    def is_binary(self) -> bool:
        """Check if the image data in the HashableImage object is binary.

        This method evaluates whether the image data contained within the
        HashableImage object is binary or not.

        Arguments:
            self (HashableImage): The HashableImage object containing image
                data.

        Returns:
            bool: Returns True if the image data is binary, False otherwise.

        Example:
            >>> image = HashableImage(data)
            >>> image.is_binary()

        Note:
            A binary image is a digital image that has only two possible
                values for each pixel.

        """
        if self._mode == "torch":
            return self._image.dtype == torch.bool
        return self._image.dtype == bool

    @jaxtyped(typechecker=beartype)
    def is_rgb(self) -> bool:
        """Check if the image in the HashableImage object is in RGB format.

        This method inspects the image stored in the HashableImage object
            and determines whether it is in RGB format.

        Arguments:
            self (HashableImage): The HashableImage object containing the
                image to be checked.

        Returns:
            bool: True if the image is in RGB format, False otherwise.

        Example:
            >>> image = HashableImage("image.jpg")
            >>> image.is_rgb()

        Note:
            RGB format is a common, three-channel color model used in
                digital imaging.

        """
        if self._mode == "torch":
            return self._image.shape[1] == 3
        return len(self._image.shape) == 3 and self._image.shape[2] == 3

    @property
    def shape(self) -> tuple[int, int] | tuple[int, int, int]:
        """Return the shape of the HashableImage object.

        This method determines and returns the shape of the HashableImage
            object. For binary images, the shape is a tuple of two integers
            representing height and width. For RGB images, the shape is a
            tuple of three integers representing height, width, and
            channels.

        Arguments:
            self (HashableImage): The HashableImage object for which the
                shape needs to be determined.

        Returns:
            Tuple[int, int]: If the image is binary, returns a tuple
                representing (height, width).
            Tuple[int, int, int]: If the image is RGB, returns a tuple
                representing (height, width, 3).

        Raises:
            ValueError: If the image is neither binary nor RGB.

        Example:
            >>> image = HashableImage(...)
            >>> image.get_shape()

        Note:
            The method does not support images other than binary or RGB.

        """
        if self.is_binary():
            return (int(self.size().height), int(self.size().width))
        if self.is_rgb():
            return (int(self.size().height), int(self.size().width), 3)
        msg = f"Image is not binary or rgb. Shape: {self.size()}, this image is {self.dtype()} and it should not happen. Report bug."
        raise ValueError(
            msg,
        )

    @jaxtyped(typechecker=beartype)
    def concat(
        self,
        other: list[HashableImage],
        mode: Literal["horizontal", "vertical"],
    ) -> HashableImage:
        """Concatenate multiple images either horizontally or vertically.

        Arguments:
            self (HashableImage): The instance of the HashableImage class.
            other (List[HashableImage]): A list of HashableImage objects to
                be concatenated with the self image.
            mode (str): A string specifying the concatenation mode. It can
                be 'horizontal' or 'vertical'.

        Returns:
            HashableImage: A new HashableImage object that represents the
                concatenated image based on the specified mode.

        Example:
            >>> img1.concat([img2, img3], "horizontal")

        Note:
            The images in the 'other' list are concatenated in the order
                they appear in the list.

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
        """Retrieve the raw image data stored in the HashableImage object.

        Arguments:
            self (HashableImage): The HashableImage object for which the raw
                image data is to be retrieved.

        Returns:
            Union[Image.Image, np.ndarray, torch.Tensor]: The raw image data
                in various formats such as PIL Image,
            UInt8 numpy array with shape '(h, w, 3)', UInt8 numpy array with
                shape '(h, w)',
            Bool numpy array with shape '(h, w)', Float torch tensor with
                shape '(1, c, h, w)',
            or Bool torch tensor with shape '(1, 1, h, w)'.

        Example:
            >>> img_data = HashableImage.get_raw_image_data()

        Note:
            The returned raw image data format depends on the original
                format of the image stored in the HashableImage object.

        """
        return self._image

    @jaxtyped(typechecker=beartype)
    def logical_and(self, other: HashableImage) -> HashableImage:
        """Perform a logical AND operation with another HashableImage.

        This method takes another HashableImage object as an input and
            performs a
        logical AND operation between the binary representations of the two
            images.
        The resulting image is returned as a new HashableImage object.

        Arguments:
            self ('HashableImage'): The HashableImage object on which the
                method is called.
            other ('HashableImage'): Another HashableImage object to perform
                the logical AND operation with.

        Returns:
            'HashableImage': A new HashableImage object representing the
                result of the logical AND operation between the two input
                images.

        Example:
            >>> img1.and_operation(img2)

        Note:
            The logical AND operation is performed on the binary
                representations of the images.

        """
        if self._mode == "torch":
            return HashableImage(
                torch.logical_and(
                    self.to_binary().tensor(),
                    other.to_binary().tensor(),
                ),
            )
        return HashableImage(
            cv2.bitwise_and(
                self.to_binary().numpy(),
                other.to_binary().numpy(),
            ),
        )

    @jaxtyped(typechecker=beartype)
    def logical_and_reduce(
        self,
        other: list[HashableImage],
    ) -> HashableImage:
        """Perform a logical AND operation on a list of HashableImage objects.

        This method takes a list of HashableImage objects and performs a
            logical AND operation on them. It returns a new HashableImage
            object with the result of the logical AND operation.

        Arguments:
            self (HashableImage): The HashableImage object on which the
                logical AND operation is performed.
            other (List[HashableImage]): A list of HashableImage objects to
                be logically ANDed with the self object.

        Returns:
            HashableImage: A new HashableImage object containing the result
                of the logical AND operation between the self object and the
                other HashableImage objects.

        Example:
            >>> self.logical_and([other_image1, other_image2])

        Note:
            The logical AND operation is performed on the pixel values of
                the HashableImage objects.

        """
        if self._mode == "torch":
            other_value = self.to_binary().tensor()
            for img in other:
                other_value = torch.logical_and(
                    other_value,
                    img.to_binary().tensor(),
                )
            return HashableImage(other_value)
        return HashableImage(
            np.logical_and.reduce(
                [self.to_binary().numpy()]
                + [img.to_binary().numpy() for img in other],
            ),
        )

    @jaxtyped(typechecker=beartype)
    def logical_or(self, other: HashableImage, /) -> HashableImage:
        """Perform a logical OR operation with another HashableImage.

        This method takes another HashableImage object as input and performs
            a logical OR operation between the binary representations of the
            two images. The resulting image is returned as a new
            HashableImage object.

        Arguments:
            self ('HashableImage'): The HashableImage object on which the
                method is called.
            other ('HashableImage'): Another HashableImage object to perform
                the logical OR operation with.

        Returns:
            'HashableImage': A new HashableImage object representing the
                result of the logical OR operation between the two input
                images.

        Example:
            >>> img1.logical_or(img2)

        Note:
            The input images must be of the same dimensions.

        """
        if self._mode == "torch":
            return HashableImage(
                torch.logical_or(
                    self.to_binary().tensor(),
                    other.to_binary().tensor(),
                ),
            )
        return HashableImage(
            cv2.bitwise_or(
                self.to_binary().numpy(),
                other.to_binary().numpy(),
            ),
        )

    @jaxtyped(typechecker=beartype)
    def logical_or_reduce(
        self,
        other: list[HashableImage],
    ) -> HashableImage:
        """Perform a logical OR operation on binary representations of.

            HashableImage objects.

        This method takes a list of HashableImage objects, converts them
            into binary representations and performs a logical OR operation
            on them. The result of this operation is used to create a new
            HashableImage object which is returned.

        Arguments:
            self (HashableImage): The HashableImage object on which the
                logical OR operation is performed.
            other (List[HashableImage]): A list of HashableImage objects to
                be combined using logical OR operation.

        Returns:
            HashableImage: A new HashableImage object representing the
                result of the logical OR operation on the input
                HashableImage objects.

        Example:
            >>> logical_or(self, [img1, img2, img3])

        Note:
            The HashableImage objects in the 'other' list must be of the
                same dimensions as the 'self' object for the logical OR
                operation to be successful.

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

    @jaxtyped(typechecker=beartype)
    def __hash__(self) -> int:
        """Calculate the hash value of a HashableImage object.

        This method generates the hash value of a HashableImage object based
            on its image data.

        Arguments:
            self (HashableImage): The HashableImage object for which the
                hash value is being calculated.

        Returns:
            int: The hash value of the HashableImage object.

        Example:
            >>> hashable_image = HashableImage(image_data)
            >>> hashable_image.calculate_hash()

        Note:
            The hash value is calculated based on the image data of the
                HashableImage object.

        """
        if self._mode == "torch":
            _bytest = hash(self._image)
        else:
            _bytest = hash(self._image.tobytes())
        frozen_set = frozenset(
            {
                self._mode,
                self.dtype(),
                _bytest,
            }
        )
        return hash(frozen_set)

    @jaxtyped(typechecker=beartype)
    def __eq__(self, other: object) -> bool:
        """Compare two HashableImage objects for equality.

        This method determines if two HashableImage objects are equal based
            on their mode and image data.

        Arguments:
            self ('HashableImage'): The HashableImage object calling the
                method.
            other ('HashableImage'): The HashableImage object to compare
                with.

        Returns:
            bool: Returns True if the two HashableImage objects are equal in
                terms of mode and image data. Returns False otherwise.

        Example:
            >>> img1.equals(img2)

        Note:
            The equality is determined based on the mode and image data of
                the HashableImage objects.

        """
        if not isinstance(other, HashableImage):
            return NotImplemented
        if self._mode != other.mode:
            return False
        if self._mode == "torch":
            return torch.equal(self._image, other._image)
        return self._image.tobytes() == other._image.tobytes()

    @lru_cache(maxsize=MAX_IMG_CACHE)
    @jaxtyped(typechecker=beartype)
    def crop_from_mask(
        self,
        mask: HashableImage,
        **kwargs: Any,
    ) -> HashableImage:
        """Crop an image based on a provided mask image.

        Arguments:
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

        Example:
            >>> crop_image(mask_image, border=5, interpolation="bilinear")

        Note:
            The mask image should be a binary image where the regions to
                keep are white and the regions to remove are black.

        """
        kwargs.setdefault("verbose", False)
        return HashableImage(
            crop_from_mask(
                self.to_rgb().numpy(), mask.to_binary().numpy(), **kwargs
            )
        )

    @lru_cache(maxsize=MAX_IMG_CACHE)
    @jaxtyped(typechecker=beartype)
    def crop_from_bbox(
        self,
        bboxes: HashableList[BoundingBox],
    ) -> HashableImage:
        """Crop an image based on the provided bounding boxes.

        This method takes a list of bounding boxes and uses them to crop the
            instance of the HashableImage class. Each bounding box in the
            list should define a region in the image that will be included
            in the cropped image. The order of the bounding boxes in the
            list does not affect the result.

        Arguments:
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

        Example:
            >>> img = HashableImage("image.jpg")
            >>> bboxes = [(10, 10, 50, 50), (100, 100, 50, 50)]
            >>> cropped_img = img.crop_image(bboxes)

        Note:
            If the bounding boxes overlap, the overlapping region will be
                included only once in the cropped image.

        """
        # set bbox to the size of the image in case it is bigger, for both float and int
        return HashableImage(
            crop_from_bbox(
                self.to_rgb().numpy(),
                [bbox.xyxyn for bbox in bboxes.to_list()],
                is_normalized=True,
            )
        )

    @lru_cache(maxsize=MAX_IMG_CACHE)
    @jaxtyped(typechecker=beartype)
    def uncrop_from_bbox(
        self,
        base: HashableImage,
        bboxes: HashableList[BoundingBox],
        *,
        resize: bool = False,
        blend_width: int = 10,
    ) -> HashableImage:
        """Uncrop an image from a specified list of bounding boxes using a.

            Least Recently Used (LRU) cache.

        This method in the HashableImage class uncrops an image from regions
            specified by a list of bounding boxes.
        It returns the uncropped image as a HashableImage object.

        Arguments:
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

        Example:
            >>> uncrop_from_bboxes(self, base, bboxes, resize=False)

        Note:
            This method uses a Least Recently Used (LRU) cache for
                performance optimization.

        """
        is_normalized = False
        _bboxes = [bbox.xyxy for bbox in bboxes.to_list()]
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
    def mask2points(
        self,
        npoints: int = 100,
        *,
        normalize: bool = False,
        rng: np.random.Generator | None = None,
        output: Literal["xy", "yx"] = "xy",
    ) -> Points:
        """Convert a mask image to a list of points.

        This method converts a mask image to a list of points. The number of
            points to generate is specified by the 'npoints' parameter.

        Arguments:
            self (HashableImage): The HashableImage object representing the
                mask image.
            npoints (int): The number of points to generate. Defaults to 100.
            normalize (bool): A boolean flag indicating whether to normalize
                the points. Defaults to False.
            rng (np.random.Generator): A random number generator. Defaults to
                None.
            output (str): A string specifying the output format. It can be
                'xy' or 'yx'. Defaults to 'xy'.

        Returns:
            List[Tuple[int, int]] | List[Tuple[float, float]]: A list of
                points generated from the mask image.

        Example:
            >>> mask2points(self, npoints=100, normalize=False, rng=None,
                output='xy')

        Note:
            The mask2points function generates points from the mask image.
                The number of points to generate is specified by the
                'npoints' parameter.

        """
        _points = mask2points(
            self.to_binary().numpy(),
            npoints=npoints,
            normalize=normalize,
            rng=rng,
            output=output,
        )
        _points_np = np.asarray(_points).astype(np.float32)
        return Points(
            _points_np, is_normalized=normalize, image_size=self.size()
        )

    @lru_cache(maxsize=MAX_IMG_CACHE)
    @jaxtyped(typechecker=beartype)
    def maskidx2bbox(
        self,
        *,
        margin: float = 0.0,
        normalized: bool = False,
        verbose: bool = False,
        closing: tuple[int, int] = (0, 0),
        opening: tuple[int, int] = (0, 0),
        area_threshold: float = 0.0,
        number_of_objects: int = -1,
    ) -> HashableList[BoundingBox]:
        """Convert a mask image with multiple objects to a list of bounding boxes.

        This method takes a mask image with multiple objects and converts it
            into a list of bounding boxes. Each bounding box represents a
            distinct object in the mask.

        Returns:
            A list of bounding boxes.

        """
        maskidx: UInt8[np.uint8, "h w"] = self.numpy()
        if len(maskidx.shape) == 2:
            unique_values: list[float] = self.unique_values()[
                0
            ]  # between 0 and 1
            bboxes: HashableList[BoundingBox] = HashableList([])
            for i in unique_values:
                if i == 0:
                    continue
                mask = HashableImage(maskidx == int(i * 255))
                bboxes.append(
                    mask.mask2bbox(
                        margin=margin,
                        normalized=normalized,
                        merge=True,
                        verbose=verbose,
                        closing=closing,
                        opening=opening,
                        area_threshold=area_threshold,
                        number_of_objects=number_of_objects,
                    )[0]
                )
            return bboxes
        msg = "maskidx must be a 2D array"
        raise ValueError(msg)

    @lru_cache(maxsize=MAX_IMG_CACHE)
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
    ) -> HashableList[BoundingBox]:
        """Convert a mask image to a bounding box in HashableList format.

        This method takes an instance of HashableImage class and additional
            keyword arguments,
        and applies the mask2bbox function to convert a mask image into a
            bounding box.
        The bounding box coordinates are then returned in a HashableList
            format.

        Arguments:
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

        Example:
            >>> mask_to_bbox(self, **kwargs)

        Note:
            The mask2bbox function must be compatible with the provided
                kwargs.

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
        all_boxes: HashableList[BoundingBox] = HashableList([])
        for box in _bbox:
            all_boxes.append(
                BoundingBox(
                    xmin=box[0],
                    ymin=box[1],
                    xmax=box[2],
                    ymax=box[3],
                    image_size=self.size(),
                )
            )
        return all_boxes

    @lru_cache(maxsize=MAX_IMG_CACHE)
    @jaxtyped(typechecker=beartype)
    def mask2squaremask(self, **kwargs: Any) -> HashableImage:
        """Convert the mask of a HashableImage object to a square mask.

        This method uses the mask2squaremask function from the image_tools
            module to convert the mask of the HashableImage object to a
            square mask.

        Arguments:
            self (HashableImage): The HashableImage object for which the
                mask needs to be converted to a square mask.
            **kwargs: Additional keyword arguments that can be passed to the
                mask2squaremask function from the image_tools module.

        Returns:
            HashableImage: A new HashableImage object with the square mask
                generated from the original mask.

        Example:
            >>> image = HashableImage(...)
            >>> square_mask_image = image.convert_to_square_mask()

        Note:
            The mask2squaremask function requires certain keyword arguments.
                Ensure these are passed to this method.

        """
        return HashableImage(
            mask2squaremask(self.to_binary().numpy(), **kwargs)
        )

    @lru_cache(maxsize=MAX_IMG_CACHE)
    @jaxtyped(typechecker=beartype)
    def blend(
        self,
        mask: HashableImage,
        alpha: float,
        *,
        with_bbox: bool,
        merge_bbox: bool = True,
    ) -> HashableImage:
        """Blend the current HashableImage object with another using a mask,.

            alpha value, and other parameters.

        Arguments:
            mask (HashableImage): The HashableImage object representing the
                mask used for blending.
            alpha (float): The transparency level of the blending operation
                (0.0 - 1.0).
            with_bbox (bool): Whether to include bounding box information in
                the blending operation.
            merge_bbox (bool, optional): Whether to merge bounding boxes
                during blending. Defaults to True.

        Returns:
            HashableImage: The HashableImage object resulting from the
                blending operation.

        Example:
            >>> blend(mask, 0.5, with_bbox=True, merge_bbox=False)

        Note:
            The blend function modifies the original HashableImage object.
                To keep the original intact, make a copy before blending.

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
    def draw_points(
        self,
        points: Points,
        color: tuple[int, int, int],
        radius: int,
        thickness: int,
        with_label: list[str] | None = None,
    ) -> HashableImage:
        """Draw circles at specified points on an image.

        Arguments:
            points (Points): A Points object containing the points at which
                circles are to be drawn.
            color (Tuple[int, int, int]): A tuple of three integers
                representing the RGB color values of the circle.
            radius (int): An integer representing the radius of each circle
                to be drawn.
            thickness (int): An integer representing the thickness of each
                circle's outline.
            from_normalized (bool, optional): A boolean flag indicating
                whether the points are normalized. If True, the points are
                assumed to be in the range [0, 1]. If False, the points are
                assumed to be in the range [0, width] and [0, height]. Defaults
                to True.
            epsilon (float, optional): A small value used for boundary
                checking. Defaults to 0.1.

        Returns:
            HashableImage: A new HashableImage object with the circles drawn
                at the specified points.

        Example:
            >>> draw_circles_on_image(points, (255, 0, 0), 5, 2, 0.1)

        Note:
            The points are assumed to be within the bounds of the image. If
                a point is near the boundary, epsilon is used to check if a
                circle can be drawn without crossing the image boundary.

        """
        canvas = self.numpy().copy()
        # points normalized
        _points = points.list_tuple_int()
        for idx, point in enumerate(_points):
            x, y = point
            cv2.circle(canvas, (x, y), radius, color, thickness)
            if with_label:
                cv2.putText(
                    canvas,
                    str(with_label[idx]),
                    (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    max(int(radius / 10), 1),
                    color,
                    2,
                )
        return HashableImage(canvas)

    @jaxtyped(typechecker=beartype)
    def morphologyEx(  # noqa: N802
        self,
        operation: Literal["erode", "dilate", "open", "close"],
        kernel: np.ndarray,
    ) -> HashableImage:
        """Perform morphological operations on an image.

        This function applies a specified morphological operation to the
            image using a given kernel.

        Arguments:
            operation (str): A string representing the morphological
                operation to be performed. It can be one of the following:
                'erode', 'dilate', 'open', or 'close'.
            kernel (np.array): A NumPy array representing the structuring
                element for the operation.

        Returns:
            HashableImage: A new instance of HashableImage with the
                morphological operation applied to the image.

        Example:
            >>> morphological_operation(1, np.array([[1, 1, 1], [1, 1, 1],
                [1, 1, 1]]))

        Note:
            The operation argument should correspond to a valid
                morphological operation type.

        """
        # https://docs.opencv.org/4.x/d9/d61/tutorial_py_morphological_ops.html
        _operation = getattr(cv2, f"MORPH_{operation.upper()}")
        return HashableImage(
            morphologyEx(self.to_binary().numpy(), _operation, kernel),
        )

    @jaxtyped(typechecker=beartype)
    def draw_polygon(
        self,
        points: list[Points],
        alpha: float = 0.5,
        add_text: str = "",
        color_text: tuple[int, int, int] = (255, 255, 0),
    ) -> HashableImage:
        """Draw a polygon on an image and return the modified image.

        This method in the 'HashableImage' class draws a polygon on an image
            using the provided points. It also allows for an optional text
            to be added to the image.

        Arguments:
            points (List[Points]): A list of Points objects containing the
                coordinates of the points that define the polygon.
            alpha (float, optional): A float value representing the
                transparency of the polygon. Defaults to 0.5.
            add_text (str, optional): A string representing the text to be
                added to the image. Defaults to ''.
            color_text (tuple[int, int, int], optional): A tuple of three
                integers representing the RGB color values of the text.
                Defaults to (255, 255, 0).

        Returns:
            HashableImage: A 'HashableImage' object with the polygon drawn
                on it and optional text added.

        Example:
            >>> draw_polygon([(10, 10), (20, 20), (10, 20)], 0.6, "Hello")

        Note:
            The points should be in the format (x, y) where x and y are
                integers. The alpha value should be between 0 and 1.

        """
        mask = HashableImage(
            polygon_to_mask(
                [point.list_tuple_int() for point in points],
                image_shape=(int(self.size().height), int(self.size().width)),
            ),
        )
        out = self.blend(mask, alpha, with_bbox=True)
        min_x = min([point.min_x() for point in points])
        min_y = min([point.min_y() for point in points])
        if add_text:
            # add text to the image in the upper left corner of the polygon
            out = out.draw_text(
                add_text,
                (min_x, min_y),
                font_size=max(self.size().min() * 0.01, 30.0),
                color=color_text,
            )
        return out

    @jaxtyped(typechecker=beartype)
    def draw_bbox(
        self,
        bbox: BoundingBox,
        alpha: float = 0.5,
        add_text: str = "",
        color: tuple[int, int, int] = (255, 255, 0),
    ) -> HashableImage:
        """Draw a bounding box on the given image and optionally add text.

        Arguments:
            image (HashableImage): The image on which the bounding box is to
                be drawn.
            bbox (Tuple[int, int, int, int]): A tuple of four integers
                representing the bounding box coordinates (x_min, y_min,
                x_max, y_max).
            alpha (float, optional): A float value representing the
                transparency of the bounding box. Defaults to 0.5.
            add_text (str, optional): A string representing the text to be
                added to the image. Defaults to an empty string.

        Returns:
            HashableImage: The original image with the bounding box drawn on
                it and optional text added.

        Example:
            >>> draw_bbox_on_image(image, (10, 20, 30, 40), 0.7, "Object")

        Note:
            The bounding box coordinates are assumed to be valid coordinates
                within the dimensions of the input image.

        """
        if bbox.is_normalized():
            msg = "The bounding box should be in absolute coordinates, not normalized."
            raise ValueError(
                msg,
            )
        mask = HashableImage(bbox2mask([bbox.xyxyn], self.size()))
        out = self.blend(mask, alpha, with_bbox=True, merge_bbox=True)
        if add_text:
            # add text to the image in the upper left corner of the bbox
            x_min, y_min, _, _ = bbox.xyxy
            out = out.draw_text(
                add_text,
                (int(x_min), int(y_min)),
                font_size=max(self.size().min() * 0.01, 30.0),
                color=color,
            )
        return out

    @jaxtyped(typechecker=beartype)
    def draw_lines(
        self,
        lines: list[tuple[tuple[int, int], tuple[int, int]]],
        color: tuple[int, int, int],
        thickness: int,
    ) -> HashableImage:
        """Draw specified lines on an image.

        This method takes a list of line coordinates, a RGB color tuple, and
            a thickness value as input,
        and draws the specified lines on the image.

        Arguments:
            lines (List[Tuple[int, int]]): A list of tuples containing the
                start and end points (in pixels)
                                            of each line to be drawn.
            color (Tuple[int, int, int]): A tuple representing the RGB color
                values (0-255) for the lines.
            thickness (int): An integer value representing the thickness of
                the lines to be drawn in pixels.

        Returns:
            HashableImage: A new HashableImage object with the specified
                lines drawn on it.

        Example:
            >>> draw_lines([(0, 0), (100, 100)], (255, 0, 0), thickness=2)

        Note:
            The start and end points of the lines are specified in pixels.
            The color is specified as an RGB tuple.
            The thickness is specified in pixels.

        """
        canvas = self.numpy()
        for line in lines:
            cv2.line(canvas, line[0], line[1], color, thickness)
        return HashableImage(canvas)

    @jaxtyped(typechecker=beartype)
    def draw_text(
        self,
        text: str,
        coord_xy: tuple[int, int],
        font_size: float = 20.0,
        color: tuple[int, int, int] = (255, 255, 255),
    ) -> HashableImage:
        """Draws text on the HashableImage object at the specified coordinates.

            with the given font size and color.

        Arguments:
            text (str): The text to be drawn on the image.
            coord_xy (tuple[int, int]): The coordinates (x, y) where the
                text will be drawn.
            font_size (float, optional): The font size of the text. Defaults
                to 20.0.
            color (tuple[int, int, int], optional): The RGB color tuple for
                the text. Defaults to white (255, 255, 255).

        Returns:
            HashableImage: A new HashableImage object with the text drawn on
                it at the specified coordinates.

        Example:
            >>> draw_text_on_image("Hello, World!", (10, 10),
                font_size=15.0, color=(255, 0, 0))

        Note:
            The text is drawn on a copy of the original HashableImage
                object, so the original image remains unchanged.

        """
        return HashableImage(
            draw_text(
                self.to_rgb().numpy(),
                text,
                coord_xy,
                font_size=font_size,
                color=color,
            ),
        )

    @jaxtyped(typechecker=beartype)
    def center_pad(
        self,
        image_size: ImageSize,
        fill: int = 0,
    ) -> HashableImage:
        """Center pad an image to a specified size with a specified fill value.

        This method in the HashableImage class is used to center pad an
            image to the given size, using the provided fill value.

        Arguments:
            image_size (Tuple[int, int]): A tuple representing the desired
                size of the image after center padding.
            fill (int): An integer value representing the fill value to be
                used for padding. Defaults to 0.

        Returns:
            HashableImage: A new HashableImage object with the image center
                padded according to the specified image_size and fill value.

        Example:
            >>> image = HashableImage(...)
            >>> padded_image = image.center_pad((500, 500), fill=255)

        Note:
            The padding is applied equally on all sides to maintain the
                image's center.

        """
        return HashableImage(
            center_pad(self.to_rgb().numpy(), image_size, fill)
        )

    @jaxtyped(typechecker=beartype)
    def get_canny_edge(
        self, threshold: tuple[int, int] = (100, 200), *, to_gray: bool = False
    ) -> HashableImage:
        """Get the Canny edge detection of the image.

        This method in the HashableImage class is used to get the Canny edge
            detection of the image.

        Arguments:
            threshold (Tuple[int, int]): A tuple representing the lower and
                upper threshold values for the edge detection. Defaults to
                (100, 200).
            to_gray (bool): A boolean flag indicating whether to convert the
                image to grayscale before applying edge detection. Defaults
                to False.

        Returns:
            HashableImage: A new HashableImage object with the Canny edge
                detection applied.

        """
        return HashableImage(
            get_canny_edge(self.to_rgb().numpy(), threshold, to_gray=to_gray)
        )

    @jaxtyped(typechecker=beartype)
    def differential_mask(
        self,
        dilation: int,
        force_steps: int | None = None,
        scale_nonmask: float | None = None,
        *,
        invert: bool = False,
    ) -> HashableImage:
        """Get the differential mask of the image.

        This method in the HashableImage class is used to get the differential
            mask of the image.

        Arguments:
            dilation (int): An integer representing the dilation value for
                the differential mask.
            force_steps (int, optional): An integer representing the number
                of steps for the differential mask. Defaults to None.
            scale_nonmask (float, optional): A float value representing the
                scale for the non-masked regions. Defaults to None.
            invert (bool, optional): A boolean flag indicating whether to
                invert the mask. Defaults to False.

        Returns:
            HashableImage: A new HashableImage object with the differential
                mask applied.

        """
        return HashableImage(
            differential_mask(
                self.to_binary().numpy(),
                dilation,
                force_steps=force_steps,
                scale_nonmask=scale_nonmask,
                invert=invert,
            )
        )

    @jaxtyped(typechecker=beartype)
    def group_regions_binary(
        self,
        *,
        closing: tuple[int, int],
        margin: float = 0.0,
        area_threshold: float = 0.0,
    ) -> list[HashableImage]:
        """Group regions in a binary image.

        This method groups regions in a binary image based on the specified
            parameters.

        Arguments:
            closing (Tuple[int, int]): A tuple representing the kernel size for
                the closing operation.
            margin (float, optional): A float value representing the margin for
                grouping regions. Defaults to 0.0.
            area_threshold (float, optional): A float value representing the
                area threshold for grouping regions. Defaults to 0.0.

        Returns:
            List[HashableImage]: A list of HashableImage objects representing
                the grouped regions.

        """
        return [
            HashableImage(region)
            for region in group_regions_from_binary(
                self.to_binary().numpy(),
                closing=closing,
                margin=margin,
                area_threshold=area_threshold,
            )
        ]

    @staticmethod
    @jaxtyped(typechecker=beartype)
    def make_image_grid(
        images: HashableDict[str, HashableList[HashableImage]],
        *,
        orientation: Literal["horizontal", "vertical"] = "horizontal",
        with_text: bool = False,
    ) -> HashableImage:
        """Arrange a dictionary of images into a grid either horizontally or.

            vertically.

        This static method in the 'HashableImage' class takes a dictionary
            of images and arranges them in a grid.
        Images are padded with black to match the maximum height and width.
            An optional text label can be included on the grid.

        Arguments:
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

        Example:
            >>> make_image_grid(images, "horizontal", with_text=True)

        Note:
            The images are padded with black to match the maximum height and
                width in the grid.

        """
        # all list should have the same number of images
        image_as_list: dict[str, list[HashableImage]] = cast(
            dict[str, list[HashableImage]],
            images.to_dict(),
        )
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

        return HashableImage(grid)

    @jaxtyped(typechecker=beartype)
    def set_minmax(self, _min: float, _max: float, /) -> HashableImage:
        """Set the minimum and maximum values of the image.

        This method sets the minimum and maximum values of the image to the
            specified values.

        Arguments:
            min (float): The minimum value to set for the image.
            max (float): The maximum value to set for the image.

        Returns:
            None

        Example:
            >>> image.set_minmax(0.0, 1.0)

        Note:
            The minimum and maximum values are used to normalize the image
                data.

        """
        data = self.tensor()
        data = (data - data.min()) / (data.max() - data.min())
        data = data * (_max - _min) + _min
        return HashableImage(data)

    @jaxtyped(typechecker=beartype)
    def __setitem__(
        self, mask: HashableImage, value: float, /
    ) -> HashableImage:
        """Set the pixel values of the image based on a mask.

        This method sets the pixel values of the image to a specified value
            based on a mask.

        Arguments:
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
        image_pt = self.tensor()
        mask_pt = mask.to_binary().tensor()
        image_pt[mask_pt.expand_as(image_pt)] = value
        return HashableImage(image_pt)


class HashableDict(MutableMapping[_KT, _VT]):
    """Hashable dictionary class."""

    def __init__(self, data: dict[_KT, _VT]) -> None:
        """Initialize an instance of the HashableDict class.

        This method converts nested dictionaries and lists within the input
            dictionary into HashableDict and HashableList objects,
            respectively, to initialize an instance of the HashableDict
            class.

        Arguments:
            self (HashableDict): The instance of the HashableDict class.
            data (dict): A dictionary containing key-value pairs where the
                values can be dictionaries or lists.

        Returns:
            None
        Example:
            >>> hash_dict = HashableDict({'key1': {'nested_key': 'value'},
                'key2': ['item1', 'item2']})

        Note:
            The HashableDict class is designed to be used in cases where a
                dictionary needs to be used as a key in another dictionary
                or added to a set, scenarios which require the dictionary to
                be hashable.

        """
        new_data: dict[_KT, _VT] = {}
        for k, v in data.items():
            if isinstance(v, dict):
                new_data[k] = cast(_VT, HashableDict(v))
            elif isinstance(v, list):
                new_data[k] = cast(_VT, HashableList(v))
            elif isinstance(v, HashableDict):
                new_data[k] = cast(_VT, HashableDict(v.to_dict()))
            elif isinstance(v, HashableList):
                new_data[k] = cast(_VT, HashableList(v.to_list()))
            else:
                new_data[k] = v
        self.__data = new_data

    def __hash__(self) -> int:
        """Calculate the hash value of a HashableDict object.

        This method computes the hash value of a HashableDict object based
            on its items. The hash value is determined by
        applying a hash function to the items of the HashableDict.

        Arguments:
            self (HashableDict): The HashableDict object for which the hash
                value is being calculated.

        Returns:
            int: The hash value of the HashableDict object.

        Example:
            >>> hashable_dict = HashableDict({'key1': 'value1', 'key2':
                'value2'})
            >>> hashable_dict.calculate_hash()

        Note:
            The hash function used may vary based on the Python interpreter
                and its version.

        """
        items = {}
        for k, v in self.__data.items():
            if isinstance(v, np.ndarray | Image.Image):
                items[k] = hash(v.tobytes())
            else:
                items[k] = hash(v)
        return hash(frozenset(items.items()))

    def __eq__(self, other: object) -> bool:
        """Compare two HashableDict instances for equality.

        This method checks if the data in the calling HashableDict instance
            is equal to the data in another HashableDict instance.

        Arguments:
            self ('HashableDict'): The instance of HashableDict calling the
                method.
            other ('HashableDict'): The other instance of HashableDict to
                compare with.

        Returns:
            bool: Returns True if the two HashableDict instances have the
                same data, otherwise returns False.

        Example:
            >>> hash_dict1 = HashableDict({'key1': 'value1', 'key2':
                'value2'})
            >>> hash_dict2 = HashableDict({'key1': 'value1', 'key2':
                'value2'})
            >>> hash_dict1.equals(hash_dict2)
            True

        """
        if not isinstance(other, HashableDict):
            return NotImplemented
        return self.__data == other.__data

    def to_dict(
        self,
    ) -> dict[_KT, _VT]:
        """Convert the HashableDict object into a dictionary.

        This method recursively converts any nested HashableDict or
            HashableList objects into standard Python dictionaries or lists,
            respectively.
        Arguments: None
        Returns:
            dict: A dictionary containing the key-value pairs of the
                HashableDict object. Any nested HashableDict or HashableList
                objects are converted into dictionaries or lists,
                respectively.

        Example:
            >>> hashable_dict = HashableDict({"key": "value"})
            >>> hashable_dict.to_dict()
            {'key': 'value'}

        Note:
            This method is useful when a standard Python dictionary
                representation of the HashableDict object is required.

        """
        to_dict: dict[_KT, _VT] = {}
        for k, v in self.__data.items():
            if isinstance(v, HashableDict):
                to_dict[k] = cast(_VT, v.to_dict())
            elif isinstance(v, HashableList):
                to_dict[k] = cast(_VT, v.to_list())
            else:
                to_dict[k] = v
        return to_dict

    def copy(self) -> HashableDict[_KT, _VT]:
        """Create a copy of the HashableDict object.

        This method generates an exact replica of the current HashableDict
            object,
        preserving all key-value pairs in the new instance.

        Arguments:
            self (HashableDict): The HashableDict object to be duplicated.

        Returns:
            HashableDict: A new HashableDict object that mirrors the
                original.

        Example:
            >>> original_dict = HashableDict({"key": "value"})
            >>> cloned_dict = original_dict.clone()

        """
        return HashableDict(self.__data.copy())

    def values(self) -> Iterable[_VT]:  # type: ignore[explicit-override, override]
        """Retrieve all values from a HashableDict.

        This method iterates over the HashableDict and returns a list
            containing all the values.

        Returns:
            List[Any]: A list containing all the values in the HashableDict.

        Example:
            >>> hashable_dict = HashableDict({'key1': 'value1', 'key2':
                'value2'})
            >>> hashable_dict.values()
            ['value1', 'value2']

        Note:
            The order of the values in the returned list is not guaranteed
                to match the order of the keys in the HashableDict.

        """
        return self.__data.values()

    def keys(self) -> Iterable[_KT]:  # type: ignore[explicit-override, override]
        """Retrieve all keys from a HashableDict.

        This method iterates over the HashableDict and returns a list of all
            keys present in the dictionary.

        Returns:
            List[Hashable]: A list containing all keys in the HashableDict.

        Example:
            >>> hash_dict = HashableDict({"a": 1, "b": 2})
            >>> hash_dict.keys()
            ['a', 'b']

        Note:
            The order of keys in the returned list is not guaranteed.

        """
        return self.__data.keys()

    def items(self) -> Iterable[tuple[_KT, _VT]]:  # type: ignore[explicit-override, override]
        """Retrieve all key-value pairs from the HashableDict.

        This method returns an iterator over the (key, value) pairs in the
            HashableDict.

        Returns:
            Iterator[Tuple[Hashable, Any]]: An iterator over the (key,
                value) pairs in the HashableDict.

        Example:
            >>> hdict = HashableDict({"a": 1, "b": 2})
            >>> list(hdict.items())
            [('a', 1), ('b', 2)]

        """
        return self.__data.items()

    def __repr__(self) -> str:
        """Return a string representation of the HashableDict object.

        This method generates a string that provides a readable
            representation of the HashableDict object. It can be used for
            debugging and logging purposes.

        Arguments:
            self (HashableDict): The instance of HashableDict object to be
                represented.

        Returns:
            str: A string representation of the HashableDict object.

        Example:
            >>> hash_dict = HashableDict({"key": "value"})
            >>> print(hash_dict)
            "{'key': 'value'}"
        Note:
            The returned string representation may not be a valid input for
                the HashableDict constructor.

        """
        return f"HashableDict: {self.__data}"

    def __getitem__(self, __name: _KT, /) -> _VT:
        """Retrieve the value associated with a specific key in a HashableDict.

            object.

        Arguments:
            __name (_KT): The key for which the associated value needs to be
                retrieved.

        Returns:
            _VT: The value associated with the specified key in the
                HashableDict object.

        Example:
            >>> hash_dict = HashableDict({'key1': 'value1', 'key2':
                'value2'})
            >>> get_value("key1")
            'value1'
        Note:
            Raises KeyError if the key is not found in the HashableDict
                object.

        """
        if __name not in self.__data:
            msg = f"Key {__name} not found in HashableDict"
            raise KeyError(msg)
        return self.__data[__name]

    def __setitem__(self, __name: _KT, __value: _VT, /) -> None:
        """Set a key-value pair in a HashableDict object.

        This method allows for setting a key-value pair in a HashableDict
            object. It takes a key and a value and associates the value with
            the key in the HashableDict object.

        Arguments:
            __name (str): The key to be set in the HashableDict object.
            __value (Any): The value to be associated with the key in the
                HashableDict object.

        Returns:
            None: This method does not return any value.

        Example:
            >>> hash_dict = HashableDict()
            >>> hash_dict.set_key_value("name", "John Doe")

        Note:
            The key must be of type string and the value can be of any type.

        """
        self.__data[__name] = __value

    def __delitem__(self, __name: _KT, /) -> None:
        """Delete an item from the HashableDict class.

        This method removes an item from the HashableDict class based on the
            provided key.

        Arguments:
            __name (_KT): The key of the item to be deleted from the
                HashableDict.

        Returns:
            None: This method does not return anything, it simply removes
                the item from the HashableDict.

        Example:
            >>> hash_dict = HashableDict({1: "a", 2: "b", 3: "c"})
            >>> hash_dict.delete_item(2)

        Note:
            After this method is called, the HashableDict will no longer
                contain an item with the provided key.

        """
        del self.__data[__name]

    def __iter__(self) -> Iterator[_KT]:
        """Make instances of the HashableDict class iterable.

        This method makes instances of the HashableDict class iterable by
            returning an iterator over the keys of the dictionary.

        Arguments:
            self (HashableDict): The instance of the HashableDict class.

        Returns:
            Iterator: An iterator object that can traverse through all the
                keys of the dictionary stored in the HashableDict instance.

        Example:
            >>> hash_dict = HashableDict({"a": 1, "b": 2})
            >>> for key in hash_dict:
            ...     print(key)

        Note:
            The iterator returned by this method allows only traversal, not
                element modification.

        """
        return iter(self.__data)

    def __len__(self) -> int:
        """Return the length of the HashableDict object.

        This method computes the length of the HashableDict object by
            returning the length of the data stored within it.

        Arguments:
            None
        Returns:
            int: An integer representing the length of the data stored
                within the HashableDict object.

        Example:
            >>> hash_dict = HashableDict({'key1': 'value1', 'key2':
                'value2'})
            >>> hash_dict.length()
            2

        """
        return len(self.__data)


class HashableList(MutableSequence[_T]):
    """Hashable list class."""

    def __init__(self, data: list[_T]) -> None:
        """Initializes an instance of the HashableList class.

        This method converts any dictionaries or lists within the input list
            to their hashable equivalents
        (HashableDict or HashableList) and stores the modified list in the
            instance.

        Arguments:
            self (HashableList): The instance of the HashableList class.
            data (List[_T]): A list of elements of any type (_T). If the
                elements are dictionaries or lists,
                              they are converted to HashableDict or
                HashableList respectively.

        Returns:
            None
        Example:
            >>> hl = HashableList([{1: "a"}, {2: "b"}, [1, 2, 3]])

        Note:
            The HashableList class is used when you need a list that can be
                used as a dictionary key.
            Regular lists and dictionaries are mutable and cannot be used as
                dictionary keys.

        """
        new_data: list[_T] = []
        for idx in range(len(data)):
            if isinstance(data[idx], dict):
                new_data.append(
                    cast(_T, HashableDict(cast(dict[_KT, _VT], data[idx]))),  # type: ignore[valid-type]
                )
            elif isinstance(data[idx], list):
                new_data.append(
                    cast(_T, HashableList(cast(list[_T], data[idx]))),
                )
            elif isinstance(data[idx], HashableDict):
                new_data.append(
                    cast(
                        _T,
                        HashableDict(
                            cast(dict[_KT, _VT], data[idx].to_dict())  # type: ignore[attr-defined, valid-type]
                        ),
                    ),
                )
            elif isinstance(data[idx], HashableList):
                new_data.append(
                    cast(
                        _T,
                        HashableList(cast(list[_T], data[idx].to_list())),  # type: ignore[attr-defined]
                    ),
                )
            else:
                new_data.append(data[idx])
        self.__data = new_data

    def __hash__(self) -> int:
        """Calculate the hash value of a HashableList object.

        This method computes the hash value of a HashableList object by
            converting its data into a frozenset and then hashing it.

        Arguments:
            self (HashableList): The HashableList object for which the hash
                value needs to be calculated.

        Returns:
            int: The hash value of the HashableList object.

        Example:
            >>> hashable_list = HashableList([1, 2, 3])
            >>> hashable_list.hash()

        Note:
            The HashableList object must contain hashable elements only.

        """
        items = []
        for idx in range(len(self.__data)):
            if isinstance(self.__data[idx], np.ndarray | Image.Image):
                items.append(hash(self.__data[idx].tobytes()))  # type: ignore[attr-defined]
            else:
                items.append(hash(self.__data[idx]))
        return hash(frozenset(items))

    def __eq__(self, other: object) -> bool:
        """Compare the hash values of two HashableList objects.

        This method compares the hash value of the HashableList object
            calling the method (self)
        with the hash value of another HashableList object (other).

        Arguments:
            self ('HashableList'): The HashableList object calling the
                method.
            other ('HashableList'): The HashableList object to compare with.

        Returns:
            bool: Returns True if the hash values of both HashableList
                objects are equal, False otherwise.
                  If the 'other' object is not an instance of HashableList,
                it returns NotImplemented.

        Example:
            >>> h1 = HashableList([1, 2, 3])
            >>> h2 = HashableList([1, 2, 3])
            >>> h1.compare_hashes(h2)
            True
        Note:
            This method uses the __hash__ method of the HashableList class
                to generate the hash values.

        """
        if not isinstance(other, HashableList):
            return NotImplemented
        return self.__hash__() == other.__hash__()

    def to_list(self) -> list[_T]:
        """Convert the HashableList object into a regular Python list.

        This method recursively converts any nested HashableDict or
            HashableList objects into their respective list representations.
        Arguments: None
        Returns:
            List: A list containing the elements of the HashableList object,
                with any nested HashableDict or HashableList objects
                converted into regular Python lists.

        Example:
            >>> hashable_list.to_list()

        Note:
            This method is useful when you need to work with regular Python
                lists instead of HashableList objects.

        """
        to_list = []
        for idx in range(len(self.__data)):
            if isinstance(self.__data[idx], HashableDict):
                to_list.append(
                    cast(
                        _T,
                        cast(
                            HashableDict[_KT, _VT],  # type: ignore[valid-type]
                            self.__data[idx],
                        ).to_dict(),
                    ),
                )
            elif isinstance(self.__data[idx], HashableList):
                to_list.append(
                    cast(
                        _T,
                        cast(HashableList[_T], self.__data[idx]).to_list(),
                    ),
                )
            else:
                to_list.append(self.__data[idx])
        return to_list

    def __repr__(self) -> str:
        """Return a string representation of the HashableList object.

        This method transforms the HashableList object into a string format.
            The string contains the class name 'HashableList' followed by
            the data stored in the object.

        Arguments:
            self (HashableList): The HashableList object itself.

        Returns:
            str: A string representation of the HashableList object. The
                string includes the class name 'HashableList' and the data
                stored in the object.

        Example:
            >>> hl = HashableList([1, 2, 3])
            >>> print(hl)
            HashableList: [1, 2, 3]

        """
        return f"HashableList: {self.__data}"

    def copy(self) -> HashableList[_T]:
        """Create a copy of the HashableList object.

        This method generates a new HashableList object by duplicating the
            data stored within the original list.

        Arguments:
            self (HashableList): The HashableList object to be copied.

        Returns:
            HashableList: A new HashableList object containing the same data
                as the original list.

        Example:
            >>> original_list = HashableList([1, 2, 3])
            >>> copied_list = original_list.copy()

        """
        return HashableList(self.__data.copy())

    def __iter__(self) -> Iterator[_T]:
        """Enable iteration over instances of the HashableList class.

        This method makes instances of the HashableList class iterable,
            allowing
        them to be used in a for loop or any other iteration context.

        Arguments:
            self (HashableList): The instance of the HashableList class.

        Returns:
            Iterator: An iterator object that enables iteration over the
                data
            stored in the HashableList instance.

        Example:
            >>> hash_list = HashableList([1, 2, 3])
            >>> for i in hash_list:
            ...     print(i)

        Note:
            This is a special method, part of the Python data model. It is
                not
            meant to be called directly, but implicitly, by Python's
                iteration
            tools like 'for' loops.

        """
        return iter(self.__data)

    @overload
    def __getitem__(self, __index: SupportsIndex, /) -> _T: ...

    @overload
    def __getitem__(self, __index: slice, /) -> HashableList[_T]: ...

    def __getitem__(
        self,
        __index: SupportsIndex | slice,
        /,
    ) -> _T | HashableList[_T]:
        """Retrieve an element or a slice of elements from the HashableList.

            object.

        This method allows for retrieving an element or a slice of elements
            from the HashableList object.

        Arguments:
            __index (int | slice): The index or slice to be retrieved from
                the HashableList object.

        Returns:
            Any: The element or slice of elements from the HashableList
                object.

        Example:
            >>> hashable_list = HashableList([1, 2, 3, 4, 5])
            >>> retrieve_element_or_slice(2)
            3
            >>> retrieve_element_or_slice(slice(1, 4))
            [2, 3, 4]

        Note:
            The HashableList object is a list that supports hash operations.

        """
        if isinstance(__index, slice):
            return HashableList(self.__data[__index])
        return self.__data[__index]

    @overload
    def __setitem__(self, key: SupportsIndex, value: _T, /) -> None: ...

    @overload
    def __setitem__(
        self,
        key: SupportsIndex,
        value: Iterable[_T],
        /,
    ) -> None: ...

    @overload
    def __setitem__(self, key: slice, value: Iterable[_T], /) -> None: ...

    def __setitem__(
        self,
        key: SupportsIndex | slice,
        value: _T | Iterable[_T],
    ) -> None:
        """Set the value of an item or slice in a HashableList object.

        This method allows for setting the value of an item or slice in a
            HashableList object.

        Arguments:
            key (Union[int, slice]): The index or slice to set the value
                for.
            value (Any): The value to set at the specified index or slice.

        Returns:
            None: This method does not return anything. It modifies the
                HashableList object in-place.

        Example:
            >>> hash_list = HashableList([1, 2, 3])
            >>> hash_list.set_item(1, "a")

        Note:
            The HashableList object must be mutable, otherwise this
                operation will raise an exception.

        """
        data_list = self.to_list()
        if isinstance(value, HashableList):
            value = value.to_list()
        data_list[key] = value  # type: ignore[assignment, index]
        self.__data = HashableList(data_list).__data

    def __len__(self) -> int:
        """Calculate the length of the HashableList object.

        This method determines the length of the HashableList object by
            returning the length of the data stored within the object.

        Arguments:
            self (HashableList): The HashableList object for which the
                length needs to be determined.

        Returns:
            int: An integer representing the length of the HashableList
                object.

        Example:
            >>> hl = HashableList([1, 2, 3])
            >>> hl.len()
            3

        """
        return len(self.__data)

    @overload
    def __delitem__(self, __index: int, /) -> None: ...

    @overload
    def __delitem__(self, __index: slice, /) -> None: ...

    def __delitem__(self, __index: int | slice, /) -> None:
        """Delete an item or a slice of items from a HashableList object.

        This method removes a single item if an integer is provided as an
            index or a range of items if a slice object is provided.

        Arguments:
            __index (Union[int, slice]): The index of the item to be deleted
                if it's an integer, or a slice object representing a range
                of items to be deleted.

        Returns:
            None: This method does not return anything.

        Example:
            >>> hashable_list = HashableList([1, 2, 3, 4, 5])
            >>> delete_item(2)
            >>> print(hashable_list)
            [1, 2, 4, 5]

        Note:
            The HashableList must support item deletion. If it doesn't, an
                error will be raised.

        """
        del self.__data[__index]

    def insert(self, __index: int, __value: _T, /) -> None:
        """Insert a value at a specified index in a HashableList object.

        Arguments:
            __index (int): An integer representing the index at which the
                value will be inserted.
            __value (_T): The value of any type that will be inserted into
                the HashableList.

        Returns:
            None: This method does not return anything.

        Example:
            >>> hashable_list.insert_value(2, "apple")

        Note:
            If the index is out of range, the value will be added at the end
                of the HashableList.

        """
        self.__data.insert(__index, __value)

    def __mul__(self, other: int) -> HashableList[_T]:
        """Multiply all elements in the HashableList by a specified integer.

        This method iterates over each element in the HashableList,
            multiplies it by the given integer value,
        and returns a new HashableList with the resulting values.

        Arguments:
            self (HashableList): The current HashableList instance.
            other (int): The integer value to multiply the elements by.

        Returns:
            HashableList: A new HashableList object containing the elements
                of the original HashableList
            multiplied by the specified integer value.

        Example:
            >>> hl = HashableList([1, 2, 3])
            >>> hl.multiply_elements(2)
            HashableList([2, 4, 6])

        Note:
            The original HashableList is not modified by this method. A new
                HashableList is returned.

        """
        return HashableList(self.__data * other)


@dataclass(config=ConfigDict(extra="forbid"), kw_only=True)
class ImageCrop:
    """Image crop class."""

    left: float
    """The left coordinate of the crop area."""
    top: float
    """The top coordinate of the crop area."""
    right: float
    """The right coordinate of the crop area."""
    bottom: float
    """The bottom coordinate of the crop area."""

    def __post_init__(self) -> None:
        """Validate and initialize the crop values of an image.

        This method is part of the 'ImageCrop' class and is used to validate
            the crop values.
        It checks if the given crop values are valid and sets a flag based
            on whether the values are normalized or not.

        Arguments:
            self (ImageCrop): An instance of the 'ImageCrop' class on which
                the method is called.

        Returns:
            None: This method does not return anything.

        Example:
            >>> image_crop = ImageCrop()
            >>> image_crop.__post_init__()

        Note:
            This method is typically called internally within the class and
                not directly by the user.

        """
        if self.left >= self.right:
            msg = f"left must be smaller than right. {self}"
            raise ValueError(msg)
        if self.top >= self.bottom:
            msg = f"top must be smaller than bottom. {self}"
            raise ValueError(msg)
        if self.left < 0 or self.top < 0 or self.right < 0 or self.bottom < 0:
            msg = f"crop values must be positive. {self}"
            raise ValueError(msg)

    def is_normalized(self) -> bool:
        """Check if the crop values are normalized.

        This method checks if the crop values are normalized, i.e., if they
            are between 0 and 1.

        Arguments:
            self (ImageCrop): The ImageCrop object for which the
                normalization is being checked.

        Returns:
            bool: True if the crop values are normalized, False otherwise.

        Example:
            >>> image_crop = ImageCrop(0.1, 0.2, 0.3, 0.4)
            >>> image_crop.is_normalized()
            True

        """
        return all(
            0 <= coord <= 1
            for coord in [self.left, self.top, self.right, self.bottom]
        )

    def __repr__(self) -> str:
        """Generate a string representation of an ImageCrop object.

        This method returns a string representation of an ImageCrop object,
            showcasing the values of its left, top, right,
        and bottom attributes. This representation can be useful for
            debugging or logging purposes.

        Arguments:
            self (ImageCrop): The ImageCrop object for which the string
                representation is being generated.

        Returns:
            str: A string representing the ImageCrop object with its left,
                top, right, and bottom attributes displayed.

        Example:
            >>> print(ImageCrop(10, 20, 30, 40))
            'ImageCrop(left=10, top=20, right=30, bottom=40)'

        """
        return f"ImageCrop(left={self.left}, top={self.top}, right={self.right}, bottom={self.bottom})"

    def __hash__(self) -> int:
        """Calculate the hash value of an ImageCrop object.

        This method calculates the hash value of an ImageCrop object based
            on its attributes.

        Arguments:
            self (ImageCrop): The ImageCrop object for which the hash value
                is being calculated.

        Returns:
            int: The hash value of the ImageCrop object.

        Example:
            >>> image_crop = ImageCrop(10, 20, 30, 40)
            >>> image_crop.__hash__()

        """
        return hash((self.left, self.top, self.right, self.bottom))

    def __eq__(self, other: object) -> bool:
        """Compare two ImageCrop instances for equality.

        This method compares two ImageCrop instances to determine if they
            are equal.

        Arguments:
            self (ImageCrop): The ImageCrop object calling the method.
            other (object): The other object to compare with.

        Returns:
            bool: True if the two ImageCrop instances are equal, False
                otherwise.

        Example:
            >>> image_crop1 = ImageCrop(10, 20, 30, 40)
            >>> image_crop2 = ImageCrop(10, 20, 30, 40)
            >>> image_crop1.__eq__(image_crop2)

        """
        if not isinstance(other, ImageCrop):
            return NotImplemented
        return (
            self.left == other.left
            and self.top == other.top
            and self.right == other.right
            and self.bottom == other.bottom
        )

    def __call__(
        self,
        image: HashableImage,
        /,
    ) -> HashableImage:
        """Crops the given image based on specified coordinates.

        Arguments:
            image (HashableImage): The image to be cropped.
            top_left (Tuple[int, int]): The coordinates of the top left
                corner of the crop area.
            bottom_right (Tuple[int, int]): The coordinates of the bottom
                right corner of the crop area.

        Returns:
            HashableImage: The cropped image.

        Example:
            >>> crop_image(image, (0, 0), (100, 100))

        Note:
            The coordinates are represented as (x, y), where x is the
                horizontal position and y is the vertical position.

        """
        image_pil = image.pil()
        if self.is_normalized():
            image_pil = TF.crop(
                image,
                self.top * image_pil.height,
                self.left * image_pil.width,
                self.bottom * image_pil.height,
                self.right * image_pil.width,
            )
        else:
            image_pil = TF.crop(
                image_pil,
                self.top,
                self.left,
                self.bottom,
                self.right,
            )
        return HashableImage(image_pil)


@dataclass(config=ConfigDict(extra="forbid"), frozen=True)
class BoundingBox:
    """Bounding box class."""

    xmin: float
    """The minimum x-coordinate of the bounding box."""
    ymin: float
    """The minimum y-coordinate of the bounding box."""
    xmax: float
    """The maximum x-coordinate of the bounding box."""
    ymax: float
    """The maximum y-coordinate of the bounding box."""
    image_size: ImageSize | None = None
    """The size of the image containing the bounding box."""

    def __post_init__(self) -> None:
        """Validate the bounding box coordinates."""
        if self.xmin >= self.xmax:
            msg = "xmin must be smaller than xmax."
            raise ValueError(msg)
        if self.ymin >= self.ymax:
            msg = "ymin must be smaller than ymax."
            raise ValueError(msg)
        if self.xmin < 0 or self.ymin < 0 or self.xmax < 0 or self.ymax < 0:
            msg = "Bounding box coordinates must be positive."
            raise ValueError(msg)
        if not self.is_normalized() and self.image_size is None:
            msg = "Image size must be provided for non-normalized bounding boxes."
            raise ValueError(msg)

    def is_normalized(self) -> bool:
        """Check if the bounding box coordinates are normalized.

        This method checks if the bounding box coordinates are normalized
            (i.e., if they are between 0 and 1).

        Arguments:
            self (BoundingBox): The BoundingBox object for which the
                normalization is being checked.

        Returns:
            bool: True if the bounding box coordinates are normalized,
                False otherwise.

        Example:
            >>> bbox = BoundingBox(0.1, 0.2, 0.3, 0.4)
            >>> bbox.is_normalized()
            True

        """
        return all(
            0 <= coord <= 1
            for coord in [self.xmin, self.ymin, self.xmax, self.ymax]
        )

    @property
    def xyxy(self) -> tuple[float, float, float, float]:
        """Calculate the minimum and maximum X and Y coordinates of the.

            bounding box.

        This method from the 'BoundingBox' class generates a tuple
            containing the X and Y coordinates
        of the minimum and maximum points of the bounding box.

        Arguments:
            None
        Returns:
            Tuple[float, float, float, float]: A tuple in the format (min_x,
                min_y, max_x, max_y)
            representing the minimum and maximum X and Y coordinates of the
                bounding box.

        Example:
            >>> bbox = BoundingBox(...)
            >>> bbox.xyxy()
            (min_x, min_y, max_x, max_y)

        Note:
            This method does not take any arguments. It calculates the
                coordinates based on the
            properties of the 'BoundingBox' instance.

        """
        if self.is_normalized() and self.image_size is not None:
            return (
                self.xmin * self.image_size.width,
                self.ymin * self.image_size.height,
                self.xmax * self.image_size.width,
                self.ymax * self.image_size.height,
            )
        if not self.is_normalized():
            return (self.xmin, self.ymin, self.xmax, self.ymax)

        msg = "Image size must be provided for normalized bounding boxes. Use xyxyn instead."
        raise ValueError(msg)

    @property
    def xywh(self) -> tuple[float, float, float, float]:
        """Calculate and return the coordinates and dimensions of a bounding.

            box.

        This function does not take any arguments. It calculates and returns
            a tuple containing
        the x, y coordinates and width, height dimensions of a bounding box.

        Returns:
            Tuple[int, int, int, int]: A tuple containing the x, y
                coordinates and width, height
            dimensions of the bounding box. The values are ordered as (x, y,
                width, height).

        Example:
            >>> get_bounding_box()
            (10, 20, 30, 40)

        Note:
            The function assumes that the bounding box is calculated based
                on some predefined conditions.

        """
        x, y, x2, y2 = self.xyxy
        return x, y, x2 - x, y2 - y

    @property
    def xyxyn(self) -> tuple[float, float, float, float]:
        """Calculate the normalized minimum and maximum X and Y coordinates.

        This method generates a tuple containing the normalized X and Y
            coordinates of the minimum and maximum points of the bounding box.

        Arguments:
            None

        Returns:
            Tuple[float, float, float, float]: A tuple in the format (min_x,
                min_y, max_x, max_y)
            representing the normalized minimum and maximum X and Y
                coordinates of the bounding box.

        Example:
            >>> bbox = BoundingBox(...)
            >>> bbox.xyxyn()

        Note:
            This method does not take any arguments. It calculates the
                coordinates based on the
            properties of the 'BoundingBox' instance.

        """
        if self.is_normalized():
            return self.xmin, self.ymin, self.xmax, self.ymax
        if self.image_size is not None:
            return (
                self.xmin / self.image_size.width,
                self.ymin / self.image_size.height,
                self.xmax / self.image_size.width,
                self.ymax / self.image_size.height,
            )

        msg = "Image size must be provided for non-normalized bounding boxes. Use xyxy instead."
        raise ValueError(msg)

    @property
    def xywhn(self) -> tuple[float, float, float, float]:
        """Calculate and return the normalized coordinates and dimensions of a.

            bounding box.

        This method calculates and returns a tuple containing the normalized
            x, y coordinates and width, height dimensions of a bounding box.

        Returns:
            Tuple[float, float, float, float]: A tuple containing the
                normalized x, y coordinates and width, height dimensions of
                the bounding box. The values are ordered as (x, y, width,
                height).

        Example:
            >>> bbox = BoundingBox(...)
            >>> bbox.xywhn()

        Note:
            This method does not take any arguments. It calculates the
                coordinates based on the properties of the 'BoundingBox'
                instance.

        """
        x, y, w, h = self.xywh
        if self.is_normalized():
            return x, y, w, h
        if self.image_size is not None:
            return (
                x / self.image_size.width,
                y / self.image_size.height,
                w / self.image_size.width,
                h / self.image_size.height,
            )

        msg = "Image size must be provided for non-normalized bounding boxes. Use xywh instead."
        raise ValueError(msg)

    def __str__(self) -> str:
        """Return a string representation of the BoundingBox object.

        This method generates a string that represents the BoundingBox
            object by displaying its minimum and maximum x and y values. No
            arguments are required for this method.

        Returns:
            str: A string representation of the BoundingBox object. The
                string includes the minimum and maximum x and y values.

        Example:
            >>> bbox = BoundingBox(0, 0, 1, 1)
            >>> print(bbox)
            'BoundingBox: Min(x=0, y=0), Max(x=1, y=1)'
        Note:
            This method is typically used for debugging purposes.

        """
        return f"xmin: {self.xmin}, ymin: {self.ymin}, xmax: {self.xmax}, ymax: {self.ymax}"

    def __hash__(self) -> int:
        """Calculate the hash value for a BoundingBox object based on its.

            attributes.

        Arguments:
            self (BoundingBox): The BoundingBox object for which the hash
                value is being calculated.

        Returns:
            int: The hash value of the BoundingBox object.

        Example:
            >>> box = BoundingBox()
            >>> calculate_hash(box)

        Note:
            The hash value is calculated based on the attributes of the
                BoundingBox object.

        """
        return hash(HashableDict(self.__dict__))

    def __eq__(self, other: object) -> bool:
        """Compare two BoundingBox objects for equality based on their hash.

            values.

        This method evaluates whether the hash values of the self and other
            BoundingBox objects are equal. If 'other' is not a BoundingBox
            object, the method returns NotImplemented.

        Arguments:
            self ('BoundingBox'): The instance of BoundingBox that calls the
                method.
            other ('BoundingBox'): Another instance of BoundingBox that is
                compared with self.

        Returns:
            bool: True if the hash values of the two BoundingBox objects are
                equal, False otherwise. Returns NotImplemented if 'other' is
                not a BoundingBox object.

        Example:
            >>> box1.equals(box2)

        Note:
            The equality of two BoundingBox objects is determined solely
                based on their hash values.

        """
        if not isinstance(other, BoundingBox):
            return NotImplemented
        return self.__hash__() == other.__hash__()


@dataclass(
    config=ConfigDict(extra="forbid", arbitrary_types_allowed=True),
    kw_only=False,
)
class Points:
    """A class to represent a set of points in an image."""

    points: np.ndarray | list[list[float]]
    """The points in the image, represented as a 2D NumPy array."""

    is_normalized: bool
    """A flag indicating whether the points are normalized."""

    image_size: ImageSize
    """The size of the image in which the points are located in case the points are not normalized."""

    def __post_init__(self) -> None:
        """Validate the Points object after initialization.

        This method is called after the Points object is initialized to
            validate the points attribute.

        Arguments:
            self (Points): The Points object to be validated.

        Returns:
            None

        Example:
            >>> points = Points(...)
            >>> points.__post_init__()

        Note:
            This method is automatically called after the Points object is
                initialized.

        """
        if isinstance(self.points, list):
            self.points = np.array(self.points)
        if not isinstance(self.points, np.ndarray):
            msg = "The 'points' attribute must be a NumPy array."
            raise TypeError(msg)
        if self.points.ndim != 2:
            msg = "The 'points' attribute must be a 2D NumPy array."
            raise ValueError(msg)

    @property
    def num_points(self) -> int:
        """Return the number of points in the Points object.

        This method returns the number of points in the Points object.

        Arguments:
            self (Points): The Points object for which the number of points
                is to be calculated.

        Returns:
            int: The number of points in the Points object.

        Example:
            >>> points = Points(...)
            >>> points.num_points

        Note:
            This method does not take any arguments. It calculates the number
                of points based on the properties of the Points object.

        """
        return self.points.shape[0]  # type: ignore[union-attr]

    @property
    def xy(self) -> Float[np.ndarray, "n 2"]:
        """Return the X and Y coordinates of the points.

        This method returns the X and Y coordinates of the points in the
            Points object.

        Arguments:
            self (Points): The Points object for which the X and Y
                coordinates are to be calculated.

        Returns:
            np.ndarray: A NumPy array containing the X and Y coordinates of
                the points.

        Example:
            >>> points = Points(...)
            >>> points.xy()

        Note:
            This method does not take any arguments. It calculates the X and Y
                coordinates based on the properties of the Points object.

        """
        if self.is_normalized:
            return self.points * np.array(
                [self.image_size.height, self.image_size.width]
            )
        return self.points

    @property
    def xyn(self) -> Float[np.ndarray, "n 2"]:
        """Return the normalized X and Y coordinates of the points.

        This method returns the normalized X and Y coordinates of the points
            in the Points object.

        Arguments:
            self (Points): The Points object for which the normalized X and
                Y coordinates are to be calculated.

        Returns:
            np.ndarray: A NumPy array containing the normalized X and Y
                coordinates of the points.

        Example:
            >>> points = Points(...)
            >>> points.xyn()

        Note:
            This method does not take any arguments. It calculates the
                normalized X and Y coordinates based on the properties of the
                Points object.

        """
        if not self.is_normalized:
            return self.points / np.array(
                [self.image_size.height, self.image_size.width]
            )
        return self.points

    def shift_points(self, shift: tuple[float, float]) -> Points:
        """Shift the points in the Points object by a specified amount.

        This method shifts the points in the Points object by a specified
            amount.

        Arguments:
            self (Points): The Points object for which the points are to be
                shifted.
            shift (Tuple[float, float]): A tuple containing the X and Y
                coordinates by which the points are to be shifted.

        Returns:
            Points: A new Points object with the shifted points.

        Example:
            >>> points = Points(...)
            >>> points.shift_points((10, 10))

        Note:
            This method shifts the points by adding the specified amount to
                the X and Y coordinates of each point.

        """
        new_points = self.xy + np.array(shift)
        return Points(
            new_points.astype(np.float32),
            is_normalized=False,
            image_size=self.image_size,
        )

    def list_tuple_int(self) -> list[tuple[int, int]]:
        """Return the points as a list of tuples of integers.

        This method returns the points in the Points object as a list of
            tuples of integers.

        Arguments:
            self (Points): The Points object for which the points are to be
                converted to a list of tuples of integers.

        Returns:
            List[Tuple[int, int]]: A list of tuples containing the X and Y
                coordinates of the points as integers.

        Example:
            >>> points = Points(...)
            >>> points.list_tuple_int()

        Note:
            This method does not take any arguments. It converts the points
                based on the properties of the Points object.

        """
        return [(int(x), int(y)) for x, y in self.xy]

    def list_tuple_float(
        self, *, normalized: bool
    ) -> list[tuple[float, float]]:
        """Return the points as a list of tuples of floats.

        This method returns the points in the Points object as a list of
            tuples of floats.

        Arguments:
            self (Points): The Points object for which the points are to be
                converted to a list of tuples of floats.

        Returns:
            List[Tuple[float, float]]: A list of tuples containing the X and
                Y coordinates of the points as floats.

        Example:
            >>> points = Points(...)
            >>> points.list_tuple_float()

        Note:
            This method does not take any arguments. It converts the points
                based on the properties of the Points object.

        """
        if normalized:
            return [(float(x), float(y)) for x, y in self.xyn]
        return [(float(x), float(y)) for x, y in self.xy]

    def min_x(self) -> int:
        """Return the minimum X coordinate of the points.

        This method returns the minimum X coordinate of the points in the
            Points object.

        Arguments:
            self (Points): The Points object for which the minimum X
                coordinate is to be calculated.

        Returns:
            int: The minimum X coordinate of the points.

        Example:
            >>> points = Points(...)
            >>> points.min_x()

        Note:
            This method does not take any arguments. It calculates the minimum
                X coordinate based on the properties of the Points object.

        """
        return int(np.min(self.xy[:, 0]))

    def min_y(self) -> int:
        """Return the minimum Y coordinate of the points.

        This method returns the minimum Y coordinate of the points in the
            Points object.

        Arguments:
            self (Points): The Points object for which the minimum Y
                coordinate is to be calculated.

        Returns:
            int: The minimum Y coordinate of the points.

        Example:
            >>> points = Points(...)
            >>> points.min_y()

        Note:
            This method does not take any arguments. It calculates the minimum
                Y coordinate based on the properties of the Points object.

        """
        return int(np.min(self.xy[:, 1]))

    def max_x(self) -> int:
        """Return the maximum X coordinate of the points.

        This method returns the maximum X coordinate of the points in the
            Points object.

        Arguments:
            self (Points): The Points object for which the maximum X
                coordinate is to be calculated.

        Returns:
            int: The maximum X coordinate of the points.

        Example:
            >>> points = Points(...)
            >>> points.max_x()

        Note:
            This method does not take any arguments. It calculates the maximum
                X coordinate based on the properties of the Points object.

        """
        return int(np.max(self.xy[:, 0]))

    def max_y(self) -> int:
        """Return the maximum Y coordinate of the points.

        This method returns the maximum Y coordinate of the points in the
            Points object.

        Arguments:
            self (Points): The Points object for which the maximum Y
                coordinate is to be calculated.

        Returns:
            int: The maximum Y coordinate of the points.

        Example:
            >>> points = Points(...)
            >>> points.max_y()

        Note:
            This method does not take any arguments. It calculates the maximum
                Y coordinate based on the properties of the Points object.

        """
        return int(np.max(self.xy[:, 1]))

    def __len__(self) -> int:
        """Return the number of points in the Points object.

        This method returns the number of points in the Points object.

        Arguments:
            self (Points): The Points object for which the number of points
                is to be calculated.

        Returns:
            int: The number of points in the Points object.

        Example:
            >>> points = Points(...)
            >>> len(points)

        Note:
            This method is used to calculate the length of the Points object.

        """
        return self.num_points

    def __hash__(self) -> int:
        """Calculate the hash value of a Points object.

        This method calculates the hash value of a Points object by
            converting its dictionary attributes into a hashable format.

        Arguments:
            self (Points): The Points object for which the hash value is
                being calculated.

        Returns:
            int: An integer representing the hash value of the Points object.

        Example:
            >>> points = Points(...)
            >>> points.__hash__()

        Note:
            Hash values are used to quickly compare dictionary keys during a
                dictionary lookup. They must be immutable and hashable.

        """
        return hash(HashableDict(self.__dict__))

    def __eq__(self, other: object) -> bool:
        """Compare two Points objects for equality.

        This method checks if two instances of the class Points are equal by
            comparing their hash values.

        Arguments:
            self (Points): The instance of the class Points calling the
                method.
            other (object): Another object to compare with the instance of
                the class Points.

        Returns:
            bool: Returns True if the hash values of the two instances are
                equal, False otherwise.
                  If the other object is not an instance of Points, it returns
                NotImplemented.

        Example:
            >>> points1 = Points(...)
            >>> points2 = Points(...)
            >>> points1 == points2

        Note:
            The equality of two Points instances doesn't mean they are the
                same object, only that their hash values are equal.

        """
        if not isinstance(other, Points):
            return NotImplemented
        return self.__hash__() == other.__hash__()
