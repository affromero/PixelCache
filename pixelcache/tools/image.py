from dataclasses import field
from itertools import product
from pathlib import Path
from typing import Literal
from urllib.request import urlopen

import cv2
import einops
import numpy as np
import torch
import torchvision.utils as tv
from beartype import beartype
from klog.path import path_exists
from jaxtyping import Bool, Float, UInt8, jaxtyped
from PIL import Image, ImageOps
from pillow_heif import register_heif_opener
from pydantic import ConfigDict
from pydantic.dataclasses import dataclass
from torchvision.io.image import (
    ImageReadMode,
    decode_jpeg,
    decode_png,
    read_file,
)
from torchvision.utils import make_grid

register_heif_opener()


_EXIF_ORIENTATION_TAG = 274


def _pil_to_tensor(img: Image.Image) -> Float[torch.Tensor, "1 c h w"]:
    """Decode a PIL image to a `1 c h w` float tensor in `[0, 1]`.

    `np.array(..., copy=True)` instead of `np.asarray(...)` because
    some PIL backends return non-writable views; `torch.from_numpy`
    warns on those.
    """
    arr = np.array(img.convert("RGB"), copy=True)
    tensor = torch.from_numpy(arr).permute(2, 0, 1)
    return tensor[None] / 255.0


@jaxtyped(typechecker=beartype)
def read_image(
    fname: str | Path,
    /,
) -> Float[torch.Tensor, "1 c h w"]:
    """Read an image into a `1 c h w` float tensor with a single decode pass.

    Decode strategy:
    - **Local JPEG/PNG without EXIF rotation:** torchvision fast path
      (`decode_jpeg` / `decode_png`). One file read, one decode.
    - **Local file with EXIF orientation tag != 1:** PIL with
      `exif_transpose` on the open handle. One decode.
    - **HEIC:** PIL (via `pillow_heif`). One decode.
    - **HTTP / HTTPS URL:** stream via `urllib.request`, decode via
      PIL. One decode.

    Args:
        fname: Local path or HTTP URL.

    Returns:
        Float tensor in `[0, 1]` with shape `1 c h w`.

    Raises:
        RuntimeError: If the source is neither a local file nor an HTTP URL.

    """
    fname_str = str(fname)
    lower = fname_str.lower()
    is_http = lower.startswith(("http://", "https://"))
    is_local = path_exists(fname_str)
    is_heic = lower.endswith(".heic")

    if is_http:
        with urlopen(fname_str, timeout=10) as resp, Image.open(resp) as img:
            return _pil_to_tensor(img)
    if is_local and is_heic:
        with Image.open(fname_str) as img:
            return _pil_to_tensor(img)
    if is_local:
        # Lazy header open: getexif() reads metadata without decoding pixels.
        with Image.open(fname_str) as img:
            orientation = img.getexif().get(_EXIF_ORIENTATION_TAG)
            if orientation is not None and orientation != 1:
                rotated = ImageOps.exif_transpose(img)
                if rotated is None:
                    msg = f"Failed to transpose image: {fname}"
                    raise RuntimeError(msg)
                return _pil_to_tensor(rotated)
        # No EXIF rotation needed. Try the torchvision fast path for
        # JPEG / PNG; fall back to PIL for everything else PIL can
        # open (WebP, BMP, TIFF, GIF, …) so we don't regress versus
        # the pre-refactor public contract.
        if lower.endswith((".jpg", ".jpeg", ".png")):
            data = read_file(fname_str)
            try:
                tensor = decode_jpeg(data, device="cpu")
            except RuntimeError:
                tensor = decode_png(data, ImageReadMode.RGB)
            return tensor[None] / 255.0
        with Image.open(fname_str) as img:
            return _pil_to_tensor(img)

    msg = f"file not supported: {fname}"
    raise RuntimeError(msg)


@jaxtyped(typechecker=beartype)
def save_image(
    img: (
        Float[torch.Tensor, "b c h w"]
        | Bool[torch.Tensor, "b c h w"]
        | Image.Image
        | UInt8[np.ndarray, "h w c"]
        | UInt8[np.ndarray, "h w"]
        | Bool[np.ndarray, "h w"]
    ),
    /,
    *,
    path: str | Path,
    nrow: int = 8,
    padding: int = 2,
    normalize: bool = True,
    scale_each: bool = False,
    pad_value: float = 0.0,
) -> None:
    """Save an image to a specified path, supporting various input types.

    Args:
        img (Union[torch.Tensor, np.ndarray, PIL.Image, List[bool]]): Input
            image data.
        path (str): The path where the image will be saved.
        nrow (int, optional): Number of images per row in the saved image
            grid. Defaults to 8.
        padding (int, optional): Padding between images in the grid.
            Defaults to 2.
        normalize (bool, optional): If True, normalizes the image data.
            Defaults to True.
        scale_each (bool, optional): If True, scales each image
            individually. Defaults to False.
        pad_value (float, optional): Padding value for the image. Defaults
            to 0.0.

    Returns:
        None: This function doesn't return anything, it saves the image to
            the specified path.

    """
    if isinstance(img, np.ndarray):
        if img.dtype == bool:
            img = (img * 255).astype(np.uint8)
        cv2.imwrite(
            str(path),
            (img[..., ::-1] if len(img.shape) == 3 else img),
        )
    elif isinstance(img, Image.Image):
        img.save(path)
    else:
        if img.dtype == torch.bool:
            img = img.float()
        tv.save_image(
            img,
            path,
            nrow=nrow,
            padding=padding,
            normalize=normalize,
            scale_each=scale_each,
            pad_value=pad_value,
        )


@jaxtyped(typechecker=beartype)
def numpy2tensor(
    imgs: (
        UInt8[np.ndarray, "h w c"]
        | UInt8[np.ndarray, "h w"]
        | Bool[np.ndarray, "h w"]
    ),
) -> Float[torch.Tensor, "1 c h w"] | Bool[torch.Tensor, "1 1 h w"]:
    """Converts a Numpy array into a tensor.

    This function takes in a Numpy array of images and converts each image
        into a tensor.
    If the input array only contains one image, the function returns a
        single tensor.
    Otherwise, it returns a list of tensors.

    Args:
        imgs (ndarray): A Numpy array of input images. Each image should be
            in the form of a multi-dimensional array.

    Returns:
        Union[List[tensor], tensor]: If multiple images are provided, a list
            of tensors is returned.
        If a single image is provided, a single tensor is returned.

    """
    if imgs.ndim == 2:
        imgs = np.expand_dims(imgs, 2)
    img_pt: Float[torch.Tensor, "1 c h w"] = torch.from_numpy(
        imgs.transpose(2, 0, 1).copy(),
    ).unsqueeze(0)
    return img_pt / 255.0 if img_pt.dtype == torch.uint8 else img_pt


@jaxtyped(typechecker=beartype)
def pil2tensor(
    img: Image.Image,
) -> Float[torch.Tensor, "1 c h w"] | Bool[torch.Tensor, "1 1 h w"]:
    """Convert a PIL Image to a tensor.

    This function takes a PIL Image as input and converts it into a tensor.
    If the resulting tensor only contains a single element, the tensor is
        returned directly.
    Otherwise, a list of tensors is returned.

    Args:
        img ('PIL Image'): The PIL Image to be converted to a tensor.

    Returns:
        Union[List[Tensor], Tensor]: The resulting tensor or list of
            tensors.

    """
    return numpy2tensor(np.asarray(img))


@jaxtyped(typechecker=beartype)
def tensor2numpy(
    tensor: Float[torch.Tensor, "b c h w"] | Bool[torch.Tensor, "b c h w"],
    *,
    output_type: type = np.uint8,
    min_max: tuple[int, int] = (0, 1),
    padding: int = 2,
) -> (
    UInt8[np.ndarray, "h w c"]
    | UInt8[np.ndarray, "h w"]
    | Bool[np.ndarray, "h w"]
):
    """Convert torch Tensors into image numpy arrays.

    This function accepts torch Tensors, clamps the values between a
        specified min and max,
    normalizes them to the range [0, 1], and then converts them to numpy
        arrays. The channel order is preserved as RGB.

    Args:
        tensor (Union[Tensor, List[Tensor]]): The input Tensor or list of
            Tensors. The function accepts three possible shapes:
            1) 4D mini-batch Tensor of shape (B x 3/1 x H x W);
            2) 3D Tensor of shape (3/1 x H x W);
            3) 2D Tensor of shape (H x W).
        output_type (numpy.dtype, optional): The desired numpy dtype of the
            output arrays. If set to ``np.uint8``, the function
            will return arrays of uint8 type with values in the range [0,
            255]. Otherwise, it will return arrays of float type
            with values in the range [0, 1]. Defaults to ``np.uint8``.
        min_max (Tuple[int, int], optional): A tuple specifying the min and
            max values for clamping. Defaults to (0, 255).

    Returns:
        Union[Tensor, List[Tensor]]: The converted numpy array(s). The
            arrays will have a shape of either (H x W x C) for 3D arrays
            or (H x W) for 2D arrays. The channel order is RGB.

    """
    if not (
        torch.is_tensor(tensor)
        or (
            isinstance(tensor, list)
            and all(torch.is_tensor(t) for t in tensor)  # E501
        )
    ):
        msg = f"tensor or list of tensors expected, got {type(tensor)}"
        raise TypeError(
            msg,
        )
    _tensor = tensor.clone().float().detach().cpu().clamp_(*min_max)
    if _tensor.size(0) == 1:
        img_grid_np = _tensor[0].numpy()
    else:
        img_grid_np: Float[np.ndarray, "3 h w"] = make_grid(  # type: ignore[no-redef]
            _tensor,
            padding=padding,
            nrow=_tensor.size(0),
            normalize=False,
        ).numpy()
    if _tensor.size(1) == 1:
        img_grid_np = img_grid_np[:1]
    img_np: Float[np.ndarray, "h w c"] = img_grid_np.transpose(1, 2, 0)
    if output_type in (np.uint8, np.uint16):
        # Unlike MATLAB, numpy.unit8/16() WILL NOT round by default.
        scale = 255.0 if output_type == np.uint8 else 65535.0
        img_np = (img_np * scale).round()
    img_np_typed = img_np.astype(output_type)
    if img_np_typed.shape[-1] == 1:
        img_np_typed = img_np_typed[..., 0]
    return img_np_typed


@jaxtyped(typechecker=beartype)
def tensor2pil(
    tensor: Float[torch.Tensor, "b c h w"] | Bool[torch.Tensor, "b c h w"],
    *,
    min_max: tuple[int, int] = (0, 1),
    padding: int = 2,
) -> Image.Image:
    """Convert torch Tensors into PIL images.

    The tensor values are first clamped to the range [min, max] and then
        normalized to the range [0, 1].

    Args:
        tensor (Union[Tensor, List[Tensor]]): The input tensor(s) to be
            converted. Accepts the following shapes:
            1) 4D mini-batch Tensor of shape (B x 3/1 x H x W);
            2) 3D Tensor of shape (3/1 x H x W);
            3) 2D Tensor of shape (H x W).
            The tensor channel should be in RGB order.
        min_max (Tuple[int, int]): The min and max values for clamping the
            tensor values.

    Returns:
        Union[Tensor, List[Tensor]]: The converted image(s) in the form of
            3D ndarray of shape (H x W x C)
        or 2D ndarray of shape (H x W). The channel order is RGB.

    """
    img_np = tensor2numpy(
        tensor,
        output_type=np.uint8 if tensor.dtype != torch.bool else bool,
        min_max=min_max,
        padding=padding,
    )
    return Image.fromarray(img_np)


def make_image_grid(
    images: list[Image.Image], rows: int, cols: int, resize: int | None = None
) -> Image.Image:
    """Prepares a single grid of images. Useful for visualization purposes.

    This function takes a list of images and arranges them in a grid with the specified number of rows and columns.
    The images can be resized to a specific size before being arranged in the grid.

    Args:
        images (List[PIL.Image.Image]): A list of PIL Image objects to be arranged in the grid.
        rows (int): The number of rows in the grid.
        cols (int): The number of columns in the grid.
        resize (int, optional): The size to which the images should be resized before arranging them in the grid. Defaults to None.

    Returns:
        PIL.Image.Image: A single PIL Image object containing the grid of images.

    """
    if resize is not None:
        images = [img.resize((resize, resize)) for img in images]

    w, h = images[0].size
    grid = Image.new("RGB", size=(cols * w, rows * h))

    for i, img in enumerate(images):
        grid.paste(img, box=(i % cols * w, i // cols * h))
    return grid


@dataclass(config=ConfigDict(extra="forbid"), kw_only=True)
class ImageSize:
    """Image size class."""

    height: int
    """The height of the image."""
    width: int
    """The width of the image."""
    is_normalized: bool = field(init=False)
    """Whether the height and width are normalized."""

    def __post_init__(self) -> None:
        """Validate the height and width attributes of the ImageSize instance.

        This method checks that the height and width attributes of an
            ImageSize instance are positive,
        within a certain range, and are either integers or floats. Raises a
            ValueError if these conditions are not met.

        Args:
            self (ImageSize): The instance of the ImageSize class.

        Returns:
            None: This method does not return any value.

        Raises:
            ValueError: If the image size does not meet the specified
                criteria.

        """
        if self.height <= 0 or self.width <= 0:
            msg = f"image size must be positive. {self}"
            raise ValueError(msg)
        # they all must be integers or all must be floats
        if isinstance(self.height, int) and isinstance(self.width, int):
            self.is_normalized = False
        elif isinstance(self.height, float) and isinstance(self.width, float):
            # must be between 0 and 1
            if self.height > 1 or self.width > 1:
                msg = f"image size must be between 0 and 1. {self}"
                raise ValueError(msg)
            self.is_normalized = True
        else:
            msg = f"all image size values must be either int or float. {self}"
            raise TypeError(
                msg,
            )

    def area(self) -> int:
        """Calculate the area of the image."""
        return self.height * self.width

    def min(self) -> int | float:
        """Return the minimum value between the height and width of an image.

            size.

        Args:
            image_size (Tuple[int, int]): A tuple containing the height and
                width of the image.

        Returns:
            Union[int, float]: The minimum value between the height and
                width of the image size.

        """
        return min(self.height, self.width)

    def max(self) -> int | float:
        """Calculate the maximum dimension of an image.

        This method in the 'ImageSize' class returns the maximum value
            between the height
        and width attributes of an image.

        Args:
            self (ImageSize instance): The instance of the 'ImageSize' class
                for which the
                                       maximum value needs to be calculated.

        Returns:
            Union[int, float]: The maximum value between the height and
                width attributes of
                              the image, which can be either an integer or a
                float.

        """
        return max(self.height, self.width)

    def product(self) -> int | float:
        """Calculate the area of an image.

        This method calculates the product of the height and width of an
            image, effectively determining its area.

        Args:
            self (ImageSize): The instance of the ImageSize class.

        Returns:
            Union[int, float]: The product of the height and width of the
                image, representing its area. The return type will be an
                integer if both height and width are integers, otherwise it
                will be a float.

        """
        return self.height * self.width

    def __eq__(self, other: object) -> bool:
        """Compare two ImageSize objects for equality based on their height and.

            width attributes.

        This method determines equality by comparing the height and width
            attributes of the
        ImageSize object calling the method and another ImageSize object.

        Args:
            self ('ImageSize'): The ImageSize object invoking the method.
            other ('ImageSize'): Another ImageSize object to compare with.

        Returns:
            bool: Returns True if the height and width of both ImageSize
                objects are equal,
                  otherwise returns False.

        """
        # compare height and width
        if not isinstance(other, ImageSize):
            msg = "This comparison can only be with an ImageSize object"
            raise TypeError(
                msg,
            )
        return not (self.height != other.height or self.width != other.width)

    def __mul__(self, other: float) -> "ImageSize":
        """Multiply the dimensions of an ImageSize object by a given value.

        This method takes an ImageSize object and a numeric value (integer
            or float) as input. It multiplies the height and width
        of the ImageSize object by the given value and returns a new
            ImageSize object with the updated dimensions.

        Args:
            self (ImageSize): The ImageSize object whose dimensions are to
                be multiplied.
            other (int | float): The numeric value by which the height and
                width of the ImageSize object will be multiplied.

        Returns:
            ImageSize: A new ImageSize object with the height and width
                multiplied by the given value.

        """
        return ImageSize(
            height=round(self.height * other), width=round(self.width * other)
        )

    def __ne__(self, other: object) -> bool:
        """Check if the current ImageSize object is not equal to another.

            object.

        Args:
            self (ImageSize): The current ImageSize object.
            other (ImageSize): The object to compare with.

        Returns:
            bool: True if the current ImageSize object is not equal to the
                other object, False otherwise.

        """
        return not self.__eq__(other)

    def __lt__(self, other: object) -> bool:
        """Compare two ImageSize objects based on their height and width.

            values.

        Args:
            self ('ImageSize'): The ImageSize object calling the method.
            other ('ImageSize'): The other ImageSize object to compare with.

        Returns:
            bool: True if the calling object's height and width are both
                less than the other object's height and width, False
                otherwise.

        """
        if not isinstance(other, ImageSize):
            msg = "This comparison can only be with an ImageSize object"
            raise TypeError(
                msg,
            )
        # compare height and width
        return bool(self.height < other.height and self.width < other.width)

    def __le__(self, other: object) -> bool:
        """Compare the size of two ImageSize objects.

        This method compares the height and width of the current ImageSize
            object (self) with another ImageSize object (other). It returns
            True if both the height and width of the current object are less
            than or equal to those of the other object. Otherwise, it
            returns False.

        Args:
            self ('ImageSize'): The current ImageSize object.
            other ('ImageSize'): Another ImageSize object to compare with
                the current object.

        Returns:
            bool: Returns True if both the height and width of the current
                object are less than or equal to those of the other object.
                Otherwise, returns False.

        """
        if not isinstance(other, ImageSize):
            msg = "This comparison can only be with an ImageSize object"
            raise TypeError(
                msg,
            )
        # compare height and width
        return bool(self.height <= other.height and self.width <= other.width)

    def __gt__(self, other: object) -> bool:
        """Compare two ImageSize objects based on their dimensions.

        This method compares two ImageSize objects based on their height and
            width attributes.
        It returns True if the calling object has greater dimensions than
            the other object in both height and width.

        Args:
            self ('ImageSize'): The calling ImageSize object.
            other ('ImageSize'): Another ImageSize object to compare with
                the calling object.

        Returns:
            bool: True if the calling object's dimensions (both height and
                width) are greater than the other object's. False otherwise.

        """
        if not isinstance(other, ImageSize):
            msg = "This comparison can only be with an ImageSize object"
            raise TypeError(
                msg,
            )
        # compare height and width
        return bool(self.height > other.height and self.width > other.width)

    def __ge__(self, other: object) -> bool:
        """Compare the size of two ImageSize objects.

        This method compares the height and width of two ImageSize objects.
            It returns True if the height and width of the calling object
            are greater than or equal to the height and width of the other
            object, otherwise returns False.

        Args:
            self ('ImageSize'): The calling ImageSize object.
            other ('ImageSize'): Another ImageSize object to compare with
                the calling object.

        Returns:
            bool: Returns True if the height and width of the calling object
                are greater than or equal to the height and width of the
                other object, otherwise returns False.

        """
        if not isinstance(other, ImageSize):
            msg = "This comparison can only be with an ImageSize object"
            raise TypeError(
                msg,
            )
        return bool(self.height >= other.height and self.width >= other.width)

    def __hash__(self) -> int:
        """Calculate the hash value of an ImageSize object.

        This method generates a unique hash value for an ImageSize object
            based on its 'height' and 'width' attributes.

        Args:
            self (ImageSize): The ImageSize object for which the hash value
                is being calculated.

        Returns:
            int: A unique integer representing the hash value of the
                ImageSize object.

        """
        return hash((self.height, self.width))

    def __repr__(self) -> str:
        """Return a string representation of the ImageSize object.

        This method generates a string that represents the ImageSize object,
            including its height and width attributes. The string is in the
            format 'ImageSize(height=height_value, width=width_value)'.

        Returns:
            str: A string representation of the ImageSize object in the
                format 'ImageSize(height=height_value, width=width_value)'.

        """
        return f"ImageSize(height={self.height}, width={self.width})"

    @staticmethod
    def from_image(
        image: (
            str
            | Image.Image
            | UInt8[np.ndarray, "h w c"]
            | UInt8[np.ndarray, "h w"]
            | Bool[np.ndarray, "h w"]
            | Float[torch.Tensor, "b c h w"]
            | Bool[torch.Tensor, "b 1 h w"]
        ),
    ) -> "ImageSize":
        """Create an ImageSize instance from various image inputs.

        This static method in the ImageSize class creates an instance of
            ImageSize based on the input image provided. It can handle
            different types of image inputs such as file paths, PIL Image
            objects, NumPy arrays, and PyTorch tensors.

        Args:
            image (Union[str, Image.Image, np.ndarray, torch.Tensor]): The
                input image to create an ImageSize instance from. It can be
                a file path (str), a PIL Image object, a NumPy array with
                shape 'h w c' or 'h w', or a PyTorch tensor with shape 'b c
                h w' or 'b 1 h w'.

        Returns:
            ImageSize: An instance of the ImageSize class representing the
                height and width of the input image.

        """
        if isinstance(image, str):
            return ImageSize.from_image(read_image(image))
        if isinstance(image, Image.Image):
            return ImageSize(height=image.height, width=image.width)
        if isinstance(image, np.ndarray):
            return ImageSize(height=image.shape[0], width=image.shape[1])
        if isinstance(image, torch.Tensor):
            return ImageSize(height=image.shape[-2], width=image.shape[-1])
        msg = f"invalid image type {type(image)}"
        raise TypeError(msg)


@jaxtyped(typechecker=beartype)
def center_pad(
    image: UInt8[np.ndarray, "h w c"],
    size: ImageSize,
    fill: int | tuple[int, int] = (0, 0),
) -> UInt8[np.ndarray, "h1 w1 c"]:
    """Pads an image to the center with a specified size and fill value.

    Args:
        image (Union[np.ndarray, PIL.Image.Image]): The input image, which
            can be either a NumPy array or a PIL Image.
        size (ImageSize): An object that contains the desired height and
            width for the padded image.
        fill (Union[int, Tuple[int, int, int]]): The fill value for padding.
            This can be either an integer or a tuple of integers.

    Returns:
        Union[np.ndarray, PIL.Image.Image]: The padded image, returned in
            the same format as the input image (either a NumPy array or a
            PIL Image).

    """
    h, w = image.shape[:2]
    h_pad = size.height // 2
    w_pad = size.width // 2
    h_mod = max(h % 2, 0)
    w_mod = max(w % 2, 0)
    new_np = np.zeros((size.height, size.width, 3), dtype=np.uint8) + fill
    new_np[
        h_pad - h // 2 : h_pad + h // 2 + h_mod,
        w_pad - w // 2 : w_pad + w // 2 + w_mod,
    ] = image
    return new_np


@jaxtyped(typechecker=beartype)
def to_binary(
    rgb: (
        Image.Image
        | UInt8[np.ndarray, "h w 3"]
        | UInt8[np.ndarray, "h w"]
        | Float[torch.Tensor, "1 c h w"]
    ),
    threshold: float = 0.0,
) -> Image.Image | Bool[np.ndarray, "h w"] | Bool[torch.Tensor, "1 1 h w"]:
    """Convert an RGB image or an array of UInt8 values to binary format.

    This function takes an image in RGB format, an array of UInt8 values, or
        a torch tensor and converts it to binary format.

    Args:
        rgb (Union[Image.Image, np.array, torch.Tensor]): An image in RGB
            format, an array of UInt8 values with shape 'h w 3' or 'h w', or
            a torch tensor with shape '1 c h w'.

    Returns:
        Union[Image.Image, np.array, torch.Tensor]: The binary version of
            the input. If the input is an image or an array, the function
            returns the binary version of the input. If the input is a torch
            tensor, the function returns the binary version of the tensor.

    """
    if threshold < 0 or threshold > 1:
        msg = "threshold should be between 0 and 1"
        raise ValueError(
            msg,
        )
    if isinstance(rgb, Image.Image | np.ndarray):
        rgb_np = np.asarray(rgb)
        if rgb_np.ndim == 3:
            rgb_np = np.logical_or.reduce(
                [
                    rgb_np[..., 0] > threshold * 255,
                    rgb_np[..., 1] > threshold * 255,
                    rgb_np[..., 2] > threshold * 255,
                ]
            )
        elif rgb_np.ndim == 2:
            rgb_np = rgb_np > threshold * 255
        if isinstance(rgb, Image.Image):
            return Image.fromarray(rgb_np)
        return rgb_np
    binary_pt: Bool[torch.Tensor, "1 1 h w"] = (
        rgb.mean(
            dim=1,
            keepdim=True,
        )
        > threshold
    )
    return binary_pt


@jaxtyped(typechecker=beartype)
def to_rgb(
    rgb: (
        Image.Image
        | UInt8[np.ndarray, "h w"]
        | Bool[np.ndarray, "h w"]
        | Float[torch.Tensor, "1 1 h w"]
        | Bool[torch.Tensor, "1 1 h w"]
    ),
) -> (
    Image.Image
    | Bool[np.ndarray, "h w 3"]
    | UInt8[np.ndarray, "h w 3"]
    | Bool[torch.Tensor, "1 3 h w"]
    | Float[torch.Tensor, "1 3 h w"]
):
    """Convert the input image to RGB format.

    Args:
        rgb (Union[np.array, PIL.Image, torch.Tensor]): The input image in
            various formats such as numpy array, PIL Image, or torch tensor.

    Returns:
        Union[np.array, PIL.Image, torch.Tensor]: The input image converted
            to RGB format.

    """
    if isinstance(rgb, np.ndarray):
        return np.asarray(rgb)[..., None].repeat(3, axis=-1)
    if isinstance(rgb, Image.Image):
        return rgb.convert("RGB")
    return einops.repeat(rgb, "1 1 h w -> 1 c h w", c=3)


@jaxtyped(typechecker=beartype)
def resize_image(
    tensor: Float[torch.Tensor, "b c h w"] | Bool[torch.Tensor, "b 1 h w"],
    /,
    resolution: int | None | ImageSize,
    mode: str,
    resize_min_max: Literal["min", "max"] = "min",
    modulo: int = 16,
) -> Float[torch.Tensor, "b c h1 w1"]:
    """Resizes the provided image to a specified resolution while maintaining.

        the aspect ratio.

    The function supports resizing based on either the minimum or maximum
        dimension and it
    can utilize different modes of interpolation.

    Args:
        input_image (ImageType): The input image to be resized.
        resolution (Union[int, None, ImageSize]): The target resolution to
            resize the image to.
                                                   Can be an integer, None,
            or an ImageSize object.
        mode (str): The interpolation mode to use during resizing.
        resize_min_max (str): Determines whether to resize based on the
            minimum or maximum dimension.
        modulo (int): The value to round the dimensions to after resizing.

    Returns:
        Tensor: The resized image as a tensor object.

    """
    height, width = tensor.shape[-2:]
    height = float(height)
    width = float(width)
    is_bool = False
    if tensor.dtype == torch.bool:
        tensor = tensor.float().repeat(1, 3, 1, 1)
        is_bool = True
    if resolution is None:
        # resize divisible by modulo
        height = round(np.round(height / modulo)) * modulo
        width = round(np.round(width / modulo)) * modulo
    elif isinstance(resolution, ImageSize):
        height, width = resolution.height, resolution.width
    else:
        if resize_min_max == "min":
            k = float(resolution) / min(height, width)  # resize with min
        else:
            k = float(resolution) / max(height, width)  # resize with max
        height *= k
        width *= k
        height = round(np.round(height / modulo)) * modulo
        width = round(np.round(width / modulo)) * modulo
    output: Float[torch.Tensor, "b c h1 w1"] = torch.nn.functional.interpolate(
        tensor,
        size=(height, width),
        mode=mode,
    )
    if is_bool:
        return output.bool()[:, :1, :, :]
    return output


@jaxtyped(typechecker=beartype)
def rgb2gray(rgb: UInt8[np.ndarray, "h w 3"]) -> UInt8[np.ndarray, "h w"]:
    """Convert an RGB image to grayscale using the luminance method.

    This function takes in an RGB image represented as a NumPy array and
        converts it to a grayscale image using the luminance method. The
        luminance method forms a weighted sum of the R, G, and B components
        of each pixel to produce a grayscale intensity.

    Args:
        rgb (np.array): A 3D NumPy array representing an RGB image. The
            dimensions represent height, width, and the three color channels
            (Red, Green, Blue).

    Returns:
        np.array: A 2D NumPy array representing the grayscale version of the
            input RGB image. The dimensions represent height and width.

    """
    return np.dot(rgb[:, :, :3], [0.299, 0.587, 0.114])


@jaxtyped(typechecker=beartype)
def tile(image: Image.Image, mode: str = "1x1") -> dict[str, Image.Image]:
    """Tile an input image into smaller images based on a specified mode.

    Args:
        image (Image.Image): The input image to be tiled.
        mode (str): The tiling mode specifying the number of tiles in the
            format 'NxM' where N is the number of rows and M is the number
            of columns. Defaults to '1x1'.

    Returns:
        Dict[str, Image.Image]: A dictionary containing the tiled images
            with keys representing the position of each tile in the format
            'NxM'.

    """
    w, h = image.size
    d_h, d_w = h // int(mode.split("x")[0]), w // int(mode.split("x")[1])
    grid = product(range(0, h - h % d_h, d_h), range(0, w - w % d_w, d_w))
    out = {}
    for i, j in grid:
        box = (j, i, j + d_w, i + d_h)
        # left, upper, right, and lower pixel coordinate.
        out[f"{i}x{j}"] = image.copy().crop(box)
    return out
