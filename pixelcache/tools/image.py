import tempfile
from dataclasses import field
from itertools import product
from pathlib import Path
from typing import Literal, cast

import cv2
import einops
import numpy as np
import requests
import torch
import torchvision.utils as tv
from beartype import beartype
from jaxtyping import Bool, Float, UInt8, jaxtyped
from PIL import Image, ImageCms
from pydantic import ConfigDict
from pydantic.dataclasses import dataclass
from torchvision.io.image import (
    ImageReadMode,
    decode_jpeg,
    decode_png,
    read_file,
)
from torchvision.utils import make_grid


@jaxtyped(typechecker=beartype)
def read_image(
    fname: str | Path,
    /,
) -> Float[torch.Tensor, "1 c h w"]:
    """Read an image from a file on disk.

    Arguments:
        fname (str): The filename of the image file to be read. It should
            include the complete path if the file is not in the same
            directory.

    Returns:
        np.array: A numpy array representation of the image, where each
            pixel is represented as a list of its RGB values.

    Example:
        >>> read_image_from_file("image.jpg")

    Note:
        This function requires the numpy and PIL libraries. Make sure they
            are installed and imported before using this function.

    """
    if Path(fname).exists():
        data = read_file(str(fname))
        try:
            tensor = decode_jpeg(data, device="cpu")
        except RuntimeError:
            tensor = decode_png(data, ImageReadMode.RGB)
    elif "http" in str(fname):
        raw_np = np.asarray(
            cast(
                Image.Image,
                Image.open(
                    requests.get(str(fname), stream=True, timeout=10).raw
                ),
            ),
        )
        tensor = torch.from_numpy(raw_np.copy()).permute(2, 0, 1)
    else:
        msg = f"file not supported: {fname}"
        raise RuntimeError(msg)
    image: Float[torch.Tensor, "1 c h w"] = tensor[None] / 255.0
    return image


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

    Arguments:
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

    Example:
        >>> save_image(img, '/path/to/save/image', nrow=10, padding=3,
            normalize=False, scale_each=True, pad_value=1.0)

    Note:
        The image data can be in the form of a torch.Tensor, numpy.ndarray,
            PIL Image, or bool arrays.

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
def compress_image(
    image: Image.Image,
    *,
    temp_dir: str | Path | None = None,
    jpeg_quality: int,
) -> str:
    """Compress an image to a JPEG file with a specified quality level.

    This function takes in an image in pillow format and compresses it to
        a JPEG file with a specified quality level.
    It saves the compressed image in a temporary directory and returns the
        path to the compressed JPEG file.

    Arguments:
        image (Image.Image): The input image to be compressed.
        temp_dir (Union[str, Path, None]): Optional temporary directory to
            save the compressed image. Defaults to None.
        jpeg_quality (int): Quality level for JPEG compression.

    Returns:
        str: Path to the compressed JPEG file.

    Example:
        >>> compress_image(image, temp_dir="/tmp", jpeg_quality=75)

    Note:
        The quality level for JPEG compression should be in the range of 1
            (worst) to 95 (best).

    """
    if temp_dir is None:
        temp_dir = Path(tempfile.gettempdir())
    jpg_file = tempfile.NamedTemporaryFile(
        dir=str(temp_dir),
        suffix=".jpg",
    ).name
    image.save(jpg_file, optimize=True, quality=jpeg_quality)
    return jpg_file


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

    Arguments:
        imgs (ndarray): A Numpy array of input images. Each image should be
            in the form of a multi-dimensional array.

    Returns:
        Union[List[tensor], tensor]: If multiple images are provided, a list
            of tensors is returned.
        If a single image is provided, a single tensor is returned.

    Example:
        >>> numpy_to_tensor(numpy_array_of_images)

    Note:
        The function assumes that the input images are already normalized
            and preprocessed.

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

    Arguments:
        img ('PIL Image'): The PIL Image to be converted to a tensor.

    Returns:
        Union[List[Tensor], Tensor]: The resulting tensor or list of
            tensors.

    Example:
        >>> pil_to_tensor(img)

    Note:
        The input image must be a PIL Image object.

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

    Arguments:
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

    Example:
        >>> convert_tensor_to_image(tensor, np.float32, (0, 255))

    Note:
        The input Tensor channel should be in RGB order.

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

    Arguments:
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

    Example:
        >>> tensor_to_image(tensor, (0, 255))

    Note:
        The input tensor values are first clamped to the specified range
            before being normalized.

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

        Arguments:
            self (ImageSize): The instance of the ImageSize class.

        Returns:
            None: This method does not return any value.

        Raises:
            ValueError: If the image size does not meet the specified
                criteria.

        Example:
            >>> img_size = ImageSize(100, 200)
            >>> img_size.__post_init__()

        Note:
            This method is automatically called after the instance has been
                initialized.

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

    def min(self) -> int | float:
        """Return the minimum value between the height and width of an image.

            size.

        Arguments:
            image_size (Tuple[int, int]): A tuple containing the height and
                width of the image.

        Returns:
            Union[int, float]: The minimum value between the height and
                width of the image size.

        Example:
            >>> min_image_dimension((800, 600))

        Note:
            If the height and width are equal, the function will return that
                common value.

        """
        return min(self.height, self.width)

    def max(self) -> int | float:
        """Calculate the maximum dimension of an image.

        This method in the 'ImageSize' class returns the maximum value
            between the height
        and width attributes of an image.

        Arguments:
            self (ImageSize instance): The instance of the 'ImageSize' class
                for which the
                                       maximum value needs to be calculated.

        Returns:
            Union[int, float]: The maximum value between the height and
                width attributes of
                              the image, which can be either an integer or a
                float.

        Example:
            >>> image_size = ImageSize(height=500, width=800)
            >>> image_size.get_max_dimension()
            800

        """
        return max(self.height, self.width)

    def product(self) -> int | float:
        """Calculate the area of an image.

        This method calculates the product of the height and width of an
            image, effectively determining its area.

        Arguments:
            self (ImageSize): The instance of the ImageSize class.

        Returns:
            Union[int, float]: The product of the height and width of the
                image, representing its area. The return type will be an
                integer if both height and width are integers, otherwise it
                will be a float.

        Example:
            >>> image = ImageSize(height=10, width=20)
            >>> image.calculate_area()
            200
        Note:
            The height and width attributes must be set for the ImageSize
                instance before calling this method.

        """
        return self.height * self.width

    def __eq__(self, other: object) -> bool:
        """Compare two ImageSize objects for equality based on their height and.

            width attributes.

        This method determines equality by comparing the height and width
            attributes of the
        ImageSize object calling the method and another ImageSize object.

        Arguments:
            self ('ImageSize'): The ImageSize object invoking the method.
            other ('ImageSize'): Another ImageSize object to compare with.

        Returns:
            bool: Returns True if the height and width of both ImageSize
                objects are equal,
                  otherwise returns False.

        Example:
            >>> img1 = ImageSize(100, 200)
            >>> img2 = ImageSize(100, 200)
            >>> img1.equals(img2)
            True

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

        Arguments:
            self (ImageSize): The ImageSize object whose dimensions are to
                be multiplied.
            other (int | float): The numeric value by which the height and
                width of the ImageSize object will be multiplied.

        Returns:
            ImageSize: A new ImageSize object with the height and width
                multiplied by the given value.

        Example:
            >>> img_size = ImageSize(10, 20)
            >>> new_img_size = img_size.multiply(2)
            >>> print(new_img_size)
            ImageSize(height=20, width=40)

        Note:
            The multiplication is performed independently on the height and
                the width of the ImageSize object.

        """
        return ImageSize(
            height=round(self.height * other), width=round(self.width * other)
        )

    def __ne__(self, other: object) -> bool:
        """Check if the current ImageSize object is not equal to another.

            object.

        Arguments:
            self (ImageSize): The current ImageSize object.
            other (ImageSize): The object to compare with.

        Returns:
            bool: True if the current ImageSize object is not equal to the
                other object, False otherwise.

        Example:
            >>> img_size1 = ImageSize(800, 600)
            >>> img_size2 = ImageSize(1024, 768)
            >>> img_size1.__ne__(img_size2)
            True
        Note:
            The equality comparison is based on the width and height
                attributes of the ImageSize objects.

        """
        return not self.__eq__(other)

    def __lt__(self, other: object) -> bool:
        """Compare two ImageSize objects based on their height and width.

            values.

        Arguments:
            self ('ImageSize'): The ImageSize object calling the method.
            other ('ImageSize'): The other ImageSize object to compare with.

        Returns:
            bool: True if the calling object's height and width are both
                less than the other object's height and width, False
                otherwise.

        Example:
            >>> img1 = ImageSize(200, 300)
            >>> img2 = ImageSize(400, 500)
            >>> img1.compare(img2)
            True
        Note:
            This method is used to compare the size of two images.

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

        Arguments:
            self ('ImageSize'): The current ImageSize object.
            other ('ImageSize'): Another ImageSize object to compare with
                the current object.

        Returns:
            bool: Returns True if both the height and width of the current
                object are less than or equal to those of the other object.
                Otherwise, returns False.

        Example:
            >>> img1 = ImageSize(200, 300)
            >>> img2 = ImageSize(250, 350)
            >>> img1.compare_size(img2)
            True
        Note:
            The comparison is done separately for height and width. Both
                dimensions of the current object need to be less than or
                equal to those of the other object for the method to return
                True.

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

        Arguments:
            self ('ImageSize'): The calling ImageSize object.
            other ('ImageSize'): Another ImageSize object to compare with
                the calling object.

        Returns:
            bool: True if the calling object's dimensions (both height and
                width) are greater than the other object's. False otherwise.

        Example:
            >>> img1 = ImageSize(200, 300)
            >>> img2 = ImageSize(100, 150)
            >>> img1.compare_size(img2)
            True
        Note:
            The comparison is based on both dimensions, so if one dimension
                is greater but the other is not, the method will return
                False.

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

        Arguments:
            self ('ImageSize'): The calling ImageSize object.
            other ('ImageSize'): Another ImageSize object to compare with
                the calling object.

        Returns:
            bool: Returns True if the height and width of the calling object
                are greater than or equal to the height and width of the
                other object, otherwise returns False.

        Example:
            >>> img1 = ImageSize(800, 600)
            >>> img2 = ImageSize(600, 400)
            >>> img1.compare_size(img2)
            True

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

        Arguments:
            self (ImageSize): The ImageSize object for which the hash value
                is being calculated.

        Returns:
            int: A unique integer representing the hash value of the
                ImageSize object.

        Example:
            >>> image_size = ImageSize(800, 600)
            >>> print(image_size.calculate_hash())

        Note:
            The hash value is unique for each unique combination of height
                and width.

        """
        return hash((self.height, self.width))

    def __repr__(self) -> str:
        """Return a string representation of the ImageSize object.

        This method generates a string that represents the ImageSize object,
            including its height and width attributes. The string is in the
            format 'ImageSize(height=height_value, width=width_value)'.
        Arguments: None
        Returns:
            str: A string representation of the ImageSize object in the
                format 'ImageSize(height=height_value, width=width_value)'.

        Example:
            >>> image_size = ImageSize(800, 600)
            >>> print(image_size)
            ImageSize(height=800, width=600)

        Note:
            This method is typically used for debugging and logging.

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

        Arguments:
            image (Union[str, Image.Image, np.ndarray, torch.Tensor]): The
                input image to create an ImageSize instance from. It can be
                a file path (str), a PIL Image object, a NumPy array with
                shape 'h w c' or 'h w', or a PyTorch tensor with shape 'b c
                h w' or 'b 1 h w'.

        Returns:
            ImageSize: An instance of the ImageSize class representing the
                height and width of the input image.

        Example:
            >>> create_from_image(image)

        Note:
            The 'h w c' and 'b c h w' denote the dimensions of the image
                (height, width, channels) and tensor (batch size, channels,
                height, width) respectively.

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
def crop_border(
    imgs: list[
        UInt8[np.ndarray, "h w 3"]
        | UInt8[np.ndarray, "h w"]
        | Bool[np.ndarray, "h w"]
    ],
    crop_border: int,
) -> list[
    UInt8[np.ndarray, "h1 w1 3"]
    | UInt8[np.ndarray, "h1 w1"]
    | Bool[np.ndarray, "h1 w1"]
]:
    """Crop borders of input images.

    This function takes in a list of images or a single image and crops the
        borders based on the specified crop_border value.
    The cropping is applied to each end of the height and width of the
        image.

    Arguments:
        imgs (Union[List[np.ndarray], np.ndarray]): Input images to be
            cropped. The images should be in the form of numpy arrays with
            shape (height, width, channels).
        crop_border (int): The number of pixels to crop from each end of the
            height and width of the image.

    Returns:
        List[np.ndarray]: A list of cropped images in the form of numpy
            arrays.

    Example:
        >>> crop_images(imgs, 10)

    Note:
        The 'crop_border' argument should be less than half of the smallest
            dimension of the input images for the function to work
            correctly.

    """
    if crop_border == 0:
        return imgs
    if isinstance(imgs, list):
        return [
            v[crop_border:-crop_border, crop_border:-crop_border, ...]
            for v in imgs
        ]
    return imgs[crop_border:-crop_border, crop_border:-crop_border]


@jaxtyped(typechecker=beartype)
def center_pad(
    image: UInt8[np.ndarray, "h w c"],
    size: ImageSize,
    fill: int | tuple[int, int] = (0, 0),
) -> UInt8[np.ndarray, "h1 w1 c"]:
    """Pads an image to the center with a specified size and fill value.

    Arguments:
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

    Example:
        >>> pad_image(image, ImageSize(200, 200), fill=(255, 255, 255))

    Note:
        The function maintains the original image type in the output. If a
            NumPy array is provided as input, the output will also be a
            NumPy array, and vice versa for a PIL Image.

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

    Arguments:
        rgb (Union[Image.Image, np.array, torch.Tensor]): An image in RGB
            format, an array of UInt8 values with shape 'h w 3' or 'h w', or
            a torch tensor with shape '1 c h w'.

    Returns:
        Union[Image.Image, np.array, torch.Tensor]: The binary version of
            the input. If the input is an image or an array, the function
            returns the binary version of the input. If the input is a torch
            tensor, the function returns the binary version of the tensor.

    Example:
        >>> convert_to_binary(rgb_image)
        >>> convert_to_binary(array)
        >>> convert_to_binary(tensor)

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

    Arguments:
        rgb (Union[np.array, PIL.Image, torch.Tensor]): The input image in
            various formats such as numpy array, PIL Image, or torch tensor.

    Returns:
        Union[np.array, PIL.Image, torch.Tensor]: The input image converted
            to RGB format.

    Example:
        >>> to_rgb(input_image)

    Note:
        The function supports multiple input formats and ensures the output
            is in RGB format.

    """
    if isinstance(rgb, np.ndarray):
        return np.asarray(rgb)[..., None].repeat(3, axis=-1)
    if isinstance(rgb, Image.Image):
        return rgb.convert("RGB")
    return einops.repeat(rgb, "1 1 h w -> 1 c h w", c=3)


@jaxtyped(typechecker=beartype)
def convert_to_space_color(
    image_np: UInt8[np.ndarray, "h w 3"],
    space: str,
    /,
    *,
    getchannel: str | None = None,
) -> UInt8[np.ndarray, "h w 3"]:
    """Convert an image to a specified color space and optionally extract a.

        specific channel.

    Arguments:
        image (Union[np.ndarray, Image.Image]): The input image, which can
            be a numpy array or a PIL Image.
        space (str): The color space to which the image should be converted.
        getchannel (Optional[str]): Optional argument to extract a specific
            channel from the image. Defaults to None.

    Returns:
        Union[np.ndarray, Image.Image]: The converted image in the specified
            color space.

    Example:
        >>> convert_color_space(image, "RGB", getchannel="R")

    Note:
        The image input should be in the form of a numpy array or PIL Image.
            The color space can be any valid color space.

    """
    if getchannel is not None and len(getchannel) > 1:
        msg = "getchannel must be a single string"
        raise TypeError(msg)

    image = Image.fromarray(image_np).convert("RGB")
    if space == "LAB":
        # Convert to Lab colourspace
        srgb_p = ImageCms.createProfile("sRGB")
        lab_p = ImageCms.createProfile("LAB")

        rgb2lab = ImageCms.buildTransformFromOpenProfiles(
            srgb_p,
            lab_p,
            "RGB",
            "LAB",
        )
        image = cast(Image.Image, ImageCms.applyTransform(image, rgb2lab))
    else:
        image = image.convert(space)
    if getchannel is not None:
        image = image.getchannel(getchannel).convert("RGB")
    return np.asarray(image)


@jaxtyped(typechecker=beartype)
def threshold_image(
    image: UInt8[np.ndarray, "h w 3"] | UInt8[np.ndarray, "h w"] | Image.Image,
    mode: Literal["<", "<=", ">", ">="],
    /,
    *,
    threshold: int,
    replace_with: Literal[0, 255],
) -> Bool[np.ndarray, "h w"] | Image.Image:
    """Apply a thresholding operation to an image based on a specified mode and.

        threshold value.

    This function converts the input image to grayscale, compares pixel
        values with the specified threshold,
    and replaces the pixels that meet the threshold condition with a
        specified value. The thresholding modes
    can be '<', '<=', '>', or '>='.

    Arguments:
        image (Union[np.array, PIL.Image.Image]): Input image in the form of
            a NumPy array or PIL Image.
        mode (str): Thresholding mode. It can be '<', '<=', '>', or '>='.
        threshold (float): Threshold value for pixel comparison.
        replace_with (float): Value to replace pixels that meet the
            threshold condition.

    Returns:
        Union[np.array, PIL.Image.Image]: Thresholded image in the form of a
            NumPy array or PIL Image.

    Example:
        >>> apply_threshold(image, ">", 0.5, 1)

    Note:
        The input image is converted to grayscale before applying the
            thresholding operation.

    """
    is_np = False
    if isinstance(image, np.ndarray):
        is_np = True
        image = Image.fromarray(image)
    # Grayscale
    image = image.convert("L")
    # Threshold
    if mode == "<":
        image = image.point(
            lambda p: replace_with if p < threshold else 255 - replace_with,
        )
    elif mode == "<=":
        image = image.point(
            lambda p: replace_with if p <= threshold else 255 - replace_with,
        )
    elif mode == ">":
        image = image.point(
            lambda p: replace_with if p > threshold else 255 - replace_with,
        )
    elif mode == ">=":
        image = image.point(
            lambda p: replace_with if p >= threshold else 255 - replace_with,
        )
    else:
        msg = f"Mode {mode} not implemented!"
        raise TypeError(msg)
    if is_np:
        image = np.asarray(image).astype(bool)
    return image


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

    Arguments:
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

    Example:
        >>> resize_image(input_image, 500, "bilinear", "min", 2)

    Note:
        Ensure the input image is in a format compatible with the function.

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

    Arguments:
        rgb (np.array): A 3D NumPy array representing an RGB image. The
            dimensions represent height, width, and the three color channels
            (Red, Green, Blue).

    Returns:
        np.array: A 2D NumPy array representing the grayscale version of the
            input RGB image. The dimensions represent height and width.

    Example:
        >>> rgb_image = np.array([[[255, 0, 0], [0, 255, 0], [0, 0, 255]]])
        >>> grayscale_image = rgb_to_grayscale(rgb_image)

    Note:
        The input RGB image should have values in the range of 0-255. The
            output grayscale image will also have values in the range of
            0-255.

    """
    return np.dot(rgb[:, :, :3], [0.299, 0.587, 0.114])


@jaxtyped(typechecker=beartype)
def get_canny_edge(
    image: UInt8[np.ndarray, "h w 3"],
    threshold: tuple[int, int] = (100, 200),
    *,
    to_gray: bool = False,
) -> UInt8[np.ndarray, "h w"]:
    """Apply the Canny edge detection algorithm to an image.

    This function takes an image as input and applies the Canny edge
        detection algorithm to it.
    It can optionally convert the image to grayscale before applying the
        edge detection.

    Arguments:
        image (np.ndarray): The input image on which the Canny edge
            detection algorithm will be applied.
        threshold (tuple[int, int]): A tuple specifying the lower and upper
            thresholds for the edge detection algorithm. Defaults to (100,
            200).
        to_gray (bool): A flag indicating whether to convert the image to
            grayscale before applying the edge detection. Defaults to False.

    Returns:
        (np.ndarray): The output image after applying the Canny edge
            detection algorithm.

    Example:
        >>> apply_canny_edge_detection(image, (100, 200), to_gray=True)

    """
    if to_gray:
        image = rgb2gray(image).astype(np.uint8)
        image = np.expand_dims(image, axis=-1)
        image = np.repeat(image, 3, axis=-1)
    return cv2.Canny(image, *threshold)


@jaxtyped(typechecker=beartype)
def cv2_inpaint(
    image: np.ndarray | Image.Image,
    mask: np.ndarray | Image.Image,
    radius: int = 3,
) -> np.ndarray | Image.Image:
    """Inpaint a given masked image using OpenCV.

    This function takes an input image and its corresponding mask, then
        applies
    the inpainting algorithm of OpenCV to reconstruct the masked parts of
        the image.

    Arguments:
        image (np.ndarray): The input image, represented as a 2D or 3D numpy
            array.
        mask (np.ndarray): The mask of the image, represented as a 2D numpy
            array.
                            The mask should have the same size as the image.
            Non-zero
                            pixels in the mask correspond to the parts of
            the image to be inpainted.

    Returns:
        np.ndarray: The inpainted image, represented as a numpy array of the
            same shape
                    as the input image.

    Example:
        >>> inpainted_image = inpaint_image(image, mask)

    Note:
        The OpenCV inpainting algorithm used in this function assumes that
            the mask
        is a binary image where non-zero pixels correspond to the parts of
            the image to be inpainted.

    """
    is_pil = isinstance(image, Image.Image)
    if is_pil:
        image = np.asarray(image)
    if isinstance(mask, Image.Image):
        mask = np.asarray(mask)
    mask = (mask.astype(bool) * 255).astype(np.uint8)
    if mask.ndim == 3:
        mask = mask[:, :, 0]
    # inpaint = cv2.inpaint(image, mask, radius, cv2.INPAINT_TELEA)
    inpaint = cv2.inpaint(image, mask, radius, cv2.INPAINT_NS)
    # Blend the original image and the inpainted image using the mask
    blended_image = (
        image * (255 - mask)[:, :, None] + inpaint * mask[:, :, None]
    )
    if is_pil:
        blended_image = Image.fromarray(blended_image)
    return blended_image


@jaxtyped(typechecker=beartype)
def add_sensor_noise_and_jpeg_compression(
    images: list[str | Path],
    *,
    noise_level: tuple[float, float] | float = 1.0,
    jpeg_quality: tuple[float, float] | float = 95.0,
    rng: np.random.Generator | None = None,
) -> list[str | Path]:
    """Add sensor noise and apply JPEG compression to a list of images.

    This function takes in a list of image file paths, adds a specified
        level of sensor noise to each image, and then applies JPEG
        compression. The sensor noise level can range from 0 (no noise) to
        255 (maximum noise), and the JPEG compression quality can range from
        0 (lowest quality, highest compression) to 100 (highest quality,
        lowest compression).

    Arguments:
        images (List[str]): List of file paths to input images. Each path
            should be a string representing the absolute or relative path to
            an image file.
        noise_level (int): Level of sensor noise to add to each image. This
            should be an integer between 0 and 255, where 0 represents no
            noise and 255 represents maximum noise.
        jpeg_quality (int): Quality level for JPEG compression to apply to
            each image. This should be an integer between 0 and 100, where 0
            represents the lowest quality (highest compression) and 100
            represents the highest quality (lowest compression).

    Returns:
        List[str]: List of file paths to the processed images. Each path is
            a string representing the absolute or relative path to a
            processed image file.

    Example:
        >>> add_noise_and_compress(["image1.jpg", "image2.jpg"], 50, 75)

    Note:
        The function will overwrite the original images with the processed
            images. Make sure to backup your original images if necessary.

    """
    is_path = isinstance(images[0], Path)
    processed_images: list[str | Path] = []
    if isinstance(noise_level, float):
        noise_level = (noise_level, noise_level)
    if isinstance(jpeg_quality, float):
        jpeg_quality = (jpeg_quality, jpeg_quality)

    if rng is None:
        rng = np.random.default_rng()

    # select random values
    noise_level = rng.integers(noise_level[0], noise_level[1])
    jpeg_quality = rng.integers(jpeg_quality[0], jpeg_quality[1])

    for image_path in images:
        # Read the input image
        img = cv2.imread(str(image_path))

        # Add sensor noise
        noise = rng.normal(0, noise_level, img.shape).astype(np.uint8)
        noisy_image = cv2.add(img, noise)

        # Apply JPEG compression
        _, temp_output_path = tempfile.mkstemp(suffix=".jpg")
        cv2.imwrite(
            temp_output_path,
            noisy_image,
            [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality],
        )
        if is_path:
            processed_images.append(Path(temp_output_path))
        else:
            processed_images.append(temp_output_path)

    return processed_images


@jaxtyped(typechecker=beartype)
def tile(image: Image.Image, mode: str = "1x1") -> dict[str, Image.Image]:
    """Tile an input image into smaller images based on a specified mode.

    Arguments:
        image (Image.Image): The input image to be tiled.
        mode (str): The tiling mode specifying the number of tiles in the
            format 'NxM' where N is the number of rows and M is the number
            of columns. Defaults to '1x1'.

    Returns:
        Dict[str, Image.Image]: A dictionary containing the tiled images
            with keys representing the position of each tile in the format
            'NxM'.

    Example:
        >>> tile_image(my_image, "2x2")

    Note:
        The image size must be evenly divisible by the number of rows and
            columns specified in the mode.

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
