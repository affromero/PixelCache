import os
import random
from pathlib import Path
from typing import cast

import cv2
import numpy as np
import requests
import torch
import torchvision.utils as tv
from beartype import beartype
from jaxtyping import Bool, Float, Int64, UInt8
from PIL import Image, ImageDraw, ImageFont
from pydantic import ConfigDict
from pydantic.dataclasses import dataclass
from torchvision.io.image import (
    ImageReadMode,
    decode_jpeg,
    decode_png,
    read_file,
)
from torchvision.utils import make_grid

max_seed_value = np.iinfo(np.uint32).max
min_seed_value = np.iinfo(np.uint32).min


@beartype
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


@beartype
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


@beartype
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


@beartype
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


@beartype
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
    height: int
    width: int

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
            height=int(self.height * other), width=int(self.width * other)
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


@beartype
def crop_border(
    imgs: np.ndarray | list[np.ndarray],
    crop_border: int,
) -> np.ndarray | list[np.ndarray]:
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


@beartype
def draw_bbox(
    image: (
        Image.Image
        | UInt8[np.ndarray, "h w 3"]
        | Float[torch.Tensor, "1 c h w"]
    ),
    bbox: list[tuple[float, float, float, float] | tuple[int, int, int, int]],
    *,
    color: str | tuple[int, int, int] = "red",
    width: int = 3,
    text: list[str] | None = None,
) -> Image.Image | UInt8[np.ndarray, "h w 3"] | Float[torch.Tensor, "1 c h w"]:
    """Draw bounding boxes on an image with specified color, width, and text.

    Arguments:
        image (Union[PIL.Image, np.array, torch.Tensor]): The image to draw
            bounding boxes on.
            This can be a PIL Image, numpy array, or torch tensor.
        bbox (List[Tuple[int, int, int, int]]): A list of tuples
            representing bounding box coordinates (x1, y1, x2, y2).
        color (str, optional): The color of the bounding box outline.
            Defaults to 'red'.
        width (int, optional): The width of the bounding box outline.
            Defaults to 3.
        text (List[str], optional): A list of strings to display inside the
            bounding boxes. Must match the length of bbox. Defaults to None.

    Returns:
        Union[PIL.Image, np.array, torch.Tensor]: An image with bounding
            boxes drawn, in the same format as the input image.

    Example:
        >>> draw_bounding_boxes(image, [(10, 10, 50, 50)], 'blue', 2,
            ['object1'])

    Note:
        The function will not modify the original image, but return a new
            one with bounding boxes drawn.

    """
    if isinstance(image, np.ndarray):
        mode = "numpy"
        image = Image.fromarray(image)
    elif isinstance(image, torch.Tensor):
        mode = "torch"
        image = tensor2pil(image)
    else:
        mode = "pil"
        image = image.copy()
    image = image.convert("RGB")
    draw = ImageDraw.Draw(image)
    if text is not None and len(text) != len(bbox):
        msg = f"If text {text} {len(text)} is given, then it must match the length of the bboxes {len(bbox)}"
        raise ValueError(
            msg,
        )
    for idx, box in enumerate(bbox):
        if isinstance(box[0], float):
            if box[0] > 1 or box[1] > 1 or box[2] > 1 or box[3] > 1:
                msg = f"box {box} is not normalized [0,1] and it is float. If should not be normalized, please convert to int"
                raise ValueError(
                    msg,
                )
            w, h = image.size
            box = (
                int(box[0] * w),
                int(box[1] * h),
                int(box[2] * w),
                int(box[3] * h),
            )  # x1, y1, x2, y2
        draw.rectangle(box, outline=color, width=width)
        if text is not None and text[idx]:
            size_text = max(max(image.size) * 0.04, 9.0)
            get_font_path = (
                Path(__file__).parent / "fonts" / "JetBrainsMono-ExtraBold.ttf"
            )
            font = ImageFont.truetype(str(get_font_path), int(size_text))
            draw.text(
                (int(box[0]), int(box[1])),
                text[idx],
                font=font,
                fill=(0, 200, 255),
                align="left",
            )
    if mode == "numpy":
        return np.asarray(image)
    if mode == "torch":
        return pil2tensor(image)
    return image


@beartype
def bbox_iou(
    boxes1: Int64[torch.Tensor, "n 4"],
    boxes2: Int64[torch.Tensor, "m 4"],
    /,
    *,
    just_intersection: bool = False,
) -> Float[torch.Tensor, "n m"]:
    """Calculate the Intersection over Union (IoU) of two sets of bounding.

        boxes.

    This function computes the IoU or just the intersection area of two sets
        of bounding boxes.
    Each bounding box is represented by a 4-dimensional vector (x1, y1, x2,
        y2), where (x1, y1)
    is the top-left corner and (x2, y2) is the bottom-right corner.

    Arguments:
        boxes1 (torch.Tensor): A tensor of shape (n, 4) representing the
            first set of bounding boxes.
        boxes2 (torch.Tensor): A tensor of shape (m, 4) representing the
            second set of bounding boxes.
        just_intersection (bool): A flag indicating whether to return only
            the intersection area
                                  or the IoU value. Defaults to False.

    Returns:
        torch.Tensor: A tensor of shape (n, m) containing the IoU values if
            just_intersection is False,
                      otherwise a tensor of shape (n, m) containing the
            intersection areas.

    Example:
        >>> calculate_iou(boxes1, boxes2, just_intersection=False)

    Note:
        The bounding boxes are assumed to be in the format (x1, y1, x2, y2),
            where (x1, y1) is the
        top-left corner and (x2, y2) is the bottom-right corner.

    """
    # Calculate intersection coordinates
    inter_ymin = torch.max(
        boxes1[:, 0].unsqueeze(1),
        boxes2[:, 0].unsqueeze(0),
    )
    inter_xmin = torch.max(
        boxes1[:, 1].unsqueeze(1),
        boxes2[:, 1].unsqueeze(0),
    )
    inter_ymax = torch.min(
        boxes1[:, 2].unsqueeze(1),
        boxes2[:, 2].unsqueeze(0),
    )
    inter_xmax = torch.min(
        boxes1[:, 3].unsqueeze(1),
        boxes2[:, 3].unsqueeze(0),
    )

    # Calculate intersection area
    inter_area = torch.clamp(inter_ymax - inter_ymin, min=0) * torch.clamp(
        inter_xmax - inter_xmin,
        min=0,
    )

    # Calculate union area
    area_box1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area_box2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    union_area = area_box1.unsqueeze(1) + area_box2.unsqueeze(0) - inter_area

    # Calculate IoU
    if just_intersection:
        return inter_area.float()
    return inter_area / union_area if union_area != 0 else torch.FloatTensor(0)


def seed_everything(seed: int | None = None, *, workers: bool = False) -> int:
    """Set the seed for pseudo-random number generators in torch, numpy, and.

        Python's random module.

    This function also sets the following environment variables:
    - ``PL_GLOBAL_SEED``: Passed to spawned subprocesses (e.g., ddp_spawn
        backend).
    - ``PL_SEED_WORKERS``: Set to 1 if ``workers=True``.

    Arguments:
        seed (int | None): The seed for the global random state in
            Lightning. If ``None``,
            the function will read the seed from the ``PL_GLOBAL_SEED``
            environment variable.
            If both are ``None`` and the ``PL_GLOBAL_SEED`` environment
            variable is not set,
            then the seed defaults to 0.
        workers (bool): If set to ``True``, configures all dataloaders
            passed to the
            Trainer with a ``worker_init_fn``. If the user already provides
            such a function
            for their dataloaders, setting this argument will have no
            influence. See also:
    :func:`~lightning_fabric.utilities.seed.pl_worker_init_function`.
            Defaults to False.

    Returns:
        None
    Example:
        >>> set_seed(42, workers=True)

    Note:
        The function does not return any value. It modifies the global state
            of several modules and environment variables.

    """
    if seed is None:
        env_seed = os.environ.get("PL_GLOBAL_SEED")
        if env_seed is None:
            seed = 0
        else:
            try:
                seed = int(env_seed)
            except ValueError:
                seed = 0
    elif not isinstance(seed, int):
        seed = int(seed)

    if not (min_seed_value <= seed <= max_seed_value):
        seed = 0

    print(f"Seed set to {seed}")
    os.environ["PL_GLOBAL_SEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)  # noqa: NPY002
    torch.manual_seed(seed)
    os.environ["PL_SEED_WORKERS"] = f"{int(workers)}"

    return seed


@beartype
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
