import random
import string
import tempfile
from collections.abc import Iterable, Iterator, MutableMapping, MutableSequence
from numbers import Number
from pathlib import Path
from typing import (
    Any,
    Literal,
    SupportsIndex,
    TypeAlias,
    TypeVar,
    cast,
    overload,
)

import cv2
import numpy as np
import torch
from beartype import beartype
from jaxtyping import Bool, Float, UInt8
from PIL import Image, ImageOps

from pixelcache.tools.cache_tools import lru_cache
from pixelcache.tools.text_tools import create_text, draw_text
from pixelcache.tools.utils import (
    ImageSize,
    make_image_grid,
    numpy2tensor,
    pil2tensor,
    read_image,
    save_image,
    tensor2numpy,
    tensor2pil,
)
from pixelcache.transforms import (
    bbox2mask,
    center_pad,
    convert_to_space_color,
    crop_from_bbox,
    crop_from_mask,
    group_regions_from_binary,
    mask2bbox,
    mask2squaremask,
    mask_blend,
    morphologyEx,
    polygon_to_mask,
    to_binary,
    uncrop_from_bbox,
)

_T = TypeVar("_T")
_KT = TypeVar("_KT")
_VT = TypeVar("_VT")

_BBOX_TYPE: TypeAlias = (
    tuple[int, int, int, int] | tuple[float, float, float, float]
)

MAX_IMG_CACHE = 5
VALID_IMAGES = Literal["pil", "numpy", "torch"]


@beartype
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


@beartype
class HashableImage:
    def __init__(
        self,
        image: (
            str
            | Path
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
            image (Union[str, Path, Image, np.ndarray, torch.Tensor,
                np.bool_]): The input image data. This can be a string file
                path, Path object, PIL Image object, numpy array, torch
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
            self.__image = image
        elif isinstance(image, str | Path):
            self.__image = read_image(image)
        elif isinstance(image, Image.Image):
            self.__image = image
        else:
            self.__image = image

        if isinstance(image, str | Path):
            self.__image_str = str(image)
        else:
            # set a tmp unique file
            self.__image_str = tempfile.NamedTemporaryFile(suffix=".png").name
            self.save(self.__image_str)

    @property
    def __mode(self) -> VALID_IMAGES:
        if isinstance(self.__image, torch.Tensor):
            return "torch"
        if isinstance(self.__image, np.ndarray):
            return "numpy"
        if isinstance(self.__image, Image.Image):
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
        return self.__image_str

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
        self.__image_str = filename

    def save(self, path: Path | str) -> None:
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
        save_image(self.__image, path=str(path), normalize=False)

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

    def downsample(self, factor: int) -> "HashableImage":
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
            height=int(self.size().height // factor),
            width=int(self.size().width // factor),
        )
        return self.resize(new_size)

    def resize(self, size: ImageSize) -> "HashableImage":
        """Resize the image to a specified size using different interpolation.

            methods based on the image mode.

        Arguments:
            self (HashableImage): The instance of the HashableImage class.
            size (ImageSize): An object containing the desired image height
                and width.

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
            if self.__mode == "torch":
                __image = torch.nn.functional.interpolate(
                    self.__image,
                    size=(height, width),
                    mode="bilinear",
                    align_corners=False,
                )
            elif self.__mode == "pil":
                __image = self.__image.resize(
                    (width, height), Image.Resampling.LANCZOS
                )
            else:
                __image = cv2.resize(
                    cast(np.ndarray, self.__image),
                    (width, height),
                    interpolation=cv2.INTER_LANCZOS4,
                )
            return HashableImage(__image)
        return self

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
        if self.__mode == "torch":
            return torch.sum(self.__image).item() == 0
        if self.__mode == "numpy":
            return np.sum(cast(np.ndarray, self.__image)).item() == 0
        return np.sum(np.asarray(self.__image)).item() == 0

    @lru_cache(maxsize=MAX_IMG_CACHE)
    def to_gray(self) -> "HashableImage":
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
        if self.__mode == "torch":
            if self.__image.shape[1] == 3:
                return HashableImage(
                    self.__image.mean(1, keepdim=True).float(),
                )
            return self
        if self.__mode == "numpy":
            if len(self.__image.shape) == 3 and self.__image.shape[2] == 3:
                return HashableImage(
                    cv2.cvtColor(self.__image, cv2.COLOR_RGB2GRAY),
                )
            return self
        return HashableImage(self.__image.convert("L"))

    @lru_cache(maxsize=MAX_IMG_CACHE)
    def flip_lr(self) -> "HashableImage":
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
        if self.__mode == "torch":
            return HashableImage(torch.flip(self.__image, [3]))
        return HashableImage(cv2.flip(self.__image, 1))

    @lru_cache(maxsize=MAX_IMG_CACHE)
    def to_rgb(self) -> "HashableImage":
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
        if self.__mode == "torch":
            if self.__image.shape[1] == 1:
                return HashableImage(self.__image.repeat(1, 3, 1, 1).float())
            return self
        if self.__mode == "numpy":
            if len(self.__image.shape) == 2:
                if self.__image.dtype == bool:
                    return HashableImage(
                        cv2.cvtColor(
                            (self.__image * 255).astype(np.uint8),
                            cv2.COLOR_GRAY2RGB,
                        ),
                    )
                return HashableImage(
                    cv2.cvtColor(self.__image, cv2.COLOR_GRAY2RGB),
                )
            return self
        return HashableImage(self.__image.convert("RGB"))

    @lru_cache(maxsize=MAX_IMG_CACHE)
    def to_binary(self, threshold: float = 0.0) -> "HashableImage":
        """Convert an image to binary format.

        This function does not take any arguments. It uses the global state
            of the program to find the image to convert.

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
            self.__mode == "torch"
            and self.__image.dtype == torch.bool
            or self.__mode == "numpy"
            and self.__image.dtype == bool
            or self.__mode == "pil"
            and self.__image.mode == "1"
        ):
            return self
        return HashableImage(to_binary(self.__image, threshold=threshold))

    @lru_cache(maxsize=MAX_IMG_CACHE)
    def unique_values(self) -> tuple[list[float], torch.Tensor, list[float]]:
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
    def split_masks(
        self,
        closing: tuple[int, int] = (0, 0),
        margin: float = 0.0,
        area_threshold: float = 0.0,
    ) -> "HashableList[HashableImage]":
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
                    self.pil(),
                    closing=closing,
                    margin=margin,
                    area_threshold=area_threshold,
                )
            ],
        )

    @lru_cache(maxsize=MAX_IMG_CACHE)
    def invert_binary(self) -> "HashableImage":
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
        if self.__mode == "torch":
            return HashableImage(~self.to_binary().tensor())
        return HashableImage(~self.to_binary().numpy())

    @lru_cache(maxsize=MAX_IMG_CACHE)
    def invert_rgb(self) -> "HashableImage":
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
        if self.__mode == "torch":
            return HashableImage(1 - self.tensor())
        return HashableImage(255 - self.numpy())

    @staticmethod
    def zeros_from_size(size: ImageSize) -> "HashableImage":
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

    def zeros_like(self) -> "HashableImage":
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
        if self.__mode == "torch":
            return HashableImage(torch.zeros_like(self.__image))
        if self.__mode == "numpy":
            return HashableImage(np.zeros_like(self.__image))
        return HashableImage(
            Image.new(self.__image.mode, self.__image.size, 0),
        )

    def ones_like(self) -> "HashableImage":
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
        if self.__mode == "torch":
            return HashableImage(torch.ones_like(self.__image))
        if self.__mode == "numpy":
            return HashableImage(np.ones_like(self.__image))
        return HashableImage(
            Image.new(
                self.__image.mode,
                self.__image.size,
                255 if self.__image.mode != "RGB" else (255, 255, 255),
            ),
        )

    def rgb2bgr(self) -> "HashableImage":
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
        if self.__mode == "numpy":
            return HashableImage(cv2.cvtColor(self.__image, cv2.COLOR_RGB2BGR))
        if self.__mode == "pil":
            return HashableImage(
                Image.fromarray(
                    cv2.cvtColor(np.asarray(self.__image), cv2.COLOR_RGB2BGR),
                ),
            )
        return HashableImage(self.__image[:, [2, 1, 0], :, :])

    def equalize_hist(self) -> "HashableImage":
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
        if self.__mode == "pil":
            return HashableImage(ImageOps.equalize(self.__image))
        return HashableImage(cv2.equalizeHist(self.numpy()))

    def to_space_color(
        self, color_space: str, getchannel: str | None = None
    ) -> "HashableImage":
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
                self.__image, color_space, getchannel=getchannel
            )
        )

    def __add__(self, other: object) -> "HashableImage":
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
        if self.__mode == "torch":
            other_value = (
                other if isinstance(other, Number) else other.tensor()
            )
            return HashableImage((self.tensor() + other_value).clamp(0, 1))
        other_value = other if isinstance(other, Number) else other.numpy()
        return HashableImage((self.numpy() + other_value).clip(0, 255))

    def __sub__(self, other: object) -> "HashableImage":
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
        if self.__mode == "torch":
            other_value = (
                other if isinstance(other, Number) else other.tensor()
            )
            return HashableImage((self.tensor() - other_value).clamp(0, 1))
        other_value = other if isinstance(other, Number) else other.numpy()
        return HashableImage((self.numpy() - other_value).clip(0, 255))

    def __mul__(self, other: object) -> "HashableImage":
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
        if self.__mode == "torch":
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

    def __truediv__(self, other: object) -> "HashableImage":
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
        if self.__mode == "torch":
            other_value = (
                other if isinstance(other, Number) else other.tensor()
            )
            return HashableImage((self.tensor() / other_value).clamp(0, 1))
        other_value = other if isinstance(other, Number) else other.numpy()
        return HashableImage(
            (self.numpy() / other_value).clip(0, 255).astype(np.uint8),
        )

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
        return ImageSize.from_image(self.__image)

    def copy(self) -> "HashableImage":
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
        if self.__mode == "torch":
            image = HashableImage(self.__image.clone())
        else:
            image = HashableImage(self.__image.copy())
        image.set_filename(self.get_filename())
        return image

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
        if self.__mode == "torch":
            value = self.__image.float().mean().item()
        else:
            value = np.mean(self.__image)
        # two decimal places
        return round(value, 2)

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
        if self.__mode == "torch":
            value = self.__image.float().std().item()
        else:
            value = np.std(self.__image)
        return round(value, 2)

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
        if self.__mode == "torch":
            value = self.__image.float().min().item()
        else:
            value = float(np.min(self.__image))
        return round(value, 2)

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
        if self.__mode == "torch":
            value = self.__image.float().max().item()
        else:
            value = float(np.max(self.__image))
        return round(value, 2)

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
        if self.__mode == "torch":
            value = self.__image.float().sum().item()
        else:
            value = float(np.sum(self.__image))
        return round(value, 2)

    def dtype(self) -> str:
        """Return a string representing the data type and channels of the.

            image.

        This method in the 'HashableImage' class determines the image's data
            type and channels based on its mode (torch or numpy).

        Arguments:
            self (HashableImage): The instance of the 'HashableImage' class.

        Returns:
            str: A string representing the data type and channels of the
                image.

        Example:
            >>> image = HashableImage(...)
            >>> print(image.dtype())

        Note:
            The image's mode (torch or numpy) influences the returned
                string.

        """
        if self.__mode == "torch":
            channels = "RGB" if self.__image.shape[1] == 3 else "L"
            return f"{self.__image.dtype} {channels}"
        if self.__mode == "numpy":
            channels = (
                "RGB"
                if len(self.__image.shape) == 3 and self.__image.shape[2] == 3
                else "L"
            )
            return f"{self.__image.dtype} {channels}"
        return str(self.__image.mode)

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
            self.get_filename() if "/tmp" not in self.get_filename() else ""  # noqa: S108
        )
        return f"HashableImage: {self.__mode} {self.dtype()} {self.size()} - mean: {self.mean()} std: {self.std()} min {self.min()} max {self.max()}{_filename}"

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
        if self.__mode == "torch":
            return tensor2pil(self.__image)
        if self.__mode == "numpy":
            return Image.fromarray(self.__image)
        return self.__image

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
        if self.__mode == "torch":
            return tensor2numpy(
                self.__image,
                output_type=(
                    bool if self.__image.dtype == torch.bool else np.uint8
                ),
            )
        if self.__mode == "numpy":
            return self.__image
        return np.asarray(self.__image)

    @lru_cache(maxsize=MAX_IMG_CACHE)
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
        if self.__mode == "torch":
            return self.__image
        if self.__mode == "numpy":
            return numpy2tensor(self.__image)
        return pil2tensor(self.__image)

    @property
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
        return self.__mode

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
        if self.__mode == "torch":
            return self.__image.dtype == torch.bool
        return self.__image.dtype == bool

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
        if self.__mode == "torch":
            return self.__image.shape[1] == 3
        return len(self.__image.shape) == 3 and self.__image.shape[2] == 3

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

    def concat(
        self,
        other: list["HashableImage"],
        mode: Literal["horizontal", "vertical"],
    ) -> "HashableImage":
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
        if self.__mode == "torch":
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
        return self.__image

    def logical_and(self, other: "HashableImage") -> "HashableImage":
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
        if self.__mode == "torch":
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

    def logical_and_reduce(
        self,
        other: list["HashableImage"],
    ) -> "HashableImage":
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
        if self.__mode == "torch":
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

    def logical_or(self, other: "HashableImage") -> "HashableImage":
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
        if self.__mode == "torch":
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
        if self.__mode == "torch":
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
        if self.__mode == "torch":
            return hash(self.__image)
        return hash(self.__image.tobytes())

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
        if self.__mode != other.mode:
            return False
        if self.__mode == "torch":
            return torch.equal(self.__image, other.__image)
        return self.__image.tobytes() == other.__image.tobytes()

    @lru_cache(maxsize=MAX_IMG_CACHE)
    def crop_from_mask(
        self,
        mask: "HashableImage",
        **kwargs: Any,
    ) -> "HashableImage":
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
    def crop_from_bbox(
        self,
        bboxes: "HashableList[_BBOX_TYPE]",
    ) -> "HashableImage":
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
        if isinstance(bboxes[0][0], float):
            _bboxes = [
                (
                    max(0.0, bbox[0]),
                    max(0.0, bbox[1]),
                    min(0.999, bbox[2]),
                    min(0.999, bbox[3]),
                )
                for bbox in bboxes.to_list()
            ]
        else:
            _bboxes = [
                (
                    max(0, bbox[0]),
                    max(0, bbox[1]),
                    min(self.size().width - 1, bbox[2]),
                    min(self.size().height - 1, bbox[3]),
                )
                for bbox in bboxes.to_list()
            ]
        return HashableImage(crop_from_bbox(self.to_rgb().numpy(), _bboxes))

    @lru_cache(maxsize=MAX_IMG_CACHE)
    def uncrop_from_bbox(
        self,
        base: "HashableImage",
        bboxes: "HashableList[_T]",
        *,
        resize: bool = False,
    ) -> "HashableImage":
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
        return HashableImage(
            uncrop_from_bbox(
                base.to_rgb().numpy(),
                self.to_rgb().numpy(),
                bboxes.to_list(),
                resize=resize,
            )
        )

    @lru_cache(maxsize=MAX_IMG_CACHE)
    def mask2bbox(self, **kwargs: Any) -> "HashableList[_BBOX_TYPE]":
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
            **kwargs: Additional keyword arguments to pass to the mask2bbox
                function.

        Returns:
            HashableList: A list containing the bounding box coordinates
                generated from the mask image.

        Example:
            >>> mask_to_bbox(self, **kwargs)

        Note:
            The mask2bbox function must be compatible with the provided
                kwargs.

        """
        return HashableList(mask2bbox(self.to_binary().numpy(), **kwargs))

    @lru_cache(maxsize=MAX_IMG_CACHE)
    def mask2squaremask(self, **kwargs: Any) -> "HashableImage":
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

    def draw_points(
        self,
        points: Float[np.ndarray, "n 2"],
        color: tuple[int, int, int],
        radius: int,
        thickness: int,
        *,
        from_normalized: bool = True,
        epsilon: float = 1e-6,
    ) -> "HashableImage":
        """Draw circles at specified points on an image.

        Arguments:
            points (np.ndarray): A numpy array of floating-point values
                representing the coordinates (X, Y) of the points where circles
                will be drawn. The shape of the array should be (n, 2),
                where n is the number of points. Each point should be in the
                range [0, 1]. The points are assumed to be within the bounds
                of the image.
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
        for point in points:
            x, y = point
            if not from_normalized:
                x /= canvas.shape[1]
                y /= canvas.shape[0]
            if (
                x < epsilon
                or y < epsilon
                or x > 1 - epsilon
                or y > 1 - epsilon
            ):
                continue
            x = int(x * canvas.shape[1])
            y = int(y * canvas.shape[0])
            cv2.circle(canvas, (x, y), radius, color, thickness)
        return HashableImage(canvas)

    def morphologyEx(  # noqa: N802
        self,
        operation: Literal["erode", "dilate", "open", "close"],
        kernel: np.ndarray,
    ) -> "HashableImage":
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

    def draw_polygon(
        self,
        points: list[tuple[int, int]],
        alpha: float = 0.5,
        add_text: str = "",
    ) -> "HashableImage":
        """Draw a polygon on an image and return the modified image.

        This method in the 'HashableImage' class draws a polygon on an image
            using the provided points. It also allows for an optional text
            to be added to the image.

        Arguments:
            points (List[Tuple[int, int]]): A list of tuples containing the
                coordinates of the points that define the polygon.
            alpha (float, optional): A float value representing the
                transparency of the polygon. Defaults to 0.5.
            add_text (str, optional): A string representing the text to be
                added to the image. Defaults to ''.

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
                points,
                image_shape=(int(self.size().height), int(self.size().width)),
            ),
        )
        out = self.blend(mask, alpha, with_bbox=True)
        if add_text:
            # add text to the image in the upper left corner of the polygon
            x_min = min([point[0] for point in points])
            y_min = min([point[1] for point in points])
            out = out.draw_text(
                add_text,
                (x_min, y_min),
                font_size=max(self.size().min() * 0.01, 30.0),
                color=(255, 255, 0),
            )
        return out

    def draw_bbox(
        self,
        bbox: tuple[int, int, int, int],
        alpha: float = 0.5,
        add_text: str = "",
        color: tuple[int, int, int] = (255, 255, 0),
    ) -> "HashableImage":
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
        mask = bbox2mask([bbox], self.size())
        out = self.blend(mask, alpha, with_bbox=True, merge_bbox=True)
        if add_text:
            # add text to the image in the upper left corner of the bbox
            x_min, y_min, _, _ = bbox
            out = out.draw_text(
                add_text,
                (x_min, y_min),
                font_size=max(self.size().min() * 0.01, 30.0),
                color=color,
            )
        return out

    def draw_lines(
        self,
        lines: list[tuple[tuple[int, int], tuple[int, int]]],
        color: tuple[int, int, int],
        thickness: int,
    ) -> "HashableImage":
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

    def draw_text(
        self,
        text: str,
        coord_xy: tuple[int, int],
        font_size: float = 20.0,
        color: tuple[int, int, int] = (255, 255, 255),
    ) -> "HashableImage":
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

    def center_pad(
        self,
        image_size: ImageSize,
        fill: int = 0,
    ) -> "HashableImage":
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

    @staticmethod
    def make_image_grid(
        images: "HashableDict[str, HashableList[HashableImage]]",
        *,
        orientation: Literal["horizontal", "vertical"] = "horizontal",
        with_text: bool = False,
    ) -> "HashableImage":
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
            dict[str, list["HashableImage"]],
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
        for idx in range(max_images):
            row_images = [imgs[idx].pil() for imgs in image_as_list.values()]
            all_images.extend(row_images)
        nrows = (
            len(list(image_as_list.values()))
            if orientation == "vertical"
            else max_images
        )
        ncols = (
            max_images
            if orientation == "vertical"
            else len(list(image_as_list.values()))
        )

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

    def set_minmax(self, _min: float, _max: float, /) -> "HashableImage":
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

    def __setitem__(
        self, mask: "HashableImage", value: float, /
    ) -> "HashableImage":
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
                items[k] = v.tobytes()
            else:
                items[k] = v
        return hash(frozenset(items))

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

    def copy(self) -> "HashableDict[_KT, _VT]":
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

    def __getitem__(self, __name: _KT) -> _VT:
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
        return self.__data[__name]

    def __setitem__(self, __name: _KT, __value: _VT) -> None:
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

    def __delitem__(self, __name: _KT) -> None:
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
                items.append(self.__data[idx].tobytes())  # type: ignore[attr-defined]
            else:
                items.append(self.__data[idx])
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

    def copy(self) -> "HashableList[_T]":
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
    def __getitem__(self, __index: slice, /) -> "HashableList[_T]": ...

    def __getitem__(
        self,
        __index: SupportsIndex | slice,
    ) -> _T | "HashableList[_T]":
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
    def __delitem__(self, __index: int) -> None: ...

    @overload
    def __delitem__(self, __index: slice) -> None: ...

    def __delitem__(self, __index: int | slice) -> None:
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

    def insert(self, __index: int, __value: _T) -> None:
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

    def __mul__(self, other: int) -> "HashableList[_T]":
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
