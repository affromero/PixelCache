"""Spatial primitive types: `ImageCrop`, `BoundingBox`, `Points`.

These are content-addressable value types (`@dataclass` via pydantic).
Each one is independently hashable and pairs with HashableImage for
mask / crop / point operations.

`Points.to_mask()` constructs a `HashableImage` — that creates a
genuine import cycle (core.py → _types.py → core.py.HashableImage).
The resolution uses a documented function-local import (see
`Points.to_mask`).
"""

from typing import TYPE_CHECKING

import numpy as np
from jaxtyping import Float
from pydantic import ConfigDict
from pydantic.dataclasses import dataclass
from torchvision.transforms import functional as TF

from pixelcache._collections import HashableDict
from pixelcache.tools.image import ImageSize

if TYPE_CHECKING:
    from pixelcache.core import HashableImage


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

        Args:
            self (ImageCrop): An instance of the 'ImageCrop' class on which
                the method is called.

        Returns:
            None: This method does not return anything.

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

        Args:
            self (ImageCrop): The ImageCrop object for which the
                normalization is being checked.

        Returns:
            bool: True if the crop values are normalized, False otherwise.

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

        Args:
            self (ImageCrop): The ImageCrop object for which the string
                representation is being generated.

        Returns:
            str: A string representing the ImageCrop object with its left,
                top, right, and bottom attributes displayed.

        """
        return f"ImageCrop(left={self.left}, top={self.top}, right={self.right}, bottom={self.bottom})"

    def __hash__(self) -> int:
        """Calculate the hash value of an ImageCrop object.

        This method calculates the hash value of an ImageCrop object based
            on its attributes.

        Args:
            self (ImageCrop): The ImageCrop object for which the hash value
                is being calculated.

        Returns:
            int: The hash value of the ImageCrop object.

        """
        return hash((self.left, self.top, self.right, self.bottom))

    def __eq__(self, other: object) -> bool:
        """Compare two ImageCrop instances for equality.

        This method compares two ImageCrop instances to determine if they
            are equal.

        Args:
            self (ImageCrop): The ImageCrop object calling the method.
            other (object): The other object to compare with.

        Returns:
            bool: True if the two ImageCrop instances are equal, False
                otherwise.

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
        image: "HashableImage",
        /,
    ) -> "HashableImage":
        """Return a HashableImage cropped to this rectangle.

        Args:
            image: Source HashableImage to crop.

        Returns:
            New HashableImage containing only the cropped region.

        """
        # Documented circular import: HashableImage lives in pixelcache.core
        # which imports _types.py for BoundingBox / Points type annotations.
        from pixelcache.core import HashableImage

        # `TF.crop(img, top, left, height, width)` — height/width, NOT
        # bottom/right. The pre-fix version passed bottom/right and
        # also passed the HashableImage object (not image_pil) in the
        # normalized branch.
        image_pil = image.pil()
        if self.is_normalized():
            h = image_pil.height
            w = image_pil.width
            cropped = TF.crop(
                image_pil,
                round(self.top * h),
                round(self.left * w),
                round((self.bottom - self.top) * h),
                round((self.right - self.left) * w),
            )
        else:
            cropped = TF.crop(
                image_pil,
                round(self.top),
                round(self.left),
                round(self.bottom - self.top),
                round(self.right - self.left),
            )
        return HashableImage(cropped)


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
            msg = f"xmin must be smaller than xmax. Got {self.xmin} and {self.xmax}"
            raise ValueError(msg)
        if self.ymin >= self.ymax:
            msg = f"ymin must be smaller than ymax. Got {self.ymin} and {self.ymax}"
            raise ValueError(msg)
        if self.xmin < 0 or self.ymin < 0 or self.xmax < 0 or self.ymax < 0:
            msg = f"Bounding box coordinates must be positive. Got {self.xmin}, {self.ymin}, {self.xmax}, {self.ymax}"
            raise ValueError(msg)
        if not self.is_normalized() and self.image_size is None:
            msg = f"Image size must be provided for non-normalized bounding boxes. Got {self.xmin}, {self.ymin}, {self.xmax}, {self.ymax}"
            raise ValueError(msg)

    def is_normalized(self) -> bool:
        """Check if the bounding box coordinates are normalized.

        This method checks if the bounding box coordinates are normalized
            (i.e., if they are between 0 and 1).

        Args:
            self (BoundingBox): The BoundingBox object for which the
                normalization is being checked.

        Returns:
            bool: True if the bounding box coordinates are normalized,
                False otherwise.

        """
        return all(
            0 <= coord <= 1
            for coord in [self.xmin, self.ymin, self.xmax, self.ymax]
        )

    def size(self) -> ImageSize:
        """Get the size of the bounding box."""
        xyxy = self.xyxy
        return ImageSize(width=xyxy[2] - xyxy[0], height=xyxy[3] - xyxy[1])

    @property
    def xyxy(self) -> tuple[int, int, int, int]:
        """Calculate the minimum and maximum X and Y coordinates of the.

            bounding box.

        This method from the 'BoundingBox' class generates a tuple
            containing the X and Y coordinates
        of the minimum and maximum points of the bounding box.

        Args:
            None
        Returns:
            Tuple[int, int, int, int]: A tuple in the format (min_x,
                min_y, max_x, max_y)
            representing the minimum and maximum X and Y coordinates of the
                bounding box.

        """
        if self.is_normalized() and self.image_size is not None:
            return (
                int(self.xmin * self.image_size.width),
                int(self.ymin * self.image_size.height),
                int(self.xmax * self.image_size.width),
                int(self.ymax * self.image_size.height),
            )
        if not self.is_normalized():
            return (
                int(self.xmin),
                int(self.ymin),
                int(self.xmax),
                int(self.ymax),
            )

        msg = "Image size must be provided for normalized bounding boxes. Use xyxyn instead."
        raise ValueError(msg)

    @property
    def xywh(self) -> tuple[int, int, int, int]:
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

        """
        x, y, x2, y2 = self.xyxy
        return int(x), int(y), int(x2 - x), int(y2 - y)

    @property
    def xyxyn(self) -> tuple[float, float, float, float]:
        """Calculate the normalized minimum and maximum X and Y coordinates.

        This method generates a tuple containing the normalized X and Y
            coordinates of the minimum and maximum points of the bounding box.

        Args:
            None

        Returns:
            Tuple[float, float, float, float]: A tuple in the format (min_x,
                min_y, max_x, max_y)
            representing the normalized minimum and maximum X and Y
                coordinates of the bounding box.

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

        """
        return f"xmin: {self.xmin}, ymin: {self.ymin}, xmax: {self.xmax}, ymax: {self.ymax}"

    def __hash__(self) -> int:
        """Calculate the hash value for a BoundingBox object based on its.

            attributes.

        Args:
            self (BoundingBox): The BoundingBox object for which the hash
                value is being calculated.

        Returns:
            int: The hash value of the BoundingBox object.

        """
        return hash(HashableDict(self.__dict__))

    def __eq__(self, other: object) -> bool:
        """Compare bounding boxes by their fields, not just their hashes.

        Hash equality is only a necessary condition. Two BoundingBoxes
        whose hashes collide but whose coordinates or image_size differ
        must compare unequal — equality must hold even under hash
        collision.
        """
        if not isinstance(other, BoundingBox):
            return NotImplemented
        return (
            self.xmin == other.xmin
            and self.ymin == other.ymin
            and self.xmax == other.xmax
            and self.ymax == other.ymax
            and self.image_size == other.image_size
        )


@dataclass(
    config=ConfigDict(extra="forbid", arbitrary_types_allowed=True),
    kw_only=False,
)
class Points:
    """A class to represent a set of points in an image."""

    points: Float[np.ndarray, "_ 2"]
    """The points in the image, represented as a 2D NumPy array."""

    is_normalized: bool
    """A flag indicating whether the points are normalized."""

    image_size: ImageSize
    """The size of the image in which the points are located in case the points are not normalized."""

    def to_mask(self) -> "HashableImage":
        """Convert the points to a uint8 mask of size `image_size`."""
        # Documented circular import: HashableImage lives in pixelcache.core
        # which imports _types.py for BoundingBox / Points type annotations.
        from pixelcache.core import HashableImage

        mask = np.zeros(
            (self.image_size.height, self.image_size.width), dtype=np.uint8
        )
        xy = self.xy
        for point in xy:
            mask[int(point[1]), int(point[0])] = 255
        return HashableImage(mask)

    def clip_to_image_bounds(self, image_size: ImageSize) -> "Points":
        """Clip points to stay within the specified image bounds.

        Filters out points that are outside the image boundaries or have negative coordinates.
        Only keeps points where 0 <= x < width and 0 <= y < height.

        Args:
            image_size (ImageSize): The target image dimensions to clip points to.

        Returns:
            Points: A new Points object containing only points within the bounds.

        """
        clipped_xy = np.empty_like(self.xy)
        clipped_xy[:, 0] = np.clip(self.xy[:, 0], 0, image_size.width - 1)
        clipped_xy[:, 1] = np.clip(self.xy[:, 1], 0, image_size.height - 1)
        return Points(
            points=clipped_xy,
            is_normalized=False,
            image_size=image_size,
        )

    @property
    def num_points(self) -> int:
        """Return the number of points in the Points object.

        This method returns the number of points in the Points object.

        Args:
            self (Points): The Points object for which the number of points
                is to be calculated.

        Returns:
            int: The number of points in the Points object.

        """
        return self.points.shape[0]

    @property
    def xy(self) -> Float[np.ndarray, "_ 2"]:
        """Return pixel-space `(x, y)` coordinates.

        If points are stored in normalized form, scale by
        `(width, height)` — `x` along width, `y` along height. The
        pre-fix code multiplied by `(height, width)`, which flipped the
        axes for non-square images.
        """
        if self.is_normalized:
            return self.points * np.array(
                [self.image_size.width, self.image_size.height]
            )
        return self.points

    @property
    def xyn(self) -> Float[np.ndarray, "_ 2"]:
        """Return normalized `(x/width, y/height)` coordinates in `[0, 1]`."""
        if not self.is_normalized:
            return self.points / np.array(
                [self.image_size.width, self.image_size.height]
            )
        return self.points

    def shift_points(self, shift: tuple[float, float]) -> "Points":
        """Shift the points in the Points object by a specified amount.

        This method shifts the points in the Points object by a specified
            amount.

        Args:
            self (Points): The Points object for which the points are to be
                shifted.
            shift (Tuple[float, float]): A tuple containing the X and Y
                coordinates by which the points are to be shifted.

        Returns:
            Points: A new Points object with the shifted points.

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

        Args:
            self (Points): The Points object for which the points are to be
                converted to a list of tuples of integers.

        Returns:
            List[Tuple[int, int]]: A list of tuples containing the X and Y
                coordinates of the points as integers.

        """
        return [(int(x), int(y)) for x, y in self.xy]

    def list_tuple_float(
        self, *, normalized: bool
    ) -> list[tuple[float, float]]:
        """Return the points as a list of tuples of floats.

        This method returns the points in the Points object as a list of
            tuples of floats.

        Args:
            self (Points): The Points object for which the points are to be
                converted to a list of tuples of floats.

        Returns:
            List[Tuple[float, float]]: A list of tuples containing the X and
                Y coordinates of the points as floats.

        """
        if normalized:
            return [(float(x), float(y)) for x, y in self.xyn]
        return [(float(x), float(y)) for x, y in self.xy]

    def min_x(self) -> int:
        """Return the minimum X coordinate of the points.

        This method returns the minimum X coordinate of the points in the
            Points object.

        Args:
            self (Points): The Points object for which the minimum X
                coordinate is to be calculated.

        Returns:
            int: The minimum X coordinate of the points.

        """
        return int(np.min(self.xy[:, 0]))

    def min_y(self) -> int:
        """Return the minimum Y coordinate of the points.

        This method returns the minimum Y coordinate of the points in the
            Points object.

        Args:
            self (Points): The Points object for which the minimum Y
                coordinate is to be calculated.

        Returns:
            int: The minimum Y coordinate of the points.

        """
        return int(np.min(self.xy[:, 1]))

    def max_x(self) -> int:
        """Return the maximum X coordinate of the points.

        This method returns the maximum X coordinate of the points in the
            Points object.

        Args:
            self (Points): The Points object for which the maximum X
                coordinate is to be calculated.

        Returns:
            int: The maximum X coordinate of the points.

        """
        return int(np.max(self.xy[:, 0]))

    def max_y(self) -> int:
        """Return the maximum Y coordinate of the points.

        This method returns the maximum Y coordinate of the points in the
            Points object.

        Args:
            self (Points): The Points object for which the maximum Y
                coordinate is to be calculated.

        Returns:
            int: The maximum Y coordinate of the points.

        """
        return int(np.max(self.xy[:, 1]))

    def __len__(self) -> int:
        """Return the number of points in the Points object.

        This method returns the number of points in the Points object.

        Args:
            self (Points): The Points object for which the number of points
                is to be calculated.

        Returns:
            int: The number of points in the Points object.

        """
        return self.num_points

    def __hash__(self) -> int:
        """Calculate the hash value of a Points object.

        This method calculates the hash value of a Points object by
            converting its dictionary attributes into a hashable format.

        Args:
            self (Points): The Points object for which the hash value is
                being calculated.

        Returns:
            int: An integer representing the hash value of the Points object.

        """
        return hash(HashableDict(self.__dict__))

    def __eq__(self, other: object) -> bool:
        """Compare Points by content, not just by hash.

        `np.array_equal` handles both shape and value comparison
        correctly; `is_normalized` and `image_size` round out the
        identity. Hash equality is only a necessary condition.
        """
        if not isinstance(other, Points):
            return NotImplemented
        return (
            np.array_equal(self.points, other.points)
            and self.is_normalized == other.is_normalized
            and self.image_size == other.image_size
        )
