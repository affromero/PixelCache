import tempfile
from pathlib import Path
from typing import Any, Literal, cast

import cv2
import einops
import numpy as np
import torch
from beartype import beartype
from jaxtyping import Bool, Float, UInt8
from PIL import Image, ImageCms

from pixelcache.tools.logger import get_logger
from pixelcache.tools.utils import (
    ImageSize,
    bbox_iou,
    draw_bbox,
    numpy2tensor,
    tensor2numpy,
)

logger = get_logger()


@beartype
def keep_n_largest_components(
    binary_mask: Bool[np.ndarray, "h w"],
    n: int,
) -> Bool[np.ndarray, "h w"]:
    """Select the n largest connected components from a binary mask image.

    This function takes a binary mask image and an integer n as input and
    returns a new binary mask image containing only the n largest connected
    components from the input mask.

    Arguments:
        binary_mask (np.array): A boolean NumPy array representing the
            binary mask image.
        n (int): An integer specifying the number of largest connected
            components to keep.

    Returns:
        np.array: A boolean NumPy array representing the binary mask image
            containing
        only the n largest connected components.

    Example:
        >>> select_largest_components(binary_mask, 3)

    Note:
        The binary mask should only contain boolean values (True or False).

    """
    # Find connected components
    _, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary_mask,
        connectivity=8,
    )

    # Calculate the size of each connected component
    component_sizes = stats[1:, cv2.CC_STAT_AREA]

    # Find the indices of the N largest connected components
    largest_component_indices = (
        np.argsort(component_sizes)[-n:] + 1
    )  # Adding 1 because stats includes the background label

    # Create a new binary mask with the N largest connected components
    largest_components_mask = np.isin(labels, largest_component_indices)

    return largest_components_mask.astype(bool)


@beartype
def reduce_masks(
    mask: Bool[np.ndarray, "h w"],
    *,
    number_of_objects: int = -1,
    closing: tuple[int, int] = (0, 0),
    opening: tuple[int, int] = (0, 0),
    area_threshold: float = 0.0,
    merge: bool = False,
    verbose: bool = True,
) -> list[Bool[np.ndarray, "h w"]]:
    """Simplify an image mask by applying morphological operations.

    This function takes an image mask or a boolean numpy array as input and
        applies various morphological operations
    to reduce the number of objects in the mask. It can perform operations
        like opening, closing, area thresholding,
    and merging of objects.

    Arguments:
        mask (Union[Image.Image, np.ndarray]): Input mask as an Image.Image
            object or a boolean numpy array.
        number_of_objects (int, optional): Number of objects to keep in the
            mask. Defaults to -1, which keeps all objects.
        closing (Tuple[int, int], optional): Tuple specifying the kernel
            size for morphological closing operation. Defaults to None.
        opening (Tuple[int, int], optional): Tuple specifying the kernel
            size for morphological opening operation. Defaults to None.
        area_threshold (int, optional): Minimum area threshold for connected
            components to be retained. Defaults to None.
        merge (bool, optional): Boolean flag to merge adjacent objects.
            Defaults to False.
        verbose (bool, optional): Boolean flag to enable/disable verbose
            logging. Defaults to False.

    Returns:
        List[np.ndarray]: A list of boolean numpy arrays representing the
            reduced masks after applying the specified operations.

    Example:
        >>> simplify_mask(mask, number_of_objects=2, closing=(5,5),
            opening=(3,3), area_threshold=500, merge=True, verbose=True)

    Note:
        The function does not modify the original mask but returns a new
            one.

    """
    # count the number of separated bynary objects
    if isinstance(mask, Image.Image):
        mask_np: Bool[np.ndarray, "h w"] = np.asarray(to_binary(mask))
    else:
        mask_np = mask
    if sum(opening) > 0:
        mask_np = morphologyEx(mask_np, cv2.MORPH_OPEN, np.ones(opening))
    if sum(closing) > 0:
        mask_np = morphologyEx(mask_np, cv2.MORPH_CLOSE, np.ones(closing))
    if area_threshold > 0:
        mask_np = remove_disconnected_regions([mask_np], area_threshold)[0]
    num_objects = cv2.connectedComponents((mask_np * 255).astype(np.uint8))[0]
    # bboxes = mask2bbox(mask_np, margin=margin)
    # num_objects = len(bboxes)
    if number_of_objects == -1:
        if verbose:
            logger.debug(f"Found {num_objects-1} objects.")
    elif num_objects >= number_of_objects + 1:  # +1 for the background
        if verbose:
            logger.debug(
                f"Found {num_objects-1} objects, but only {number_of_objects} are allowed. Keeping the {number_of_objects} largest objects.",
            )
        # only keep the largest objects
        mask_np = keep_n_largest_components(mask_np, n=number_of_objects)
    elif verbose:
        logger.error(
            f"Found {num_objects-1} objects, and only {number_of_objects} are allowed.",
        )
    # separate the connectedcomponent in different masks
    mask_segm: UInt8[np.ndarray, "h w"] = cv2.connectedComponentsWithStats(
        (mask_np * 255).astype(np.uint8),
    )[1]
    # split segm into different masks
    mask_list: list[Bool[np.ndarray, "h w"]] = []
    for i in range(1, mask_segm.max() + 1):
        mask_i: Bool[np.ndarray, "h w"] = np.zeros_like(mask_np)
        mask_i[mask_segm == i] = True
        if merge and len(mask_list):
            mask_i = np.logical_or(mask_i, mask_list[-1])
        mask_list.append(mask_i)
    if merge:
        mask_list = [mask_list[-1]]
    return mask_list


@beartype
def remove_small_regions(
    mask: np.ndarray,
    area_thresh: float,
    mode: Literal["holes", "islands"],
) -> tuple[np.ndarray, bool]:
    """Removes small disconnected regions and holes in a given mask.

    This function operates on a binary mask, removing regions and holes that
        are smaller than a specified size. It also returns an indicator flag
        showing whether the original mask has been modified.

    Arguments:
        mask (np.array): A binary mask with 1s indicating the regions of
            interest and 0s elsewhere.
        min_size (int): The minimum size of regions or holes to be retained
            in the mask.
        connectivity (int): The connectivity defining the neighborhood of a
            pixel. Defaults to 2.

    Returns:
        Tuple[np.array, bool]: A tuple containing the modified mask and a
            boolean indicating whether the mask has been modified.

    Example:
        >>> remove_small_regions(mask, 100, 2)

    Note:
        The mask should be a 2D numpy array. The function uses morphological
            operations to remove small regions and holes.

    """
    correct_holes = mode == "holes"
    working_mask = (correct_holes ^ mask).astype(np.uint8)
    n_labels, regions, stats, _ = cv2.connectedComponentsWithStats(
        working_mask,
        8,
    )
    sizes = stats[:, -1][1:]  # Row 0 is background label
    small_regions = [i + 1 for i, s in enumerate(sizes) if s < area_thresh]
    if len(small_regions) == 0:
        return mask, False
    fill_labels = [0, *small_regions]
    if not correct_holes:
        fill_labels = [i for i in range(n_labels) if i not in fill_labels]
        # If every region is below threshold, keep largest
        if len(fill_labels) == 0:
            fill_labels = [int(np.argmax(sizes)) + 1]
    mask = np.isin(regions, fill_labels)
    return mask, True


def group_regions_from_binary(
    bbox_img: Bool[np.ndarray, "h w"],
    /,
    *,
    closing: tuple[int, int],
    margin: float = 0.0,
    area_threshold: float = 0.0,
) -> list[Bool[np.ndarray, "h w"]]:
    """Group regions in a binary or PIL image based on specific parameters.

    Arguments:
        bbox_img (Union[np.ndarray, PIL.Image.Image]): A NumPy array or PIL
            image representing the binary image.
        closing (Tuple[int, int]): A tuple of two integers specifying the
            kernel size for the morphological closing operation.
        margin (float): Margin value for grouping regions. This represents
            the distance between regions that should be considered as a
            single group.
        area_threshold (float): Area threshold for grouping regions. This
            represents the minimum area that a group of regions should have
            to be considered as a valid group.

    Returns:
        List[Union[np.ndarray, PIL.Image.Image]]: A list of NumPy arrays or
            PIL images representing the grouped regions in the input image.

    Example:
        >>> group_regions(bbox_img, (3, 3), 0.5, 100)

    Note:
        The function performs a morphological closing operation on the
            binary image before grouping regions. This operation helps to
            close small holes in the regions.

    """
    bbox_img = morphologyEx(bbox_img, cv2.MORPH_CLOSE, np.ones(closing))
    image_size = ImageSize.from_image(bbox_img)
    # reduce_masks
    list_bbox = mask2bbox(
        bbox_img,
        margin=margin,
        area_threshold=area_threshold,
    )
    square_mask = [bbox2mask([i], image_size) for i in list_bbox]
    return [np.logical_and(i, bbox_img) for i in square_mask]


@beartype
def remove_disconnected_regions(
    masks: list[Bool[np.ndarray, "h w"]],
    area_thresh: float | list[float] = 0.0,
    /,
) -> list[Bool[np.ndarray, "h w"]]:
    """Remove disconnected regions from a mask or a list of masks based on an.

        area threshold.

    Arguments:
        masks (Union[np.array, List[np.array]]): A boolean numpy array or a
            list of boolean numpy arrays representing masks.
        area_thresh (Union[float, List[float]]): A float or a list of floats
            specifying the area threshold for removing disconnected regions.
            Each float represents the minimum area size to keep a region.

    Returns:
        Union[np.array, List[np.array]]: A boolean numpy array or a list of
            boolean numpy arrays with disconnected regions removed. If the
            area of a disconnected region in the mask is less than the area
            threshold, that region is removed from the mask.

    Example:
        >>> remove_disconnected_regions(mask, 50)
        or
        >>> remove_disconnected_regions([mask1, mask2], [50, 100])

    Note:
        If a list of masks and a single area threshold are provided, the
            same area threshold is applied to all masks.

    """
    if isinstance(area_thresh, float) and area_thresh == 0.0:
        return masks
    if isinstance(area_thresh, float):
        area_thresh_list: list[float] = [area_thresh] * len(masks)
    elif isinstance(area_thresh, list) and len(area_thresh) != len(masks):
        msg = (
            "if area_thresh is a list, it must be the same length as the masks"
        )
        raise ValueError(
            msg,
        )
    else:
        area_thresh_list: list[float] = area_thresh  # type: ignore[no-redef]

    if any(i >= 1.0 or i < 0 for i in area_thresh_list):
        msg = "area_thresh should be between 0 and 1, just a percentage of the total area"
        raise ValueError(
            msg,
        )

    fine_masks: list[Bool[np.ndarray, "h w"]] = []
    for mask, area_rel in zip(masks, area_thresh_list, strict=False):
        area_abs = np.prod([i * area_rel for i in mask.shape[:2]])
        mask_removed: Bool[np.ndarray, "h w"] = remove_small_regions(
            mask,
            area_thresh=area_abs,
            mode="holes",
        )[0]
        mask_removed = remove_small_regions(
            mask_removed,
            area_thresh=area_abs,
            mode="islands",
        )[0]
        fine_masks.append(mask_removed)
    return fine_masks


@beartype
def morphologyEx(  # noqa: N802
    mask: Bool[np.ndarray, "h w"],
    mode: int,
    kernel: Float[np.ndarray, "n n"],
    /,
    *,
    struct: str | None = None,
) -> Bool[np.ndarray, "h w"]:
    """Apply a morphological operation to a binary image mask.

    This function performs a morphological operation (specified by the mode)
        on a binary image mask using a given structuring element (kernel).
        The type of structuring element can be optionally specified.

    Arguments:
        mask (Union[PIL.Image.Image, np.ndarray]): A binary image mask,
            either as a PIL Image or a NumPy array.
        mode (int): An integer specifying the morphological operation mode.
        kernel (np.ndarray): A NumPy array representing the structuring
            element for the operation.
        struct (Optional[str]): An optional parameter specifying the type of
            structuring element to use. Defaults to None.

    Returns:
        Union[PIL.Image.Image, np.ndarray]: The morphologically transformed
            mask, either as a PIL Image or a NumPy array.

    Example:
        >>> morphological_operation(mask, 1, kernel, struct="disk")

    Note:
        The mode argument corresponds to different morphological operations
            such as erosion, dilation, etc.

    """
    mask255 = (mask * 255).astype(np.uint8)

    if struct is not None and struct == "elipse":
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        np.asarray(
            [
                [0, 0, 1, 0, 0],
                [1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1],
                [0, 0, 1, 0, 0],
            ],
            dtype=np.uint8,
        )

    return cv2.morphologyEx(mask255, mode, kernel).astype(bool)


def mask2points(
    binary_mask: Bool[np.ndarray, "h w"],
    /,
    npoints: int = 100,
    *,
    normalize: bool = False,
    rng: np.random.Generator | None = None,
    output: Literal["xy", "yx"] = "xy",
) -> list[tuple[int, int]] | list[tuple[float, float]]:
    """Convert a numpy image mask into a list of points.

    This function takes a numpy image mask and converts it into a list of
        points. The number of points to be generated can be specified.

    Arguments:
        binary_mask (bool[np.ndarray, "h w"]): A image mask to convert into
            points.
        npoints (int): The number of points to generate. Defaults to 100.
            if -1, all points are returned.
        normalize (bool): Whether to return normalized point coordinates.
            Defaults to False.

    Returns:
        List[Tuple[int, int]]: A list of tuples containing integer or float
            values representing the points.

    Example:
        >>> mask2points(hash_mask, 100, normalize=True, verbose=True)

    Note:
        This function is particularly useful for image processing tasks where
            points are required.

    """
    coords_yx = np.argwhere(binary_mask)
    if len(coords_yx) == 0:
        msg = "No points found in the mask"
        raise ValueError(msg)
    if rng is None:
        rng = np.random.default_rng()
    if npoints == -1:
        npoints = len(coords_yx)
    rand_ind = rng.choice(len(coords_yx), npoints, replace=False)
    points = coords_yx[rand_ind].tolist()
    if normalize:
        h, w = binary_mask.shape
        _points = [(y / w, x / h) for y, x in points]
    else:
        _points = [(int(y), int(x)) for y, x in points]
    if output == "xy":
        _points = [(x, y) for y, x in _points]
    return _points


@beartype
def mask2bbox(
    binary_mask: Bool[np.ndarray, "h w"],
    /,
    margin: float,
    *,
    normalized: bool = False,
    merge: bool = False,
    verbose: bool = True,
    closing: tuple[int, int] = (0, 0),
    opening: tuple[int, int] = (0, 0),
    area_threshold: float = 0.0,
    **kwargs: Any,
) -> list[tuple[int, int, int, int]] | list[tuple[float, float, float, float]]:
    """Convert a numpy image mask into bounding boxes.

    This function takes a numpy image mask and converts it into bounding
        boxes. It also allows for various image processing operations such
        as opening, closing, and merging masks.

    Arguments:
        binary_mask: A binary image mask to convert into bounding boxes.
        margin (float): A value to adjust the bounding box size.
        normalized (bool): Whether to return normalized bounding box
            coordinates.
        merge (bool): Whether to merge overlapping bounding boxes.
        verbose (bool): A flag for verbose output.
        closing (tuple): Two integers specifying the closing operation
            parameters.
        opening (tuple): Two integers specifying the opening operation
            parameters.
        area_threshold (float): A value to remove disconnected regions based
            on area.
        **kwargs: Additional keyword arguments.

    Returns:
        list: A list of tuples containing integer or float values
            representing the bounding boxes.

    Example:
        >>> convert_mask_to_bboxes(hash_mask, 0.5, True, False, True, (1,1),
            (2,2), 0.1)

    Note:
        This function is particularly useful for image processing tasks
            where bounding boxes are required.

    """
    if area_threshold > 0:
        binary_mask = remove_disconnected_regions(
            [binary_mask], area_threshold
        )[0]
    # connected components
    if sum(opening) > 0:
        binary_mask = morphologyEx(
            binary_mask, cv2.MORPH_OPEN, np.ones(opening)
        )
    if sum(closing) > 0:
        binary_mask = morphologyEx(
            binary_mask, cv2.MORPH_CLOSE, np.ones(closing)
        )
    if not merge:
        mask_list = reduce_masks(
            binary_mask,
            number_of_objects=-1,
            verbose=verbose,
            **kwargs,
        )
    else:
        mask_list = [binary_mask]
    bboxes = []
    for image in mask_list:
        rows = np.any(image, axis=1)
        cols = np.any(image, axis=0)
        try:
            ymin, ymax = np.where(rows)[0][[0, -1]]
            xmin, xmax = np.where(cols)[0][[0, -1]]
        except IndexError:
            if verbose:
                logger.warning("No bbox found")
            ymin, ymax, xmin, xmax = 0, 0, 0, 0
        # big margin for small bbox, small margin for big bbox
        h, w = image.shape
        bbox_h = ymax - ymin
        bbox_w = xmax - xmin
        ymin = max(0, ymin - bbox_h * margin)
        ymax = min(h - 1, ymax + bbox_h * margin)
        if ymin == ymax:
            ymin = max(0, ymin - 1)
            ymax = min(h - 1, ymax + 1)
        xmin = max(0, xmin - bbox_w * margin)
        xmax = min(w - 1, xmax + bbox_w * margin)
        if xmin == xmax:
            xmin = max(0, xmin - 1)
            xmax = min(w - 1, xmax + 1)
        bbox = (
            int(np.round(xmin)),
            int(np.round(ymin)),
            int(np.round(xmax)),
            int(np.round(ymax)),
        )
        bboxes.append(bbox)
    # check if bboxes overlap whenever margin is enabled
    if len(bboxes) > 1:
        for i in range(len(bboxes) - 1):
            for j in range(i + 1, len(bboxes)):
                if (
                    bbox_iou(
                        torch.LongTensor(bboxes[i])[None],
                        torch.LongTensor(bboxes[j])[None],
                    )[0][0].item()
                    > 0
                ):
                    # merge bboxes
                    merge_box = (
                        min(bboxes[i][0], bboxes[j][0]),
                        min(bboxes[i][1], bboxes[j][1]),
                        max(bboxes[i][2], bboxes[j][2]),
                        max(bboxes[i][3], bboxes[j][3]),
                    )
                    bboxes[i] = bboxes[j] = merge_box
        bboxes = list(set(bboxes))
    if normalized:
        bboxes = [
            (
                bbox[0] / image.shape[1],
                bbox[1] / image.shape[0],
                bbox[2] / image.shape[1],
                bbox[3] / image.shape[0],
            )
            for bbox in bboxes
        ]
    if merge and len(bboxes):
        # get min and max
        xmin = min([_bbox[0] for _bbox in bboxes])
        ymin = min([_bbox[1] for _bbox in bboxes])
        xmax = max([_bbox[2] for _bbox in bboxes])
        ymax = max([_bbox[3] for _bbox in bboxes])
        bboxes = [(xmin, ymin, xmax, ymax)]
    if normalized:
        return [
            (
                float(_bbox[0]),
                float(_bbox[1]),
                float(_bbox[2]),
                float(_bbox[3]),
            )
            for _bbox in bboxes
        ]
    return [
        (int(_bbox[0]), int(_bbox[1]), int(_bbox[2]), int(_bbox[3]))
        for _bbox in bboxes
    ]


@beartype
def bbox2mask(
    bbox: (
        list[tuple[float, float, float, float]]
        | list[tuple[int, int, int, int]]
    ),
    image_size: ImageSize,
) -> Bool[np.ndarray, "h w"]:
    """Generate a binary mask from bounding box coordinates and image size.

    This function takes a list of bounding boxes and an image size as input
        and generates a binary mask based on the bounding box coordinates.
        If the bounding box coordinates are normalized, it converts them to
        pixel values.

    Arguments:
        bbox (List[Tuple[Union[float, int]]]): A list of tuples containing
            the bounding box coordinates (x1, y1, x2, y2) either as floats
            or integers.
        image_size (Tuple[int, int]): A tuple representing the size of the
            image (height, width).

    Returns:
        torch.Tensor: A binary mask represented as a torch tensor with True
            values inside the bounding boxes and False values outside.

    Example:
        >>> bbox2mask([(0.1, 0.2, 0.3, 0.4)], (512, 512))

    Note:
        The bounding box coordinates can be either normalized or in pixel
            values. If normalized, the function will convert them to pixel
            values based on the provided image size.

    """
    height, width = image_size.height, image_size.width
    zeros = np.zeros((height, width), dtype=bool)
    for box in bbox:
        if isinstance(box[0], float):
            if box[0] > 1 or box[1] > 1 or box[2] > 1 or box[3] > 1:
                msg = "box is not normalized"
                raise ValueError(msg)
            box = (
                int(box[0] * width),
                int(box[1] * height),
                int(box[2] * width),
                int(box[3] * height),
            )  # x1, y1, x2, y2
        box = (int(box[0]), int(box[1]), int(box[2]), int(box[3]))
        zeros[box[1] : box[3], box[0] : box[2]] = True
    return zeros


@beartype
def mask2squaremask(
    mask: Bool[np.ndarray, "h w"],
    margin: float,
    **kwargs: Any,
) -> Bool[np.ndarray, "h w"]:
    """Convert a mask image into a square mask image with a specified margin.

    This function takes a mask image and converts it into a square mask
        image by adding a margin around the bounding box of the original
        mask.

    Arguments:
        mask (np.ndarray): The original mask image.
        margin (float): The margin to be added around the bounding box of
            the mask.
        **kwargs: Additional keyword arguments.

    Returns:
        np.ndarray: A square mask image with the specified margin added
            around the bounding box of the original mask.

    Example:
        >>> square_mask(original_mask, 0.5)

    Note:
        The margin is added around the bounding box of the original mask to
            create the square mask.

    """
    bbox = mask2bbox(mask, margin, **kwargs)
    return bbox2mask(bbox, ImageSize.from_image(mask))


@beartype
def resize_squaremask(
    mask: Bool[np.ndarray, "h w"],
    size: ImageSize,
    **kwargs: Any,
) -> Bool[np.ndarray, "h w"]:
    """Resize an image mask to a square shape.

    This function takes in an image mask and resizes it to a square shape
        based on the specified size.
    It first resizes the image using the 'nearest-exact' mode and then
        converts it into a square mask with a specified background value.

    Arguments:
        mask (np.ndarray): The input image mask to be resized.
        size (ImageSize): The desired size of the square mask.
        **kwargs (Any): Additional keyword arguments for the mask2squaremask
            function.

    Returns:
        np.ndarray: The resized square mask.

    Example:
        >>> resize_square_mask(mask, size, background_value=0)

    Note:
        The 'nearest-exact' mode is used for resizing to maintain the
            original mask values as much as possible.

    """
    mask_pt = numpy2tensor(mask)
    mask_pt = resize_image(mask_pt, size, mode="nearest-exact")
    return mask2squaremask(tensor2numpy(mask_pt), 0.0, **kwargs)


@beartype
def resize_image(
    tensor: Float[torch.Tensor, "b c h w"] | Bool[torch.Tensor, "b 1 h w"],
    /,
    resolution: int | None | ImageSize,
    mode: str,
    resize_min_max: Literal["min", "max"] = "min",
    modulo: int = 16,
) -> Float[torch.Tensor, "b c h w"]:
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
        height = int(np.round(height / modulo)) * modulo
        width = int(np.round(width / modulo)) * modulo
    elif isinstance(resolution, ImageSize):
        height, width = resolution.height, resolution.width
    else:
        if resize_min_max == "min":
            k = float(resolution) / min(height, width)  # resize with min
        else:
            k = float(resolution) / max(height, width)  # resize with max
        height *= k
        width *= k
        height = int(np.round(height / modulo)) * modulo
        width = int(np.round(width / modulo)) * modulo
    output: Float[torch.Tensor, "b c h w"] = torch.nn.functional.interpolate(
        tensor,
        size=(height, width),
        mode=mode,
    )
    if is_bool:
        return output.bool()[:, :1, :, :]
    return output


@beartype
def crop_from_mask(
    image: UInt8[np.ndarray, "h w c"],
    mask: Bool[np.ndarray, "h w"],
    margin: float = 0.0,
    **kwargs: Any,
) -> UInt8[np.ndarray, "h w c"]:
    """Crop an image based on a bounding box defined by a mask.

    This function takes an image and a mask as input, along with an optional
        margin value, and returns a cropped version
    of the image based on the bounding box defined by the mask.

    Arguments:
        image (np.ndarray): The original image to be cropped.
        mask (np.ndarray): The mask used to define the bounding box for
            cropping.
        margin (float | None): The margin around the mask bounding box.
            Defaults to 0.0.
        **kwargs: Additional keyword arguments that can be passed to the
            function.

    Returns:
        np.ndarray: The cropped version of the original image based on
            the bounding box defined by the mask.

    Example:
        >>> crop_image(image, mask, margin=0.1)

    Note:
        The margin is expressed as a fraction of the image's size. For
            example, a margin of 0.1 adds a 10% border around the mask.

    """
    return crop_from_bbox(image, mask2bbox(mask, margin, **kwargs))


@beartype
def crop_from_bbox(
    image_np: UInt8[np.ndarray, "h w c"],
    /,
    bboxes: (
        list[tuple[int, int, int, int]]
        | list[tuple[float, float, float, float]]
    ),
) -> UInt8[np.ndarray, "h w c"]:
    """Crops an image based on provided bounding box coordinates.

    This function takes an image and a list of bounding boxes as input and
        crops the image based on the bounding box coordinates provided. It
        supports both normalized and non-normalized bounding box
        coordinates.

    Arguments:
        image (np.ndarray): The input image to be cropped.
        bboxes (List[Tuple[Union[int, float]]]): A list of tuples containing
            the bounding box coordinates. Each tuple should have either four
            integers or four floats representing (xmin, ymin, xmax, ymax)
            coordinates of the bounding box.

    Returns:
        np.ndarray: A new np.ndarray object representing the cropped
            image based on the provided bounding box coordinates.

    Example:
        >>> crop_image(image, [(10, 10, 50, 50)])

    Note:
        If the bounding box coordinates are given as floats, they are
            assumed to be normalized to the range [0, 1].

    """
    if isinstance(bboxes[0][0], float) and np.mean(bboxes) < 1:
        is_normalized = True
    elif isinstance(bboxes[0][0], int) and np.mean(bboxes) > 1:
        is_normalized = False
    else:
        msg = f"invalid bbox {bboxes}. bboxes must be either normalized or not normalized for float or int, respectively"
        raise ValueError(
            msg,
        )

    # get min, max from bboxes
    xmin = min([_bbox[0] for _bbox in bboxes])
    ymin = min([_bbox[1] for _bbox in bboxes])
    xmax = max([_bbox[2] for _bbox in bboxes])
    ymax = max([_bbox[3] for _bbox in bboxes])
    if is_normalized:
        xmin *= image_np.shape[1]
        ymin *= image_np.shape[0]
        xmax *= image_np.shape[1]
        ymax *= image_np.shape[0]
    xmin = int(max(0, xmin))
    ymin = int(max(0, ymin))
    xmax = int(min(image_np.shape[1] - 1, xmax))
    ymax = int(min(image_np.shape[0] - 1, ymax))
    return image_np[ymin:ymax, xmin:xmax]


@beartype
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


@beartype
def uncrop_from_bbox(
    base_image: UInt8[np.ndarray, "h w c"],
    image: UInt8[np.ndarray, "h w c"],
    bboxes: (
        list[tuple[int, int, int, int]]
        | list[tuple[float, float, float, float]]
    ),
    *,
    resize: bool = False,
) -> UInt8[np.ndarray, "h w c"]:
    """Uncrop an image from given bounding boxes and return the resulting.

        image.

    Arguments:
        base_image (Image): The base image from which to uncrop the image.
        image (Image): The image to be uncropped.
        bboxes (List[Tuple[int, int, int, int]]): A list of bounding boxes.
            Each bounding box is represented as a tuple of (x_min, y_min,
            x_max, y_max) or (x_min_norm, y_min_norm, x_max_norm,
            y_max_norm).
        resize (bool, optional): A flag indicating whether to resize the
            image to fit the bounding boxes. Defaults to False.

    Returns:
        np.ndarray: The uncropped image as a np.ndarray object.

    Example:
        >>> uncrop_image(base_image, image, bboxes, resize=True)

    Note:
        The bounding boxes can be in normalized or absolute coordinates. If
            the resize flag is True, the image will be resized to fit the
            bounding boxes.

    """
    if isinstance(bboxes[0][0], float) and np.mean(bboxes) < 1:
        is_normalized = True
    elif isinstance(bboxes[0][0], int) and np.mean(bboxes) > 1:
        is_normalized = False
    else:
        msg = f"invalid bbox {bboxes}. bbox must be either normalized or not normalized for float or int, respectively"
        raise ValueError(
            msg,
        )

    # get min, max from bboxes
    xmin = min([_bbox[0] for _bbox in bboxes])
    ymin = min([_bbox[1] for _bbox in bboxes])
    xmax = max([_bbox[2] for _bbox in bboxes])
    ymax = max([_bbox[3] for _bbox in bboxes])
    out_image = base_image.copy()
    image_size = ImageSize.from_image(base_image)
    height = image_size.height
    width = image_size.width
    if is_normalized:
        xmin *= width
        ymin *= height
        xmax *= width
        ymax *= height
    xmin = int(max(0, xmin))
    ymin = int(max(0, ymin))
    xmax = int(min(width - 1, xmax))
    ymax = int(min(height - 1, ymax))
    if resize:
        image = tensor2numpy(
            resize_image(
                numpy2tensor(image),
                ImageSize(width=xmax - xmin, height=ymax - ymin),
                "bilinear",
            )
        )
    out_image[ymin:ymax, xmin:xmax] = image
    return out_image


@beartype
def center_pad(
    image: UInt8[np.ndarray, "h w c"],
    size: ImageSize,
    fill: int | tuple[int, int] = (0, 0),
) -> UInt8[np.ndarray, "h w c"]:
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
    h_mod = max(size.height % 2, 0)
    w_mod = max(size.width % 2, 0)
    new_np = np.zeros((size.height, size.width, 3), dtype=np.uint8) + fill
    new_np[
        h_pad - h // 2 : h_pad + h // 2 + h_mod,
        w_pad - w // 2 : w_pad + w // 2 + w_mod,
    ] = image
    return new_np


@beartype
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
        rgb_np = np.asarray(rgb).astype(bool)
        if rgb_np.ndim == 3:
            rgb_np = rgb_np[..., 0]
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


@beartype
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


@beartype
def mask_blend(
    image: UInt8[np.ndarray, "h w 3"],
    mask: UInt8[np.ndarray, "h w 3"] | Bool[np.ndarray, "h w"],
    alpha: float,
    *,
    with_bbox: bool = True,
    merge_bbox: bool = True,
) -> UInt8[np.ndarray, "h w 3"]:
    """Blend an image with a mask based on a given alpha value.

    Arguments:
        image (np.ndarray): The input image to be blended with the mask.
        mask (np.ndarray): The mask to be applied to the image.
        alpha (float): The blending factor, determining the degree of
            transparency for the mask.
        with_bbox (bool, optional): Flag to include bounding boxes in the
            output. Defaults to True.
        merge_bbox (bool, optional): Flag to merge bounding boxes in the
            output. Defaults to True.

    Returns:
        np.ndarray: The blended image resulting from the application of
            the mask on the input image.

    Example:
        >>> mask_blend(image, mask, 0.5, with_bbox=True, merge_bbox=True)

    Note:
        The alpha value ranges from 0.0 (full visibility of the image, mask
            fully transparent) to 1.0 (full visibility of the mask, image
            fully transparent).

    """
    is_mask_binary = mask.ndim == 2
    if not is_mask_binary:
        blend = alpha * mask + (1 - alpha) * image
        blend = np.clip(blend, 0, 255).astype(np.uint8)
    else:
        blend = image.copy()
        blend[~mask.astype(bool)] = image[~mask.astype(bool)] * alpha
        blend = np.clip(blend, 0, 255).astype(np.uint8)
    if with_bbox:
        # draw bbox
        binary_mask = to_binary(mask) if not is_mask_binary else mask
        if not merge_bbox:
            # closing the mask just for better visualization
            binary_mask = morphologyEx(binary_mask, "close", np.ones((5, 5)))

        blend = draw_bbox(
            blend,
            mask2bbox(
                binary_mask,
                margin=0.001,
                merge=merge_bbox,
                verbose=False,
                area_threshold=0.0,
            ),
        )
    return cast(np.ndarray, blend)


@beartype
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


@beartype
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


def polygon_to_mask(
    polygon: list[tuple[int, int]],
    image_shape: tuple[int, int],
) -> Bool[np.ndarray, "h w"]:
    """Convert a polygon into a segmentation mask.

    This function takes a list of polygon vertices and an image shape, and
        returns a binary mask with the polygon area filled.

    Arguments:
        polygon (List[Tuple[int, int]]): List of (x, y) coordinates
            representing the vertices of the polygon. The coordinates are
            expected to be integers.
        image_shape (Tuple[int, int]): Shape of the image (height, width)
            for which the mask is to be generated. Both height and width are
            expected to be integers.

    Returns:
        np.ndarray: A 2D numpy array of the same shape as the input image,
            where the area within the polygon is filled with 1s and the rest
            with 0s.

    Example:
        >>> convert_polygon_to_mask([(10, 10), (20, 10), (20, 20), (10,
            20)], (30, 30))

    Note:
        The vertices of the polygon should be provided in either clockwise
            or counter-clockwise order.

    """
    # Create an empty mask
    mask = np.zeros(image_shape, dtype=np.uint8)

    # Convert polygon to an array of points
    pts = np.array(polygon, dtype=np.int32)

    # Fill the polygon with white color (255)
    cv2.fillPoly(mask, [pts], color=(255,))

    return mask.astype(bool)
