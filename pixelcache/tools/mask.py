from typing import Any, Literal, cast

import cv2
import numpy as np
import torch
from beartype import beartype
from jaxtyping import Bool, Float, Int64, UInt8, jaxtyped
from PIL import Image

from pixelcache.tools.bbox import bbox_iou, crop_from_bbox
from pixelcache.tools.draw import draw_bbox
from pixelcache.tools.image import (
    ImageSize,
    numpy2tensor,
    resize_image,
    tensor2numpy,
    to_binary,
)
from pixelcache.tools.logger import get_logger

logger = get_logger()


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
        normalized=True,
    )
    square_mask = [bbox2mask([i], image_size) for i in list_bbox]
    return [np.logical_and(i, bbox_img) for i in square_mask]


@jaxtyped(typechecker=beartype)
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
            area_thresh=int(area_abs),
            mode="holes",
        )[0]
        mask_removed = remove_small_regions(
            mask_removed,
            area_thresh=int(area_abs),
            mode="islands",
        )[0]
        fine_masks.append(mask_removed)
    return fine_masks


@jaxtyped(typechecker=beartype)
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
        _points = [(round(y), round(x)) for y, x in points]
    if output == "xy":
        _points = [(x, y) for y, x in _points]
    return _points


@jaxtyped(typechecker=beartype)
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
    number_of_objects: int = -1,
) -> list[tuple[float, float, float, float]]:
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
            representing the bounding boxes. x1, y1, x2, y2.

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
            number_of_objects=number_of_objects,
            verbose=verbose,
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
        ymax = min(h, ymax + bbox_h * margin)
        if ymin == ymax:
            # too small
            continue
        xmin = max(0, xmin - bbox_w * margin)
        xmax = min(w, xmax + bbox_w * margin)
        if xmin == xmax:
            # too small
            continue
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
    if merge and len(bboxes) > 0:
        # get min and max
        xmin = min([_bbox[0] for _bbox in bboxes])
        ymin = min([_bbox[1] for _bbox in bboxes])
        xmax = max([_bbox[2] for _bbox in bboxes])
        ymax = max([_bbox[3] for _bbox in bboxes])
        bboxes = [(xmin, ymin, xmax, ymax)]
    return [
        (
            float(_bbox[0]),
            float(_bbox[1]),
            float(_bbox[2]),
            float(_bbox[3]),
        )
        for _bbox in bboxes
    ]


@jaxtyped(typechecker=beartype)
def bbox2mask(
    bbox: list[tuple[float, float, float, float]],
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
                msg = f"box is not normalized. {box}, image size: {image_size} - it should be normalized [0, 1]"
                raise ValueError(msg)
            box = (
                round(box[0] * width),
                round(box[1] * height),
                round(box[2] * width),
                round(box[3] * height),
            )  # x1, y1, x2, y2
        box = (int(box[0]), int(box[1]), int(box[2]), int(box[3]))
        zeros[box[1] : box[3], box[0] : box[2]] = True
    return zeros


@jaxtyped(typechecker=beartype)
def mask2squaremask(
    mask: Bool[np.ndarray, "h w"],
    margin: float,
    *,
    merge: bool = False,
    verbose: bool = True,
    closing: tuple[int, int] = (0, 0),
    opening: tuple[int, int] = (0, 0),
    area_threshold: float = 0.0,
    number_of_objects: int = -1,
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
    bbox = mask2bbox(
        mask,
        margin=margin,
        normalized=True,
        merge=merge,
        verbose=verbose,
        closing=closing,
        opening=opening,
        area_threshold=area_threshold,
        number_of_objects=number_of_objects,
    )
    return bbox2mask(bbox, ImageSize.from_image(mask))


@jaxtyped(typechecker=beartype)
def resize_squaremask(
    mask: Bool[np.ndarray, "h w"],
    size: ImageSize,
    **kwargs: Any,
) -> Bool[np.ndarray, "h1 w1"]:
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


@jaxtyped(typechecker=beartype)
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
        if merge and len(mask_list) > 0:
            mask_i = np.logical_or(mask_i, mask_list[-1])
        mask_list.append(mask_i)
    if merge:
        mask_list = [mask_list[-1]]
    return mask_list


@jaxtyped(typechecker=beartype)
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


@jaxtyped(typechecker=beartype)
def remove_small_regions(
    mask: Bool[np.ndarray, "h w"],
    area_thresh: int,
    mode: Literal["holes", "islands"],
    connectivity: int = 8,
) -> tuple[Bool[np.ndarray, "h w"], bool]:
    """Removes small disconnected regions in a binary mask based on the specified mode.

    This function operates on a binary mask, removing regions that are smaller than a
    specified area threshold. The mode determines whether to remove small background
    regions (holes) or small foreground regions (islands).

    Arguments:
        mask (np.array): A binary mask with 1s indicating the foreground and 0s
            indicating the background.
        area_thresh (int): The minimum area (in pixels) for regions to be retained.
            Regions smaller than this threshold will be removed.
        mode (Literal["holes", "islands"]): Specifies the type of regions to remove:
            - "holes": Removes small background regions (0s) completely surrounded by
              foreground (1s).
            - "islands": Removes small foreground regions (1s) completely surrounded by
              background (0s).

    Returns:
        Tuple[np.array, bool]: A tuple containing:
            - The modified mask with small regions removed.
            - A boolean indicating whether the mask was modified (True if changes were made).

    Example:
        >>> # Remove small holes (background regions)
        >>> cleaned_mask, modified = remove_small_regions(mask, 100, mode="holes")
        >>> # Remove small islands (foreground regions)
        >>> cleaned_mask, modified = remove_small_regions(mask, 100, mode="islands")

    Note:
        - If all regions are smaller than the threshold, the largest region is retained.
        - The function preserves the overall structure of the mask while removing noise.

    """
    correct_holes = mode == "holes"
    working_mask = (correct_holes ^ mask).astype(np.uint8)
    n_labels, regions, stats, _ = cv2.connectedComponentsWithStats(
        working_mask,
        connectivity=connectivity,
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


@jaxtyped(typechecker=beartype)
def crop_from_mask(
    image: UInt8[np.ndarray, "h w c"],
    mask: Bool[np.ndarray, "h w"],
    margin: float = 0.0,
    *,
    normalized: bool = False,
    merge: bool = False,
    verbose: bool = True,
    closing: tuple[int, int] = (0, 0),
    opening: tuple[int, int] = (0, 0),
    area_threshold: float = 0.0,
    number_of_objects: int = -1,
) -> UInt8[np.ndarray, "h1 w1 c"]:
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
        normalized (bool): Whether the mask is normalized. Defaults to False.
        merge (bool): Whether to merge overlapping bounding boxes. Defaults
            to False.
        verbose (bool): A flag for verbose output. Defaults to True.
        closing (Tuple[int, int]): The kernel size for morphological closing
            operation. Defaults to (0, 0).
        opening (Tuple[int, int]): The kernel size for morphological opening
            operation. Defaults to (0, 0).
        area_threshold (float): The area threshold for grouping regions.
            Defaults to 0.0.
        number_of_objects (int): The number of objects to keep in the mask.
            Defaults to -1.


    Returns:
        np.ndarray: The cropped version of the original image based on
            the bounding box defined by the mask.

    Example:
        >>> crop_image(image, mask, margin=0.1)

    Note:
        The margin is expressed as a fraction of the image's size. For
            example, a margin of 0.1 adds a 10% border around the mask.

    """
    return crop_from_bbox(
        image,
        mask2bbox(
            mask,
            margin,
            normalized=normalized,
            merge=merge,
            verbose=verbose,
            closing=closing,
            opening=opening,
            area_threshold=area_threshold,
            number_of_objects=number_of_objects,
        ),
        is_normalized=normalized,
    )


@jaxtyped(typechecker=beartype)
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


@jaxtyped(typechecker=beartype)
def mask_to_polygon(
    mask: Bool[np.ndarray, "h w"], *, min_area: float = 100.0
) -> list[list[tuple[int, int]]]:
    """Convert a boolean mask into a polygon.

    This function takes a boolean mask represented as a NumPy array and
        converts it into a polygon. The polygon is represented as a list of
        tuples where each tuple contains the x and y coordinates of a point
        on the polygon.

    Args:
        mask (Bool[np.ndarray, "h w"]): The boolean mask to convert.
        min_area (float): The minimum area of the polygon to return.

    Returns:
        list[list[tuple[int, int]]]: A list of polygons, where each polygon is a list of (x, y) coordinates.

    """
    # Find contours in the binary mask
    contours, _ = cv2.findContours(
        (mask * 255).astype(np.uint8),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    # Find the contour with the largest area
    largest_contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest_contour) < min_area:
        return []

    all_valid_polygons = []
    for contour in contours:
        if cv2.contourArea(contour) < min_area:
            continue
        # Extract the vertices of the contour
        _polygon = contour.reshape(-1, 2).tolist()
        polygon = [tuple(i) for i in _polygon]
        all_valid_polygons.append(polygon)
    return all_valid_polygons


def polygon_to_mask(
    polygons: list[list[tuple[int, int]]],
    image_shape: tuple[int, int],
) -> Bool[np.ndarray, "h w"]:
    """Convert a polygon into a segmentation mask.

    This function takes a list of polygon vertices and an image shape, and
        returns a binary mask with the polygon area filled.

    Arguments:
        polygon (List[List[Tuple[int, int]]]): List of list of (x, y) coordinates
            representing the vertices of the polygon. The coordinates are
            expected to be integers.
        image_shape (Tuple[int, int]): Shape of the image (height, width)
            for which the mask is to be generated. Both height and width are
            expected to be integers.

    Returns:
        np.ndarray: A 2D numpy array of the same shape as the input image,
            where the area within the polygon is filled with 1s and the rest
            with 0s.


    """
    # Create an empty mask
    mask = np.zeros(image_shape, dtype=np.uint8)

    # Convert polygon to an array of points
    for polygon in polygons:
        pts = np.array(polygon, dtype=np.int32)

        # Fill the polygon with white color (255)
        cv2.fillPoly(mask, [pts], color=(255,))

    return mask.astype(bool)


def refine_masks(
    masks_pt: Bool[torch.Tensor, "b 1 h w"],
    min_area: float = 100.0,
) -> Bool[torch.Tensor, "b 1 h w"]:
    """Refine input masks by converting them to polygons and back to masks.

    This function takes a binary mask as input, converts it to a set of polygons,
    and then converts these polygons back to a binary mask. This process can help
    to smooth the mask and remove small holes or inconsistencies.

    Arguments:
        mask (np.ndarray): A binary mask to be refined. This should be a 2D
            numpy array where 1s represent the mask and 0s represent the
            background.

    Returns:
        np.ndarray: The refined mask, in the same format as the input mask.

    Example:
        >>> refined_mask = refine_masks(original_mask)

    Note:
        This function can be slow for large masks, as the conversion to
        polygons and back to masks is computationally intensive.

    """
    refined_mask = torch.zeros_like(masks_pt)
    for idx, mask in enumerate(masks_pt):
        shape = mask.shape[-2:]
        polygon = mask_to_polygon(
            tensor2numpy(mask[None], output_type=bool), min_area=min_area
        )
        mask = polygon_to_mask(polygon, shape)
        refined_mask[idx] = numpy2tensor(mask)[0]

    return refined_mask


def differential_mask(
    mask: Bool[np.ndarray, "h w"],
    dilation: int,
    force_steps: int | None = None,
    scale_nonmask: float | None = None,
    *,
    invert: bool = False,
) -> UInt8[np.ndarray, "h w"]:
    """Apply a differential mask to the given image mask.

    This function takes an input image mask and applies a differential mask
        to it based on the specified parameters such as dilation, force
        steps, invert, and scale nonmask.

    Arguments:
        mask (bool numpy): The input image mask to which the differential
            mask will be applied.
        dilation (int): The dilation factor for the mask.
        force_steps (int | None): The number of force steps for the
            differential mask. Defaults to None.
        invert (bool): Flag to invert the output mask. Defaults to False.
        scale_nonmask (float | None): The scaling factor for non-mask
            regions. Defaults to None.

    Returns:
        bool numpy: The output image with the applied differential mask.

    Example:
        >>> apply_differential_mask(mask, 2, force_steps=3, invert=True,
            scale_nonmask=1.5)

    Note:
        The function modifies the original mask, so if preservation of the
            original mask is needed, a copy should be passed.

    """
    mask_np = (mask * 255).astype(np.uint8)
    # get contours of mask, fill with diff, dilate, get contours, fill remaining with diff, repeat
    count = 1
    last_mask = np.zeros_like(mask_np)
    out_mask = np.zeros_like(mask_np)
    if force_steps is not None:
        linspace_count = np.linspace(0, 255, force_steps).astype(np.uint8)
    while True:
        # do not fill the same region twice
        contours = cv2.findContours(
            mask_np,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        fill_value = (
            count if force_steps is None else linspace_count[count - 1].item()
        )
        temp_mask = cv2.drawContours(
            np.zeros_like(mask_np),
            contours[0],
            -1,
            255 - fill_value,
            -1,
        )
        out_mask = np.maximum(out_mask, temp_mask)
        if (
            np.all(mask_np == last_mask)
            or count == 255
            or (force_steps is not None and count == force_steps)
        ):
            break
        last_mask = mask_np
        mask_np = cv2.dilate(mask_np, np.ones((dilation, dilation)))
        count += 1
    out: np.ndarray = (
        (out_mask - out_mask.min()) / (out_mask.max() - out_mask.min()) * 255
    )
    out_uint = out[:, :, None].repeat(3, axis=-1)
    if scale_nonmask is not None:
        out_uint[out_uint != 255] *= scale_nonmask
    if invert:
        out_uint = 255 - out_uint
    return out_uint[..., 0].astype(np.uint8)


@jaxtyped(typechecker=beartype)
def align_masks(
    base: Bool[torch.Tensor, "n h w"],
    anchors: Bool[torch.Tensor, "o m h w"],
    /,
) -> tuple[Bool[torch.Tensor, "o m h w"], Int64[torch.Tensor, "o m"]]:
    """Align anchor masks based on the intersection over union (IOU) matrix.

        with the base mask.

    Arguments:
        base (torch.Tensor): A Boolean tensor representing the base mask.
                             It has a shape of 'n h w', where 'n' is the
            number of base masks,
                             'h' is the height of the masks, and 'w' is the
            width of the masks.
        anchors (torch.Tensor): A Boolean tensor representing the anchor
            masks.
                                It has a shape of 'o m h w', where 'o' is
            the number of objects,
                                'm' is the number of anchor masks per
            object, 'h' is the height of the masks,
                                and 'w' is the width of the masks.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: A tuple containing two elements:
            - aligned_indexes (torch.Tensor): A Boolean tensor representing
            the aligned anchor masks.
                                               It has a shape of 'o m h w'.
            - indexes (torch.Tensor): An Int64 tensor containing the indexes
            of the closest masks in the base
                                      for each anchor mask. It has a shape
            of 'o m'.

    Example:
        >>> base = torch.rand((10, 32, 32))
        >>> anchors = torch.rand((5, 20, 32, 32))
        >>> aligned_indexes, indexes = align_masks(base, anchors)

    Note:
        The alignment is done based on the intersection over union (IOU)
            between the base and anchor masks.

    """
    # all masks correspond to the same image.
    aligned_indexes = anchors.clone()
    indexes = torch.zeros((anchors.size(0), anchors.size(1))).long()
    for i in range(anchors.size(0)):
        iou_matrix: Float[torch.Tensor, "m m"] = mask_iou(
            base[None],
            anchors[i][None],
            just_intersection=True,
        )
        closest_indices = torch.argmax(iou_matrix, dim=0)
        # check indices are different values, otherwise it would mean different anchor bboxes are close to the same base one
        if len(closest_indices.unique()) != closest_indices.size(0):
            msg = f"masks in anchor {anchors[i]} point to the same mask in {base}. Check ids {closest_indices}."
            raise ValueError(
                msg,
            )
        if closest_indices.tolist() != list(range(closest_indices.size(0))):
            logger.debug(
                f"[orange1]Aligning mask applied here! {closest_indices.tolist()} for anchors {i}[/orange1]",
            )
        # replace with new indices
        indexes[i] = closest_indices
        aligned_indexes[i] = aligned_indexes[i][closest_indices]
    return aligned_indexes, indexes


@jaxtyped(typechecker=beartype)
def mask_iou(
    mask1: Bool[np.ndarray | torch.Tensor, "m n h w"],
    mask2: Bool[np.ndarray | torch.Tensor, "m n h w"],
    /,
    *,
    just_intersection: bool = False,
) -> Float[np.ndarray | torch.Tensor, "n n"]:
    """Calculate the Intersection over Union (IoU) of two binary mask arrays.

    This function computes the IoU, a measure of overlap, between two binary
        mask arrays. The masks represent objects in an image. Optionally, it
        can return only the intersection value instead of the IoU.

    Arguments:
        mask1 (np.ndarray): Binary array representing the first object mask
            in the image.
        mask2 (np.ndarray): Binary array representing the second object mask
            in the image.
        just_intersection (bool, optional): Flag to return only the
            intersection value. If False, the function returns the IoU.
            Defaults to False.

    Returns:
        float: IoU value between the two masks if 'just_intersection' is
            False. If 'just_intersection' is True, returns the intersection
            value.

    Example:
        >>> mask_iou(mask1, mask2, just_intersection=False)

    Note:
        The masks should be binary arrays of the same shape, where '1'
            represents the object and '0' represents the background.

    """
    # Calculate IoU between two masks
    if mask1.shape != mask2.shape:
        msg = f"mask1 and mask2 must have the same shape. {mask1.shape} and {mask2.shape}"
        raise ValueError(
            msg,
        )
    if type(mask1) is not type(mask2):
        msg = "mask1 and mask2 type must be the same"
        raise TypeError(msg)

    if isinstance(mask1, torch.Tensor):
        mask2_tp = mask2.transpose(0, 1)
        intersection = torch.logical_and(mask1, mask2_tp).sum(dim=(2, 3))
        union = torch.logical_or(mask1, mask2_tp).sum(dim=(2, 3))
    else:
        mask2_tp = mask2.transpose(1, 0, 2, 3)
        intersection = np.logical_and(mask1, mask2_tp).sum(axis=(2, 3))
        union = np.logical_or(mask1, mask2_tp).sum(axis=(2, 3))
    if (union == 0).any():
        intersection[union == 0] = 0
        union[union == 0] = 1
    if just_intersection:
        return intersection.float()
    return intersection / union


@jaxtyped(typechecker=beartype)
def mask_intersection(
    mask1: Bool[np.ndarray, "h w"],
    mask2: Bool[np.ndarray, "h w"],
    target: Bool[np.ndarray, "h w"] | None = None,
) -> float | None:
    """Calculate the Intersection over Union (IoU) of two boolean masks.

    Arguments:
        mask1 (np.array): A boolean numpy array representing the first mask.
        mask2 (np.array): A boolean numpy array representing the second
            mask.
        target (np.array | None): A boolean numpy array representing the
            target mask. Defaults to None.

    Returns:
        float: The IoU value as a float, or 0.0 if the intersection is
            empty. If the target mask is provided, it calculates the IoU
            based on the target mask.

    Example:
        >>> calculate_iou(mask1, mask2, target=target)

    Note:
        The Intersection over Union (IoU) is a measure of the overlap
            between two sets. In this context, it is used to evaluate the
            overlap between two masks.

    """
    # Calculate intersection between two masks
    if mask1.shape != mask2.shape:
        msg = f"mask1 and mask2 must have the same shape. {mask1.shape} and {mask2.shape}"
        raise ValueError(
            msg,
        )
    intersection = np.logical_and(mask1, mask2).sum()
    if intersection == 0:
        return 0.0
    if target is None:
        target_sum = min(mask2.sum(), mask1.sum())
    else:
        target_sum = target.sum()
    return intersection / target_sum


@jaxtyped(typechecker=beartype)
def contour_from_mask(
    mask: Image.Image | Bool[np.ndarray, "h w"],
) -> list[np.ndarray]:
    """Extract the contours from a given image mask.

    This function takes in an image mask or a boolean NumPy array and
        returns a list of contours found in the mask.

    Arguments:
        mask (Union[Image.Image, np.array]): Either an Image.Image object or
            a boolean NumPy array representing the mask.

    Returns:
        List[np.array]: A list of NumPy arrays containing the contours found
            in the mask.

    Example:
        >>> find_contours(mask)

    Note:
        The input mask can be either a PIL Image object or a boolean NumPy
            array.

    """
    if isinstance(mask, Image.Image):
        mask = np.asarray(mask.convert("L"))
    if mask.ndim == 3:
        mask = mask[:, :, 0]
    mask = mask.astype(bool) * 255.0
    mask = mask.astype(np.uint8)
    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_NONE,
    )
    return list(contours)
