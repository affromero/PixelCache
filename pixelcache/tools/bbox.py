import numpy as np
import torch
from beartype import beartype
from jaxtyping import Float, Int64, UInt8, jaxtyped

from pixelcache.tools.image import (
    ImageSize,
    numpy2tensor,
    resize_image,
    tensor2numpy,
)


@jaxtyped(typechecker=beartype)
def crop_from_bbox(
    image_np: UInt8[np.ndarray, "h w c"],
    /,
    bboxes: list[tuple[float, float, float, float]],
    *,
    is_normalized: bool,
) -> UInt8[np.ndarray, "h1 w1 c"]:
    """Crops an image based on provided bounding box coordinates.

    This function takes an image and a list of bounding boxes as input and
        crops the image based on the bounding box coordinates provided. It
        supports both normalized and non-normalized bounding box
        coordinates.

    Arguments:
        image (np.ndarray): The input image to be cropped.
        bboxes (List[Tuple[float, float, float, float]]): A list of bounding
            boxes. Each bounding box is represented as a tuple of (x_min,
            y_min, x_max, y_max) or (x_min_norm, y_min_norm, x_max_norm,
            y_max_norm).
        is_normalized (bool): A flag indicating whether the bounding box
            coordinates are normalized.

    Returns:
        np.ndarray: A new np.ndarray object representing the cropped
            image based on the provided bounding box coordinates.

    Example:
        >>> crop_image(image, [(10, 10, 50, 50)])

    Note:
        If the bounding box coordinates are given as floats, they are
            assumed to be normalized to the range [0, 1].

    """
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
    xmin = int(round(max(0, xmin)))
    ymin = int(round(max(0, ymin)))
    xmax = int(round(min(image_np.shape[1], xmax)))
    ymax = int(round(min(image_np.shape[0], ymax)))
    return image_np[ymin:ymax, xmin:xmax]


@jaxtyped(typechecker=beartype)
def uncrop_from_bbox(
    base_image: UInt8[np.ndarray, "h w c"],
    image: UInt8[np.ndarray, "h1 w1 c"],
    bboxes: list[tuple[float, float, float, float]],
    *,
    is_normalized: bool,
    resize: bool = False,
    blend_width: int = 0,
) -> UInt8[np.ndarray, "h w c"]:
    """Uncrop an image from given bounding boxes and return the resulting.

        image.

    Arguments:
        base_image (Image): The base image from which to uncrop the image.
        image (Image): The image to be uncropped.
        bboxes (List[Tuple[float, float, float, float]]): A list of bounding
            boxes. Each bounding box is represented as a tuple of (x_min,
            y_min, x_max, y_max) or (x_min_norm, y_min_norm, x_max_norm,
            y_max_norm).
        is_normalized (bool): A flag indicating whether the bounding box
            coordinates are normalized.
        resize (bool): A flag indicating whether to resize the image to fit
            the bounding boxes. Defaults to False.
        blend_width (int): Width of the blending region in pixels. Defaults to 10.

    Returns:
        np.ndarray: The uncropped image with blended edges.

    Example:
        >>> uncrop_image(base_image, image, bboxes, resize=True, blend_width=15)

    Note:
        The bounding boxes can be in normalized or absolute coordinates. If
            the resize flag is True, the image will be resized to fit the
            bounding boxes.

    """
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
    xmin = round(max(0, xmin))
    ymin = round(max(0, ymin))
    xmax = round(min(width, xmax))
    ymax = round(min(height, ymax))
    if resize:
        image = tensor2numpy(
            resize_image(
                numpy2tensor(image),
                ImageSize(width=xmax - xmin, height=ymax - ymin),
                "bilinear",
            )
        )
    # Create a mask for blending
    mask = np.ones_like(image, dtype=np.float32)
    if blend_width > 0:
        # Apply blending to the edges
        for c in range(mask.shape[2]):
            mask[:blend_width, :, c] *= np.linspace(0, 1, blend_width)[:, None]
            mask[-blend_width:, :, c] *= np.linspace(1, 0, blend_width)[
                :, None
            ]
            mask[:, :blend_width, c] *= np.linspace(0, 1, blend_width)
            mask[:, -blend_width:, c] *= np.linspace(1, 0, blend_width)

    # Blend the image into the base_image
    out_image[ymin:ymax, xmin:xmax] = (
        out_image[ymin:ymax, xmin:xmax] * (1 - mask) + image * mask
    ).astype(np.uint8)
    return out_image


@jaxtyped(typechecker=beartype)
def increase_bbox(
    bboxes: list[tuple[float, float, float, float]],
    image_size: ImageSize,
    /,
    *,
    is_normalized: bool,
    margin: float,
) -> list[tuple[float, float, float, float]]:
    """Increase the size of bounding boxes by a specified margin without.

        exceeding image boundaries.

    Arguments:
        bboxes (List[Tuple[float, float, float, float]]): A list of bounding
            boxes. Each bounding box is represented as a tuple of (x_min,
            y_min, x_max, y_max) or (x_min_norm, y_min_norm, x_max_norm,
            y_max_norm).
        image_size (ImageSize): An object representing the size of the
            image.
        is_normalized (bool): A flag indicating whether the bounding box
            coordinates are normalized.
        margin (float): A value specifying the margin by which to increase
            the bounding boxes.

    Returns:
        List[Tuple[float, float, float, float]]: A list of bounding boxes
            with increased size.

    Example:
        >>> increase_bbox([(10, 10, 20, 20)], ImageSize(100, 100), 5)

    Note:
        The function ensures that the expanded bounding boxes do not exceed
            the image boundaries.

    """
    # increase margin depending on how big the bounding box is,
    # eg., margin of 0.2 should increase more for a very small bounding box than for a very large bounding box
    new_bboxes = []
    for bbox in bboxes:
        if is_normalized:
            bbox = (
                bbox[0] * image_size.width,
                bbox[1] * image_size.height,
                bbox[2] * image_size.width,
                bbox[3] * image_size.height,
            )
        # apply margin to the bbox, margin is a fraction of the bbox size
        bbox_h = bbox[3] - bbox[1]
        bbox_w = bbox[2] - bbox[0]
        ratio_height = round(margin * image_size.height / bbox_h)
        ratio_width = round(margin * image_size.width / bbox_w)
        # if these ratios are too high, then the margin affects more the bounding box, which means the bounding box was very small
        if not is_normalized:
            new_bboxes.append(
                (
                    max(0, bbox[0] - ratio_width),
                    max(0, bbox[1] - ratio_height),
                    min(image_size.width, bbox[2] + ratio_width),
                    min(image_size.height, bbox[3] + ratio_height),
                ),
            )
        else:
            new_bboxes.append(
                (
                    max(0, (bbox[0] - ratio_width) / image_size.width),
                    max(0, (bbox[1] - ratio_height) / image_size.height),
                    min(0.9999, (bbox[2] + ratio_width) / image_size.width),
                    min(0.9999, (bbox[3] + ratio_height) / image_size.height),
                ),
            )
    return new_bboxes


@jaxtyped(typechecker=beartype)
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


@jaxtyped(typechecker=beartype)
def bbox_intersection(
    box1: tuple[int, int, int, int],
    box2: tuple[int, int, int, int],
) -> float:
    """Calculate the Intersection over Union (IoU) of two bounding boxes.

    This function generates two images based on the input bounding box
        coordinates. It then computes the area of intersection
    between the two images and returns the IoU, a measure of the overlap
        between the two bounding boxes.

    Arguments:
        box1 (Tuple[int, int, int, int]): A tuple representing the
            coordinates of the first bounding box (x1, y1, x2, y2).
        box2 (Tuple[int, int, int, int]): A tuple representing the
            coordinates of the second bounding box (x3, y3, x4, y4).

    Returns:
        float: The Intersection over Union (IoU) of the two bounding boxes,
            represented as a float.

    Example:
        >>> bbox_intersection((1, 1, 2, 2), (1, 1, 3, 3))

    """
    # Calculate intersection between two bounding boxes
    # convevrt to binary images using max dim
    x1, y1, x2, y2 = box1
    x3, y3, x4, y4 = box2
    img1 = np.zeros((max(y2, y4), max(x2, x4)))
    img2 = np.zeros((max(y2, y4), max(x2, x4)))
    img1[y1:y2, x1:x2] = 1
    img2[y3:y4, x3:x4] = 1
    intersection = np.logical_and(img1, img2).sum()
    # is small included in big?
    if intersection == img1.sum() or intersection == img2.sum():
        return 1.0
    is_bigger = img1.sum() > img2.sum()
    return intersection / (img1.sum() if is_bigger else img2.sum())


@jaxtyped(typechecker=beartype)
def align_bounding_boxes(
    base: Int64[torch.Tensor, "n 4"],
    anchors: Int64[torch.Tensor, "o m 4"],
    /,
) -> tuple[Int64[torch.Tensor, "o m 4"], Int64[torch.Tensor, "o m"]]:
    """Aligns bounding boxes from a base set to a set of anchor bounding boxes.

        based on their intersection over union (IOU) values.

    Arguments:
        base (torch.Tensor): A tensor representing the base bounding boxes.
            It should have a shape of (n, 4), where n is the number of base
            bounding boxes and 4 represents the coordinates of each bounding
            box.
        anchors (torch.Tensor): A tensor representing the anchor bounding
            boxes. It should have a shape of (o, m, 4), where o is the
            number of objects, m is the number of anchor bounding boxes per
            object, and 4 represents the coordinates of each bounding box.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: A tuple containing two tensors:
        1. aligned_indexes (torch.Tensor): A tensor representing the aligned
            anchor bounding boxes. It has the same shape as the input
            'anchors' tensor (o, m, 4).
        2. indexes (torch.Tensor): A tensor representing the indexes of the
            closest base bounding boxes for each anchor bounding box. It has
            a shape of (o, m), where o is the number of objects and m is the
            number of anchor bounding boxes per object.

    Example:
        >>> align_bounding_boxes(base, anchors)

    Note:
        The alignment is based on the Intersection over Union (IOU) values
            between the base bounding boxes and the anchor bounding boxes.

    """
    # all bounding boxes are in the same image
    # for instance, there are 4 people in the image, and each person has 3 bounding boxes
    # then args can be something like [person1_bbox1, person1_bbox2, person1_bbox3], [person2_bbox3, person2_bbox1, person2_bbox2], ...
    # the goal is to align the bounding boxes so that they are in the same order as the base
    aligned_indexes = anchors.clone()
    indexes = torch.zeros((anchors.size(0), anchors.size(1))).long()
    for i in range(anchors.size(0)):
        iou_matrix: Float[torch.Tensor, "n m"] = bbox_iou(base, anchors[i])
        closest_indices = torch.argmax(iou_matrix, dim=0)
        # check indices are different values, otherwise it would mean different anchor bboxes are close to the same base one
        if len(closest_indices.unique()) != closest_indices.size(0):
            msg = f"bounding boxes in anchor {anchors[i]} point to the same base bbox {base}. Check ids {closest_indices}."
            raise ValueError(
                msg,
            )
        # replace with new indices
        indexes[i] = closest_indices
        aligned_indexes[i] = aligned_indexes[i][closest_indices]
    return aligned_indexes, indexes


@jaxtyped(typechecker=beartype)
def points2bbox(
    list_points: list[list[tuple[int, int]]],
    /,
) -> list[tuple[int, int, int, int]]:
    """Convert a list of points to bounding boxes.

    This function takes a list of points represented as tuples of integers
        and converts them into bounding boxes
    represented as tuples of integers (xmin, ymin, xmax, ymax).

    Arguments:
        list_points (List[List[Tuple[int, int]]]): A list of lists where
            each inner list contains tuples of integers
        representing points (x, y).

    Returns:
        List[Tuple[int, int, int, int]]: A list of tuples where each tuple
            represents a bounding box in the format
        (xmin, ymin, xmax, ymax).

    Example:
        >>> convert_to_bounding_boxes([[(1, 2), (3, 4)], [(5, 6), (7, 8)]])
        [(1, 2, 3, 4), (5, 6, 7, 8)]

    Note:
        The bounding box is calculated by finding the minimum and maximum x
            and y values among the points.

    """
    bboxes = []
    for points in list_points:
        # extract min/max from the list of points
        x, y = zip(*points, strict=False)
        xmin = min(x)
        xmax = max(x)
        ymin = min(y)
        ymax = max(y)
        bboxes.append((xmin, ymin, xmax, ymax))
    return bboxes
