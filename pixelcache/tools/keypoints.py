import torch
from beartype import beartype
from jaxtyping import Float, jaxtyped

from pixelcache.tools.image import ImageSize


@jaxtyped(typechecker=beartype)
def kpts2bbox(
    kpts: (
        Float[torch.Tensor, "n 3"]
        | Float[torch.Tensor, "n 2"]
        | Float[torch.Tensor, "b n 3"]
        | Float[torch.Tensor, "b n 2"]
    ),
    *,
    normalized: bool,
    image_size: ImageSize | None = None,
) -> list[tuple[int, int, int, int]] | list[tuple[float, float, float, float]]:
    """Generate bounding boxes from given keypoints.

    Arguments:
        kpts (torch.Tensor): A tensor containing keypoints in one of the
            following formats: 'n 3', 'n 2', 'b n 3', or 'b n 2'.
        normalized (bool): A flag indicating whether the keypoints are
            normalized.
        image_size (ImageSize | None): Optional. Specifies the size of the
            image. Defaults to None.

    Returns:
        List[Tuple[Union[int, float]]]: A list of tuples containing the
            bounding box coordinates (x1, y1, x2, y2) for each set of
            keypoints. The coordinates can be either integers or floats
            based on the normalization.

    Example:
        >>> get_bounding_boxes(kpts, normalized, image_size=image_size)

    Note:
        The keypoints tensor is expected to be in one of the specified
            formats.

    """
    # kpts: b x n x 3 (x, y, v) in image coordinate
    # bbox: b x 4 (x1, y1, x2, y2)
    # v: 0 - not visible, 1 - visible, 2 - occluded
    if kpts.ndim == 2:
        kpts = kpts.unsqueeze(0)
    if kpts.shape[-1] == 2:
        kpts = torch.cat((kpts, torch.ones_like(kpts[..., :1])), dim=-1)
    # check kpts is in image coordinate [h, w]
    if kpts.max() <= 1 and kpts.sum() > 0:
        msg = "kpts must be in image coordinate [h, w]"
        raise ValueError(msg)
    if normalized:
        if image_size is None:
            msg = "image_size must be provided when normalized is True"
            raise ValueError(
                msg,
            )
        kpts[..., 0] /= image_size.width
        kpts[..., 1] /= image_size.height

    if kpts.shape[0] == 0:
        msg = "kpts is empty"
        raise ValueError(msg)
    x1: Float[torch.Tensor, " b"] = kpts[:, :, 0].min(dim=1)[0]
    y1: Float[torch.Tensor, " b"] = kpts[:, :, 1].min(dim=1)[0]
    x2: Float[torch.Tensor, " b"] = kpts[:, :, 0].max(dim=1)[0]
    y2: Float[torch.Tensor, " b"] = kpts[:, :, 1].max(dim=1)[0]
    if not normalized:
        x11 = x1.long()
        y11 = y1.long()
        x22 = x2.long()
        y22 = y2.long()
    else:
        x11 = x1
        y11 = y1
        x22 = x2
        y22 = y2
    return [
        (_x1.item(), _y1.item(), _x2.item(), _y2.item())
        for _x1, _y1, _x2, _y2 in zip(x11, y11, x22, y22, strict=False)
    ]
