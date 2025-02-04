from pathlib import Path

import numpy as np
import torch
from beartype import beartype
from jaxtyping import Float, UInt8, jaxtyped
from PIL import Image, ImageDraw, ImageFont

from pixelcache.tools.image import pil2tensor, tensor2pil


@jaxtyped(typechecker=beartype)
def draw_bbox(
    image: (
        Image.Image
        | UInt8[np.ndarray, "h w 3"]
        | Float[torch.Tensor, "1 c h w"]
    ),
    bbox: list[tuple[float, float, float, float]],
    *,
    is_normalized: bool = False,
    color: str | tuple[int, int, int] = "red",
    width: int = 3,
    text: list[str] | None = None,
) -> Image.Image | UInt8[np.ndarray, "h w 3"] | Float[torch.Tensor, "1 c h w"]:
    """Draw bounding boxes on an image with specified color, width, and text.

    Arguments:
        image (Union[PIL.Image, np.array, torch.Tensor]): The image to draw
            bounding boxes on.
            This can be a PIL Image, numpy array, or torch tensor.
        bbox (List[Tuple[float, float, float, float]]): A list of bounding
            boxes to draw. Each bounding box is a tuple of floats in the
            format (x1, y1, x2, y2) where x1, y1 is the top-left corner and
            x2, y2 is the bottom-right corner.
        If is_normalized is True, the bounding box coordinates are
            normalized to [0, 1] and should be floats.
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
        if is_normalized:
            if box[0] > 1 or box[1] > 1 or box[2] > 1 or box[3] > 1:
                msg = f"box {box} is not normalized [0,1] and it is float. If should not be normalized, please convert to int"
                raise ValueError(
                    msg,
                )
            w, h = image.size
            box = (
                round(box[0] * w),
                round(box[1] * h),
                round(box[2] * w),
                round(box[3] * h),
            )  # x1, y1, x2, y2
        draw.rectangle(box, outline=color, width=width)
        if text is not None and text[idx]:
            size_text = max(max(image.size) * 0.04, 9.0)
            get_font_path = (
                Path(__file__).parent / "fonts" / "JetBrainsMono-ExtraBold.ttf"
            )
            font = ImageFont.truetype(str(get_font_path), round(size_text))
            draw.text(
                (round(box[0]), round(box[1])),
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
