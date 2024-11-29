from pathlib import Path
from typing import Literal

import numpy as np
import torch
from beartype import beartype
from PIL import Image, ImageDraw, ImageFont

from pixelcache.tools.logger import get_logger
from pixelcache.tools.utils import ImageSize, read_image

logger = get_logger()


@beartype
def get_font(font_path: str, text_size: float) -> ImageFont.FreeTypeFont:
    """Generate a FreeTypeFont object using a specified font file and text.

        size.

    This function accepts a font file path and a text size, verifies if the
        font file exists,
    and returns a FreeTypeFont object for the specified font.

    Arguments:
        font_path (str): The path to the font file. This must be a valid
            path to a .ttf or .otf file.
        text_size (float): The size of the text. This must be a positive
            number representing the desired font size.

    Returns:
        ImageFont.FreeTypeFont: A FreeTypeFont object corresponding to the
            specified font and text size.
        This can be used to render text with the specified font and size.

    Example:
        >>> generate_font("/path/to/font.ttf", 12)

    Note:
        Raises a FileNotFoundError if the specified font file does not
            exist.

    """
    if not Path(font_path).is_file():
        msg = f"Font file not found {font_path}"
        raise RuntimeError(msg)
    return ImageFont.truetype(font_path, int(text_size))


@beartype
def get_font_path(font: str = "JetBrainsMono-Regular", /) -> str:
    """Retrieve the file path of a specified font.

    This function accepts a font name as a string and returns the file path
        of the corresponding font file. If the font file does not exist, a
        TypeError is raised.

    Arguments:
        font (str): The name of the font for which the file path is to be
            retrieved.

    Returns:
        str: The file path of the font file corresponding to the input font
            name.

    Example:
        >>> get_font_path("Arial")

    Note:
        Make sure the font name is correct and the corresponding font file
            exists.

    """
    font_path = Path("pixelcache") / "fonts" / f"{font}.ttf"
    if not font_path.exists():
        msg = f"File {font_path!s} not found. Please check path"
        raise TypeError(msg)
    return str(font_path)


@beartype
def create_text(
    img_path: str | Path | np.ndarray,
    texts: list[str],
    background: str = "white",
    orientation: Literal["horizontal", "vertical"] = "horizontal",
    font_path: str | Path | None = None,
) -> np.ndarray:
    """Generate an image with texts displayed on it based on provided.

        parameters.

    Arguments:
        img_path (str | np.array): Path to the image file or a numpy array
            representing the image.
        texts (List[str]): List of strings containing the texts to be
            displayed on the image.
        background (str): Background color of the image. Defaults to
            'white'.
        orientation (str): Orientation of the text, either 'horizontal' or
            'vertical'. Defaults to 'horizontal'.
        font_path (str): Path to the font file to be used for the text.
            Defaults to 'JetBrainsMono-ExtraBold'.

    Returns:
        np.array: A numpy array representing the image with the specified
            texts displayed on it.

    Example:
        >>> create_text_image('path/to/image.jpg', ['Hello', 'World'],
            background='black', orientation='vertical',
            font_path='path/to/font.ttf')

    Note:
        The returned numpy array can be directly used for image processing
            tasks.

    """
    foreground_tuple = (0, 0, 0) if background == "white" else (255, 255, 255)
    background_tuple = (255, 255, 255) if background == "white" else (0, 0, 0)

    if font_path is None:
        font_path = get_font_path()

    font_path = str(Path(font_path).absolute())
    if not Path(font_path).is_file():
        msg = f"Font file not found {font_path}"
        raise RuntimeError(msg)

    if isinstance(img_path, str | Path):
        image = read_image(img_path)
    else:
        image = img_path

    if torch.is_tensor(image):
        msg = f"str, or Image, or numpy array expected, got {type(image)}"
        raise RuntimeError(
            msg,
        )
    if image.ndim != 3:
        logger.error("expected HxWx3 arrays, got {image.shape}")

    if orientation not in ["horizontal", "vertical"]:
        msg = f"Text orientation {orientation} not supported"
        raise RuntimeError(msg)

    if orientation == "vertical":
        image = np.rot90(image, k=-1)
        texts.reverse()

    white_image = []
    # first get the best size for all texts
    sizes = []
    for text in texts:
        _, font_size = display_string(
            text=text,
            size=ImageSize(
                height=image.shape[0],
                width=image.shape[1] // len(texts),
            ),
            font_path=font_path,
        )
        sizes.append(font_size)
    best_size = min(sizes)
    for text in texts:
        disp_str = display_string(
            text=text,
            size=ImageSize(
                height=image.shape[0],
                width=image.shape[1] // len(texts),
            ),
            background=background_tuple,
            foreground=foreground_tuple,
            font_path=font_path,
            force_size=best_size,
        )[0]
        white_image.append(disp_str)
    white_image = np.hstack(white_image)
    white_image = remove_white_text(white_image)
    image = np.vstack((white_image, image))

    if orientation == "vertical":
        image = np.rot90(image, k=1)
    return image


def textsize(text: str, font: ImageFont.FreeTypeFont) -> tuple[int, int]:
    """Get the size of the text when rendered with the specified font."""
    # im = Image.new(mode="P", size=(0, 0))
    # draw = ImageDraw.Draw(im)
    # _, _, width, height = draw.textbbox((0, 0), text=text, font=font)
    # return width, height
    _, _, width, height = font.getbbox(text)
    return int(width), int(height)


@beartype
def display_string(
    *,
    text: str,
    size: ImageSize,
    font_path: str | None = None,
    justification: str = "center",
    force_size: int | None = None,
    background: tuple[int, int, int] = (255, 255, 255),
    foreground: tuple[int, int, int] = (0, 0, 0),
) -> tuple[Image.Image, int]:
    """Create a black image with wrapped text, adjusting the font size and justification.

    Args:
        text (str): Text to display on the image.
        size (ImageSize): Width and height of the image.
        justification (str): Justification for the text ('left', 'center', 'right').
        font_path (str): Path to the font file to use. Defaults to None.
        font_size (int): Font size of the text.
        background (Tuple[int, int, int], optional): Background color of the image. Defaults to (255, 255, 255).
        foreground (Tuple[int, int, int], optional): Foreground color of the text. Defaults to (0, 0, 0).

    Returns:
        Image.Image: A PIL Image object with the text rendered on it.
        int: The font size used for rendering the text.

    """
    if font_path is None:
        font_path = get_font_path()
    # Create a black image
    _image_size = (size.width, size.height)
    img = Image.new("RGB", _image_size, color=background)
    draw = ImageDraw.Draw(img)

    if force_size is not None:
        font_size = force_size
    else:
        # Load the default font
        initial_font_size = (
            min(_image_size) // 50
        )  # Initial font size - will be increase to fit the text
        font_size = initial_font_size
        font = ImageFont.truetype(font_path, size=font_size)
        # Estimate maximum font size
        last_font_size = font_size
        while True:
            test_size = textsize(text, font=font)
            if test_size[0] > _image_size[0] or test_size[1] > _image_size[1]:
                break
            font_size = max(
                initial_font_size, font_size + 1
            )  # Increase font size until it fits
            if font_size == last_font_size:
                break
            last_font_size = font_size
            font = ImageFont.truetype(font_path, size=font_size)

        font_size = int(
            font_size * 0.8
        )  # Slightly reduce font size for better fit
    font = ImageFont.truetype(font_path, size=font_size)

    # Text Wrapping
    lines = []
    for line in text.split("\n"):
        words = line.split()
        current_line: list[str] = []
        for word in words:
            test_line = " ".join([*current_line, word])
            width, _ = textsize(test_line, font=font)
            if width <= _image_size[0]:
                current_line.append(word)
            else:
                lines.append(" ".join(current_line))
                current_line = [word]
        lines.append(" ".join(current_line))

    # Determine vertical position for central alignment
    total_height = len(lines) * textsize(lines[0], font=font)[1]
    current_y = (_image_size[1] - total_height) // 2

    # Draw text onto the image
    for line in lines:
        width, height = textsize(line, font=font)
        if justification == "center":
            x = (_image_size[0] - width) // 2
        elif justification == "right":
            x = _image_size[0] - width
        else:
            x = 0  # 'left' justification is default

        draw.text((x, current_y), line, font=font, fill=foreground)
        current_y += height

    return img, font_size


@beartype
def remove_white_text(image: np.ndarray) -> np.ndarray:
    """Remove white text from the top and bottom of an image.

    This function accepts an image in the form of a NumPy array, checks if
        the input is a string or a PIL Image,
    removes any white text present at the top and bottom of the image, and
        returns the modified image.

    Arguments:
        image (np.ndarray): A NumPy array representing an image. The
            function checks if this is a string or a PIL Image.

    Returns:
        np.ndarray: The modified image as a NumPy array, with white text
            removed from the top and bottom.

    Example:
        >>> remove_white_text(image_array)

    Note:
        The function does not modify the original image array, but returns a
            new one.

    """
    if isinstance(image, str):
        image = Image.open(image)
    if isinstance(image, Image.Image):
        image = np.asarray(image)
    for start_row in range(image.shape[0]):
        if not (image[start_row] == 255).all().item():
            break
    for end_row in range(image.shape[0] - 1, -1, -1):
        if not (image[end_row] == 255).all().item():
            break
    return image[start_row - 2 : min(end_row + 2, image.shape[0] - 1)]


@beartype
def draw_text(
    image: np.ndarray | Image.Image,
    text: str,
    position: tuple[int, int],
    font_path: str | Path | None = None,
    font_size: float = 20.0,
    color: tuple[int, int, int] = (255, 255, 255),
) -> np.ndarray | Image.Image:
    """Draw specified text on an image at a given position using a specific.

        font and color.

    Arguments:
        image (Union[np.ndarray, PIL.Image.Image]): The input image, which
            can either be a NumPy array or a PIL Image object.
        text (str): The text to be drawn on the image.
        position (Tuple[int, int]): A tuple containing the x and y
            coordinates where the text should be drawn.
        font_path (str, optional): The font file to be used for drawing the
            text. Defaults to 'default_font_path'.
        font_size (float, optional): The size of the font to be used for
            drawing the text. Defaults to 20.0.
        color (Tuple[int, int, int], optional): The color of the text to be
            drawn on the image. Defaults to white (255, 255, 255).

    Returns:
        Union[np.ndarray, PIL.Image.Image]: The modified image with the text
            drawn on it. The return type matches the input type: if a NumPy
            array was input, a NumPy array is returned; if a PIL Image
            object was input, a PIL Image object is returned.

    Example:
        >>> draw_text_on_image(image, 'Hello World!', (50, 50), 'Arial.ttf',
            30, (0, 0, 0))

    Note:
        The position is defined from the top-left corner of the image, with
            positive x going right and positive y going down.

    """
    if isinstance(image, Image.Image):
        is_pil = True
        image = np.asarray(image)
    else:
        is_pil = False
    if font_path is None:
        font_path = get_font_path()
    font = get_font(str(font_path), font_size)
    image = Image.fromarray(image)
    draw = ImageDraw.Draw(image)
    draw.text(position, text, font=font, fill=color)
    if is_pil:
        return image
    return np.asarray(image)
