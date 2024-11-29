from pathlib import Path

from pixelcache import HashableDict, HashableImage, HashableList, ImageSize
from pixelcache.tools.logger import get_logger

logger = get_logger()


def main() -> None:
    """Main function to demonstrate blending of two images."""
    image0 = "https://images.pexels.com/photos/18624700/pexels-photo-18624700/free-photo-of-a-vintage-typewriter.jpeg"
    image1 = Path("pixelcache") / "assets" / "pixel_cache.png"
    images_hash = [HashableImage(image) for image in [image0, image1]]
    for image in images_hash:
        logger.info(f"Image: {image} - Hash: {hash(image)}")
    logger.info(f"Hash for list of images: {hash(HashableList(images_hash))}")
    image_size = images_hash[1].size()
    logger.info(f"Image size: {image_size} - Resizing all to this size")
    resized_images = [image.resize(image_size) for image in images_hash]
    # crop images
    increased_size_pad = ImageSize(
        width=image_size.width + 1000, height=image_size.height + 1000
    )
    mask = (
        images_hash[1]
        .center_pad(increased_size_pad, fill=255)
        .resize(image_size)
        .to_space_color("HSV", getchannel="S")
        .to_binary(0.3)
    )
    cropped = resized_images[1].crop_from_mask(mask)
    uncropped = cropped.uncrop_from_bbox(
        base=resized_images[0], bboxes=mask.mask2bbox(margin=0.0), resize=True
    )
    output_debug = HashableDict(
        {
            "image base": HashableList([resized_images[0]]),
            "image reference": HashableList([resized_images[1]]),
            "cropped_image": HashableList([cropped.resize(image_size)]),
            "uncropped_image": HashableList([uncropped]),
        }
    )
    output = image1.parent / (str(image1.stem) + "_demo_cropUncrop.jpg")
    HashableImage.make_image_grid(
        output_debug, orientation="horizontal", with_text=True
    ).save(output)
    logger.success(f"Output saved to: {output}")


if __name__ == "__main__":
    main()
