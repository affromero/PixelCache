from pathlib import Path

from pixelcache import HashableDict, HashableImage, HashableList
from pixelcache.tools.logger import get_logger

logger = get_logger()


def main() -> None:
    """Main function to demonstrate blending of two images."""
    image0 = "https://images.pexels.com/photos/28811907/pexels-photo-28811907/free-photo-of-majestic-elk-standing-in-forest-clearing.jpeg"
    image1 = Path("pixelcache") / "assets" / "pixel_cache.png"
    images_hash = [HashableImage(image) for image in [image0, image1]]
    for image in images_hash:
        logger.info(f"Image: {image} - Hash: {hash(image)}")
    logger.info(f"Hash for list of images: {hash(HashableList(images_hash))}")
    image_size = images_hash[1].size()
    logger.info(f"Image size: {image_size} - Resizing all to this size")
    resized_images = [image.resize(image_size) for image in images_hash]
    # blend images
    blended_image = resized_images[0].blend(
        resized_images[1], alpha=0.5, with_bbox=False
    )
    # binarize second image
    blended_image_binarized = resized_images[0].blend(
        resized_images[1].to_binary(0.5).invert_binary(),
        alpha=0.2,
        with_bbox=True,
    )
    output_debug = HashableDict(
        {
            "image base": HashableList([resized_images[0]]),
            "image reference": HashableList([resized_images[1]]),
            "blended_image": HashableList([blended_image]),
            "blended_image_binarized": HashableList([blended_image_binarized]),
        }
    )
    output = image1.parent / (str(image1.stem) + "_demo_blend.jpg")
    HashableImage.make_image_grid(
        output_debug, orientation="horizontal", with_text=True
    ).save(output)
    logger.success(f"Output saved to: {output}")


if __name__ == "__main__":
    main()
