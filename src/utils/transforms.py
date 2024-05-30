from PIL import Image


class HorizontalCenterCrop:
    """
    Applies a horizontal center crop to an image.

    Args:
        width (int): The desired width of the cropped image.

    Attributes:
        width (int): The desired width of the cropped image.

    Methods:
        __call__(img: Image.Image) -> Image.Image: Applies a horizontal center crop to the input image.
    """

    def __init__(self, width: int):
        self.width = width

    def __call__(self, img: Image.Image) -> Image.Image:
        """
        Applies a horizontal center crop to the input image.

        Args:
            img (PIL.Image.Image): The input image.

        Returns:
            PIL.Image.Image: The cropped image.
        """
        width, height = img.size
        left = (width - self.width) // 2
        top = 0
        right = left + self.width
        bottom = height
        return img.crop((left, top, right, bottom))
