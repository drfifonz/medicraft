from PIL import Image


class HorizontalCenterCrop:
    def __init__(self, width: int):
        self.width = width

    def __call__(self, img: Image.Image) -> Image.Image:
        width, height = img.size
        left = (width - self.width) // 2
        top = 0
        right = left + self.width
        bottom = height
        return img.crop((left, top, right, bottom))
