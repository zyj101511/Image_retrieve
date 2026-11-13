"""Decode raw LMDB image bytes with EXIF orientation applied."""
from io import BytesIO

from PIL import Image, ImageOps


def decode_image(data):
    with Image.open(BytesIO(data)) as image:
        return ImageOps.exif_transpose(image).convert("RGB")
