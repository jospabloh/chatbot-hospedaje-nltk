"""Small helpers shared by the tests."""
import io

from PIL import Image


def make_image(fmt="PNG", size=(120, 90), color=(90, 140, 200)) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", size, color).save(buffer, format=fmt)
    return buffer.getvalue()
