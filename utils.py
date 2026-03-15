from pathlib import Path
from typing import BinaryIO, List, Union

from PIL import Image

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def load_image_file(image_source: Union[BinaryIO, Path]) -> Image.Image:
    """Open an uploaded file/path and convert it to RGB."""
    return Image.open(image_source).convert("RGB")


def caption_to_download_text(caption: str) -> bytes:
    """Convert caption text to bytes for Streamlit download button."""
    return caption.encode("utf-8")


def list_example_images(directory: Union[str, Path]) -> List[Path]:
    """Return sorted image paths from the example_images directory."""
    image_dir = Path(directory)
    if not image_dir.exists():
        return []

    images = [
        path
        for path in image_dir.iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    ]
    return sorted(images)
