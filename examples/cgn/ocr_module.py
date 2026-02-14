# Copyright 2025 Google LLC.
# Demo 示例：图片 OCR 模块，将图片识别为文本。
# 可选依赖：pip install pytesseract Pillow
# 并安装 Tesseract：https://github.com/tesseract-ocr/tesseract

"""OCR module: image path or PIL Image -> text string."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Union

if TYPE_CHECKING:
    from PIL import Image

# 支持传入文件路径或 PIL.Image
_ImageInput = Union[str, "Image.Image"]

try:
    import pytesseract
    from PIL import Image

    _HAS_OCR = True
except ImportError:
    _HAS_OCR = False
    Image = None  # type: ignore
    pytesseract = None  # type: ignore


def ocr_image(image_input: _ImageInput, lang: str | None = None) -> str:
    """从图片中识别文字。

    Args:
        image_input: 图片路径（str）或 PIL.Image 对象。
        lang: Tesseract 语言代码，如 "eng", "chi_sim", "jpn"。默认 None 使用 Tesseract 默认。

    Returns:
        识别出的文本字符串。

    Raises:
        RuntimeError: 未安装 pytesseract/Pillow 或 Tesseract 未安装/未在 PATH 中。
    """
    if not _HAS_OCR:
        raise RuntimeError(
            "OCR 需要安装: pip install pytesseract Pillow，并安装 Tesseract (https://github.com/tesseract-ocr/tesseract)"
        )
    if isinstance(image_input, str):
        if not os.path.isfile(image_input):
            raise FileNotFoundError(f"图片不存在: {image_input}")
        img = Image.open(image_input)
    else:
        img = image_input
    kwargs = {}
    if lang:
        kwargs["lang"] = lang
    return pytesseract.image_to_string(img, **kwargs).strip()
