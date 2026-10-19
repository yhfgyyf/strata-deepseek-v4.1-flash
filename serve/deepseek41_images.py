"""DeepSeek-V4.1 image preprocessing, using the existing NumPy/Pillow dependencies.

Geometry and patch layout follow the model's supplied MIT-licensed
inference/image_processor.py. EXIF orientation is deliberately preserved, as
in that reference; it is part of the input pixels rather than a server policy.
"""
from __future__ import annotations

from dataclasses import dataclass
import io
import math

import numpy as np
from PIL import Image, ImageOps

IMAGE_TOKEN = 129264
PATCH_SIZE = 14
DOWNSAMPLE = 3
MAX_TOKENS = 1024
MIN_PIXELS = 295936


@dataclass(frozen=True)
class ImagePatches:
    values: np.ndarray
    height: int
    width: int
    llm_height: int
    llm_width: int

    @property
    def tokens(self):
        return self.llm_height * (self.llm_width + 1) + 2


def image_grid(width: int, height: int, max_tokens: int = MAX_TOKENS):
    if width < 1 or height < 1 or not 4 <= max_tokens <= MAX_TOKENS:
        raise ValueError("invalid image dimensions or token budget")
    if width * height < MIN_PIXELS:
        ratio = math.sqrt(MIN_PIXELS / (width * height))
        width, height = int(width * ratio), int(height * ratio)
    best_w = math.ceil(width / PATCH_SIZE) * PATCH_SIZE
    best_h = math.ceil(height / PATCH_SIZE) * PATCH_SIZE

    def grid(h, w):
        return math.ceil((h // PATCH_SIZE) / DOWNSAMPLE), math.ceil((w // PATCH_SIZE) / DOWNSAMPLE)

    nh, nw = grid(best_h, best_w)
    if nh * (nw + 1) + 2 > max_tokens:
        ratio = height / width
        max_w = math.sqrt((max_tokens - 2) / ratio + 0.25) - 0.5
        max_h = max_w * ratio
        cell = PATCH_SIZE * DOWNSAMPLE
        if max_w < 1:
            best_h, best_w = (max_tokens - 2) // 2 * cell, cell
        elif max_h < 1:
            best_h, best_w = cell, (max_tokens - 3) * cell
        else:
            beta = min(math.floor(max_w) * cell / width, math.floor(max_h) * cell / height)
            best_h = math.floor(height * beta / PATCH_SIZE) * PATCH_SIZE
            best_w = math.floor(width * beta / PATCH_SIZE) * PATCH_SIZE
        nh, nw = grid(best_h, best_w)
    if nh < 1 or nw < 1 or nh * (nw + 1) + 2 > max_tokens:
        raise ValueError("image grid exceeds the token budget")
    return nh, nw, best_h, best_w


def round_bf16(values: np.ndarray):
    values = np.ascontiguousarray(values, dtype=np.float32)
    bits = values.view(np.uint32)
    rounded = (bits + np.uint32(0x7fff) + ((bits >> 16) & np.uint32(1))) & np.uint32(0xffff0000)
    return rounded.view(np.float32)


def preprocess(data: bytes, max_tokens: int = MAX_TOKENS):
    try:
        with Image.open(io.BytesIO(data)) as source:
            image = source.convert("RGB")
    except (OSError, ValueError) as error:
        raise ValueError("invalid image") from error
    lh, lw, h, w = image_grid(image.width, image.height, max_tokens)
    image = ImageOps.pad(image, (w, h), color=(127, 127, 127))
    # Keep each FP32 operation separate to match torch's FP32 -> BF16 boundary.
    pixels = np.asarray(image, dtype=np.float32).transpose(2, 0, 1) / np.float32(255)
    pixels = round_bf16((pixels - np.float32(0.5)) / np.float32(0.5))
    vh, vw = h // PATCH_SIZE, w // PATCH_SIZE
    patches = pixels.reshape(3, vh, PATCH_SIZE, vw, PATCH_SIZE).transpose(1, 3, 0, 2, 4)
    patches = np.ascontiguousarray(patches.reshape(vh * vw, 3, PATCH_SIZE, PATCH_SIZE))
    return ImagePatches(patches, vh, vw, lh, lw)


def expand_images(ids: list[int], patches: list[ImagePatches]):
    if ids.count(IMAGE_TOKEN) != len(patches):
        raise ValueError("image placeholder count does not match image inputs")
    output, spans, image_index = [], [], 0
    for token in ids:
        if token != IMAGE_TOKEN:
            output.append(token)
        else:
            image = patches[image_index]
            spans.append((len(output), image))
            output.extend([IMAGE_TOKEN] * image.tokens)
            image_index += 1
    return output, spans
