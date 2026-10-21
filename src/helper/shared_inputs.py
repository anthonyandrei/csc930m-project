"""Shared StreetSurfaceVis pixels and fixed five-class evaluation protocol."""

from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

SURFACE_TYPES = ("asphalt", "concrete", "paving_stones", "sett", "unpaved")
LABEL_IDS = {name: number for number, name in enumerate(SURFACE_TYPES)}
IMAGE_SHAPE = (96, 128, 3)
SEED = 42


def prepare_image(path):
    """Round crop bounds to nearest even; Lanczos fit, then centered black pad.

    Pixel order is height, width, RGB. Flatten in C order (RGB varies fastest).
    An odd spare padding pixel goes to the bottom/right.
    """
    try:
        with Image.open(path) as source:
            width, height = source.size
            box = tuple(round(value) for value in
                        (width * 0.25, height * 0.5, width * 0.75, height))
            if box[2] <= box[0] or box[3] <= box[1]:
                raise ValueError("Image is too small for the lower-center crop")
            crop = source.convert("RGB").crop(box)
            fitted = ImageOps.contain(crop, (128, 96), method=Image.Resampling.LANCZOS)
            prepared = Image.new("RGB", (128, 96), (0, 0, 0))
            prepared.paste(fitted, ((128 - fitted.width) // 2, (96 - fitted.height) // 2))
            return np.asarray(prepared, dtype=np.float32) / np.float32(255.0)
    except (OSError, ValueError) as error:
        raise ValueError(f"Cannot prepare required image {path}: {error}") from error


def prepare_split(records, image_dir):
    """Prepare one assigned split in row order; fail instead of dropping IDs.

    Filenames use image IDs under image_dir, so saved VM paths stay portable.
    The flat array is a view of images, without a second pixel allocation.
    """
    ids = records["mapillary_image_id"]
    if records.empty or ids.isna().any() or ids.duplicated().any():
        raise ValueError("A split needs nonempty, unique image IDs")
    if not ids.map(lambda value: isinstance(value, str) and value.isdigit()).all():
        raise ValueError("Image IDs must be digit strings")
    if records["split"].nunique(dropna=False) != 1 or not records["split"].isin(["train", "val", "test"]).all():
        raise ValueError("Every record must belong to the same train, val, or test split")
    labels = records["surface_type"].map(LABEL_IDS)
    if labels.isna().any():
        raise ValueError("Unknown or missing surface-type label")
    images = np.empty((len(records), *IMAGE_SHAPE), dtype=np.float32)
    for index, image_id in enumerate(ids):
        images[index] = prepare_image(Path(image_dir) / f"{image_id}.jpg")
    return {"image_ids": ids.to_numpy(copy=True), "labels": labels.to_numpy(dtype=np.int64),
            "images": images, "flat": images.reshape(len(records), -1)}
