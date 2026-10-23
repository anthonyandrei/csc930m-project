"""Shared StreetSurfaceVis pixels and fixed five-class evaluation protocol."""

from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageOps
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)

SURFACE_TYPES = ("asphalt", "concrete", "paving_stones", "sett", "unpaved")
LABEL_IDS = {name: number for number, name in enumerate(SURFACE_TYPES)}
IMAGE_SHAPE = (96, 128, 3)
SEED = 42


def prepare_image(path):
    """Return a lower-center road crop as normalized float32 RGB pixels.

    Keep the middle half of the width and bottom half of the height. Round
    bounds to the nearest integer, with halfway ties going to the even integer.
    Resize to fit 128 by 96 without stretching, using Lanczos interpolation,
    then center on black padding. Output shape is (96, 128, 3), in [0, 1].
    An odd spare padding pixel goes to the bottom/right.
    """
    try:
        with Image.open(path) as source:
            width, height = source.size
            box = tuple(
                round(value)
                for value in (width * 0.25, height * 0.5, width * 0.75, height)
            )
            if box[2] <= box[0] or box[3] <= box[1]:
                raise ValueError("Image is too small for the lower-center crop")
            crop = source.convert("RGB").crop(box)
            # Fit within the target while preserving proportions; paste into the padded canvas.
            fitted = ImageOps.contain(crop, (128, 96), method=Image.Resampling.LANCZOS)
            prepared = Image.new("RGB", (128, 96), (0, 0, 0))
            prepared.paste(
                fitted, ((128 - fitted.width) // 2, (96 - fitted.height) // 2)
            )
            return np.asarray(prepared, dtype=np.float32) / np.float32(255.0)
    except (OSError, ValueError) as error:
        raise ValueError(f"Cannot prepare required image {path}: {error}") from error


def prepare_split(records, image_dir):
    """Prepare one assigned split in row order; fail instead of dropping IDs.

    records needs unique digit-string mapillary_image_id values, surface_type,
    and one shared train/val/test split value. Filenames use IDs under image_dir,
    so saved VM paths stay portable. Return aligned image_ids, int64 labels,
    images shaped (N, 96, 128, 3), and flat features shaped (N, 36864).
    flat shares image storage; RGB channels vary fastest within each row.
    """
    ids = records["mapillary_image_id"]
    if records.empty or ids.isna().any() or ids.duplicated().any():
        raise ValueError("A split needs nonempty, unique image IDs")
    if not ids.map(lambda value: isinstance(value, str) and value.isdigit()).all():
        raise ValueError("Image IDs must be digit strings")
    if (
        records["split"].nunique(dropna=False) != 1
        or not records["split"].isin(["train", "val", "test"]).all()
    ):
        raise ValueError(
            "Every record must belong to the same train, val, or test split"
        )
    labels = records["surface_type"].map(LABEL_IDS)
    if labels.isna().any():
        raise ValueError("Unknown or missing surface-type label")
    # Allocate once and fill in record order so IDs, labels, and pixel rows stay aligned.
    images = np.empty((len(records), *IMAGE_SHAPE), dtype=np.float32)
    for index, image_id in enumerate(ids):
        images[index] = prepare_image(Path(image_dir) / f"{image_id}.jpg")
    return {
        "image_ids": ids.to_numpy(copy=True),
        "labels": labels.to_numpy(dtype=np.int64),
        "images": images,
        "flat": images.reshape(len(records), -1),
    }


def evaluate_predictions(labels, predictions):
    """Score integer IDs in fixed class order; undefined class scores are zero.

    Confusion-matrix rows are actual labels; columns are predicted labels.
    Inputs are equal-length 1D class-ID vectors. Return accuracy, macro_f1,
    a per_class score table, and a labeled confusion_matrix. Macro-F1 is the
    unweighted mean of all five class F1 scores, including absent classes.
    """
    truth, predicted = np.asarray(labels), np.asarray(predictions)
    for values in (truth, predicted):
        if values.ndim != 1 or not len(values) or values.dtype.kind not in "iu":
            raise ValueError("Labels and predictions must be nonempty 1D integer IDs")
        if not np.isin(values, range(len(SURFACE_TYPES))).all():
            raise ValueError("Class IDs must be in the fixed range 0 to 4")
    if truth.shape != predicted.shape:
        raise ValueError("Each label needs exactly one prediction")
    # Fix the class list so missing predictions or classes cannot change the report order.
    class_ids = list(range(len(SURFACE_TYPES)))
    precision, recall, f1, support = precision_recall_fscore_support(
        truth, predicted, labels=class_ids, average=None, zero_division=0
    )
    return {
        "accuracy": float(accuracy_score(truth, predicted)),
        "macro_f1": float(f1.mean()),
        "per_class": pd.DataFrame(
            {"precision": precision, "recall": recall, "f1": f1, "support": support},
            index=SURFACE_TYPES,
        ),
        "confusion_matrix": pd.DataFrame(
            confusion_matrix(truth, predicted, labels=class_ids),
            index=pd.Index(SURFACE_TYPES, name="actual"),
            columns=pd.Index(SURFACE_TYPES, name="predicted"),
        ),
    }


def rank_candidates(candidates):
    """Rank by descending validation macro-F1, then accuracy; return at most five.

    Callers supply full-precision validation scores from evaluate_predictions
    and retain settings alongside them. Exact ties keep input grid order.
    The returned entries are the original candidate dictionaries; test scores
    are never used.
    """
    for candidate in candidates:
        for key in ("validation_macro_f1", "validation_accuracy"):
            score = candidate[key]
            if not np.isfinite(score) or not 0 <= score <= 1:
                raise ValueError(f"{key} must be a finite score between zero and one")
    return sorted(
        candidates,
        key=lambda row: (-row["validation_macro_f1"], -row["validation_accuracy"]),
    )[:5]
