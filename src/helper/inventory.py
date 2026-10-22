"""Audit StreetSurfaceVis release metadata and its 256-pixel image directory."""

import filecmp
import hashlib
import importlib.metadata
import os
import platform
import sys

import pandas as pd
from PIL import Image

EXPECTED_TOTAL_RECORDS = 9122
EXPECTED_TRAIN_RECORDS = 8346
EXPECTED_TEST_RECORDS = 776

EXPECTED_CLASSES = {
    "asphalt",
    "concrete",
    "paving_stones",
    "sett",
    "unpaved",
}

EXPECTED_QUALITIES = {
    "bad",
    "excellent",
    "good",
    "intermediate",
    "very_bad",
}

EXPECTED_COLUMNS = [
    "mapillary_image_id",
    "user_id",
    "user_name",
    "captured_at",
    "longitude",
    "latitude",
    "train",
    "surface_type",
    "surface_quality",
]


def get_environment_info():
    """Return the Python, OS, and package versions shown in the notebook."""
    packages = ["numpy", "pandas", "opencv-python", "pillow", "scikit-learn"]
    versions = {}
    for pkg in packages:
        try:
            versions[pkg] = importlib.metadata.version(pkg)
        except importlib.metadata.PackageNotFoundError:
            versions[pkg] = "not installed"

    return {
        "python_version": sys.version.split()[0],
        "os": f"{platform.system()} {platform.release()}",
        "packages": versions,
    }


def load_and_validate_metadata(csv_path):
    """Load the full release CSV and summarize its schema, labels, and split.

    Duplicate counts are rows beyond the first matching image ID or complete
    CSV record. The function reports unexpected values without changing rows.
    Returns the loaded DataFrame and a summary dictionary.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Metadata file not found at: {csv_path}")

    df = pd.read_csv(csv_path, dtype={"mapillary_image_id": str})

    missing_cols = set(EXPECTED_COLUMNS) - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns in CSV: {missing_cols}")

    total_rows = len(df)
    train_count = int(df["train"].sum())
    test_count = total_rows - train_count
    unique_ids = df["mapillary_image_id"].nunique()
    null_counts = df.isnull().sum().to_dict()

    classes_found = set(df["surface_type"].dropna().unique())
    unexpected_classes = classes_found - EXPECTED_CLASSES

    qualities_found = set(df["surface_quality"].dropna().unique())
    unexpected_qualities = qualities_found - EXPECTED_QUALITIES

    train_values_found = set(df["train"].dropna().unique())
    invalid_train_flags = train_values_found - {True, False, "True", "False"}

    audit_summary = {
        "total_records": total_rows,
        "expected_total": EXPECTED_TOTAL_RECORDS,
        "train_records": train_count,
        "expected_train": EXPECTED_TRAIN_RECORDS,
        "test_records": test_count,
        "expected_test": EXPECTED_TEST_RECORDS,
        "unique_ids": unique_ids,
        "duplicate_ids": total_rows - unique_ids,
        "fully_repeated_rows": int(df.duplicated().sum()),
        "has_nulls": any(v > 0 for v in null_counts.values()),
        "unexpected_classes": list(unexpected_classes),
        "unexpected_qualities": list(unexpected_qualities),
        "invalid_train_flags": list(invalid_train_flags),
    }

    return df, audit_summary


def audit_image_inventory(metadata_df, image_dir):
    """Match each metadata row to a JPG and check whether Pillow can open it.

    Returns the row-level inventory, count summary, missing image IDs,
    decode failures as (ID, error, byte size), and extra filenames. A decode
    failure is excluded from the later byte comparison.
    """
    if not os.path.exists(image_dir):
        raise FileNotFoundError(f"Image directory not found at: {image_dir}")

    dir_files = set(os.listdir(image_dir))

    expected_ids = metadata_df["mapillary_image_id"].tolist()
    missing_ids = []
    decode_failures = []
    readable_paths = {}

    for img_id in expected_ids:
        filename = f"{img_id}.jpg"
        file_path = os.path.join(image_dir, filename)

        if filename not in dir_files:
            missing_ids.append(img_id)
            continue

        file_size = os.path.getsize(file_path)

        try:
            with Image.open(file_path) as img:
                img.load()
            readable_paths[img_id] = file_path
        except Exception as err:
            decode_failures.append((img_id, str(err), file_size))

    extra_files = [
        f
        for f in dir_files
        if not f.startswith(".") and f not in {f"{i}.jpg" for i in expected_ids}
    ]

    inventory_df = metadata_df.copy()
    inventory_df["file_path"] = inventory_df["mapillary_image_id"].map(
        lambda x: os.path.join(image_dir, f"{x}.jpg")
    )
    inventory_df["is_present"] = inventory_df["file_path"].map(os.path.exists)
    inventory_df["is_readable"] = inventory_df["mapillary_image_id"].isin(
        readable_paths
    )

    summary = {
        "expected_images": len(expected_ids),
        "found_images": len(expected_ids) - len(missing_ids),
        "readable_images": len(readable_paths),
        "missing_images": len(missing_ids),
        "decode_failures": len(decode_failures),
        "extra_files": len(extra_files),
        "unusable_images": len(missing_ids) + len(decode_failures),
    }

    return inventory_df, summary, missing_ids, decode_failures, extra_files


def audit_duplicate_images(inventory_df):
    """Find byte-identical files among distinct readable inventory paths.

    SHA-256 narrows candidate matches; a byte comparison confirms each group.
    Returns group counts, every member's ID, labels and published split flag,
    and paths that could not be read while hashing. Within-partition and
    cross-split group counts can overlap for groups with several members.
    """
    readable = inventory_df.loc[inventory_df["is_readable"]]
    # Repeated metadata rows should not make a file count as its own duplicate.
    paths = readable["file_path"].drop_duplicates()
    digests = {}
    read_failures = []

    for path in paths:
        try:
            digest = hashlib.sha256()
            with open(path, "rb") as image_file:
                # Read 1 MiB at a time; the empty byte string ends the loop at EOF.
                for chunk in iter(lambda: image_file.read(1024 * 1024), b""):
                    digest.update(chunk)
            # Files with the same hash become candidates for a full byte comparison.
            digests.setdefault(digest.digest(), []).append(path)
        except OSError:
            read_failures.append(path)

    groups = []
    for candidates in digests.values():
        confirmed = []
        for path in candidates:
            for group in confirmed:
                # Compare with one member of each group, even when hashes match.
                if filecmp.cmp(path, group[0], shallow=False):
                    group.append(path)
                    break
            else:
                # Python runs this else only if the inner loop never reaches break.
                confirmed.append([path])
        groups.extend(group for group in confirmed if len(group) > 1)

    columns = ["mapillary_image_id", "surface_type", "surface_quality", "train"]
    members = []
    for number, group in enumerate(groups, 1):
        # Restore each file's IDs and labels without repeating identical metadata rows.
        rows = readable.loc[
            readable["file_path"].isin(group), columns
        ].drop_duplicates()
        rows = rows.assign(group=number)
        members.append(rows)
    group_df = (
        pd.concat(members, ignore_index=True)
        if members
        else pd.DataFrame(columns=["group", *columns])
    )

    summary = {
        "compared_files": len(paths) - len(read_failures),
        "read_failures": len(read_failures),
        "identical_file_groups": len(groups),
        "affected_image_ids": group_df["mapillary_image_id"].nunique(),
        "within_train_groups": 0,
        "within_test_groups": 0,
        "cross_split_groups": 0,
        "conflicting_surface_type_groups": 0,
        "conflicting_surface_quality_groups": 0,
    }
    for _, rows in group_df.groupby("group"):
        flags = set(rows["train"])
        # Count distinct IDs per split; one group can count within and across splits.
        summary["within_train_groups"] += (
            rows.loc[rows["train"] == True, "mapillary_image_id"].nunique() >= 2
        )
        summary["within_test_groups"] += (
            rows.loc[rows["train"] == False, "mapillary_image_id"].nunique() >= 2
        )
        summary["cross_split_groups"] += flags == {True, False}
        # Multiple labels for identical files indicate a labeling conflict.
        summary["conflicting_surface_type_groups"] += rows["surface_type"].nunique() > 1
        summary["conflicting_surface_quality_groups"] += (
            rows["surface_quality"].nunique() > 1
        )

    return summary, group_df, read_failures
