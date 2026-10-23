"""Filter audited records, assign grouped splits, and verify their separation."""

import os
from typing import Any

import pandas as pd


def filter_usable_dataset(
    inventory_df: pd.DataFrame,
    duplicate_groups: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Return usable inventory rows and exclusion counts with their image IDs.

    Inputs are the image inventory and byte-identical groups from the audit.
    Exclude every member of a group if either type or quality labels disagree.
    Readable duplicates with consistent labels remain for grouped splitting.
    """
    # Compare distinct label pairs within each group to detect conflicting copies.
    conflicting_ids = []
    for _, rows in duplicate_groups.groupby("group"):
        labels = rows[["surface_type", "surface_quality"]].drop_duplicates()
        if len(labels) > 1:
            conflicting_ids.extend(rows["mapillary_image_id"].tolist())

    unreadable_ids = inventory_df.loc[
        ~inventory_df["is_readable"], "mapillary_image_id"
    ].tolist()

    # An image can fail both checks; count it once in the combined exclusions.
    excluded_ids = sorted(list(set(conflicting_ids + unreadable_ids)))

    usable_df = inventory_df.loc[
        ~inventory_df["mapillary_image_id"].isin(excluded_ids)
    ].copy()

    exclusions_dict = {
        "unreadable_count": len(unreadable_ids),
        "unreadable_ids": unreadable_ids,
        "conflicting_duplicate_count": len(conflicting_ids),
        "conflicting_duplicate_ids": conflicting_ids,
        "total_excluded": len(excluded_ids),
        "excluded_ids": excluded_ids,
    }

    return usable_df, exclusions_dict


def generate_dataset_splits(
    usable_df: pd.DataFrame,
    duplicate_groups: pd.DataFrame,
    train_val_ratio: float = 0.8,
    random_state: int = 42,
) -> pd.DataFrame:
    """Divide the published training pool by class while keeping duplicates together.

    Pass filtered inventory rows and audited duplicate groups. train_val_ratio
    is the target training fraction of the published training pool; the rest
    goes to validation. Whole groups make this ratio approximate. Return a
    copy with group_id and split columns, preserving the remaining test rows.
    Conflicting labels must be filtered first. Published train/test duplicates
    are not reassigned here; verify_dataset_splits rejects that leakage.
    """
    df = usable_df.copy()

    # Treat duplicate copies as one allocation unit; other images get their own group.
    dup_map = dict(
        zip(duplicate_groups["mapillary_image_id"], duplicate_groups["group"])
    )
    df["group_id"] = df["mapillary_image_id"].map(
        lambda x: f"dup_group_{dup_map[x]}" if x in dup_map else f"single_{x}"
    )

    df["split"] = "unassigned"
    test_mask = df["train"] == False
    df.loc[test_mask, "split"] = "test"

    train_pool_mask = df["train"] == True
    train_pool = df.loc[train_pool_mask].copy()

    # Collapse each training group into its class, member IDs, and image count.
    group_records = []
    for gid, rows in train_pool.groupby("group_id"):
        group_records.append(
            {
                "group_id": gid,
                "surface_type": rows["surface_type"].iloc[0],
                "image_ids": rows["mapillary_image_id"].tolist(),
                "size": len(rows),
            }
        )
    group_df = pd.DataFrame(group_records)

    val_ids = []
    train_ids = []

    # Shuffle groups within each class, then fill a validation target measured in images.
    for _, cls_groups in group_df.groupby("surface_type"):
        shuffled = cls_groups.sample(frac=1.0, random_state=random_state)
        total_images = cls_groups["size"].sum()
        target_val = int(round(total_images * (1.0 - train_val_ratio)))

        # Take whole groups that fit; allow the first to overshoot a positive target.
        curr_val = 0
        for _, row in shuffled.iterrows():
            if (curr_val + row["size"] <= target_val) or (
                curr_val == 0 and target_val > 0
            ):
                val_ids.extend(row["image_ids"])
                curr_val += row["size"]
            else:
                train_ids.extend(row["image_ids"])

    df.loc[df["mapillary_image_id"].isin(train_ids), "split"] = "train"
    df.loc[df["mapillary_image_id"].isin(val_ids), "split"] = "val"

    return df


def verify_dataset_splits(
    split_df: pd.DataFrame,
    duplicate_groups: pd.DataFrame,
) -> dict[str, Any]:
    """Assert that assigned rows form valid splits, then return counts and class totals.

    split_df must retain the published train flag and assigned split column.
    duplicate_groups supplies audited group membership. Checks reject shared
    IDs, changes to the remaining published test set, duplicates spanning
    splits, and incomplete or repeated assignments. Failures raise AssertionError.
    """
    train_ids = set(split_df.loc[split_df["split"] == "train", "mapillary_image_id"])
    val_ids = set(split_df.loc[split_df["split"] == "val", "mapillary_image_id"])
    test_ids = set(split_df.loc[split_df["split"] == "test", "mapillary_image_id"])

    train_val_overlap = train_ids & val_ids
    train_test_overlap = train_ids & test_ids
    val_test_overlap = val_ids & test_ids
    is_disjoint = (
        len(train_val_overlap) == 0
        and len(train_test_overlap) == 0
        and len(val_test_overlap) == 0
    )

    original_test_ids = set(
        split_df.loc[split_df["train"] == False, "mapillary_image_id"]
    )
    test_preserved = test_ids == original_test_ids

    # Join retained duplicate members to their assignments and flag groups in multiple splits.
    leakage_groups = []
    present_dups = duplicate_groups.loc[
        duplicate_groups["mapillary_image_id"].isin(split_df["mapillary_image_id"])
    ]
    merged_dups = present_dups.merge(
        split_df[["mapillary_image_id", "split"]], on="mapillary_image_id"
    )
    for group_num, rows in merged_dups.groupby("group"):
        if rows["split"].nunique() > 1:
            leakage_groups.append(group_num)
    zero_leakage = len(leakage_groups) == 0

    total_records = len(split_df)
    # With disjoint sets, this equality also catches unassigned or repeated image IDs.
    coverage_ok = len(train_ids) + len(val_ids) + len(test_ids) == total_records

    assert is_disjoint, (
        f"Partitions overlap! Overlaps: TV={len(train_val_overlap)}, TT={len(train_test_overlap)}, VT={len(val_test_overlap)}"
    )
    assert test_preserved, "Published test set was altered!"
    assert zero_leakage, f"Duplicate groups leaked across splits: {leakage_groups}"
    assert coverage_ok, "Partition counts do not sum to total records!"

    counts = split_df["split"].value_counts().to_dict()
    proportions = (
        (split_df["split"].value_counts(normalize=True) * 100).round(2).to_dict()
    )
    type_distribution = pd.crosstab(
        split_df["surface_type"], split_df["split"], margins=True
    )

    return {
        "is_disjoint": is_disjoint,
        "test_preserved": test_preserved,
        "zero_leakage": zero_leakage,
        "coverage_ok": coverage_ok,
        "split_counts": counts,
        "split_proportions_pct": proportions,
        "type_distribution": type_distribution,
    }


def save_split_assignments(
    split_df: pd.DataFrame,
    output_path: str = "data/splits.csv",
) -> str:
    """Write available ID, split, label, and path columns to CSV; return its path.

    Create the parent directory if needed and overwrite an existing file.
    Temporary audit and grouping columns are omitted.
    """
    columns = [
        "mapillary_image_id",
        "split",
        "surface_type",
        "surface_quality",
        "file_path",
    ]
    cols_to_keep = [col for col in columns if col in split_df.columns]

    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    split_df[cols_to_keep].to_csv(output_path, index=False)
    return output_path
