import os
from typing import Tuple, Dict, Any
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold


def filter_usable_dataset(
    inventory_df: pd.DataFrame,
    duplicate_groups: pd.DataFrame,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Filter out unreadable images and conflicting duplicate pairs."""
    # Step 1: Find image IDs in duplicate groups with conflicting labels
    conflicting_ids = []
    for _, rows in duplicate_groups.groupby("group"):
        labels = rows[["surface_type", "surface_quality"]].drop_duplicates()
        if len(labels) > 1:
            conflicting_ids.extend(rows["mapillary_image_id"].tolist())

    # Step 2: Find unreadable image IDs (decode failures)
    unreadable_ids = inventory_df.loc[
        ~inventory_df["is_readable"], "mapillary_image_id"
    ].tolist()

    # Step 3: Combine all excluded IDs into a unique sorted list
    excluded_ids = sorted(list(set(conflicting_ids + unreadable_ids)))

    # Step 4: Keep only usable images
    usable_df = inventory_df.loc[
        ~inventory_df["mapillary_image_id"].isin(excluded_ids)
    ].copy()

    # Step 5: Build audit log
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
    """Generate train/val/test splits preventing duplicate leakage and preserving test."""
    df = usable_df.copy()

    # Step 1: Create a grouping column so duplicate groups stay together
    dup_map = dict(
        zip(duplicate_groups["mapillary_image_id"], duplicate_groups["group"])
    )
    df["group_id"] = df["mapillary_image_id"].map(
        lambda x: f"dup_group_{dup_map[x]}" if x in dup_map else f"single_{x}"
    )

    # Step 2: Assign published test records directly to 'test'
    df["split"] = "unassigned"
    test_mask = df["train"] == False
    df.loc[test_mask, "split"] = "test"

    # Step 3: Extract the training pool candidates
    train_pool_mask = df["train"] == True
    train_pool = df.loc[train_pool_mask].copy()

    # Step 4: Use StratifiedGroupKFold to enforce stratification AND grouping
    # n_splits=5 provides an 80% train / 20% val partition
    n_splits = round(1.0 / (1.0 - train_val_ratio))
    sgkf = StratifiedGroupKFold(
        n_splits=n_splits, shuffle=True, random_state=random_state
    )

    train_idx_rel, val_idx_rel = next(
        sgkf.split(
            X=train_pool,
            y=train_pool["surface_type"],
            groups=train_pool["group_id"],
        )
    )

    train_ids = train_pool.iloc[train_idx_rel]["mapillary_image_id"].values
    val_ids = train_pool.iloc[val_idx_rel]["mapillary_image_id"].values

    # Step 5: Assign split labels
    df.loc[df["mapillary_image_id"].isin(train_ids), "split"] = "train"
    df.loc[df["mapillary_image_id"].isin(val_ids), "split"] = "val"

    return df


def verify_dataset_splits(
    split_df: pd.DataFrame,
    duplicate_groups: pd.DataFrame,
) -> Dict[str, Any]:
    """Verify disjointness, test preservation, zero duplicate leakage, and coverage."""
    train_ids = set(split_df.loc[split_df["split"] == "train", "mapillary_image_id"])
    val_ids = set(split_df.loc[split_df["split"] == "val", "mapillary_image_id"])
    test_ids = set(split_df.loc[split_df["split"] == "test", "mapillary_image_id"])

    # Check 1: Mutually disjoint sets
    train_val_overlap = train_ids & val_ids
    train_test_overlap = train_ids & test_ids
    val_test_overlap = val_ids & test_ids
    is_disjoint = (
        len(train_val_overlap) == 0
        and len(train_test_overlap) == 0
        and len(val_test_overlap) == 0
    )

    # Check 2: Test preservation
    original_test_ids = set(
        split_df.loc[split_df["train"] == False, "mapillary_image_id"]
    )
    test_preserved = test_ids == original_test_ids

    # Check 3: Zero duplicate leakage across splits
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

    # Check 4: Complete coverage
    total_records = len(split_df)
    coverage_ok = len(train_ids) + len(val_ids) + len(test_ids) == total_records

    # Mathematical assertions
    assert is_disjoint, f"Partitions overlap! Overlaps: TV={len(train_val_overlap)}, TT={len(train_test_overlap)}, VT={len(val_test_overlap)}"
    assert test_preserved, "Published test set was altered!"
    assert zero_leakage, f"Duplicate groups leaked across splits: {leakage_groups}"
    assert coverage_ok, "Partition counts do not sum to total records!"

    counts = split_df["split"].value_counts().to_dict()
    proportions = (
        split_df["split"].value_counts(normalize=True) * 100
    ).round(2).to_dict()
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
    """Save the final reusable split table keyed by mapillary_image_id."""
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