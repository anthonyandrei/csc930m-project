"""Helpers for Phase 2 split balance, confusion summaries, and error-image grids."""

from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

from src.helper.shared_inputs import SURFACE_TYPES, evaluate_predictions

def split_class_balance_table(split_df):
    """Return per-split counts and within-split percentages for all surface types."""
    split_order = ("train", "val", "test")
    columns = {}
    for split_name in split_order:
        subset = split_df.loc[split_df["split"] == split_name]
        split_total = len(subset)
        if split_total == 0:
            raise ValueError(f"Split '{split_name}' has zero rows")
        counts = subset["surface_type"].value_counts().reindex(SURFACE_TYPES, fill_value=0)
        percentages = (100.0 * counts / split_total).round(2)
        columns[f"{split_name}_count"] = counts
        columns[f"{split_name}_pct"] = percentages

    table = pd.DataFrame(columns, index=pd.Index(SURFACE_TYPES, name="surface_type"))
    totals = {
        col: int(table[col].sum()) if col.endswith("_count") else round(float(table[col].sum()), 2)
        for col in table.columns
    }
    table.loc["total"] = totals
    return table

def recover_baseline_validation_predictions(model_key, namespace, artifact_dir=None):
    """Look up validation predictions from memory or saved artifacts without retraining."""
    selected_var = f"{model_key}_selected"
    if selected_var in namespace and isinstance(namespace[selected_var], dict):
        selected = namespace[selected_var]
        if "predictions" in selected:
            return {
                "available": True,
                "source": f"in-memory {selected_var}['predictions']",
                "predictions": np.asarray(selected["predictions"], dtype=np.int64),
            }

    if artifact_dir is not None:
        base = Path(artifact_dir)
        csv_path = base / f"{model_key}_val_predictions.csv"
        if csv_path.exists():
            frame = pd.read_csv(csv_path)
            if "prediction" in frame.columns:
                return {
                    "available": True,
                    "source": str(csv_path),
                    "predictions": frame["prediction"].to_numpy(dtype=np.int64),
                }

    return {
        "available": False,
        "source": None,
        "reason": (
            f"No in-memory '{selected_var}' or saved '{model_key}_val_predictions.csv' found; "
            "baseline training/tuning was not rerun."
        ),
    }

def summarize_confusion_errors(labels, predictions, top_n=5):
    """Return evaluation report and top off-diagonal confusion pairs."""
    report = evaluate_predictions(labels, predictions)
    cm = report["confusion_matrix"]
    total_errors = int((np.asarray(labels) != np.asarray(predictions)).sum())

    rows = []
    for actual in SURFACE_TYPES:
        actual_total = int(cm.loc[actual].sum())
        for predicted in SURFACE_TYPES:
            if actual == predicted:
                continue
            count = int(cm.loc[actual, predicted])
            if count == 0:
                continue
            rows.append(
                {
                    "actual": actual,
                    "predicted": predicted,
                    "errors": count,
                    "pct_of_actual_class": round(100.0 * count / actual_total, 2) if actual_total else 0.0,
                    "pct_of_all_errors": round(100.0 * count / total_errors, 2) if total_errors else 0.0,
                }
            )

    pairs_df = pd.DataFrame(rows)
    if not pairs_df.empty:
        pairs_df = pairs_df.sort_values("errors", ascending=False).head(top_n).reset_index(drop=True)
    return report, pairs_df


def select_representative_examples(split_records, labels, predictions, per_row=4):
    """Select representative correct and misclassified examples from one aligned split."""
    records = split_records.reset_index(drop=True)
    y_true = np.asarray(labels, dtype=np.int64)
    y_pred = np.asarray(predictions, dtype=np.int64)
    if len(records) != len(y_true) or len(y_true) != len(y_pred):
        raise ValueError("split_records, labels, and predictions must have the same length")

    correct_items = []
    used_indices = set()
    preferred_order = ("concrete", "sett", "unpaved", "paving_stones", "asphalt")
    for surface in preferred_order:
        if len(correct_items) >= per_row:
            break
        class_id = SURFACE_TYPES.index(surface)
        matches = np.where((y_true == class_id) & (y_pred == class_id))[0]
        if len(matches) > 0:
            idx = int(matches[0])
            used_indices.add(idx)
            correct_items.append(
                {
                    "image_id": str(records.loc[idx, "mapillary_image_id"]),
                    "file_path": str(records.loc[idx, "file_path"]),
                    "actual": SURFACE_TYPES[int(y_true[idx])],
                    "predicted": SURFACE_TYPES[int(y_pred[idx])],
                }
            )

    _, top_pairs = summarize_confusion_errors(y_true, y_pred, top_n=10)
    wrong_items = []
    for pair in top_pairs.itertuples():
        if len(wrong_items) >= per_row:
            break
        true_id = SURFACE_TYPES.index(pair.actual)
        pred_id = SURFACE_TYPES.index(pair.predicted)
        matches = np.where((y_true == true_id) & (y_pred == pred_id))[0]
        for match in matches:
            idx = int(match)
            if idx not in used_indices:
                used_indices.add(idx)
                wrong_items.append(
                    {
                        "image_id": str(records.loc[idx, "mapillary_image_id"]),
                        "file_path": str(records.loc[idx, "file_path"]),
                        "actual": pair.actual,
                        "predicted": pair.predicted,
                    }
                )
                break

    return {"correct": correct_items, "incorrect": wrong_items}


def error_examples_figure(examples, title):
    """Render a 2-row Pillow figure of representative correct and misclassified images."""
    correct_items = examples.get("correct", [])
    wrong_items = examples.get("incorrect", [])
    cols = max(len(correct_items), len(wrong_items), 1)
    cell_w, cell_h = 245, 250
    figure = Image.new("RGB", (cols * cell_w + 20, 2 * cell_h + 65), "white")
    draw = ImageDraw.Draw(figure)
    draw.text((15, 10), title, fill="black")

    rows = [
        ("Correctly Classified (Validation)", correct_items, 38),
        ("Misclassified (Validation)", wrong_items, 38 + cell_h + 15),
    ]
    for row_label, items, top_y in rows:
        draw.text((15, top_y), row_label, fill="black")
        for col_idx, item in enumerate(items):
            left_x = 15 + col_idx * cell_w
            with Image.open(item["file_path"]) as img:
                thumb = img.convert("RGB")
                thumb.thumbnail((cell_w - 20, 175))
                figure.paste(thumb, (left_x, top_y + 22))
            draw.text((left_x, top_y + 202), f"ID: {item['image_id']}", fill="black")
            draw.text(
                (left_x, top_y + 218),
                f"True: {item['actual']} | Pred: {item['predicted']}",
                fill="black",
            )
    return figure
