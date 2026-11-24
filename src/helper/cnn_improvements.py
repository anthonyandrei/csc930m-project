"""Controlled CNN improvement experiments reusing the baseline CNN architecture and trainer."""

import numpy as np
import pandas as pd
import torch

from src.helper.cnn import require_cuda, train_cnn_candidate
from src.helper.shared_inputs import SURFACE_TYPES, evaluate_predictions, rank_candidates


def compute_class_weights(train_labels):
    """Return balanced per-class weights N / (5 * count) from training labels only."""
    labels = np.asarray(train_labels, dtype=np.int64)
    counts = np.bincount(labels, minlength=len(SURFACE_TYPES))
    if np.any(counts == 0):
        raise ValueError("Every surface type must appear at least once in train_labels")
    weights = len(labels) / (len(SURFACE_TYPES) * counts.astype(np.float32))
    return weights.astype(np.float32)


def _training_tensors(split, device):
    """Copy prepared NHWC images and class IDs to the training device."""
    images = torch.from_numpy(np.ascontiguousarray(split["images"].transpose(0, 3, 1, 2)))
    labels = torch.as_tensor(np.array(split["labels"], copy=True), dtype=torch.long, device=device)
    return images.to(device), labels


def run_improvement_suite(train, validation):
    """Run single-factor and combined CNN improvement trials under controlled settings."""
    device = require_cuda()
    x_train, y_train = _training_tensors(train, device)
    x_val, y_val = _training_tensors(validation, device)
    balanced_weights = compute_class_weights(train["labels"])

    specs = [
        ("class_weights", True, False, 0.0),
        ("early_stopping", False, True, 0.0),
        ("dropout_0.5", False, False, 0.5),
        ("class_weights+early_stopping", True, True, 0.0),
        ("class_weights+dropout_0.5", True, False, 0.5),
        ("early_stopping+dropout_0.5", False, True, 0.5),
        ("all_three", True, True, 0.5),
    ]
    candidates = []
    for idx, (name, cw, es, dp) in enumerate(specs, start=1):
        run = train_cnn_candidate(
            x_train,
            y_train,
            x_val,
            y_val,
            validation["images"],
            validation["labels"],
            depth=3,
            filters=(16, 16, 32),
            rate=0.03,
            device=device,
            dropout_p=dp,
            class_weights=balanced_weights if cw else None,
            early_stopping=es,
            patience=3,
        )
        scores = run["scores"]
        candidates.append(
            {
                "name": name,
                "class_weights": cw,
                "early_stopping": es,
                "dropout_p": dp,
                "epochs_run": run["epochs_run"],
                "best_epoch": run["best_epoch"],
                "validation_macro_f1": scores["macro_f1"],
                "validation_accuracy": scores["accuracy"],
                "concrete_recall": float(scores["per_class"].loc["concrete", "recall"]),
                "concrete_tp": int(scores["confusion_matrix"].loc["concrete", "concrete"]),
                "train_losses": run["train_losses"],
                "val_losses": run["val_losses"],
                "predictions": run["predictions"],
                "model": run["model"],
            }
        )
        print(f"Completed {idx}/{len(specs)} trials", flush=True)

    top_five = rank_candidates(candidates)
    summary_cols = [
        "name",
        "epochs_run",
        "best_epoch",
        "validation_macro_f1",
        "validation_accuracy",
        "concrete_recall",
        "concrete_tp",
    ]
    return {
        "candidates": candidates,
        "top_five": top_five,
        "best": top_five[0],
        "top_five_table": pd.DataFrame(top_five)[summary_cols],
        "device": str(device),
    }


def run_weighted_followup(train, validation, baseline):
    """Compare two epoch-40 checkpoints against the exact saved weighted baseline.

    Split dictionaries contain the prepared images and labels. The baseline is
    the selected record from the earlier improvement suite. Return the table
    and the winning trial, or None if the baseline retains its place.
    """
    baseline_scores = evaluate_predictions(validation["labels"], baseline["predictions"])
    baseline_f1 = baseline_scores["macro_f1"]
    if baseline["name"] != "class_weights" or baseline["epochs_run"] != 20:
        raise ValueError("Expected the 20-epoch class-weighted validation baseline")
    device = require_cuda()
    x_train, y_train = _training_tensors(train, device)
    x_val, y_val = _training_tensors(validation, device)
    weights = compute_class_weights(train["labels"])
    trials = []
    for name, rate, changes in (
        ("Fixed 0.03", 0.03, {}),
        ("Step decay", 0.1, {11: 0.01, 21: 0.001}),
    ):
        run = train_cnn_candidate(
            x_train, y_train, x_val, y_val, validation["images"], validation["labels"],
            depth=3, filters=(16, 16, 32), rate=rate, device=device,
            class_weights=weights, epochs=40, rate_changes=changes,
        )
        scores = run["scores"]
        trials.append({
            **run,
            "name": name,
            "validation_macro_f1": scores["macro_f1"],
            "validation_accuracy": scores["accuracy"],
            "concrete_recall": scores["per_class"].loc["concrete", "recall"],
            "concrete_tp": scores["confusion_matrix"].loc["concrete", "concrete"],
        })
        print(f"Completed {name} at epoch {run['epochs_run']}", flush=True)

    qualified = [
        trial for trial in trials
        if trial["validation_macro_f1"] > baseline_f1 and trial["concrete_tp"] >= 46
    ]
    winner = max(qualified, key=lambda trial: trial["validation_macro_f1"], default=None)
    columns = ["name", "epochs_run", "validation_macro_f1", "concrete_recall", "concrete_tp"]
    reference = {
        "name": "Fixed 0.03 (baseline)",
        "epochs_run": baseline["epochs_run"],
        "validation_macro_f1": baseline_f1,
        "concrete_recall": baseline_scores["per_class"].loc["concrete", "recall"],
        "concrete_tp": baseline_scores["confusion_matrix"].loc["concrete", "concrete"],
    }
    table = pd.DataFrame([reference, *trials], columns=columns)
    table["selected"] = [winner is None, *(trial is winner for trial in trials)]
    return {"table": table, "trials": trials, "winner": winner}
