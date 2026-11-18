"""Controlled CNN improvement experiments reusing the baseline CNN architecture and trainer."""

import numpy as np
import pandas as pd
import torch

from src.helper.cnn import require_cuda, train_cnn_candidate
from src.helper.shared_inputs import SURFACE_TYPES, rank_candidates


def compute_class_weights(train_labels):
    """Return balanced per-class weights N / (5 * count) from training labels only."""
    labels = np.asarray(train_labels, dtype=np.int64)
    counts = np.bincount(labels, minlength=len(SURFACE_TYPES))
    if np.any(counts == 0):
        raise ValueError("Every surface type must appear at least once in train_labels")
    weights = len(labels) / (len(SURFACE_TYPES) * counts.astype(np.float32))
    return weights.astype(np.float32)


def run_improvement_suite(train, validation):
    """Run single-factor and combined CNN improvement trials under controlled settings."""
    device = require_cuda()
    x_train = torch.from_numpy(np.ascontiguousarray(train["images"].transpose(0, 3, 1, 2))).to(device)
    y_train = torch.as_tensor(np.array(train["labels"], copy=True), dtype=torch.long, device=device)
    x_val = torch.from_numpy(np.ascontiguousarray(validation["images"].transpose(0, 3, 1, 2))).to(device)
    y_val = torch.as_tensor(np.array(validation["labels"], copy=True), dtype=torch.long, device=device)
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
