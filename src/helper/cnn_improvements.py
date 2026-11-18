"""Controlled CNN improvement experiments: class weighting, early stopping, and dropout."""

import numpy as np
import pandas as pd
import torch
from torch import nn

from src.helper.cnn import predict_cnn, require_cuda
from src.helper.shared_inputs import SURFACE_TYPES, evaluate_predictions, rank_candidates


def compute_class_weights(train_labels):
    """Return balanced per-class weights N / (5 * count) from training labels only."""
    labels = np.asarray(train_labels, dtype=np.int64)
    counts = np.bincount(labels, minlength=len(SURFACE_TYPES))
    if np.any(counts == 0):
        raise ValueError("Every surface type must appear at least once in train_labels")
    weights = len(labels) / (len(SURFACE_TYPES) * counts.astype(np.float32))
    return weights.astype(np.float32)

def build_improved_cnn(dropout_p=0.0):
    """Build the controlled 3-block (16, 16, 32) CNN with optional classifier dropout."""
    layers = []
    channels = 3
    for width in (16, 16, 32):
        convolution = nn.Conv2d(channels, width, 3, stride=1, padding=1)
        nn.init.kaiming_normal_(convolution.weight, nonlinearity="relu")
        nn.init.zeros_(convolution.bias)
        layers.extend((convolution, nn.ReLU(), nn.MaxPool2d(2, 2)))
        channels = width

    flattened_size = channels * (96 // 8) * (128 // 8)
    layers.append(nn.Flatten())
    if dropout_p > 0.0:
        layers.append(nn.Dropout(p=dropout_p))
    layers.append(nn.Linear(flattened_size, len(SURFACE_TYPES)))
    return nn.Sequential(*layers)

def train_improved_cnn_candidate(
    train,
    validation,
    *,
    name,
    use_class_weights=False,
    use_early_stopping=False,
    dropout_p=0.0,
    patience=3,
    device=None,
):
    """Train one controlled CNN improvement trial and return validation results."""
    if device is None:
        device = require_cuda()
    torch.manual_seed(42)

    x_train = torch.from_numpy(np.ascontiguousarray(train["images"].transpose(0, 3, 1, 2))).to(device)
    y_train = torch.as_tensor(np.array(train["labels"], copy=True), dtype=torch.long, device=device)
    x_val = torch.from_numpy(np.ascontiguousarray(validation["images"].transpose(0, 3, 1, 2))).to(device)
    y_val = torch.as_tensor(np.array(validation["labels"], copy=True), dtype=torch.long, device=device)

    model = build_improved_cnn(dropout_p=dropout_p).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.03)
    weight_tensor = (
        torch.tensor(compute_class_weights(train["labels"]), dtype=torch.float32, device=device)
        if use_class_weights
        else None
    )
    criterion = nn.CrossEntropyLoss(weight=weight_tensor)

    train_losses, val_losses = [], []
    best_val_loss = float("inf")
    best_epoch = 20
    best_state = None
    no_improve = 0

    for epoch in range(1, 21):
        model.train()
        order = torch.randperm(len(x_train), device=device)
        train_loss_sum = 0.0
        for start in range(0, len(order), 64):
            indices = order[start : start + 64]
            optimizer.zero_grad()
            loss = criterion(model(x_train[indices]), y_train[indices])
            loss.backward()
            optimizer.step()
            train_loss_sum += loss.item() * len(indices)
        train_losses.append(train_loss_sum / len(order))

        model.eval()
        val_loss_sum = 0.0
        with torch.no_grad():
            for start in range(0, len(x_val), 64):
                batch_x = x_val[start : start + 64]
                batch_y = y_val[start : start + 64]
                val_loss_sum += criterion(model(batch_x), batch_y).item() * len(batch_y)
        val_loss = val_loss_sum / len(x_val)
        val_losses.append(val_loss)

        if use_early_stopping:
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= patience:
                    break

    if use_early_stopping and best_state is not None:
        model.load_state_dict(best_state)

    predictions = predict_cnn(model, validation["images"])
    scores = evaluate_predictions(validation["labels"], predictions)
    concrete_recall = float(scores["per_class"].loc["concrete", "recall"])
    concrete_tp = int(scores["confusion_matrix"].loc["concrete", "concrete"])
    return {
        "name": name,
        "class_weights": use_class_weights,
        "early_stopping": use_early_stopping,
        "dropout_p": dropout_p,
        "epochs_run": len(val_losses),
        "best_epoch": best_epoch if use_early_stopping else len(val_losses),
        "validation_macro_f1": scores["macro_f1"],
        "validation_accuracy": scores["accuracy"],
        "concrete_recall": concrete_recall,
        "concrete_tp": concrete_tp,
        "train_losses": train_losses,
        "val_losses": val_losses,
        "predictions": predictions,
        "model": model,
    }

def run_improvement_suite(train, validation):
    """Run single-factor and combined CNN improvement trials under controlled settings."""
    device = require_cuda()
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
    for name, cw, es, dp in specs:
        result = train_improved_cnn_candidate(
            train,
            validation,
            name=name,
            use_class_weights=cw,
            use_early_stopping=es,
            dropout_p=dp,
            device=device,
        )
        candidates.append(result)
        print(
            f"Completed {name}: val_macro_f1={result['validation_macro_f1']:.4f}, "
            f"concrete_recall={result['concrete_tp']}/210",
            flush=True,
        )

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
    single_factor_df = pd.DataFrame(candidates[:3])[summary_cols]
    top_five_df = pd.DataFrame(top_five)[summary_cols]
    return {
        "candidates": candidates,
        "top_five": top_five,
        "best": top_five[0],
        "single_factor_table": single_factor_df,
        "top_five_table": top_five_df,
        "device": str(device),
    }