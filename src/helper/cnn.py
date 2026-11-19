"""Custom CNN search using the shared five-class evaluation protocol."""

from time import perf_counter

import numpy as np
import torch
from torch import nn

from src.helper.shared_inputs import SURFACE_TYPES, evaluate_predictions, rank_candidates

FILTERS = ((16, 16, 32), (32, 32, 64))
RATES = (0.01, 0.03, 0.05)


def require_cuda():
    """Return the working CUDA device with the most free memory.

    Probe each device with a GPU operation. Raise RuntimeError if CUDA is
    unavailable or no device can run the operation. Print the selected device.
    """
    if not torch.cuda.is_available():
        raise RuntimeError("CNN training requires a working CUDA GPU")

    working_devices = []
    for index in range(torch.cuda.device_count()):
        try:
            device = torch.device(f"cuda:{index}")
            (torch.ones(1, device=device) + 1).cpu()
            torch.cuda.synchronize(device)
            free_bytes, _ = torch.cuda.mem_get_info(device)
            working_devices.append((free_bytes, index))
        except Exception:
            continue

    if not working_devices:
        raise RuntimeError("CNN training requires a compatible CUDA GPU")

    device = torch.device(f"cuda:{max(working_devices)[1]}")
    print(
        f"CNN training device: {torch.cuda.get_device_name(device)} ({device})",
        flush=True,
    )
    return device


def build_cnn(depth, filters, dropout_p=0.0):
    """Build a CNN for 96 by 128 RGB images with five output scores.

    Depth selects one to three blocks from the filter-count sequence. Each
    block has a He-initialized convolution, ReLU, and max-pooling; the last
    feature maps feed optional dropout and one linear classifier.
    """
    layers = []
    channels = 3
    for width in filters[:depth]:
        convolution = nn.Conv2d(channels, width, 3, stride=1, padding=1)
        nn.init.kaiming_normal_(convolution.weight, nonlinearity="relu")
        nn.init.zeros_(convolution.bias)
        layers.extend((convolution, nn.ReLU(), nn.MaxPool2d(2, 2)))
        channels = width

    pooled_height = 96 // 2**depth
    pooled_width = 128 // 2**depth
    flattened_size = channels * pooled_height * pooled_width
    layers.append(nn.Flatten())
    if dropout_p > 0.0:
        layers.append(nn.Dropout(p=dropout_p))
    layers.append(nn.Linear(flattened_size, len(SURFACE_TYPES)))
    return nn.Sequential(*layers)


def predict_cnn(model, images, batch_size=64):
    """Return class IDs in image order for nonempty NHWC float32 images.

    Evaluate batches on the model's device without gradients. This puts the
    model in evaluation mode and returns a one-dimensional NumPy array.
    """
    device = next(model.parameters()).device
    model.eval()
    predictions = []
    with torch.no_grad():
        for start in range(0, len(images), batch_size):
            images_batch = images[start : start + batch_size]
            channels_first = np.ascontiguousarray(images_batch.transpose(0, 3, 1, 2))
            batch = torch.from_numpy(channels_first).to(device)
            predictions.append(model(batch).argmax(1).cpu().numpy())
    return np.concatenate(predictions)


def train_cnn_candidate(
    x_train,
    y_train,
    x_val,
    y_val,
    val_images,
    val_labels,
    *,
    depth,
    filters,
    rate,
    device,
    dropout_p=0.0,
    class_weights=None,
    early_stopping=False,
    patience=3,
    epochs=20,
    batch_size=64,
    seed=42,
):
    """Train one CNN candidate with plain SGD and exact weighted-mean loss tracking."""
    torch.manual_seed(seed)
    model = build_cnn(depth, filters, dropout_p=dropout_p).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=rate)
    weight_tensor = (
        torch.tensor(class_weights, dtype=torch.float32, device=device)
        if class_weights is not None
        else None
    )
    train_criterion = nn.CrossEntropyLoss(weight=weight_tensor)
    eval_criterion = nn.CrossEntropyLoss(weight=weight_tensor, reduction="sum")

    train_losses, val_losses = [], []
    best_val_loss = float("inf")
    best_epoch = epochs
    best_state = None
    no_improve = 0

    for epoch in range(1, epochs + 1):
        model.train()
        order = torch.randperm(len(x_train), device=device)
        train_loss_sum = 0.0
        train_weight_sum = 0.0
        for start in range(0, len(order), batch_size):
            indices = order[start : start + batch_size]
            batch_y = y_train[indices]
            optimizer.zero_grad()
            logits = model(x_train[indices])
            loss = train_criterion(logits, batch_y)
            loss.backward()
            optimizer.step()
            batch_w = weight_tensor[batch_y].sum().item() if weight_tensor is not None else len(batch_y)
            train_loss_sum += loss.item() * batch_w
            train_weight_sum += batch_w
        train_losses.append(train_loss_sum / train_weight_sum)

        model.eval()
        val_loss_sum = 0.0
        val_weight_sum = 0.0
        with torch.no_grad():
            for start in range(0, len(x_val), batch_size):
                batch_x = x_val[start : start + batch_size]
                batch_y = y_val[start : start + batch_size]
                val_loss_sum += eval_criterion(model(batch_x), batch_y).item()
                val_weight_sum += (
                    weight_tensor[batch_y].sum().item() if weight_tensor is not None else len(batch_y)
                )
        val_loss = val_loss_sum / val_weight_sum
        val_losses.append(val_loss)

        if early_stopping:
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= patience:
                    break

    if early_stopping and best_state is not None:
        model.load_state_dict(best_state)

    predictions = predict_cnn(model, val_images)
    scores = evaluate_predictions(val_labels, predictions)
    return {
        "model": model,
        "predictions": predictions,
        "scores": scores,
        "train_losses": train_losses,
        "val_losses": val_losses,
        "epochs_run": len(val_losses),
        "best_epoch": best_epoch if early_stopping else len(val_losses),
    }


def tune_cnn(train, validation):
    """Train 18 CUDA candidates and return the validation-selected final model.

    Each split dictionary contains nonempty normalized NHWC images and aligned
    integer labels. Each candidate trains for 20 epochs, and records retain
    validation predictions and both loss histories. Rank by validation macro-F1,
    then accuracy, then grid order; test data is not used.

    Return the model, all records, top five, selected record, device, and elapsed
    seconds. Print the device and progress. Raise RuntimeError if CUDA cannot
    run. The returned model remains on the selected device.
    """
    device = require_cuda()
    started = perf_counter()
    train_images = np.ascontiguousarray(train["images"].transpose(0, 3, 1, 2))
    x_train = torch.from_numpy(train_images).to(device)
    del train_images
    y_train = torch.as_tensor(np.array(train["labels"], copy=True), dtype=torch.long, device=device)
    val_images = np.ascontiguousarray(validation["images"].transpose(0, 3, 1, 2))
    x_val = torch.from_numpy(val_images).to(device)
    del val_images
    y_val = torch.as_tensor(np.array(validation["labels"], copy=True), dtype=torch.long, device=device)

    candidates = []
    best_model = None
    for depth in (1, 2, 3):
        for filters in FILTERS:
            for rate in RATES:
                run = train_cnn_candidate(
                    x_train,
                    y_train,
                    x_val,
                    y_val,
                    validation["images"],
                    validation["labels"],
                    depth=depth,
                    filters=filters,
                    rate=rate,
                    device=device,
                )
                candidate = {
                    "depth": depth,
                    "filters": filters[:depth],
                    "learning_rate": rate,
                    "validation_macro_f1": run["scores"]["macro_f1"],
                    "validation_accuracy": run["scores"]["accuracy"],
                    "predictions": run["predictions"],
                    "train_losses": run["train_losses"],
                    "val_losses": run["val_losses"],
                }
                candidates.append(candidate)
                if rank_candidates(candidates)[0] is candidate:
                    best_model = run["model"]
                print(f"Completed {len(candidates)}/18 configurations", flush=True)

    top_five = rank_candidates(candidates)
    return {
        "model": best_model,
        "candidates": candidates,
        "top_five": top_five,
        "best_candidate": top_five[0],
        "device": str(device),
        "tuning_seconds": perf_counter() - started,
    }
