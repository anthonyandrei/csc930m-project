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


def build_cnn(depth, filters):
    """Build a CNN for 96 by 128 RGB images with five output scores.

    Depth selects one to three blocks from the filter-count sequence. Each
    block has a He-initialized convolution, ReLU, and max-pooling; the last
    feature maps feed one linear classifier.
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
    layers.extend((nn.Flatten(), nn.Linear(flattened_size, len(SURFACE_TYPES))))
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
    y_train = torch.as_tensor(train["labels"], dtype=torch.long, device=device)
    val_images = np.ascontiguousarray(validation["images"].transpose(0, 3, 1, 2))
    x_val = torch.from_numpy(val_images).to(device)
    del val_images
    y_val = torch.as_tensor(validation["labels"], dtype=torch.long, device=device)

    candidates = []
    best_model = None
    for depth in (1, 2, 3):
        for filters in FILTERS:
            for rate in RATES:
                torch.manual_seed(42)
                model = build_cnn(depth, filters).to(device)
                optimizer = torch.optim.SGD(model.parameters(), lr=rate)
                criterion = nn.CrossEntropyLoss()
                train_losses, val_losses = [], []

                for epoch in range(20):
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
                            loss = criterion(model(batch_x), batch_y)
                            val_loss_sum += loss.item() * len(batch_y)
                    val_losses.append(val_loss_sum / len(x_val))

                predictions = predict_cnn(model, validation["images"])
                scores = evaluate_predictions(validation["labels"], predictions)
                candidate = {
                    "depth": depth,
                    "filters": filters[:depth],
                    "learning_rate": rate,
                    "validation_macro_f1": scores["macro_f1"],
                    "validation_accuracy": scores["accuracy"],
                    "predictions": predictions,
                    "train_losses": train_losses,
                    "val_losses": val_losses,
                }
                candidates.append(candidate)
                if rank_candidates(candidates)[0] is candidate:
                    best_model = model
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
