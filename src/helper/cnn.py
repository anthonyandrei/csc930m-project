"""Custom CNN search using the shared five-class evaluation protocol."""
from time import perf_counter
import numpy as np
import torch
from torch import nn
from src.helper.shared_inputs import SURFACE_TYPES, evaluate_predictions, rank_candidates

FILTERS = ((16, 16, 32), (32, 32, 64))
RATES = (0.01, 0.03, 0.05)


def require_cuda():
    """Return a tested CUDA device; raise RuntimeError if CUDA cannot run."""
    if not torch.cuda.is_available():
        raise RuntimeError("CNN training requires a working CUDA GPU")
    try:
        device = torch.device("cuda:0")
        (torch.ones(1, device=device) + 1).cpu()
        torch.cuda.synchronize(device)
    except Exception as error:
        raise RuntimeError("CNN training requires a compatible CUDA GPU") from error
    print(f"CNN training device: {torch.cuda.get_device_name(device)} ({device})", flush=True)
    return device


def build_cnn(depth, filters):
    """Build one to three blocks from a filter prefix for 96 by 128 RGB images."""
    layers = []
    channels = 3
    for width in filters[:depth]:
        conv = nn.Conv2d(channels, width, 3, stride=1, padding=1)
        nn.init.kaiming_normal_(conv.weight, nonlinearity="relu")
        nn.init.zeros_(conv.bias)
        layers.extend((conv, nn.ReLU(), nn.MaxPool2d(2, 2)))
        channels = width
    layers.extend((nn.Flatten(), nn.Linear(channels * (96 // 2**depth) * (128 // 2**depth), len(SURFACE_TYPES))))
    return nn.Sequential(*layers)


def predict_cnn(model, images, batch_size=64):
    """Return class IDs in image order for NHWC float32 images on the model device."""
    device = next(model.parameters()).device
    model.eval()
    predictions = []
    with torch.no_grad():
        for start in range(0, len(images), batch_size):
            batch = torch.from_numpy(np.ascontiguousarray(images[start:start + batch_size].transpose(0, 3, 1, 2))).to(device)
            predictions.append(model(batch).argmax(1).cpu().numpy())
    return np.concatenate(predictions)


def tune_cnn(train, validation):
    """Train 18 CUDA candidates on prepared splits and return the validation winner.

    The split dictionaries contain normalized NHWC images and integer labels.
    Candidate records retain validation predictions and per-epoch losses.
    Raises RuntimeError if CUDA is unavailable or its probe fails.
    """
    device = require_cuda()
    started = perf_counter()
    x_train = torch.from_numpy(np.ascontiguousarray(train['images'].transpose(0, 3, 1, 2))).to(device)
    y_train = torch.as_tensor(train['labels'], dtype=torch.long, device=device)
    x_val = torch.from_numpy(np.ascontiguousarray(validation['images'].transpose(0, 3, 1, 2))).to(device)
    y_val = torch.as_tensor(validation['labels'], dtype=torch.long, device=device)
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
                    total = 0.0
                    for start in range(0, len(order), 64):
                        indices = order[start:start + 64]
                        optimizer.zero_grad()
                        loss = criterion(model(x_train[indices]), y_train[indices])
                        loss.backward()
                        optimizer.step()
                        total += loss.item() * len(indices)
                    train_losses.append(total / len(order))
                    model.eval()
                    total = 0.0
                    with torch.no_grad():
                        for start in range(0, len(x_val), 64):
                            total += criterion(model(x_val[start:start + 64]), y_val[start:start + 64]).item() * len(y_val[start:start + 64])
                    val_losses.append(total / len(x_val))
                predictions = predict_cnn(model, validation['images'])
                scores = evaluate_predictions(validation['labels'], predictions)
                candidate = dict(depth=depth, filters=filters[:depth], learning_rate=rate,
                                 validation_macro_f1=scores['macro_f1'], validation_accuracy=scores['accuracy'],
                                 predictions=predictions, train_losses=train_losses, val_losses=val_losses)
                candidates.append(candidate)
                if rank_candidates(candidates)[0] is candidate:
                    best_model = model
                print(f"Completed {len(candidates)}/18 configurations", flush=True)
    top_five = rank_candidates(candidates)
    return dict(model=best_model, candidates=candidates, top_five=top_five,
                best_candidate=top_five[0], device=str(device), tuning_seconds=perf_counter()-started)
