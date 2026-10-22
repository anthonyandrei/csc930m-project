"""Feedforward Neural Network baseline using standard SGD and PyTorch with automatic GPU acceleration."""

from time import perf_counter

import numpy as np
from PIL import Image, ImageDraw
import torch
import torch.nn as nn

from src.helper.shared_inputs import (
    IMAGE_SHAPE,
    SURFACE_TYPES,
    evaluate_predictions,
    rank_candidates,
)

GRID_LAYERS = (1, 2, 3, 4, 5)
GRID_NEURONS = (32, 64, 128)
GRID_LEARNING_RATES = (0.01, 0.03, 0.05)
INPUT_DIM = int(np.prod(IMAGE_SHAPE))
NUM_CLASSES = len(SURFACE_TYPES)


def get_device():
    """Select the best available device, picking the GPU with the most free VRAM if available."""
    if torch.cuda.is_available():
        try:
            free_mem = []
            for i in range(torch.cuda.device_count()):
                free, _ = torch.cuda.mem_get_info(i)
                free_mem.append((free, i))
            free_mem.sort(reverse=True)
            return torch.device(f"cuda:{free_mem[0][1]}")
        except Exception:
            return torch.device("cuda:0")
    return torch.device("cpu")


def build_mlp(input_dim, num_classes, num_layers, hidden_neurons):
    """Build a sequential feedforward network with ReLU hidden activations."""
    if num_layers < 1:
        raise ValueError("Network must have at least 1 hidden layer")
    layers = []
    current_dim = input_dim
    for _ in range(num_layers):
        layers.append(nn.Linear(current_dim, hidden_neurons))
        layers.append(nn.ReLU())
        current_dim = hidden_neurons
    layers.append(nn.Linear(current_dim, num_classes))
    return nn.Sequential(*layers)


def train_mlp_candidate(
    train_features,
    train_labels,
    val_features,
    val_labels,
    num_layers,
    hidden_neurons,
    learning_rate,
    epochs=20,
    batch_size=64,
    seed=42,
    device=None,
):
    """Train one MLP configuration using standard mini-batch SGD and record loss history."""
    if device is None:
        device = get_device()

    torch.manual_seed(seed)
    np.random.seed(seed)

    x_train = torch.as_tensor(train_features, dtype=torch.float32)
    y_train = torch.as_tensor(train_labels, dtype=torch.int64)
    x_val = torch.as_tensor(val_features, dtype=torch.float32).to(device)
    y_val = torch.as_tensor(val_labels, dtype=torch.int64).to(device)

    model = build_mlp(
        input_dim=train_features.shape[1],
        num_classes=NUM_CLASSES,
        num_layers=num_layers,
        hidden_neurons=hidden_neurons,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    num_samples = len(x_train)
    train_losses = []
    val_losses = []

    for _ in range(epochs):
        model.train()
        permutation = torch.randperm(num_samples)
        epoch_loss_sum = 0.0

        for start_idx in range(0, num_samples, batch_size):
            batch_indices = permutation[start_idx : start_idx + batch_size]
            batch_x = x_train[batch_indices].to(device)
            batch_y = y_train[batch_indices].to(device)

            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()

            epoch_loss_sum += loss.item() * len(batch_indices)

        train_losses.append(epoch_loss_sum / num_samples)

        model.eval()
        with torch.no_grad():
            val_logits = model(x_val)
            val_loss = criterion(val_logits, y_val).item()
            val_losses.append(val_loss)

    model.eval()
    with torch.no_grad():
        val_predictions = model(x_val).argmax(dim=1).cpu().numpy()

    scores = evaluate_predictions(val_labels, val_predictions)

    return {
        "model": model,
        "validation_macro_f1": scores["macro_f1"],
        "validation_accuracy": scores["accuracy"],
        "predictions": val_predictions,
        "train_losses": train_losses,
        "val_losses": val_losses,
        "device": device,
    }


def tune_neural_network(
    train_features,
    train_labels,
    val_features,
    val_labels,
    epochs=20,
    batch_size=64,
    seed=42,
    device=None,
):
    """Execute the full 45-candidate Cartesian grid search and extract top five."""
    if device is None:
        device = get_device()
    print(f"Neural Network training device: {device}")

    started = perf_counter()
    candidates = []
    trained_models = []

    for num_layers in GRID_LAYERS:
        for hidden_neurons in GRID_NEURONS:
            for lr in GRID_LEARNING_RATES:
                result = train_mlp_candidate(
                    train_features=train_features,
                    train_labels=train_labels,
                    val_features=val_features,
                    val_labels=val_labels,
                    num_layers=num_layers,
                    hidden_neurons=hidden_neurons,
                    learning_rate=lr,
                    epochs=epochs,
                    batch_size=batch_size,
                    seed=seed,
                    device=device,
                )
                candidate_record = {
                    "num_layers": num_layers,
                    "hidden_neurons": hidden_neurons,
                    "learning_rate": lr,
                    "validation_macro_f1": result["validation_macro_f1"],
                    "validation_accuracy": result["validation_accuracy"],
                    "predictions": result["predictions"],
                    "train_losses": result["train_losses"],
                    "val_losses": result["val_losses"],
                    "model_idx": len(trained_models),
                }
                candidates.append(candidate_record)
                trained_models.append(result["model"])

    top_five = rank_candidates(candidates)
    best_candidate = top_five[0]
    best_model = trained_models[best_candidate["model_idx"]]

    return {
        "model": best_model,
        "candidates": candidates,
        "top_five": top_five,
        "best_candidate": best_candidate,
        "tuning_seconds": perf_counter() - started,
        "device": str(device),
    }


def predict_neural_network(model, features, device=None):
    """Predict integer class labels for a 2D float32 pixel matrix."""
    if device is None:
        try:
            device = next(model.parameters()).device
        except StopIteration:
            device = get_device()

    model.eval()
    with torch.no_grad():
        x_tensor = torch.as_tensor(features, dtype=torch.float32).to(device)
        logits = model(x_tensor)
        return logits.argmax(dim=1).cpu().numpy()


def plot_learning_curves(train_losses, val_losses, width=640, height=360):
    """Render training and validation loss curves using Pillow (zero external plotting dependency)."""
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)

    pad_left, pad_right, pad_top, pad_bottom = 60, 30, 40, 45
    plot_w = width - pad_left - pad_right
    plot_h = height - pad_top - pad_bottom

    all_losses = train_losses + val_losses
    min_loss = 0.0
    max_loss = max(all_losses) * 1.1 if max(all_losses) > 0 else 1.0
    epochs = len(train_losses)
    denom = max(1, epochs - 1)

    for i in range(5):
        y_val = min_loss + (max_loss - min_loss) * i / 4.0
        y_pos = int(
            pad_top + plot_h - (y_val - min_loss) / (max_loss - min_loss) * plot_h
        )
        draw.line(
            [(pad_left, y_pos), (pad_left + plot_w, y_pos)],
            fill="#e0e0e0",
            width=1,
        )
        draw.text((10, y_pos - 6), f"{y_val:.2f}", fill="#555555")

    for ep in [1, 5, 10, 15, 20]:
        if ep <= epochs:
            x_pos = int(pad_left + (ep - 1) / denom * plot_w)
            draw.line(
                [(x_pos, pad_top), (x_pos, pad_top + plot_h)],
                fill="#f0f0f0",
                width=1,
            )
            draw.text((x_pos - 6, pad_top + plot_h + 8), str(ep), fill="#555555")

    draw.line(
        [(pad_left, pad_top), (pad_left, pad_top + plot_h)], fill="black", width=2
    )
    draw.line(
        [(pad_left, pad_top + plot_h), (pad_left + plot_w, pad_top + plot_h)],
        fill="black",
        width=2,
    )

    def to_coords(losses):
        coords = []
        for i, val in enumerate(losses):
            x = pad_left + int(i / denom * plot_w)
            y = pad_top + plot_h - int(
                (val - min_loss) / (max_loss - min_loss) * plot_h
            )
            coords.append((x, y))
        return coords

    t_coords = to_coords(train_losses)
    v_coords = to_coords(val_losses)

    for i in range(len(t_coords) - 1):
        draw.line([t_coords[i], t_coords[i + 1]], fill="#1f77b4", width=3)
    for i in range(len(v_coords) - 1):
        draw.line([v_coords[i], v_coords[i + 1]], fill="#ff7f0e", width=3)

    draw.text(
        (pad_left + plot_w // 4, 12),
        "Training vs. Validation Loss (Cross-Entropy)",
        fill="black",
    )
    draw.text((pad_left + plot_w // 2 - 15, height - 20), "Epoch", fill="black")

    draw.line([(width - 200, 15), (width - 170, 15)], fill="#1f77b4", width=3)
    draw.text((width - 165, 10), "Train Loss", fill="black")
    draw.line([(width - 100, 15), (width - 70, 15)], fill="#ff7f0e", width=3)
    draw.text((width - 65, 10), "Val Loss", fill="black")

    return image
