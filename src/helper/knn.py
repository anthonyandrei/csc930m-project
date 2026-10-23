"""Exact, batched CUDA pixel KNN using the shared validation protocol."""

from time import perf_counter

import numpy as np
import torch

from src.helper.shared_inputs import (
    SURFACE_TYPES,
    evaluate_predictions,
    rank_candidates,
)

K_VALUES = tuple(range(1, 21))


def _features(values, width=None):
    """Require finite float32 feature rows, optionally matching a known row width."""
    values = np.asarray(values)
    if (
        values.ndim != 2 or not all(values.shape) or values.dtype != np.float32
        or not np.isfinite(values).all()
    ):
        raise ValueError("Features must be a nonempty, finite float32 matrix")
    if width is not None and values.shape[1] != width:
        raise ValueError("Query features must have the training feature count")
    return values


class GPUKNN:
    """Keep training pixels on one GPU; copy only query batches and results.

    Each feature row is a flattened normalized image; labels are class IDs
    in SURFACE_TYPES order. cdist computes Euclidean distances from each query
    batch to all training rows. Only the nearest indices return to the CPU.
    Sorted neighbor prefixes support every smaller k without another search.
    """

    def __init__(self, features, labels, *, device, batch_size):
        """Store at least 20 labeled training rows on CUDA; start with k set to 20."""
        if type(batch_size) is not int or batch_size < 1:
            raise ValueError("batch_size must be a positive integer")
        self.device = torch.device(device)
        if self.device.type != "cuda" or not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for GPU KNN; select an available CUDA device and kernel")
        features = _features(features)
        labels = np.asarray(labels)
        if (
            labels.shape != (len(features),) or labels.dtype.kind not in "iu"
            or not np.isin(labels, range(len(SURFACE_TYPES))).all()
        ):
            raise ValueError("Training labels must be integer class IDs 0 through 4, one per row")
        if len(features) < max(K_VALUES):
            raise ValueError("The declared search needs at least 20 training images")
        self.batch_size = batch_size
        self.n_neighbors = max(K_VALUES)
        self.n_samples_fit_ = len(features)
        self.labels = labels.copy()
        try:
            with torch.inference_mode():
                self.training_features = torch.tensor(features, device=self.device)
            torch.cuda.synchronize(self.device)
        except torch.cuda.OutOfMemoryError as error:
            raise RuntimeError("GPU KNN cannot store training pixels; select a GPU with more free memory") from error

    @torch.inference_mode()
    def neighbor_indices(self, features, k):
        """Return an (image count, k) array of nearest training-row indices.

        k must be a positive integer no larger than the training set. Rows are
        sorted by distance; equal distances have no guaranteed neighbor order.
        """
        features = _features(features, self.training_features.shape[1])
        indices = np.empty((len(features), k), dtype=np.int64)
        try:
            for start in range(0, len(features), self.batch_size):
                stop = min(start + self.batch_size, len(features))
                queries = torch.as_tensor(features[start:stop], device=self.device)
                # Compare this batch with all training rows, limiting the distance matrix size.
                distances = torch.cdist(
                    queries, self.training_features, p=2,
                    compute_mode="use_mm_for_euclid_dist",
                )
                nearest = distances.topk(k, largest=False, sorted=True).indices
                indices[start:stop] = nearest.cpu().numpy()
            torch.cuda.synchronize(self.device)
        except torch.cuda.OutOfMemoryError as error:
            raise RuntimeError("GPU KNN query ran out of memory; reduce batch_size or select a freer GPU") from error
        return indices

    def predict(self, features):
        """Give each neighbor one vote; tied classes choose the lowest class ID."""
        # Convert neighbor indices to class labels, then count votes for each query row.
        neighbors = self.labels[self.neighbor_indices(features, self.n_neighbors)]
        votes = np.stack(
            [(neighbors == label).sum(axis=1) for label in range(len(SURFACE_TYPES))],
            axis=1,
        )
        return votes.argmax(axis=1)

    def get_params(self):
        """Return the current neighbor count, distance rule, and execution settings."""
        return {
            "n_neighbors": self.n_neighbors, "weights": "uniform",
            "algorithm": "brute", "metric": "euclidean",
            "device": str(self.device), "batch_size": self.batch_size,
        }


def tune_knn(
    train_features, train_labels, validation_features, validation_labels,
    *, device="cuda:0", batch_size=128,
):
    """Fit on training only; reuse one validation query for every declared k.

    Inputs are 2D float32 normalized pixel matrices and class-ID vectors.
    Return the model, all candidate scores and predictions, the top five,
    sorted neighbor labels, and search/total timings in seconds. Rank by
    validation macro-F1, then accuracy, then grid order. The model keeps its
    original training set and uses the selected k; no test data is used.
    """
    if torch.device(device).type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for GPU KNN; select an available CUDA device and kernel")
    torch.cuda.synchronize(device)
    started = perf_counter()
    model = GPUKNN(train_features, train_labels, device=device, batch_size=batch_size)
    search_started = perf_counter()
    indices = model.neighbor_indices(validation_features, max(K_VALUES))
    search_seconds = perf_counter() - search_started
    neighbor_labels = np.asarray(train_labels)[indices]
    votes = np.zeros((len(validation_features), len(SURFACE_TYPES)), dtype=int)
    rows = np.arange(len(validation_features))
    candidates = []
    # Add the next nearest label to each row: accumulated votes now cover exactly k neighbors.
    for k in K_VALUES:
        votes[rows, neighbor_labels[:, k - 1]] += 1
        # argmax returns the first maximum, so tied votes favor the lowest class ID.
        predictions = votes.argmax(axis=1)
        scores = evaluate_predictions(validation_labels, predictions)
        candidates.append(
            {
                "k": k,
                "validation_macro_f1": scores["macro_f1"],
                "validation_accuracy": scores["accuracy"],
                "predictions": predictions,
            }
        )
    top_five = rank_candidates(candidates)
    model.n_neighbors = top_five[0]["k"]
    return {
        "model": model,
        "candidates": candidates,
        "top_five": top_five,
        "neighbor_labels": neighbor_labels,
        "validation_search_seconds": search_seconds,
        "tuning_seconds": perf_counter() - started,
    }
