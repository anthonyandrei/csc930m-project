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

    cdist forms a batch-by-training distance matrix, never a three-dimensional
    pixel difference array. Sorted top-k prefixes support every smaller k.
    """

    def __init__(self, features, labels, *, device, batch_size):
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
        features = _features(features, self.training_features.shape[1])
        indices = np.empty((len(features), k), dtype=np.int64)
        try:
            for start in range(0, len(features), self.batch_size):
                stop = min(start + self.batch_size, len(features))
                queries = torch.as_tensor(features[start:stop], device=self.device)
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
        neighbors = self.labels[self.neighbor_indices(features, self.n_neighbors)]
        votes = np.stack(
            [(neighbors == label).sum(axis=1) for label in range(len(SURFACE_TYPES))],
            axis=1,
        )
        return votes.argmax(axis=1)

    def get_params(self):
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

    Inputs are the existing normalized pixels and encoded labels. The returned
    model keeps its original training set and uses the validation-selected k.
    Candidate predictions remain available for the selected validation report.
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
    for k in K_VALUES:
        votes[rows, neighbor_labels[:, k - 1]] += 1
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
