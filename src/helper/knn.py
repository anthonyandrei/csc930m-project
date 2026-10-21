"""Small raw-pixel KNN search using the shared validation protocol."""

from time import perf_counter

import numpy as np
from sklearn.neighbors import KNeighborsClassifier

from src.helper.shared_inputs import SURFACE_TYPES, evaluate_predictions, rank_candidates

K_VALUES = tuple(range(1, 21))


def tune_knn(train_features, train_labels, validation_features, validation_labels):
    """Fit on training only; reuse one validation query for every declared k.

    Inputs are the existing normalized pixels and encoded labels. The returned
    model keeps its original training set and uses the validation-selected k.
    Candidate predictions remain available for the selected validation report.
    """
    started = perf_counter()
    model = KNeighborsClassifier(
        n_neighbors=max(K_VALUES), weights="uniform", algorithm="brute",
        metric="euclidean", n_jobs=1,
    ).fit(train_features, train_labels)
    search_started = perf_counter()
    indices = model.kneighbors(validation_features, return_distance=False)
    search_seconds = perf_counter() - search_started
    neighbor_labels = np.asarray(train_labels)[indices]
    votes = np.zeros((len(validation_features), len(SURFACE_TYPES)), dtype=int)
    rows = np.arange(len(validation_features))
    candidates = []
    for k in K_VALUES:
        votes[rows, neighbor_labels[:, k - 1]] += 1
        predictions = votes.argmax(axis=1)
        scores = evaluate_predictions(validation_labels, predictions)
        candidates.append({
            "k": k, "validation_macro_f1": scores["macro_f1"],
            "validation_accuracy": scores["accuracy"], "predictions": predictions,
        })
    top_five = rank_candidates(candidates)
    model.set_params(n_neighbors=top_five[0]["k"])
    return {
        "model": model, "candidates": candidates, "top_five": top_five,
        "neighbor_labels": neighbor_labels,
        "validation_search_seconds": search_seconds,
        "tuning_seconds": perf_counter() - started,
    }
