"""
evaluate.py

Evaluation metrics for the recommendation models: RMSE plus
ranking-quality metrics (precision@k, recall@k, F1@k).
"""

from collections import defaultdict

import pandas as pd
from surprise import accuracy


def precision_recall_at_k(model, testset, k: int = 10, threshold: float = 3.5) -> dict:
    """
    Compute RMSE, precision@k, recall@k, and F1@k for a fitted model.

    Note: the notebook version of this function relied on a module-level
    `testset` variable rather than a parameter. It's passed explicitly
    here so the function has no hidden dependencies and can be reused
    across different models and test sets.

    Parameters
    ----------
    model : a fitted surprise algorithm
    testset : the surprise testset to evaluate on
    k : int, default 10
        Cutoff for the top-k recommendations considered.
    threshold : float, default 3.5
        Rating at or above which an item is considered "relevant".

    Returns
    -------
    dict with keys: 'rmse', 'precision', 'recall', 'f1'
        (The notebook version printed these instead of returning them;
        returning a dict makes the results usable elsewhere, e.g. to
        build a comparison table — see evaluate_all_models below.)
    """
    predictions = model.test(testset)

    user_est_true = defaultdict(list)
    for uid, _, true_r, est, _ in predictions:
        user_est_true[uid].append((est, true_r))

    precisions, recalls = {}, {}
    for uid, user_ratings in user_est_true.items():
        user_ratings.sort(key=lambda x: x[0], reverse=True)

        n_rel = sum((true_r >= threshold) for (_, true_r) in user_ratings)
        n_rec_k = sum((est >= threshold) for (est, _) in user_ratings[:k])
        n_rel_and_rec_k = sum(
            ((true_r >= threshold) and (est >= threshold)) for (est, true_r) in user_ratings[:k]
        )

        # Precision/recall are undefined (0/0) when nothing was recommended
        # or nothing was relevant; both are set to 0 in that case, matching
        # the notebook's convention.
        precisions[uid] = n_rel_and_rec_k / n_rec_k if n_rec_k != 0 else 0
        recalls[uid] = n_rel_and_rec_k / n_rel if n_rel != 0 else 0

    precision = round(sum(precisions.values()) / len(precisions), 3)
    recall = round(sum(recalls.values()) / len(recalls), 3)
    # Guard against 0/0 when precision and recall are both 0 — the notebook's
    # F1 formula would raise ZeroDivisionError in that edge case.
    f1 = round((2 * precision * recall) / (precision + recall), 3) if (precision + recall) else 0.0
    rmse = accuracy.rmse(predictions, verbose=False)

    return {"rmse": rmse, "precision": precision, "recall": recall, "f1": f1}


def evaluate_all_models(models: dict, testset, k: int = 10, threshold: float = 3.5) -> pd.DataFrame:
    """
    Run precision_recall_at_k across several fitted models and return one
    comparison table — a programmatic version of this project's final
    results table.

    Parameters
    ----------
    models : dict
        Mapping of model name -> fitted surprise algorithm, e.g.
        {"User-User CF (tuned)": sim_user_user_optimized,
         "SVD (tuned)": svd_algo_optimized}
    testset : the surprise testset to evaluate all models on
    k, threshold : passed through to precision_recall_at_k

    Returns
    -------
    pd.DataFrame
        One row per model, columns: rmse, precision, recall, f1.
    """
    rows = {name: precision_recall_at_k(model, testset, k=k, threshold=threshold) for name, model in models.items()}
    return pd.DataFrame(rows).T
