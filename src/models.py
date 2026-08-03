"""
models.py

The three recommendation approaches used in this project:
  1. Rank-based (popularity) recommender
  2. KNN-based collaborative filtering (user-user and item-item)
  3. SVD matrix factorization

The notebook built baseline and tuned versions of each KNN/SVD model as
separate, near-identical cells. Here they're consolidated into single
parameterized functions instead, so the same function serves both the
baseline and tuned case.
"""

import pandas as pd
from surprise import SVD, KNNBasic
from surprise.model_selection import GridSearchCV

# Hyperparameter grids actually searched during tuning in the notebook,
# kept here so the tuning process is reproducible without re-reading it.
USER_USER_PARAM_GRID = {
    "k": [20, 30, 40],
    "min_k": [3, 6, 9],
    "sim_options": {"name": ["msd", "cosine", "pearson"], "user_based": [True]},
}

ITEM_ITEM_PARAM_GRID = {
    "k": [10, 20, 30],
    "min_k": [3, 6, 9],
    "sim_options": {"name": ["msd", "cosine"], "user_based": [False]},
}

SVD_PARAM_GRID = {
    "n_epochs": [10, 20, 30],
    "lr_all": [0.001, 0.005, 0.01],
    "reg_all": [0.2, 0.4, 0.6],
}


# ---------------------------------------------------------------------
# Model 1: Rank-based (popularity) recommender
# ---------------------------------------------------------------------

def compute_popularity_scores(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute average rating and rating count per product.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain 'prod_id' and 'rating' columns.

    Returns
    -------
    pd.DataFrame
        Indexed by prod_id, columns 'Average Rating' and 'Rating Count',
        sorted by Average Rating descending.
    """
    average_rating = df.groupby("prod_id")["rating"].mean()
    rating_count = df.groupby("prod_id")["rating"].count()
    final_rating = pd.DataFrame({"Average Rating": average_rating, "Rating Count": rating_count})
    return final_rating.sort_values(by="Average Rating", ascending=False)


def get_top_n_products(data: pd.DataFrame, n: int = 5, min_interactions: int = 50) -> pd.DataFrame:
    """
    Return the top-n products by average rating, restricted to products
    with at least `min_interactions` ratings.

    Parameters
    ----------
    data : pd.DataFrame
        Output of compute_popularity_scores.
    n : int, default 5
    min_interactions : int, default 50

    Returns
    -------
    pd.DataFrame
    """
    recommendations = data[data["Rating Count"] >= min_interactions]
    recommendations = recommendations.sort_values(by=["Average Rating", "Rating Count"], ascending=False)
    return recommendations.head(n)


# ---------------------------------------------------------------------
# Model 2: Collaborative filtering (KNN, user-user and item-item)
# ---------------------------------------------------------------------

def train_knn_model(
    trainset,
    k: int = 40,
    min_k: int = 6,
    sim_name: str = "cosine",
    user_based: bool = True,
) -> KNNBasic:
    """
    Build and fit a KNNBasic collaborative filtering model.

    Set user_based=True for user-user similarity, or False for
    item-item similarity. Defaults reproduce this project's tuned
    user-user configuration — pass ITEM_ITEM's best params (k=30,
    min_k=6, sim_name="msd", user_based=False) for that variant instead.

    The notebook's baselines used KNNBasic(sim_options=..., verbose=False)
    with k/min_k left at the surprise library's own defaults — call
    KNNBasic(sim_options={"name": "cosine", "user_based": True},
    verbose=False).fit(trainset) directly to reproduce that exact
    comparison point.

    Returns
    -------
    surprise.KNNBasic
        Fitted model.
    """
    sim_options = {"name": sim_name, "user_based": user_based}
    model = KNNBasic(k=k, min_k=min_k, sim_options=sim_options, verbose=False)
    model.fit(trainset)
    return model


# ---------------------------------------------------------------------
# Model 3: Model-based CF (SVD matrix factorization)
# ---------------------------------------------------------------------

def train_svd_model(
    trainset,
    n_epochs: int = 20,
    lr_all: float = 0.01,
    reg_all: float = 0.2,
    random_state: int = 1,
) -> SVD:
    """
    Build and fit an SVD matrix factorization model.

    Defaults reproduce this project's tuned configuration. The
    notebook's untuned baseline was simply SVD(random_state=1) with the
    surprise library's own defaults for everything else — call that
    directly if you need that exact comparison point.

    Returns
    -------
    surprise.SVD
        Fitted model.
    """
    model = SVD(random_state=random_state, n_epochs=n_epochs, lr_all=lr_all, reg_all=reg_all)
    model.fit(trainset)
    return model


# ---------------------------------------------------------------------
# Hyperparameter tuning (shared by any surprise algorithm)
# ---------------------------------------------------------------------

def tune_hyperparameters(algo_class, param_grid: dict, data, cv: int = 3, measure: str = "rmse"):
    """
    Run GridSearchCV for any surprise algorithm class (KNNBasic, SVD, ...).

    The notebook repeated this pattern three times (once per model); it's
    one generic function here since GridSearchCV itself is generic.

    Parameters
    ----------
    algo_class : type
        The algorithm class itself, e.g. KNNBasic or SVD (not an instance).
    param_grid : dict
        See USER_USER_PARAM_GRID / ITEM_ITEM_PARAM_GRID / SVD_PARAM_GRID
        above for this project's actual search spaces.
    data : surprise.Dataset
    cv : int, default 3
    measure : str, default "rmse"

    Returns
    -------
    (best_params, best_score, cv_results_df)
    """
    gs = GridSearchCV(algo_class, param_grid, measures=[measure], cv=cv, n_jobs=-1)
    gs.fit(data)
    return gs.best_params[measure], gs.best_score[measure], pd.DataFrame(gs.cv_results)


# ---------------------------------------------------------------------
# Recommendation generation
# ---------------------------------------------------------------------

def get_recommendations(data: pd.DataFrame, user_id: str, top_n: int, algo) -> list:
    """
    Generate the top-n recommended products for a user from a fitted model.

    Predicts a rating for every product the user hasn't already rated
    and returns the highest-predicted ones.

    Parameters
    ----------
    data : pd.DataFrame
        Must contain 'user_id', 'prod_id', 'rating' columns.
    user_id : str
    top_n : int
    algo : a fitted surprise algorithm (from train_knn_model / train_svd_model)

    Returns
    -------
    list of (prod_id, predicted_rating) tuples, sorted descending by rating.
    """
    user_item_interactions_matrix = data.pivot(index="user_id", columns="prod_id", values="rating")
    non_interacted_products = (
        user_item_interactions_matrix.loc[user_id][user_item_interactions_matrix.loc[user_id].isnull()]
        .index.tolist()
    )

    recommendations = []
    for item_id in non_interacted_products:
        est = algo.predict(user_id, item_id).est
        recommendations.append((item_id, est))

    recommendations.sort(key=lambda x: x[1], reverse=True)
    return recommendations[:top_n]
