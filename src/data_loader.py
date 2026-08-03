"""
data_loader.py

Data loading, filtering, and dataset preparation for the Amazon
Electronics recommendation system.
"""

import pandas as pd
from surprise import Dataset, Reader
from surprise.model_selection import train_test_split as surprise_train_test_split


def load_ratings(path: str) -> pd.DataFrame:
    """
    Load the raw Amazon ratings CSV.

    The source file has no header row; columns are user_id, prod_id,
    rating, timestamp. The timestamp column is dropped since nothing
    downstream uses it.

    Parameters
    ----------
    path : str
        Path to ratings_Electronics.csv (or an equivalent file with the
        same column order).

    Returns
    -------
    pd.DataFrame
        Columns: user_id, prod_id, rating.
    """
    df = pd.read_csv(path, names=["user_id", "prod_id", "rating", "timestamp"])
    return df.drop("timestamp", axis=1).copy()


def filter_by_min_interactions(
    df: pd.DataFrame,
    min_user_ratings: int = 50,
    min_product_ratings: int = 5,
) -> pd.DataFrame:
    """
    Restrict the data to users and products with enough ratings to
    support meaningful collaborative filtering.

    Users are filtered first, then products are filtered on the
    user-filtered result — the same two-stage order used in the
    notebook. (The notebook built this with manual counting loops;
    this version does the same filtering with pandas' value_counts()
    for the same result, more concisely.)

    Parameters
    ----------
    df : pd.DataFrame
        Must contain 'user_id' and 'prod_id' columns.
    min_user_ratings : int, default 50
        Minimum number of ratings a user must have to be kept.
    min_product_ratings : int, default 5
        Minimum number of ratings a product must have to be kept.

    Returns
    -------
    pd.DataFrame
        The filtered dataframe.
    """
    user_counts = df["user_id"].value_counts()
    keep_users = user_counts[user_counts >= min_user_ratings].index
    df_users_filtered = df[df["user_id"].isin(keep_users)]

    prod_counts = df_users_filtered["prod_id"].value_counts()
    keep_prods = prod_counts[prod_counts >= min_product_ratings].index
    df_final = df_users_filtered[df_users_filtered["prod_id"].isin(keep_prods)]

    return df_final


def build_surprise_dataset(df: pd.DataFrame, rating_scale: tuple = (1, 5)) -> Dataset:
    """
    Wrap a ratings dataframe into a surprise Dataset, ready for
    train_test_split or GridSearchCV.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain 'user_id', 'prod_id', 'rating' columns.
    rating_scale : tuple, default (1, 5)
        Minimum and maximum possible rating values.

    Returns
    -------
    surprise.Dataset
    """
    reader = Reader(rating_scale=rating_scale)
    return Dataset.load_from_df(df[["user_id", "prod_id", "rating"]], reader)


def split_dataset(data: Dataset, test_size: float = 0.2, random_state: int = 1):
    """
    Split a surprise Dataset into trainset/testset objects.

    Thin wrapper around surprise's own train_test_split, kept here so
    every data-preparation step lives in one place.

    Returns
    -------
    (trainset, testset)
    """
    return surprise_train_test_split(data, test_size=test_size, random_state=random_state)
