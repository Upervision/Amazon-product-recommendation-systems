# Amazon Product Recommendation System

Building and comparing recommendation engines on Amazon Electronics ratings data — from a simple popularity baseline to tuned collaborative filtering and matrix factorization — to identify the approach that best balances accuracy, personalization, and practicality.

## Overview

E-commerce platforms rely on recommendation systems to surface relevant products from catalogs too large to browse manually. This project implements and evaluates three recommendation paradigms on Amazon's Electronics category ratings — ratings only, with no review text or product metadata, to keep the modeling task free of content bias — and compares them on both prediction accuracy and recommendation quality.

Full analysis: [`./RecommendationSystem.ipynb`](RecommendationSystem./RecommendationSystem.ipynb)

## Dataset

- **Source**: [Amazon product ratings data](http://jmcauley.ucsd.edu/data/amazon/), Julian McAuley (UCSD)
- **Raw size**: 7,824,482 ratings
- **Filtering**: restricted to users with ≥ 50 ratings and products with ≥ 5 ratings, to keep the user-item matrix dense enough to model meaningfully
- **Final size**: 65,290 ratings across 1,540 unique users and 5,689 unique products

## Approach

| Model | Technique |
|---|---|
| **1. Rank-Based** | Popularity recommender using average rating, with a minimum-interactions threshold to filter out low-volume outliers |
| **2. Collaborative Filtering** | Memory-based CF in both directions — User-User and Item-Item similarity — via `KNNBasic` |
| **3. Model-Based CF** | Matrix factorization via `SVD`, learning latent user/item factors |

Every collaborative filtering model was built twice — a default-parameter baseline, then re-tuned with `GridSearchCV` (3-fold cross-validation) over similarity metric, neighborhood size, learning rate, and regularization — so the effect of tuning is measured directly rather than assumed.

## Evaluation

Models were assessed on a held-out test split using:
- **RMSE** — rating prediction accuracy
- **Precision@10 / Recall@10 / F1@10** — ranking quality of the top-10 recommendations (relevance threshold = 3.5)

### Results

| Model | RMSE ↓ | Precision@10 | Recall@10 | F1@10 |
|---|---|---|---|---|
| User-User CF (baseline) | 1.0260 | 0.844 | 0.862 | 0.853 |
| User-User CF (tuned) | 0.9759 | 0.834 | 0.896 | 0.864 |
| Item-Item CF (baseline) | 1.0147 | 0.826 | 0.853 | 0.839 |
| Item-Item CF (tuned) | 0.9751 | 0.829 | 0.892 | 0.859 |
| SVD (baseline) | 0.9104 | 0.837 | 0.880 | 0.858 |
| **SVD (tuned)** | **0.9014** | **0.841** | 0.880 | **0.860** |

Hyperparameter tuning improved every collaborative filtering model. The tuned SVD model achieved the lowest prediction error and the best overall precision/recall balance, making it the strongest candidate for deployment.

**Best hyperparameters found:**
- User-User CF: `k=40, min_k=6, similarity=cosine`
- Item-Item CF: `k=30, min_k=6, similarity=msd`
- SVD: `n_epochs=20, lr_all=0.01, reg_all=0.2`

## Recommendations

- **Deploy the tuned SVD model** as the primary recommendation engine — it delivers the most accurate personalized predictions.
- **Use popularity-based recommendations for cold-start users** who have little or no rating history.
- **Consider a hybrid strategy**, blending popularity, item-item similarity, and matrix factorization for added robustness.
- **Retrain periodically** — customer preferences shift over time, and static models degrade.
- **Encourage post-purchase ratings** to reduce data sparsity and improve future recommendation quality.

## Tech Stack

- Python, pandas, NumPy
- [`scikit-surprise`](http://surpriselib.com/) — `KNNBasic`, `SVD`, `GridSearchCV`
- Matplotlib, Seaborn — EDA and evaluation visualizations

## Notebook Contents

1. Data loading, filtering, and exploratory analysis
2. Model 1 — Rank-based (popularity) recommender
3. Model 2 — Collaborative filtering: User-User and Item-Item (`KNNBasic`), baseline and tuned
4. Model 3 — Model-based CF: `SVD` matrix factorization, baseline and tuned
5. Evaluation: RMSE, Precision@k / Recall@k / F1@k across all models
6. Business recommendations and conclusion

## Setup

```bash
pip install scikit-surprise pandas numpy matplotlib seaborn
```

Update the dataset path in the notebook to point to your local copy of the ratings file, then run top to bottom.

## Acknowledgments

Dataset courtesy of Julian McAuley (UCSD) — [Amazon product data](http://jmcauley.ucsd.edu/data/amazon/). Completed as an applied project for the MIT IDSS AI and Data Science Program (delivered via Great Learning).
