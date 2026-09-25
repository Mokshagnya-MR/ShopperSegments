"""Customer segmentation with RFM + K-Means on the UCI Online Retail II dataset.

Usage:
    python main.py                  # full analysis with k = 5
    python main.py --k 4            # try a different number of clusters
    python main.py --skip-extras    # skip the model comparison and stability checks

Each step prints its results. Charts go to images/ and tables to outputs/.
The reusable logic lives in segmentation.py and the charts in plots.py, so this
file reads like the project outline.
"""
import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, davies_bouldin_score, silhouette_score
from sklearn.mixture import GaussianMixture

import plots
import segmentation as seg

warnings.filterwarnings("ignore")
pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 20)
pd.set_option("display.float_format", lambda x: f"{x:,.2f}")

OUT_DIR = Path("outputs")


def section(title):
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


# ----------------------------------------------------------------------------
# Step 1-2: load and clean
# ----------------------------------------------------------------------------
def load_and_clean(data_path):
    """Load the raw transactions, then remove rows that don't describe real purchases.

    Cleaning order: missing customer -> cancellations -> bad quantity/price ->
    postage/fee codes -> duplicates. Each step's row count is logged so the
    README can show exactly what was removed and why.
    """
    section("1. Load data")
    raw = seg.load_cached(data_path)  # first run parses the Excel (~40 s), later runs use a parquet cache
    print(f"{len(raw):,} rows | {raw['InvoiceDate'].min():%Y-%m-%d} to {raw['InvoiceDate'].max():%Y-%m-%d}")
    print("\nMissing values per column:\n" + raw.isna().sum().to_string())

    section("2. Clean")
    clean, log = seg.clean_transactions(raw)
    print(log.to_string(index=False))
    plots.cleaning_steps(log)
    return clean, log


# ----------------------------------------------------------------------------
# Step 3-4: RFM features and preprocessing
# ----------------------------------------------------------------------------
def build_features(clean):
    """Collapse transactions to one row per customer: Recency, Frequency, Monetary."""
    section("3. RFM features")
    rfm = seg.build_rfm(clean)
    print(f"{len(rfm):,} customers\n")
    print(rfm.describe().T)
    print("\nSkewness (0 = symmetric; >1 = long right tail):\n" + rfm[seg.RFM_COLS].skew().to_string())
    return rfm


def preprocess(rfm):
    """log1p, then StandardScaler.

    K-Means uses Euclidean distance, so:
      * features on bigger scales (Monetary in £1000s) would dominate -> standardise
      * long right tails let a few huge customers pull centroids around -> log first
    The log must come before scaling; scaling alone doesn't change a distribution's shape.
    """
    section("4. Preprocess (log1p + StandardScaler)")
    X, _ = seg.preprocess(rfm)
    X_df = pd.DataFrame(X, columns=seg.RFM_COLS, index=rfm.index)
    print("Skewness after transform:\n" + X_df.skew().to_string())
    plots.rfm_distributions(rfm, X_df)
    return X


# ----------------------------------------------------------------------------
# Step 5: choose k
# ----------------------------------------------------------------------------
def evaluate_k(X, chosen_k, k_range=range(2, 11)):
    """Fit K-Means for each k and record inertia (elbow) and silhouette (separation).

    The two metrics often disagree. The final choice also has to make business
    sense: k = 5 maps onto Champions / Loyal / New / At Risk / Lost.
    """
    section("5. Choose k (elbow + silhouette)")
    rows = []
    for k in k_range:
        km = seg.fit_kmeans(X, k)
        rows.append({"k": k,
                     "inertia": km.inertia_,
                     "silhouette": silhouette_score(X, km.labels_),
                     "davies_bouldin": davies_bouldin_score(X, km.labels_)})
    scores = pd.DataFrame(rows).set_index("k")
    print(scores)
    plots.choose_k(scores, chosen_k)
    return scores


# ----------------------------------------------------------------------------
# Step 6-8: fit, interpret, visualise, recommend
# ----------------------------------------------------------------------------
def fit_segments(rfm, k):
    """Fit the final model and describe each segment in original (unscaled) units."""
    section(f"6. Fit K-Means (k={k}) and profile segments")
    segmented, X, model, _ = seg.segment_customers(rfm, k=k)
    profile = seg.segment_profile(segmented)
    print(profile.drop(columns="Action"))
    return segmented, profile, X, model


def visualise(rfm, segmented, profile, X, model):
    section("7. Charts")
    order = list(profile.index)
    segments = segmented["Segment"].values

    plots.snake_plot(X, segments, order)

    pca = PCA(n_components=2, random_state=seg.RANDOM_STATE)
    P = pca.fit_transform(X)
    plots.pca_scatter(P, pca.transform(model.cluster_centers_), segments, order, pca.explained_variance_ratio_)
    print("PCA loadings:\n", pd.DataFrame(pca.components_, columns=seg.RFM_COLS, index=["PC1", "PC2"]).round(2))

    plots.size_vs_revenue(profile)

    # Pareto: rank customers by spend and accumulate their share of revenue.
    spend = rfm["Monetary"].sort_values(ascending=False)
    cum_rev = (spend.cumsum() / spend.sum() * 100).values
    cum_cust = np.arange(1, len(spend) + 1) / len(spend) * 100
    customers_for = lambda rev_pct: cum_cust[np.searchsorted(cum_rev, rev_pct)]
    plots.pareto(cum_cust, cum_rev, {50: customers_for(50), 80: customers_for(80)})
    print(f"Top 10% of customers = {cum_rev[len(spend) // 10 - 1]:.1f}% of revenue")
    print(f"50% of revenue comes from the top {customers_for(50):.1f}% of customers; "
          f"80% from the top {customers_for(80):.1f}%")

    plots.rfm_boxplots(segmented, order)
    return P


def recommend(profile):
    section("8. Recommended actions")
    for name, row in profile.iterrows():
        print(f"{name:<10} {row.CustomerShare:5.1f}% of customers, {row.RevenueShare:5.1f}% of revenue")
        print(f"           -> {row.Action}")


# ----------------------------------------------------------------------------
# Step 9: stretch goals
# ----------------------------------------------------------------------------
def compare_algorithms(rfm, segmented, X, P, kmeans_labels, k):
    """Do Gaussian Mixture and hierarchical (Ward) clustering find the same structure?

    Cluster ids are arbitrary, so clusters are re-numbered 1..k by mean spend
    before comparing. ARI (adjusted Rand index) = 1 means an identical partition.
    """
    section("9a. K-Means vs Gaussian Mixture vs Agglomerative (Ward)")
    labels = {
        "K-Means": kmeans_labels,
        "Gaussian Mixture": GaussianMixture(k, random_state=seg.RANDOM_STATE, n_init=5).fit(X).predict(X),
        "Agglomerative (Ward)": AgglomerativeClustering(k, linkage="ward").fit_predict(X),
    }

    def rank_by_spend(lab):
        order = rfm["Monetary"].groupby(lab).mean().sort_values(ascending=False).index
        return pd.Series(lab).map({c: i + 1 for i, c in enumerate(order)}).values

    comparison = pd.DataFrame({
        name: {"silhouette": silhouette_score(X, lab),
               "davies_bouldin": davies_bouldin_score(X, lab),
               "ARI_vs_kmeans": adjusted_rand_score(kmeans_labels, lab),
               "smallest_cluster": np.bincount(lab).min(),
               "largest_cluster": np.bincount(lab).max()}
        for name, lab in labels.items()}).T
    print(comparison)

    ranked = {name: rank_by_spend(lab) for name, lab in labels.items()}
    for name in ["Agglomerative (Ward)", "Gaussian Mixture"]:
        table = pd.crosstab(segmented["Segment"].values, ranked[name], normalize="index") * 100
        table.index.name, table.columns.name = "K-Means segment", f"{name} cluster (1 = top spend)"
        print(f"\nWhere each K-Means segment lands under {name} (row %):\n{table.round(0)}")

    plots.model_comparison(P, ranked, comparison["silhouette"].to_dict(), k)
    return comparison


def check_stability(rfm, segmented, X, kmeans_labels, k, n_seeds=20):
    """Refit with different random seeds; a stable model gives (almost) the same clusters."""
    section(f"9b. Stability across {n_seeds} random seeds")
    rows = []
    for s in range(n_seeds):
        lab = seg.fit_kmeans(X, k, random_state=s).labels_
        names = pd.Series(lab).map(seg.name_clusters(rfm, lab, X)).values
        rows.append({"seed": s,
                     "ARI": adjusted_rand_score(kmeans_labels, lab),
                     "same_segment_name": (names == segmented["Segment"].values).mean()})
    stab = pd.DataFrame(rows)
    print(stab[["ARI", "same_segment_name"]].agg(["mean", "min", "max"]))
    plots.stability(stab, seg.RANDOM_STATE)
    return stab


# ----------------------------------------------------------------------------
# Save + entry point
# ----------------------------------------------------------------------------
def save_tables(tables):
    section("10. Save outputs")
    OUT_DIR.mkdir(exist_ok=True)
    for name, (df, keep_index) in tables.items():
        df.round(4).to_csv(OUT_DIR / f"{name}.csv", index=keep_index)
        print(f"  saved outputs/{name}.csv")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", default="online_retail_II.xlsx", help="path to the transactions file")
    parser.add_argument("--k", type=int, default=5, help="number of clusters (default 5)")
    parser.add_argument("--skip-extras", action="store_true", help="skip model comparison and stability checks")
    args = parser.parse_args()

    clean, clean_log = load_and_clean(args.data)
    rfm = build_features(clean)
    X = preprocess(rfm)
    scores = evaluate_k(X, args.k)
    segmented, profile, X, model = fit_segments(rfm, args.k)
    P = visualise(rfm, segmented, profile, X, model)
    recommend(profile)

    tables = {
        "customer_segments": (segmented, True),
        "segment_profile": (profile, True),
        "cleaning_log": (clean_log, False),
        "k_selection": (scores, True),
    }
    if not args.skip_extras:
        tables["model_comparison"] = (compare_algorithms(rfm, segmented, X, P, model.labels_, args.k), True)
        tables["stability"] = (check_stability(rfm, segmented, X, model.labels_, args.k), False)
    save_tables(tables)


if __name__ == "__main__":
    main()
