"""Chart functions for the segmentation analysis. Each one saves a PNG to images/."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # draw to files, no window needed
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd
import seaborn as sns

from segmentation import RFM_COLS

IMG_DIR = Path("images")
PALETTE = {"Champions": "#2a9d8f", "Loyal": "#264653", "New": "#8ab17d",
           "At Risk": "#e76f51", "Lost": "#9e9e9e"}

sns.set_theme(style="whitegrid")


def _save(fig, name):
    IMG_DIR.mkdir(exist_ok=True)
    fig.savefig(IMG_DIR / name, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved images/{name}")


def cleaning_steps(clean_log):
    steps = clean_log.iloc[1:]
    fig, ax = plt.subplots(figsize=(9, 3.8))
    ax.barh(steps["step"], steps["rows_removed"], color="#e76f51")
    for y, (n, p) in enumerate(zip(steps["rows_removed"], steps["pct_of_raw_removed"])):
        ax.text(n + 2000, y, f"{n:,}  ({p:.1f}%)", va="center", fontsize=9)
    ax.invert_yaxis()
    ax.set_xlim(0, steps["rows_removed"].max() * 1.35)
    ax.xaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f"{x / 1000:,.0f}k"))
    ax.set_title(f"Rows removed by each cleaning step "
                 f"({clean_log.iloc[0].rows_remaining:,} → {clean_log.iloc[-1].rows_remaining:,})")
    ax.set_xlabel("rows removed")
    _save(fig, "01_cleaning_steps.png")


def rfm_distributions(rfm, X_scaled):
    fig, axes = plt.subplots(2, 3, figsize=(13, 6.5))
    for i, col in enumerate(RFM_COLS):
        sns.histplot(rfm[col], bins=60, ax=axes[0, i], color="#264653")
        axes[0, i].set_title(f"{col}: raw (skew {rfm[col].skew():.1f})")
        sns.histplot(X_scaled[col], bins=60, ax=axes[1, i], color="#2a9d8f")
        axes[1, i].set_title(f"{col}: log1p + scaled (skew {X_scaled[col].skew():.2f})")
        for ax in axes[:, i]:
            ax.set_xlabel(""); ax.set_ylabel("")
    fig.suptitle("Log transform removes the heavy right skew", y=1.01, fontsize=13)
    fig.tight_layout()
    _save(fig, "02_rfm_distributions.png")


def choose_k(scores, chosen_k):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4))
    a1.plot(scores.index, scores["inertia"], "o-", color="#264653")
    a1.set(title="Elbow curve", xlabel="k", ylabel="inertia (WCSS)")
    a2.plot(scores.index, scores["silhouette"], "o-", color="#e76f51")
    a2.set(title="Silhouette score", xlabel="k", ylabel="silhouette")
    for a in (a1, a2):
        a.axvline(chosen_k, ls="--", color="grey", lw=1)
        a.text(chosen_k + 0.15, a.get_ylim()[1], f"chosen k = {chosen_k}", fontsize=9, color="grey", va="top")
    fig.tight_layout()
    _save(fig, "03_choose_k.png")


def snake_plot(X, segments, order):
    means = (pd.DataFrame(X, columns=RFM_COLS).assign(Segment=segments)
             .groupby("Segment").mean().loc[order])
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for s, row in means.iterrows():
        ax.plot(RFM_COLS, row.values, "o-", lw=2.5, label=s, color=PALETTE[s])
    ax.axhline(0, color="black", lw=0.8)
    ax.set(title="Snake plot: standardised log-RFM per segment", ylabel="z-score (0 = average customer)")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", frameon=False)
    _save(fig, "04_snake_plot.png")


def pca_scatter(P, centroids_2d, segments, order, explained):
    fig, ax = plt.subplots(figsize=(8, 6))
    for s in order:
        m = segments == s
        ax.scatter(P[m, 0], P[m, 1], s=8, alpha=0.55, color=PALETTE[s], label=f"{s} ({m.sum():,})")
    ax.scatter(centroids_2d[:, 0], centroids_2d[:, 1], marker="X", s=220, c="black",
               edgecolor="white", lw=1.5, label="centroids")
    ax.set(title=f"Customers in PCA space ({explained.sum():.0%} of variance shown)",
           xlabel=f"PC1 ({explained[0]:.0%}): overall engagement / value",
           ylabel=f"PC2 ({explained[1]:.0%}): mostly recency (higher = lapsed)")
    leg = ax.legend(frameon=False)
    for h in leg.legend_handles:
        h.set_sizes([40]); h.set_alpha(1)
    _save(fig, "05_pca_clusters.png")


def size_vs_revenue(profile):
    share = profile[["CustomerShare", "RevenueShare"]]
    x, w = np.arange(len(share)), 0.38
    fig, ax = plt.subplots(figsize=(9, 4.5))
    b1 = ax.bar(x - w / 2, share["CustomerShare"], w, label="% of customers", color="#b0bec5")
    b2 = ax.bar(x + w / 2, share["RevenueShare"], w, label="% of revenue",
                color=[PALETTE[s] for s in share.index])
    ax.bar_label(b1, fmt="%.0f%%", fontsize=9)
    ax.bar_label(b2, fmt="%.0f%%", fontsize=9, fontweight="bold")
    ax.set_xticks(x, share.index)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter())
    top = profile.iloc[0]
    ax.set_title(f"{profile.index[0]} are {top.CustomerShare:.0f}% of customers "
                 f"but {top.RevenueShare:.0f}% of revenue")
    ax.legend(frameon=False)
    _save(fig, "06_size_vs_revenue.png")


def pareto(cum_cust, cum_rev, marks):
    """marks: {revenue %: customer % needed to reach it}"""
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(cum_cust, cum_rev, color="#264653", lw=2.5)
    ax.plot([0, 100], [0, 100], ls="--", color="grey", lw=1, label="perfect equality")
    for rev_pct, cust_pct in marks.items():
        ax.plot([cust_pct, cust_pct, 0], [0, rev_pct, rev_pct], ls=":", color="#e76f51")
        ax.annotate(f"top {cust_pct:.1f}% of customers → {rev_pct}% of revenue",
                    (cust_pct, rev_pct), xytext=(cust_pct + 5, rev_pct - 7), fontsize=9)
    ax.xaxis.set_major_formatter(mtick.PercentFormatter())
    ax.yaxis.set_major_formatter(mtick.PercentFormatter())
    ax.set(xlim=(0, 100), ylim=(0, 101), xlabel="customers (ranked by spend)",
           ylabel="cumulative revenue", title="Revenue is highly concentrated")
    ax.legend(frameon=False, loc="lower right")
    _save(fig, "07_pareto.png")


def rfm_boxplots(segmented, order):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, col in zip(axes, RFM_COLS):
        sns.boxplot(data=segmented, x="Segment", y=col, order=order, hue="Segment",
                    palette=PALETTE, ax=ax, showfliers=False, legend=False)
        ax.set(title=col, xlabel="")
        if col != "Recency":
            ax.set_yscale("log")
        ax.tick_params(axis="x", rotation=30)
    fig.suptitle("Unscaled RFM by segment (outliers hidden; log axis for F and M)", y=1.03)
    fig.tight_layout()
    _save(fig, "08_rfm_boxplots.png")


def model_comparison(P, ranked_labels, silhouettes, k):
    """ranked_labels: {model name: labels 1..k, where 1 = highest-spend cluster}"""
    colors = sns.color_palette("viridis", k)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharex=True, sharey=True)
    for ax, (name, r) in zip(axes, ranked_labels.items()):
        for i in range(1, k + 1):
            ax.scatter(P[r == i, 0], P[r == i, 1], s=5, alpha=0.5, color=colors[i - 1],
                       label=f"cluster {i}" + (" (highest spend)" if i == 1 else ""))
        ax.set_title(f"{name}  (silhouette {silhouettes[name]:.2f})")
    fig.tight_layout()
    leg = fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncol=k,
                     bbox_to_anchor=(0.5, -0.08), frameon=False)
    for h in leg.legend_handles:
        h.set_sizes([30]); h.set_alpha(1)
    fig.suptitle("Same data, three algorithms (PCA projection; clusters ranked by mean spend)", y=1.02)
    _save(fig, "09_model_comparison.png")


def stability(stab, ref_seed):
    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.bar(stab["seed"], stab["ARI"], color="#2a9d8f")
    ax.set(ylim=(0.9, 1.0), xlabel="random seed", ylabel=f"ARI vs. seed {ref_seed}",
           title=f"Cluster stability across {len(stab)} seeds (mean ARI {stab['ARI'].mean():.3f})")
    ax.set_xticks(stab["seed"])
    _save(fig, "10_stability.png")
