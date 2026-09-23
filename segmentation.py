"""RFM customer segmentation pipeline for the UCI Online Retail II dataset.

Shared by the notebook, the CLI (main.py) and the Streamlit app (app.py) so that
all three produce identical segments.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

RANDOM_STATE = 42
RFM_COLS = ["Recency", "Frequency", "Monetary"]

# Stock codes that are fees, postage or bookkeeping entries rather than products.
NON_PRODUCT_CODES = {
    "POST", "DOT", "C2", "M", "D", "ADJUST", "ADJUST2",
    "BANK CHARGES", "CRUK", "TEST001", "TEST002", "AMAZONFEE", "S", "B",
}

# Column aliases so the loader also accepts the original "Online Retail" (I) export.
COLUMN_ALIASES = {
    "InvoiceNo": "Invoice",
    "UnitPrice": "Price",
    "CustomerID": "Customer ID",
    "Customer_ID": "Customer ID",
}
REQUIRED_COLUMNS = ["Invoice", "StockCode", "Quantity", "InvoiceDate", "Price", "Customer ID"]

SEGMENT_ACTIONS = {
    "Champions": "Reward them: early access to new ranges, a VIP tier and referral incentives. No discounts needed.",
    "Loyal": "Upsell and cross-sell: bundles, volume pricing and a loyalty programme to push them into Champions.",
    "New": "Onboard fast: a welcome series and a time-limited second-order incentive to build the habit.",
    "At Risk": "Win back now: personalised 'we miss you' offer based on past purchases, plus a quick feedback survey.",
    "Lost": "Low-cost reactivation only: one seasonal campaign, then suppress from paid channels.",
}
SEGMENT_ORDER = ["Champions", "Loyal", "New", "At Risk", "Lost"]


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def load_transactions(source) -> pd.DataFrame:
    """Load transactions from an .xlsx (all sheets stacked), .csv or .parquet file.

    `source` may be a path or a file-like object with a `.name` attribute
    (e.g. a Streamlit upload).
    """
    name = str(getattr(source, "name", source)).lower()
    str_cols = {"Invoice": str, "InvoiceNo": str, "StockCode": str, "Description": str}
    if name.endswith((".xlsx", ".xls")):
        sheets = pd.read_excel(source, sheet_name=None, dtype=str_cols)
        df = pd.concat(sheets.values(), ignore_index=True)
    elif name.endswith(".parquet"):
        df = pd.read_parquet(source)
    else:
        df = pd.read_csv(source, dtype=str_cols, encoding_errors="replace")

    df = df.rename(columns=COLUMN_ALIASES)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}. Expected {REQUIRED_COLUMNS}.")
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    df["Quantity"] = pd.to_numeric(df["Quantity"], errors="coerce")
    df["Price"] = pd.to_numeric(df["Price"], errors="coerce")
    return df


def load_cached(xlsx_path: str | Path, cache_path: str | Path = "data_raw.parquet") -> pd.DataFrame:
    """Read the Excel file once (~40 s) and reuse a parquet cache afterwards."""
    cache_path = Path(cache_path)
    if cache_path.exists():
        return pd.read_parquet(cache_path)
    df = load_transactions(xlsx_path)
    df.to_parquet(cache_path)
    return df


# --------------------------------------------------------------------------- #
# Cleaning
# --------------------------------------------------------------------------- #
def clean_transactions(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply the cleaning steps in order and log how many rows each one removes."""
    steps = [
        ("Missing Customer ID", lambda d: d.dropna(subset=["Customer ID"])),
        ("Cancelled invoices (start with 'C')", lambda d: d[~d["Invoice"].astype(str).str.startswith("C")]),
        ("Non-positive quantity", lambda d: d[d["Quantity"] > 0]),
        ("Zero or negative price", lambda d: d[d["Price"] > 0]),
        ("Non-product codes (postage, fees, manual)", lambda d: d[~d["StockCode"].astype(str).str.upper().isin(NON_PRODUCT_CODES)]),
        ("Exact duplicate rows", lambda d: d.drop_duplicates()),
    ]
    log = [{"step": "Raw data", "rows_removed": 0, "rows_remaining": len(df)}]
    for label, fn in steps:
        before = len(df)
        df = fn(df)
        log.append({"step": label, "rows_removed": before - len(df), "rows_remaining": len(df)})

    df = df.copy()
    df["Customer ID"] = df["Customer ID"].astype(float).astype(int)
    df["Revenue"] = df["Quantity"] * df["Price"]
    log = pd.DataFrame(log)
    log["pct_of_raw_removed"] = log["rows_removed"] / log.loc[0, "rows_remaining"] * 100
    return df.reset_index(drop=True), log


# --------------------------------------------------------------------------- #
# Feature engineering
# --------------------------------------------------------------------------- #
def build_rfm(df: pd.DataFrame, snapshot_date: pd.Timestamp | None = None) -> pd.DataFrame:
    """One row per customer: Recency (days), Frequency (unique invoices), Monetary (spend).

    Also returns AvgOrderValue and Tenure (days since first purchase) as extra
    descriptors; they are not used for clustering.
    """
    if snapshot_date is None:
        snapshot_date = df["InvoiceDate"].max().normalize() + pd.Timedelta(days=1)
    rfm = df.groupby("Customer ID").agg(
        LastPurchase=("InvoiceDate", "max"),
        FirstPurchase=("InvoiceDate", "min"),
        Frequency=("Invoice", "nunique"),
        Monetary=("Revenue", "sum"),
    )
    rfm["Recency"] = (snapshot_date - rfm["LastPurchase"]).dt.days
    rfm["Tenure"] = (snapshot_date - rfm["FirstPurchase"]).dt.days
    rfm["AvgOrderValue"] = rfm["Monetary"] / rfm["Frequency"]
    return rfm[RFM_COLS + ["AvgOrderValue", "Tenure"]]


# --------------------------------------------------------------------------- #
# Preprocessing + clustering
# --------------------------------------------------------------------------- #
def preprocess(rfm: pd.DataFrame) -> tuple[np.ndarray, StandardScaler]:
    """log1p to tame the right skew, then standardise so each feature weighs equally."""
    logged = np.log1p(rfm[RFM_COLS])
    scaler = StandardScaler()
    return scaler.fit_transform(logged), scaler


def fit_kmeans(X: np.ndarray, k: int, random_state: int = RANDOM_STATE) -> KMeans:
    return KMeans(n_clusters=k, random_state=random_state, n_init=10).fit(X)


def name_clusters(rfm: pd.DataFrame, labels: np.ndarray, X: np.ndarray) -> dict[int, str]:
    """Map cluster ids to business names using the cluster centroids in scaled space.

    Each centroid is compared with the average customer (0 after scaling):
      * recent     = recency below average (scaled recency < 0)
      * valuable   = frequency + monetary above average
    recent & valuable  -> Champions (highest value) / Loyal (the rest)
    recent & not       -> New
    lapsed & valuable  -> At Risk
    lapsed & not       -> Lost
    Rule-based naming keeps labels stable when cluster ids change between seeds.
    """
    cent = pd.DataFrame(X, columns=RFM_COLS).groupby(labels).mean()
    value = cent["Frequency"] + cent["Monetary"]
    recent = cent["Recency"] < 0
    valuable = value > 0

    names: dict[int, str] = {}
    best = value[recent & valuable]
    for c in cent.index:
        if recent[c] and valuable[c]:
            names[c] = "Champions" if c == best.idxmax() else "Loyal"
        elif recent[c]:
            names[c] = "New"
        elif valuable[c]:
            names[c] = "At Risk"
        else:
            names[c] = "Lost"

    # Disambiguate duplicates (only happens for k != 5), ranking by value.
    dupes = {n for n in names.values() if list(names.values()).count(n) > 1}
    counts: dict[str, int] = {}
    for c in value.sort_values(ascending=False).index:
        n = names[c]
        if n in dupes:
            counts[n] = counts.get(n, 0) + 1
            names[c] = f"{n} {counts[n]}"
    return names


def segment_customers(rfm: pd.DataFrame, k: int = 5, random_state: int = RANDOM_STATE):
    """Full modelling step: preprocess, fit K-Means, attach cluster ids and names."""
    X, scaler = preprocess(rfm)
    model = fit_kmeans(X, k, random_state)
    names = name_clusters(rfm, model.labels_, X)
    out = rfm.copy()
    out["Cluster"] = model.labels_
    out["Segment"] = out["Cluster"].map(names)
    return out, X, model, scaler


def segment_profile(segmented: pd.DataFrame) -> pd.DataFrame:
    """Average unscaled RFM per segment plus size and revenue share."""
    prof = segmented.groupby("Segment").agg(
        Customers=("Recency", "size"),
        Recency=("Recency", "mean"),
        Frequency=("Frequency", "mean"),
        Monetary=("Monetary", "mean"),
        AvgOrderValue=("AvgOrderValue", "mean"),
        Revenue=("Monetary", "sum"),
    )
    prof["CustomerShare"] = prof["Customers"] / prof["Customers"].sum() * 100
    prof["RevenueShare"] = prof["Revenue"] / prof["Revenue"].sum() * 100
    base = {s: s.rsplit(" ", 1)[0] if s[-1].isdigit() else s for s in prof.index}
    prof = prof.loc[sorted(prof.index, key=lambda s: (SEGMENT_ORDER.index(base[s]), s))]
    prof["Action"] = [SEGMENT_ACTIONS[base[s]] for s in prof.index]
    return prof


def run_pipeline(df_raw: pd.DataFrame, k: int = 5):
    """Raw transactions -> (segmented customers, profile, cleaning log)."""
    clean, log = clean_transactions(df_raw)
    rfm = build_rfm(clean)
    segmented, *_ = segment_customers(rfm, k)
    return segmented, segment_profile(segmented), log
