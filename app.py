"""Streamlit demo: upload transactions, get RFM segments back.

Run:  streamlit run app.py
"""
import io

import pandas as pd
import plotly.express as px
import streamlit as st

import segmentation as seg

PALETTE = {"Champions": "#2a9d8f", "Loyal": "#264653", "New": "#8ab17d",
           "At Risk": "#e76f51", "Lost": "#9e9e9e"}

st.set_page_config(page_title="Customer Segmentation", layout="wide")
st.title("Customer Segmentation (RFM + K-Means)")
st.write(
    "Upload transaction data (.xlsx, .csv or .parquet) with the columns "
    "`Invoice, StockCode, Quantity, InvoiceDate, Price, Customer ID`. "
    "The original *Online Retail* column names (`InvoiceNo`, `UnitPrice`, `CustomerID`) also work."
)

upload = st.file_uploader("Transactions file", type=["xlsx", "csv", "parquet"])
use_sample = st.checkbox("Use the bundled sample (Online Retail II, Dec 2010 – Dec 2011)", value=upload is None)
k = st.sidebar.slider("Number of clusters (k)", 2, 8, 5,
                      help="k = 5 maps onto Champions / Loyal / New / At Risk / Lost.")


@st.cache_data(show_spinner="Reading file…")
def read(data: bytes, name: str) -> pd.DataFrame:
    buf = io.BytesIO(data)
    buf.name = name
    return seg.load_transactions(buf)


@st.cache_data(show_spinner="Loading sample…")
def read_sample() -> pd.DataFrame:
    return seg.load_transactions("data/sample_transactions.csv.gz")


@st.cache_data(show_spinner="Cleaning and clustering…")
def run(df: pd.DataFrame, k: int):
    return seg.run_pipeline(df, k)


if upload is not None:
    raw = read(upload.getvalue(), upload.name)
elif use_sample:
    raw = read_sample()
else:
    st.stop()

try:
    segmented, profile, log = run(raw, k)
except ValueError as e:
    st.error(str(e))
    st.stop()

base = lambda s: s.rsplit(" ", 1)[0] if s[-1].isdigit() else s
colors = {s: PALETTE[base(s)] for s in profile.index}

c1, c2, c3, c4 = st.columns(4)
c1.metric("Transactions (raw)", f"{log.iloc[0].rows_remaining:,}")
c2.metric("After cleaning", f"{log.iloc[-1].rows_remaining:,}")
c3.metric("Customers", f"{len(segmented):,}")
top = profile.iloc[0]
c4.metric(f"{profile.index[0]} revenue share", f"{top.RevenueShare:.0f}%",
          f"{top.CustomerShare:.0f}% of customers", delta_color="off")

st.subheader("Segment profile")
st.dataframe(
    profile.style.format({"Recency": "{:.0f} d", "Frequency": "{:.1f}", "Monetary": "£{:,.0f}",
                          "AvgOrderValue": "£{:,.0f}", "Revenue": "£{:,.0f}",
                          "CustomerShare": "{:.1f}%", "RevenueShare": "{:.1f}%"}),
    width="stretch",
)

left, right = st.columns(2)
shares = profile[["CustomerShare", "RevenueShare"]].reset_index().melt(
    id_vars="Segment", var_name="Metric", value_name="Percent")
left.plotly_chart(px.bar(shares, x="Segment", y="Percent", color="Metric", barmode="group",
                         title="Customers vs revenue by segment",
                         color_discrete_sequence=["#b0bec5", "#264653"]), width="stretch")
right.plotly_chart(px.scatter(segmented.reset_index(), x="Recency", y="Monetary", color="Segment",
                              log_y=True, opacity=0.6, color_discrete_map=colors,
                              hover_data=["Customer ID", "Frequency"],
                              title="Recency vs spend (log scale)"), width="stretch")

with st.expander("Cleaning log"):
    st.dataframe(log, width="stretch")

st.subheader("Customers")
seg_filter = st.multiselect("Filter segments", list(profile.index), default=list(profile.index))
view = segmented[segmented["Segment"].isin(seg_filter)].round(2)
st.dataframe(view, width="stretch")
st.download_button("Download segments as CSV", view.to_csv().encode(), "customer_segments.csv", "text/csv")
