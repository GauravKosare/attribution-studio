"""Dynamic Multi-Touch Attribution Dashboard.

Fully input-driven: generate synthetic data with your own parameters, or upload your
own touchpoints/conversions/spend CSVs (schema in docs/02_data_dictionary.md), and
every model, chart, and table below recomputes from that input. Nothing is hardcoded
to a fixed channel list or a fixed dataset.

Run with:
    streamlit run dashboards/app.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data_generator import DEFAULT_CHANNELS, generate_dataset
from src.journey_builder import build_journeys
from src.attribution import rule_based, markov, shapley
from src.roi import compute_roi, recommend_reallocation

st.set_page_config(page_title="Multi-Touch Attribution", layout="wide")
st.title("📊 Multi-Touch Attribution & Budget Reallocation")
st.caption(
    "Feed it any touchpoint/conversion/spend data — synthetic or your own — and it "
    "rebuilds journeys, runs every attribution model, and recomputes ROI live."
)

# ---------------------------------------------------------------------------
# 1. DATA INPUT (dynamic — synthetic generator OR file upload)
# ---------------------------------------------------------------------------
st.sidebar.header("1. Data source")
source = st.sidebar.radio("Choose input", ["Generate synthetic data", "Upload my own CSVs"])

if source == "Generate synthetic data":
    st.sidebar.subheader("Synthetic data parameters")
    n_users = st.sidebar.slider("Number of users", 500, 20000, 5000, step=500)
    days = st.sidebar.slider("Date range (days)", 30, 180, 90, step=10)
    base_rate = st.sidebar.slider("Base conversion rate", 0.01, 0.30, 0.06, step=0.01)
    seed = st.sidebar.number_input("Random seed", value=42, step=1)

    selected_channels = st.sidebar.multiselect(
        "Channels to include", list(DEFAULT_CHANNELS.keys()), default=list(DEFAULT_CHANNELS.keys())
    )
    channels_cfg = {c: DEFAULT_CHANNELS[c] for c in selected_channels} or DEFAULT_CHANNELS

    if st.sidebar.button("Generate", type="primary") or "touchpoints" not in st.session_state:
        tp, conv, spend = generate_dataset(
            n_users=n_users, channels=channels_cfg, days=days,
            base_conversion_rate=base_rate, seed=int(seed),
        )
        st.session_state["touchpoints"] = tp
        st.session_state["conversions"] = conv
        st.session_state["channel_spend"] = spend
else:
    st.sidebar.subheader("Upload CSVs (schema: docs/02_data_dictionary.md)")
    tp_file = st.sidebar.file_uploader("touchpoints.csv", type="csv")
    conv_file = st.sidebar.file_uploader("conversions.csv", type="csv")
    spend_file = st.sidebar.file_uploader("channel_spend.csv", type="csv")
    if tp_file and conv_file and spend_file:
        st.session_state["touchpoints"] = pd.read_csv(tp_file)
        st.session_state["conversions"] = pd.read_csv(conv_file)
        st.session_state["channel_spend"] = pd.read_csv(spend_file)
    elif "touchpoints" not in st.session_state:
        st.info("Upload all three CSVs to continue, or switch to synthetic data.")
        st.stop()

touchpoints = st.session_state["touchpoints"]
conversions = st.session_state["conversions"]
channel_spend = st.session_state["channel_spend"]

# ---------------------------------------------------------------------------
# 2. JOURNEY CONSTRUCTION (adjustable rules — docs/02 §4)
# ---------------------------------------------------------------------------
st.sidebar.header("2. Journey rules")
lookback_days = st.sidebar.slider("Lookback window (days)", 7, 120, 30)
collapse_minutes = st.sidebar.slider("Collapse repeat touches within (min)", 0, 120, 30)

journeys = build_journeys(
    touchpoints, conversions, lookback_days=lookback_days, collapse_repeats_minutes=collapse_minutes
)

if journeys.empty:
    st.warning("No journeys could be built from this data. Check your input schema.")
    st.stop()

# ---------------------------------------------------------------------------
# 3. SUMMARY
# ---------------------------------------------------------------------------
c1, c2, c3, c4 = st.columns(4)
c1.metric("Total journeys", f"{len(journeys):,}")
c2.metric("Conversions", f"{int(journeys['converted'].sum()):,}")
conv_rate = journeys["converted"].mean() * 100
c3.metric("Conversion rate", f"{conv_rate:.1f}%")
c4.metric("Median path length", f"{journeys['n_touches'].median():.0f}")

# ---------------------------------------------------------------------------
# 4. RUN ALL ATTRIBUTION MODELS (dynamic on whatever channels are present)
# ---------------------------------------------------------------------------
with st.spinner("Running attribution models..."):
    rule_based_credit = rule_based.run_all(journeys)
    markov_credit = markov.run(journeys)
    shapley_credit = shapley.run(journeys)
    all_credit = pd.concat([rule_based_credit, markov_credit, shapley_credit], ignore_index=True)

model_labels = {
    "first_touch": "First-touch", "last_touch": "Last-touch", "linear": "Linear",
    "time_decay": "Time-decay", "position_based": "Position-based (U)",
    "markov": "Markov (removal effect)", "shapley": "Shapley value",
}
all_credit["model_label"] = all_credit["model"].map(model_labels)

st.divider()
st.subheader("Channel credit across models")
pivot = all_credit.pivot_table(index="channel", columns="model_label", values="credit_share", fill_value=0)
pivot = pivot[[v for v in model_labels.values() if v in pivot.columns]]
st.dataframe(pivot.style.format("{:.1%}").background_gradient(cmap="Blues", axis=1), width='stretch')

fig_bar = px.bar(
    all_credit, x="channel", y="credit_share", color="model_label", barmode="group",
    labels={"credit_share": "Credit share", "channel": "Channel", "model_label": "Model"},
    title="Channel credit share by attribution model",
)
fig_bar.update_layout(yaxis_tickformat=".0%")
st.plotly_chart(fig_bar, width='stretch')

# highlight biggest divergence (last-touch vs markov) — the headline insight
if "last_touch" in pivot.columns.map(lambda x: x) or "Last-touch" in pivot.columns:
    if "Last-touch" in pivot.columns and "Markov (removal effect)" in pivot.columns:
        diverge = (pivot["Markov (removal effect)"] - pivot["Last-touch"]).sort_values()
        under = diverge.index[0]
        over = diverge.index[-1]
        st.info(
            f"**Biggest divergence:** Last-touch over-credits **{over}** by "
            f"{diverge[over]*100:+.1f} pts vs. Markov, and under-credits **{under}** by "
            f"{diverge[under]*100:+.1f} pts. This is the actionable gap between naive "
            f"and data-driven attribution."
        )

# ---------------------------------------------------------------------------
# 5. SANKEY DIAGRAM OF JOURNEY PATHS
# ---------------------------------------------------------------------------
st.divider()
st.subheader("Customer journey paths (Sankey)")
top_n = st.slider("Show top N paths (rest grouped as 'Other')", 5, 50, 20)

converted_paths = journeys[journeys["converted"]]
path_counts = converted_paths["path_str"].value_counts()
top_paths = path_counts.head(top_n).index

nodes, node_idx = [], {}
links_src, links_dst, links_val = [], [], []


def get_node(label: str) -> int:
    if label not in node_idx:
        node_idx[label] = len(nodes)
        nodes.append(label)
    return node_idx[label]


for path_str, count in path_counts.items():
    path = path_str.split(" > ") if path_str else []
    if not path:
        continue
    display_path = path if path_str in top_paths else ["Other paths"]
    for i, ch in enumerate(display_path[: min(len(display_path), 6)]):
        label = f"{i+1}. {ch}"
        src = get_node(label if i == 0 else f"{i}. {display_path[i-1]}")
        dst = get_node(label)
        links_src.append(src)
        links_dst.append(dst)
        links_val.append(count)

if nodes:
    fig_sankey = go.Figure(
        go.Sankey(
            node=dict(label=nodes, pad=15, thickness=15),
            link=dict(source=links_src, target=links_dst, value=links_val),
        )
    )
    fig_sankey.update_layout(height=500, margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig_sankey, width='stretch')
else:
    st.info("Not enough converted journeys to draw a Sankey diagram.")

# ---------------------------------------------------------------------------
# 6. ROI & BUDGET REALLOCATION (dynamic model selector)
# ---------------------------------------------------------------------------
st.divider()
st.subheader("ROI & recommended budget reallocation")

model_choice = st.selectbox(
    "Attribution model for ROI", options=list(model_labels.keys()),
    format_func=lambda k: model_labels[k], index=list(model_labels.keys()).index("markov"),
)
shift_fraction = st.slider("Reallocation aggressiveness (% of underperformer spend to shift)", 0, 100, 20) / 100

total_conversions = int(journeys["converted"].sum())
total_revenue = float(journeys["conversion_value"].sum())

roi = compute_roi(all_credit, channel_spend, total_conversions, total_revenue, model_choice)
roi = recommend_reallocation(roi, shift_fraction=shift_fraction)

display_roi = roi[[
    "channel", "spend", "spend_share", "credit_share", "attributed_conversions",
    "attributed_revenue", "roas", "recommended_spend", "delta_spend", "delta_spend_pct",
]].copy()
display_roi.columns = [
    "Channel", "Spend", "Spend %", "Credit %", "Attr. Conversions", "Attr. Revenue",
    "ROAS", "Recommended Spend", "Δ Spend", "Δ Spend %",
]

st.dataframe(
    display_roi.style.format(
        {
            "Spend": "${:,.0f}", "Spend %": "{:.1%}", "Credit %": "{:.1%}",
            "Attr. Conversions": "{:,.0f}", "Attr. Revenue": "${:,.0f}", "ROAS": "{:.2f}x",
            "Recommended Spend": "${:,.0f}", "Δ Spend": "${:,.0f}", "Δ Spend %": "{:+.1f}%",
        },
        na_rep="—",
    ).background_gradient(subset=["ROAS"], cmap="RdYlGn"),
    width='stretch',
)

fig_roi = go.Figure()
fig_roi.add_bar(name="Current spend", x=roi["channel"], y=roi["spend"])
fig_roi.add_bar(name="Recommended spend", x=roi["channel"], y=roi["recommended_spend"])
fig_roi.update_layout(barmode="group", title="Current vs. recommended spend by channel")
st.plotly_chart(fig_roi, width='stretch')

st.download_button(
    "Download ROI table (CSV)", display_roi.to_csv(index=False).encode(), "roi_summary.csv", "text/csv"
)
st.download_button(
    "Download channel credit table (CSV)", all_credit.to_csv(index=False).encode(),
    "channel_credit.csv", "text/csv",
)

st.divider()
st.caption(
    "Directional, not causal — see docs/05_validation_report_template.md and "
    "docs/06_business_recommendation_template.md for the full methodology and caveats."
)
