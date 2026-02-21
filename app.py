import streamlit as st
import pandas as pd

import model_core as model

st.set_page_config(page_title="DCF App", layout="wide")
st.title("DCF Model (Interactive)")

# ── Ticker ────────────────────────────────────────────────────────────────────
ticker = st.text_input("Ticker", value="HEIA.AS",
                       help="Yahoo Finance ticker, e.g. AAPL, HEIA.AS, VOD.L")

# ── Overrides ─────────────────────────────────────────────────────────────────
st.subheader("Overrides (optional)")
st.caption("Leave any field blank to use the value fetched from Yahoo Finance or the built-in default.")

col1, col2, col3 = st.columns(3)
with col1:
    st.markdown("**Market / Cost of Capital**")
    rf          = st.text_input("Risk-free rate (rf)",       value="", placeholder="e.g. 0.04  →  4%")
    erp         = st.text_input("Equity risk premium (erp)", value="", placeholder="e.g. 0.05  →  5%")
    beta        = st.text_input("Beta",                       value="", placeholder="e.g. 1.2")

with col2:
    st.markdown("**Debt & Capital Structure**")
    cost_debt   = st.text_input("Cost of debt",              value="", placeholder="e.g. 0.035  →  3.5%")
    wD          = st.text_input("Debt weight (wD)",          value="", placeholder="e.g. 0.30  →  30%")
    cost_equity = st.text_input("Cost of equity (optional)", value="", placeholder="e.g. 0.08  →  8%")

with col3:
    st.markdown("**Operating Assumptions**")
    terminal_g  = st.text_input("Terminal growth (g)",       value="", placeholder="e.g. 0.025  →  2.5%")
    rev_growth  = st.text_input("Revenue growth",            value="", placeholder="e.g. 0.06  →  6%")
    ebit_margin = st.text_input("EBIT margin",               value="", placeholder="e.g. 0.15  →  15%")
    tax_rate    = st.text_input("Tax rate",                  value="", placeholder="e.g. 0.25  →  25%")

user_inputs = {
    "rf": rf, "erp": erp, "beta": beta,
    "cost_debt": cost_debt, "wD": wD, "cost_equity": cost_equity,
    "terminal_g": terminal_g, "rev_growth": rev_growth,
    "ebit_margin": ebit_margin, "tax_rate": tax_rate,
}

# ── Comparable Companies ──────────────────────────────────────────────────────
st.subheader("Comparable Companies (optional)")
st.caption(
    "Enter ticker symbols for your chosen peers. Leave all blank to let the model "
    "auto-select peers from Yahoo Finance. Data is fetched automatically."
)

# Initialise session state for peer rows
if "peer_count" not in st.session_state:
    st.session_state.peer_count = 4

col_add, col_remove, _ = st.columns([1, 1, 6])
with col_add:
    if st.button("＋ Add peer"):
        st.session_state.peer_count += 1
with col_remove:
    if st.button("－ Remove last") and st.session_state.peer_count > 1:
        st.session_state.peer_count -= 1

peer_tickers = []
peer_cols = st.columns(min(st.session_state.peer_count, 4))  # max 4 per row
for i in range(st.session_state.peer_count):
    col = peer_cols[i % 4]
    with col:
        val = st.text_input(
            f"Peer {i+1}",
            value="",
            placeholder="e.g. UB.AS",
            key=f"peer_{i}",
        )
        peer_tickers.append(val.strip())

# Filter out blanks
user_peers = [p.upper() for p in peer_tickers if p.strip()]

if user_peers:
    st.info(f"Will use **{len(user_peers)}** user-supplied peer(s): {', '.join(user_peers)}")
else:
    st.info("No peers entered — model will auto-select peers from Yahoo Finance.")

# ── Run ───────────────────────────────────────────────────────────────────────
run = st.button("Run model", type="primary")

# ── Helpers ───────────────────────────────────────────────────────────────────
def pretty_df(df, decimals=2):
    df2 = df.copy()
    for c in df2.columns:
        if pd.api.types.is_numeric_dtype(df2[c]):
            df2[c] = df2[c].map(lambda v: "" if pd.isna(v) else f"{v:,.{decimals}f}")
    return df2

def show_forecast(df, decimals=2):
    """Set Year as index so it renders as a row label, not a data column."""
    d = df.copy()
    if "Year" in d.columns:
        d = d.set_index("Year")
    for c in d.columns:
        if pd.api.types.is_numeric_dtype(d[c]):
            d[c] = d[c].map(lambda v: "" if pd.isna(v) else f"{v:,.{decimals}f}")
    return d

def style_assumptions(df):
    def row_style(row):
        src = row["Source"]
        if "User Input" in src:
            color = "#d4edda"
        elif any(x in src for x in ["Yahoo", "Damodaran", "CAPM", "Computed"]):
            color = "#cce5ff"
        else:
            color = "#f8f9fa"
        return [f"background-color: {color}"] * len(row)
    return df.style.apply(row_style, axis=1)

# ── Output ────────────────────────────────────────────────────────────────────
if run:
    with st.spinner("Running model…"):
        res = model.handle_report(ticker, user_inputs=user_inputs,
                                  user_peers=user_peers if user_peers else None)

    st.success("Done")

    # Key metrics
    st.subheader("Key Output")
    c1, c2 = st.columns(2)
    with c1:
        st.metric("DCF Value per Share",
                  f"{res['vps']:,.2f}" if res["vps"] is not None else "NA")
    with c2:
        st.metric("As-of Price",
                  f"{res['px']:,.2f}" if res["px"] is not None else "NA")

    # Assumptions
    st.subheader("Model Assumptions")
    st.caption("Every key driver and where it came from.")
    if res.get("assumptions_df") is not None:
        st.dataframe(style_assumptions(res["assumptions_df"]),
                     use_container_width=True, hide_index=True)
        st.caption(
            "🟢 **User Input** — value you typed above  |  "
            "🔵 **Yahoo / Computed** — fetched or derived automatically  |  "
            "⬜ **Default** — built-in fallback"
        )

    # Forecast tables
    st.subheader("Forecast Income Statement")
    st.table(show_forecast(res["df_is"]))

    st.subheader("Forecast Balance Sheet")
    st.table(show_forecast(res["df_bs"]))

    st.subheader("Forecast Cash Flow")
    st.table(show_forecast(res["df_cf"]))

    st.subheader("DCF Bridge")
    st.table(pretty_df(res["bridge"], decimals=2))

    # Comps
    st.subheader("Comps (Peers)")
    if res.get("comps_df") is not None and not res["comps_df"].empty:
        # Show company name alongside ticker if available
        st.table(pretty_df(res["comps_df"], decimals=2))
    else:
        st.info("No comps data available.")

    st.subheader("Comps Valuation")
    if res.get("comp_val_df") is not None and not res["comp_val_df"].empty:
        st.table(pretty_df(res["comp_val_df"], decimals=2))

    # Sensitivity
    st.subheader("Sensitivity Table (Value per share)")
    st.dataframe(res["sens_table"], use_container_width=True)

    st.subheader("Sensitivity Heatmap")
    st.pyplot(res["heatmap_fig"])

    # Comps multiples chart
    st.subheader("Peer Multiples")
    if res.get("multiples_fig") is not None:
        st.pyplot(res["multiples_fig"])
    else:
        st.info("Not enough peer data to plot multiples.")

    # Cash flow Sankey
    st.subheader("Cash Flow Allocation")
    st.caption("How operating cash flow is allocated in the first forecast year.")
    if res.get("sankey_fig") is not None:
        st.pyplot(res["sankey_fig"])
    else:
        st.info("Cash flow allocation chart unavailable.")

    # Scenarios
    st.subheader("Scenarios")
    st.table(pretty_df(res["scen_df"], decimals=2))
