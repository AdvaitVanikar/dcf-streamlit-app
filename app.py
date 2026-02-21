import streamlit as st
import pandas as pd

# IMPORTANT: change this to your file name (without .py)
# Example: if your big file is dcf_model.py -> import dcf_model as model
import model_core as model


st.set_page_config(page_title="DCF App", layout="wide")
st.title("DCF Model (Interactive)")

# Input controls
ticker = st.text_input("Ticker", value="HEIA.AS", help="Yahoo Finance ticker symbol, e.g. AAPL, HEIA.AS, VOD.L")

st.subheader("Overrides (optional)")
st.caption("Leave any field blank to use the value fetched from Yahoo Finance or the built-in default.")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("**Market / Cost of Capital**")
    rf = st.text_input("Risk-free rate (rf)", value="", placeholder="e.g. 0.04  →  4%")
    erp = st.text_input("Equity risk premium (erp)", value="", placeholder="e.g. 0.05  →  5%")
    beta = st.text_input("Beta", value="", placeholder="e.g. 1.2")

with col2:
    st.markdown("**Debt & Capital Structure**")
    cost_debt = st.text_input("Cost of debt", value="", placeholder="e.g. 0.035  →  3.5%")
    wD = st.text_input("Debt weight (wD)", value="", placeholder="e.g. 0.30  →  30%")
    cost_equity = st.text_input("Cost of equity (optional override)", value="", placeholder="e.g. 0.08  →  8%")

with col3:
    st.markdown("**Operating Assumptions**")
    terminal_g = st.text_input("Terminal growth (g)", value="", placeholder="e.g. 0.025  →  2.5%")
    rev_growth = st.text_input("Revenue growth", value="", placeholder="e.g. 0.06  →  6%")
    ebit_margin = st.text_input("EBIT margin", value="", placeholder="e.g. 0.15  →  15%")
    tax_rate = st.text_input("Tax rate", value="", placeholder="e.g. 0.25  →  25%")

user_inputs = {
    "rf": rf,
    "erp": erp,
    "beta": beta,
    "cost_debt": cost_debt,
    "wD": wD,
    "cost_equity": cost_equity,
    "terminal_g": terminal_g,
    "rev_growth": rev_growth,
    "ebit_margin": ebit_margin,
    "tax_rate": tax_rate,
}

run = st.button("Run model", type="primary")

def pretty_df(df, decimals=2):
    df2 = df.copy()
    for c in df2.columns:
        if pd.api.types.is_numeric_dtype(df2[c]):
            df2[c] = df2[c].map(lambda v: "" if pd.isna(v) else f"{v:,.{decimals}f}")
    return df2

def style_assumptions(df):
    """Colour-code the Source column."""
    def row_style(row):
        src = row["Source"]
        if "User Input" in src:
            color = "#d4edda"
        elif "Yahoo" in src or "Damodaran" in src:
            color = "#cce5ff"
        else:
            color = "#f8f9fa"
        return [f"background-color: {color}"] * len(row)
    return df.style.apply(row_style, axis=1)

if run:
    with st.spinner("Running model..."):
        res = model.handle_report(ticker, user_inputs=user_inputs)

    st.success("Done")

    # KEY METRICS
    st.subheader("Key Output")
    c1, c2 = st.columns(2)
    with c1:
        st.metric("DCF Value per Share", f"{res['vps']:,.2f}" if res["vps"] is not None else "NA")
    with c2:
        st.metric("As-of Price", f"{res['px']:,.2f}" if res["px"] is not None else "NA")

    # ASSUMPTIONS TABLE
    st.subheader("Model Assumptions")
    st.caption("Every key driver and where it came from.")
    if "assumptions_df" in res and res["assumptions_df"] is not None:
        st.dataframe(
            style_assumptions(res["assumptions_df"]),
            use_container_width=True,
            hide_index=True,
        )
        st.caption(
            "🟢 **User Input** — value you typed above  |  "
            "🔵 **Yahoo Finance / Damodaran** — fetched automatically  |  "
            "⬜ **Default / Computed** — built-in fallback or derived value"
        )
    else:
        st.info("Assumptions summary not available.")

    # FORECAST TABLES
    st.subheader("Forecast Income Statement")
    st.table(pretty_df(res["df_is"], decimals=2))

    st.subheader("Forecast Balance Sheet")
    st.table(pretty_df(res["df_bs"], decimals=2))

    st.subheader("Forecast Cash Flow")
    st.table(pretty_df(res["df_cf"], decimals=2))

    st.subheader("DCF Bridge")
    st.table(pretty_df(res["bridge"], decimals=2))

    st.subheader("Comps (Peers)")
    st.table(pretty_df(res["comps_df"], decimals=2))

    st.subheader("Comps Valuation")
    if res["comp_val_df"] is not None and not res["comp_val_df"].empty:
        st.table(pretty_df(res["comp_val_df"], decimals=2))

    st.subheader("Sensitivity Table (Value per share)")
    st.dataframe(res["sens_table"], use_container_width=True)

    st.subheader("Sensitivity Heatmap")
    st.pyplot(res["heatmap_fig"])

    st.subheader("Scenarios")
    st.table(pretty_df(res["scen_df"], decimals=2))

    st.subheader("Revenue Scenarios Plot")
    st.pyplot(res["revenue_fig"])
