import streamlit as st
import pandas as pd

# IMPORTANT: change this to your file name (without .py)
# Example: if your big file is dcf_model.py -> import dcf_model as model
import model_core as model


st.set_page_config(page_title="DCF App", layout="wide")
st.title("DCF Model (Interactive)")

# Input controls
ticker = st.text_input("Ticker", value="HEIA.AS")

st.subheader("Overrides (optional)")
col1, col2, col3 = st.columns(3)

with col1:
    rf = st.text_input("Risk-free (rf)", value="")
    erp = st.text_input("Equity risk premium (erp)", value="")
    beta = st.text_input("Beta", value="")

with col2:
    cost_debt = st.text_input("Cost of debt", value="")
    wD = st.text_input("Debt weight (wD)", value="")

with col3:
    cost_equity = st.text_input("Cost of equity (optional)", value="")
    terminal_g = st.text_input("Terminal growth (g)", value="")

user_inputs = {
    "rf": rf,
    "erp": erp,
    "beta": beta,
    "cost_debt": cost_debt,
    "wD": wD,
    "cost_equity": cost_equity,
    "terminal_g": terminal_g,
}

run = st.button("Run model")

def pretty_df(df, decimals=2):
    df2 = df.copy()
    for c in df2.columns:
        if pd.api.types.is_numeric_dtype(df2[c]):
            df2[c] = df2[c].map(lambda v: "" if pd.isna(v) else f"{v:,.{decimals}f}")
    return df2

if run:
    with st.spinner("Running model..."):
        res = model.handle_report(ticker, user_inputs=user_inputs)

    st.success("Done")

    st.subheader("Key output")
    c1, c2 = st.columns(2)
    with c1:
        st.metric("DCF Value per Share", f"{res['vps']:,.2f}" if res["vps"] is not None else "NA")
    with c2:
        st.metric("As-of Price", f"{res['px']:,.2f}" if res["px"] is not None else "NA")

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
