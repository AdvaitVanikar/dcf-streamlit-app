import streamlit as st
import io
import contextlib

import model_core

st.set_page_config(page_title="DCF App", layout="wide")
st.title("DCF Valuation App")

with st.sidebar:
    st.header("Inputs")
    ticker = st.text_input("Ticker", value="PRS.MC")

    st.subheader("WACC inputs (optional)")
    cost_debt = st.text_input("Cost of debt (decimal)", value="", placeholder="e.g. 0.045")
    cost_equity = st.text_input("Cost of equity (decimal)", value="", placeholder="e.g. 0.10")
    wD = st.text_input("Weight of debt (0-1)", value="", placeholder="e.g. 0.20")

    st.subheader("Or use rf + ERP + beta (optional)")
    rf = st.text_input("Risk-free rf (decimal)", value="", placeholder="blank = Yahoo ^TNX")
    erp = st.text_input("Equity risk premium ERP (decimal)", value="", placeholder="blank = Damodaran/default")
    beta = st.text_input("Beta", value="", placeholder="blank = Yahoo")

    st.subheader("Terminal")
    terminal_g = st.text_input("Terminal growth g (decimal)", value="", placeholder="e.g. 0.025")

    run = st.button("Run DCF")

if run:
    if not ticker.strip():
        st.warning("Please enter a ticker.")
    else:
        st.info("Running model...")

        user_inputs = {
            "cost_debt": cost_debt,
            "cost_equity": cost_equity,
            "wD": wD,
            "rf": rf,
            "erp": erp,
            "beta": beta,
            "terminal_g": terminal_g,
        }

        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            model_core.handle_report(ticker.strip(), user_inputs=user_inputs)

        output_text = buffer.getvalue()

        st.subheader("Model Output")
        with st.expander("Show full output", expanded=True):
            st.text(output_text)
