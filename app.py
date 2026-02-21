import streamlit as st
import io
from contextlib import redirect_stdout

import model_core  # <-- this must match your big file name (without .py)

st.set_page_config(page_title="DCF Valuation App", layout="wide")
st.title("DCF Valuation App")

ticker = st.text_input("Ticker", value="PRS.MC")

st.subheader("Optional overrides (leave blank to use defaults)")

c1, c2, c3 = st.columns(3)
with c1:
    rf = st.text_input("Risk-free rate (rf) e.g. 0.04", value="")
    erp = st.text_input("Equity risk premium (ERP) e.g. 0.05", value="")
with c2:
    beta = st.text_input("Beta e.g. 1.0", value="")
    cost_debt = st.text_input("Cost of debt e.g. 0.045", value="")
with c3:
    cost_equity = st.text_input("Cost of equity (optional) e.g. 0.09", value="")
    wD = st.text_input("Weight of debt (wD) 0-1 e.g. 0.30", value="")

terminal_g = st.text_input("Terminal growth (g) e.g. 0.025", value="")

if st.button("Run DCF"):
    user_inputs = {
        "rf": rf,
        "erp": erp,
        "beta": beta,
        "cost_debt": cost_debt,
        "cost_equity": cost_equity,
        "wD": wD,
        "terminal_g": terminal_g,
    }

    # Capture print() output from model into Streamlit
    buf = io.StringIO()
    with redirect_stdout(buf):
        model_core.handle_report(ticker.strip(), user_inputs=user_inputs)

    st.subheader("Model Output")
    st.code(buf.getvalue())
