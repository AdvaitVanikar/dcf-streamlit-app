import streamlit as st
import io
import contextlib

import model_core  # this imports your big DCF file

st.set_page_config(page_title="DCF App", layout="wide")
st.title("DCF Valuation App")

ticker = st.text_input("Ticker", value="PRS.MC")

st.write("Optional: you can add inputs later (WACC, g, etc). For now we run the report.")

run = st.button("Run DCF")

if run:
    if not ticker.strip():
        st.warning("Please enter a ticker.")
    else:
        st.info("Running model...")

        # Capture all print() output from your existing script
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            model_core.handle_report(ticker.strip())

        output_text = buffer.getvalue()

        st.subheader("Model Output")
        st.text(output_text)
