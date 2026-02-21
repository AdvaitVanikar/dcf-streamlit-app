import streamlit as st

st.title("My DCF App")

st.write("Hello, your Streamlit app is working!")

ticker = st.text_input("Enter a ticker", value="AAPL")

if st.button("Run"):
    st.write("You entered:", ticker)
