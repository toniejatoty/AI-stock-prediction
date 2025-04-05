import streamlit as st
import yfinance as yf
import matplotlib.pyplot as plt
from visualization import show_plot, show_all_historical_data
from main import get_predictions

st.title("📈 Stock Price Predictor (Linear + LSTM)")

ticker = st.text_input("Enter stock ticker (e.g. AAPL):", "AAPL")
if not ticker:
    st.warning("Please enter a ticker!")
    st.stop()


fig_all,fig_linear,fig_lstm = get_predictions(30, ticker)  


st.subheader("Historical Data (since IPO)")
st.pyplot(fig_all)



col1, col2 = st.columns(2)

with col1:
    st.subheader("Linear Regression Prediction")
    st.pyplot(fig_linear)

with col2:
    st.subheader("LSTM Prediction")
    st.pyplot(fig_lstm)