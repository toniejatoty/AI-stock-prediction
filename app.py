import streamlit as st
from main import get_predictions
import pandas as pd
from threading import Event
from tensorflow.keras.callbacks import Callback # type: ignore

stop_training = Event()
class StopTrainingCallback(Callback):
    def on_epoch_end(self, epoch, logs=None):
        if stop_training.is_set():
            self.model.stop_training = True
            st.warning("Trening przerwany przez użytkownika!")

st.title("📈 Stock Price Predictor (Linear + LSTM)")
with st.sidebar:
    st.header("Parametry modelu")
    ticker = st.text_input("Symbol akcji", "AAPL").upper()
    days_to_predict = st.number_input("Liczba dni do przewidzenia", 1, 365, 30)
    days_to_train = st.number_input("Liczba dni do uczenia", 30, 1000, 365)
    epochs = st.number_input("Liczba epok (LSTM)", 1, 500, 50)
    start_date = st.date_input("Data początkowa danych", value=pd.to_datetime("1900-01-01"))



if st.button("Uruchom predykcję"):
    try:
        stop_training.clear()
        with st.spinner("Pobieranie danych i trenowanie modeli..."):
            fig_all,fig_linear,fig_lstm = get_predictions(days_to_predict, ticker,days_to_train,epochs,start_date)    


            st.subheader("Historical Data (since IPO)")
            st.pyplot(fig_all)



            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Linear Regression Prediction")
                st.pyplot(fig_linear)

            with col2:
                st.subheader("LSTM Prediction")
                st.pyplot(fig_lstm)
    except ValueError as e:
        st.error(f"Błąd: {str(e)}")
        st.info("Check wheather ticker exists and try again")
if st.button("Przerwij trening"):
    stop_training.set()
    st.warning("Trwa przerywanie treningu...")           