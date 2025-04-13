import gradio as gr
import pandas as pd
from main import get_predictions

should_stop = False

def train_model(ticker, days_to_predict, days_to_train, epochs, start_date):
    global should_stop
    should_stop = False
    try:
        fig_all, fig_linear, fig_lstm = get_predictions(
            days_to_predict,
            ticker,
            days_to_train,
            epochs,
            pd.to_datetime(start_date),
            stop_check=lambda: should_stop
        )
        return ("Training finished successfully!", fig_all, fig_linear, fig_lstm)
    except Exception as e:
        return (f"Error: {str(e)}", None, None, None)

def stop_training():
    global should_stop
    should_stop = True
    return "Stopping training..."

demo = gr.Blocks()

with demo:
    gr.Markdown("# 📈 Stock Price Predictor (Linear + LSTM)")

    with gr.Row():
        ticker = gr.Textbox(label="Ticker", value="NVDA")
        days_to_predict = gr.Slider(1, 365, value=30, label="Days to predict")
        days_to_train = gr.Slider(1, 1000, value=365, label="Days to train")
        epochs = gr.Slider(1, 500, value=50, label="Epochs")
        start_date = gr.Textbox(label="Start date (YYYY-MM-DD)", value="1900-01-01")

    with gr.Row():
        start_button = gr.Button("Predict")
        stop_button = gr.Button("Stop Training")

    status_output = gr.Textbox(label="Status")
    fig_all_output = gr.Plot(label="Historical Data")
    fig_linear_output = gr.Plot(label="Linear Regression Prediction")
    fig_lstm_output = gr.Plot(label="LSTM Prediction")

    start_button.click(
        fn=train_model,
        inputs=[ticker, days_to_predict, days_to_train, epochs, start_date],
        outputs=[status_output, fig_all_output, fig_linear_output, fig_lstm_output]
    )

    stop_button.click(
        fn=stop_training,
        inputs=[],
        outputs=[status_output]
    )

demo.launch()
