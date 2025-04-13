import gradio as gr
import pandas as pd
from main import get_predictions

should_stop = False

def run_model(ticker, days_to_predict,start_date, days_to_train, epochs ,loss_function,optimizer_name,learning_rate,batch_size, lstm_layers):
    global should_stop
    should_stop = False
    progress=gr.Progress()
    for layer_config in lstm_layers:
        units = gr.Number(label="Units", value=50)
        layer_activation = gr.Dropdown(choices=["relu", "tanh", "sigmoid"], label="Activation", value="tanh")
        dropout = gr.Slider(0.0, 0.5, value=0.2, step=0.05, label="Dropout")
        recurrent_dropout = gr.Slider(0.0, 0.5, value=0.2, step=0.05, label="Recurrent Dropout")

    def update_progress(epoch, total_epochs):
        progress((epoch / total_epochs), desc=f"Epoch {epoch}/{total_epochs}")
    try:
        fig_all, fig_linear, fig_lstm = get_predictions(
            days_to_predict,
            ticker,
            pd.to_datetime(start_date),
            days_to_train,
            epochs,
            loss_function,
            optimizer_name,
            learning_rate,
            batch_size,
            lambda: should_stop,
            update_progress
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
        days_to_predict = gr.Slider(1, 365, value=5, label="Days to predict")
        start_date = gr.Textbox(label="Start date (YYYY-MM-DD)", value="1900-01-01")
    with gr.Row():
        days_to_train = gr.Slider(1, 1000, value=365, label="Days to train")
        epochs = gr.Slider(1, 500, value=50, label="Epochs")
        loss_function = gr.Dropdown(choices=["mse", "mae"], label="Loss function", value="mse")
        optimizer_name = gr.Dropdown(choices=["adam", "sgd", "rmsprop"], label="Optimizer", value="adam")
        learning_rate = gr.Slider(0.0001, 0.1, value=0.001, step=0.0001, label="Learning rate")
        batch_size = gr.Slider(8, 512, step=8, value=32, label="Batch Size")
    
    with gr.Row():
        start_button = gr.Button("Predict")
        stop_button = gr.Button("Stop Training")

    with gr.Row():
        lstm_layers_df = gr.Dataframe(
        headers=["units", "dropout", "return_sequences"],
        datatype=["number", "number", "bool"],
        row_count=2,  
        col_count=(3, "fixed"),
        label="LSTM Layers Configuration"
    )
    status_output = gr.Textbox(label="Status")
    fig_all_output = gr.Plot(label="Historical Data")
    fig_linear_output = gr.Plot(label="Linear Regression Prediction")
    fig_lstm_output = gr.Plot(label="LSTM Prediction")

    start_button.click(
        run_model,
        inputs=[ticker, days_to_predict, start_date, days_to_train, epochs,loss_function,optimizer_name,learning_rate,batch_size, lstm_layers_df],
        outputs=[status_output, fig_all_output, fig_linear_output, fig_lstm_output]
    )
    
    stop_button.click(
        stop_training,
        inputs=[],
        outputs=[status_output]
    )



    
demo.launch()
