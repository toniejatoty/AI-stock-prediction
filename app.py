import traceback
import gradio as gr
import numpy as np
from main import get_predictions

should_stop = False
MAX_LAYERS = 5
ACTIVATIONS = ['tanh', 'relu', 'sigmoid', 'linear']
RECURRENT_ACTIVATIONS = ['sigmoid', 'hard_sigmoid', 'tanh']

def run_model(LINEAR_check, XGB_check, LSTM1_check, LSTM2_check, ticker, days_to_predict, start_date, days_to_train,
              n_estimators, learning_rate_xgb, max_depth, XGB_early_stopping,
              epochs, loss_function, optimizer_name, learning_rate_lstm, batch_size, LSTM_early_stopping, num_layers, *LSTM_layers_info):
    global should_stop
    should_stop = False
    progress = gr.Progress()
    lstm_layers = []
    params_per_layer = 5
    charts = np.array([LINEAR_check, XGB_check, LSTM1_check, LSTM2_check])
    XGB_params = {
        "n_estimators": n_estimators,
        "learning_rate": learning_rate_xgb,
        "max_depth": max_depth,
        "early_stopping_rounds":XGB_early_stopping,
        "eval_metric": "rmse" if loss_function == "mse" else loss_function}
    for i in range(0, num_layers * params_per_layer, params_per_layer):
        layer_params = LSTM_layers_info[i:i+params_per_layer]
        lstm_layers.append({
            'units': layer_params[0],
            'dropout': layer_params[1],
            'recurrent_dropout': layer_params[2],
            'activation': layer_params[3],
            'recurrent_activation': layer_params[4],
        })


    def update_progress(name, epoch, total_epochs, loss, val_loss):
        progress((epoch / total_epochs), desc=f"I am in {name}, Progress {epoch}/{total_epochs}, loss = {np.round(loss,5)}, val_loss = {np.round(val_loss,5)}")

    try:
        fig_all, fig_linear, fig_xgb, fig_lstm,fig_lstm2, status = get_predictions(
            charts,
            days_to_predict,
            ticker,
            start_date,
            days_to_train,
            XGB_params,
            epochs,
            loss_function,
            optimizer_name,
            learning_rate_lstm,
            batch_size,
            LSTM_early_stopping,
            lambda: should_stop,
            update_progress,
            lstm_layers
        )
        print(f"status = {status}")
        return (status, fig_all, fig_linear,fig_xgb, fig_lstm, fig_lstm2)
    except Exception as e:
        error_trace = traceback.format_exc()
        print(f"Error: {str(e)}\nTraceback:\n{error_trace}")
        return (f"Error: {str(e)}", None, None, None, None, None)

def stop_training():
    global should_stop
    should_stop = True
    return "Stopping training..."

demo = gr.Blocks()

with demo:
    with gr.Row( ):
        with gr.Column(scale=2):
            gr.Markdown("# 📈 Stock Price Predictor (Linear + XGBRegressor +  LSTM)")
        with gr.Column(scale=1):
            LINEAR_check = gr.Checkbox(value=True,label='Linear',interactive=True)
            XGB_check = gr.Checkbox(value=True,label='XGBRegression',interactive=True)
        with gr.Column(scale=1):
            LSTM1_check = gr.Checkbox(value=True,label='LSTM1',interactive=True)
            LSTM2_check = gr.Checkbox(value=True,label='LSTM2',interactive=True)
    gr.Markdown("## General Information")
    with gr.Row():
        ticker = gr.Textbox(label="Ticker", value="INTC")
        days_to_predict = gr.Slider(1, 365, value=30, label="Days to predict")
        start_date = gr.Textbox(label="Start date (YYYY-MM-DD)", value="1900-01-01")
        days_to_train = gr.Slider(1, 1000, value=60, label="Days to train")
        loss_function = gr.Dropdown(choices=["mse", "mae"], label="Loss function", value="mse")
    gr.Markdown("<hr style='border: 1px solid #ddd; width: 100%;' />")
    gr.Markdown("## XGBRegressor")
    with gr.Row():
        n_estimators = gr.Slider(1, 1000, value=300, step=1, label="Number of estimators")
        learning_rate_xgb = gr.Slider(0.0001, 0.5, value=0.05, step=0.0001, label="Learning rate")
        max_depth = gr.Slider(1, 20,value=6, step=1, label="Max Depth")
        XGB_early_stopping=gr.Slider(1,500,value=100, label="Early stopping patience")
    gr.Markdown("<hr style='border: 1px solid #ddd; width: 100%;' />")
    gr.Markdown("## LSTM")
    with gr.Row():
        epochs = gr.Slider(1, 500, value=100, label="Epochs")
        optimizer_name = gr.Dropdown(choices=["adam", "sgd", "rmsprop"], label="Optimizer", value="adam")
        learning_rate_lstm = gr.Slider(0.0001, 0.1, value=0.001, step=0.0001, label="Learning rate")
        batch_size = gr.Slider(8, 512, step=8, value=32, label="Batch Size")
        LSTM_early_stopping=gr.Slider(1,500,value=10, label="Early stopping patience")
    num_of_layers_def=2    
    with gr.Row():
        num_layers = gr.Slider(1, MAX_LAYERS, value=num_of_layers_def, step=1, label="Number of LSTM Layers")
        update_layers_btn = gr.Button("Update number of Layers")
    
    lstm_layers_ui = []
    separators = []
   
    with gr.Group(visible=True):
        gr.Markdown("### LSTM Layers Configuration")
        for i in range(MAX_LAYERS):
            with gr.Row(visible=(i < num_of_layers_def)) as layer_row:
                units = gr.Number(value=50, label="Neurons")
                dropout = gr.Slider(0.0, 0.9, value=0.2, label="Dropout")
                recurrent_dropout = gr.Slider(0.0, 0.9, value=0.0, label="Recurrent Dropout")
                activation = gr.Dropdown(ACTIVATIONS, value='tanh', label="Activation")
                recurrent_activation = gr.Dropdown(RECURRENT_ACTIVATIONS, value='sigmoid', label="Recurrent Activation")

                lstm_layers_ui.append((layer_row, units, dropout, recurrent_dropout, activation, recurrent_activation))
            if i < MAX_LAYERS - 1:
                separator = gr.HTML("<hr style='margin: 15px 0; border-top: 2px solid #ccc'>", 
                                  visible=(i < 1))
                separators.append(separator)
    
    with gr.Row():
        start_button = gr.Button("Predict", variant="primary")
        stop_button = gr.Button("Stop Training", variant="stop")

    status_output = gr.Textbox(label="Status")
    fig_historical_output = gr.Plot(label="Historical Data")
    fig_linear_output = gr.Plot(label="Linear Regression Prediction")
    fig_xgb_output = gr.Plot(label="XGB Regression Prediction")
    fig_lstm_output = gr.Plot(label="LSTM Prediction")
    fig_lstm2_output = gr.Plot(label="LSTM 2 Prediction")

    def update_layer_visibility(num_layers):
        layer_updates = [gr.update(visible=(i < num_layers)) for i in range(MAX_LAYERS)]
        separator_updates = [gr.update(visible=(i < num_layers - 1)) for i in range(MAX_LAYERS - 1)]
        return layer_updates + separator_updates

    update_layers_btn.click(
        fn=update_layer_visibility,
        inputs=num_layers,
        outputs=[row for row, *_ in lstm_layers_ui] + separators
    )
    start_button.click(
        fn=run_model,
        inputs=[LINEAR_check, XGB_check, LSTM1_check, LSTM2_check, ticker, days_to_predict, start_date, days_to_train,
        n_estimators, learning_rate_xgb, max_depth, XGB_early_stopping,
        epochs, loss_function, optimizer_name, learning_rate_lstm, batch_size,LSTM_early_stopping, num_layers]+
       [comp for (_, *comps) in lstm_layers_ui for comp in comps], 
        outputs=[status_output, fig_historical_output, fig_linear_output,fig_xgb_output, fig_lstm_output, fig_lstm2_output]
    )

    stop_button.click(
        fn=stop_training,
        inputs=[],
        outputs=[status_output]
    )

demo.launch()