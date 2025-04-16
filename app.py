import gradio as gr
import pandas as pd
from main import get_predictions

should_stop = False
MAX_LAYERS = 5
ACTIVATIONS = ['tanh', 'relu', 'sigmoid', 'linear']
RECURRENT_ACTIVATIONS = ['sigmoid', 'hard_sigmoid', 'tanh']

def run_model(ticker, days_to_predict, start_date, days_to_train,
              n_estimators, learning_rate_gradian, max_depth,
              epochs, loss_function, optimizer_name, learning_rate_lstm, batch_size, num_layers, *args):
    global should_stop
    should_stop = False
    progress = gr.Progress()

    lstm_layers = []
    params_per_layer = 6
    num_layers = int(num_layers)
    gradian_params = {
        "n_estimators": int(n_estimators),
        "learning_rate": float(learning_rate_gradian),
        "max_depth": None if max_depth is None else int(max_depth)
    }
    for i in range(0, num_layers * params_per_layer, params_per_layer):
        if i+params_per_layer-1 < len(args):
            layer_params = args[i:i+params_per_layer]
            if layer_params[0] is not None:
                lstm_layers.append({
                    'units': layer_params[0],
                    'dropout': layer_params[1],
                    'recurrent_dropout': layer_params[2],
                    'activation': layer_params[3],
                    'recurrent_activation': layer_params[4],
                    'return_sequences': layer_params[5] if i < (num_layers-1)*params_per_layer else False
                })

    def update_progress(epoch, total_epochs):
        progress((epoch / total_epochs), desc=f"Epoch {epoch}/{total_epochs}")

    try:
        fig_all, fig_linear, fig_gradian, fig_lstm = get_predictions(
            days_to_predict,
            ticker,
            pd.to_datetime(start_date),
            days_to_train,
            gradian_params,
            epochs,
            loss_function,
            optimizer_name,
            learning_rate_lstm,
            batch_size,
            lambda: should_stop,
            update_progress,
            lstm_layers
        )
        return ("Training finished successfully!", fig_all, fig_linear,fig_gradian, fig_lstm)
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
        with gr.Group(visible=True) as layers_group:
            gr.Markdown("#General informations")
            ticker = gr.Textbox(label="Ticker", value="NVDA")
            days_to_predict = gr.Slider(1, 365, value=5, label="Days to predict")
            start_date = gr.Textbox(label="Start date (YYYY-MM-DD)", value="1900-01-01")
            days_to_train = gr.Slider(1, 1000, value=365, label="Days to train")

    with gr.Row():
        with gr.Group(visible=True) as layers_group:
            gr.Markdown("#Gradian Boost")
            n_estimators = gr.Slider(10, 500, value=100, step=10, label="Number of estimators")
            learning_rate_gradian = gr.Slider(0.0001, 0.1, value=0.001, step=0.0001, label="Learning rate")
            max_depth = gr.Slider(2, 50, value=None, step=1, label="Max Depth (None=unlimited)")
    
    with gr.Row():
        with gr.Group(visible=True) as layers_group:
            gr.Markdown("#LSTM")
            
            epochs = gr.Slider(1, 500, value=50, label="Epochs")
            loss_function = gr.Dropdown(choices=["mse", "mae"], label="Loss function", value="mse")
            optimizer_name = gr.Dropdown(choices=["adam", "sgd", "rmsprop"], label="Optimizer", value="adam")
            learning_rate_lstm = gr.Slider(0.0001, 0.1, value=0.001, step=0.0001, label="Learning rate")
            batch_size = gr.Slider(8, 512, step=8, value=32, label="Batch Size")
    num_of_layers_def=2    
    with gr.Row():
        num_layers = gr.Slider(1, MAX_LAYERS, value=num_of_layers_def, step=1, label="Number of LSTM Layers")
        update_layers_btn = gr.Button("Update Layers Configuration")
    
    lstm_layers_ui = []
    separators = []
   
    with gr.Group(visible=True) as layers_group:
        gr.Markdown("### LSTM Layers Configuration")
        for i in range(MAX_LAYERS):
            with gr.Row(visible=(i < num_of_layers_def)) as layer_row:
                with gr.Column():
                    with gr.Row():
                        with gr.Column():
                            units = gr.Number(value=50, label="Neurons")
                            dropout = gr.Slider(0.0, 0.9, value=0.2, label="Dropout")
                        
                        with gr.Column():
                            recurrent_dropout = gr.Slider(0.0, 0.9, value=0.0, label="Recurrent Dropout")
                            activation = gr.Dropdown(ACTIVATIONS, value='tanh', label="Activation")
                        
                        with gr.Column():
                            recurrent_activation = gr.Dropdown(RECURRENT_ACTIVATIONS, value='sigmoid', label="Recurrent Activation")
                            return_sequences = gr.Checkbox(value=(i < MAX_LAYERS-1), label="Return Sequences")
                
                lstm_layers_ui.append((layer_row, units, dropout, recurrent_dropout, 
                                     activation, recurrent_activation, return_sequences))
            if i < MAX_LAYERS - 1:
                separator = gr.HTML("<hr style='margin: 15px 0; border-top: 2px solid #ccc'>", 
                                  visible=(i < 1))
                separators.append(separator)
    
    with gr.Row():
        start_button = gr.Button("Predict", variant="primary")
        stop_button = gr.Button("Stop Training", variant="stop")

    status_output = gr.Textbox(label="Status")
    fig_all_output = gr.Plot(label="Historical Data")
    fig_linear_output = gr.Plot(label="Linear Regression Prediction")
    fig_gradian_output = gr.Plot(label="Gradian Boost Prediction")
    fig_lstm_output = gr.Plot(label="LSTM Prediction")

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
        inputs=[ticker, days_to_predict, start_date, days_to_train,
        n_estimators, learning_rate_gradian, max_depth,
        epochs, loss_function, optimizer_name, learning_rate_lstm, batch_size, num_layers] +
       [comp for (_, *comps) in lstm_layers_ui for comp in comps],
        outputs=[status_output, fig_all_output, fig_linear_output,fig_gradian_output, fig_lstm_output]
    )
    
    stop_button.click(
        fn=stop_training,
        inputs=[],
        outputs=[status_output]
    )

demo.launch()