from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from tensorflow.keras.models import Sequential  # type: ignore
from tensorflow.keras.layers import LSTM, Dense, Dropout  # type: ignore
from tensorflow.keras.optimizers import Adam, SGD, RMSprop, Adagrad # type: ignore

def predict_stock_prices(
    df_org,
    days_in_future_to_predict,
    days_to_train,
    epochs,
    loss_function,
    optimizer_name,
    learning_rate,
    batch_size,
    stop_check,
    progress_callback,
    lstm_layers
):
    df_org=df_org[['Close']]
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df_org)
    X, y = create_sequences(scaled_data, days_to_train, df_org.columns.get_loc("Close"))

    X_train, X_test = X[:-days_in_future_to_predict], X[-days_in_future_to_predict:]
    y_train, y_test = y[:-days_in_future_to_predict], y[-days_in_future_to_predict:]

    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], df_org.shape[1]))
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], df_org.shape[1]))

    model = get_model(X_train.shape[1], df_org.shape[1],optimizer_name, learning_rate,loss_function,lstm_layers)

    for epoch in range(epochs):
        progress_callback(epoch, epochs)
        if stop_check():
            print(f"Training stopped at epoch {epoch}")
            break

        print(f"{epoch} / {epochs}")
        X_val = get_X_test(
            X_test, model, df_org, days_to_train, days_in_future_to_predict
        )
        model.fit(
            X_train,
            y_train,
            epochs=1,
            batch_size=batch_size,
            validation_data=(X_val, y_test),
        )
    predictions = model.predict(X_test)
    predictions = inverse_scaller(predictions, df_org, scaler)

    test_predictions = get_predicted_new_prices(
        df_org,
        days_in_future_to_predict,
        model,
        days_to_train,
        scaler,
        X_test,
        y_test,
        X_test[0],
    )

    future_predictions = get_predicted_new_prices(
        df_org, days_in_future_to_predict, model, days_to_train, scaler, X_test, y_test
    )
    score = get_score(test_predictions,y_test,loss_function,scaler,df_org)

    return (test_predictions, future_predictions,score,predictions)  # /predictions to return

############################# functions

def create_sequences(df, days_to_train, close_index):
    X = []
    y = []
    for i in range(days_to_train, len(df)):
        X.append(df[i - days_to_train : i])
        y.append(df[i, close_index])
    return np.array(X), np.array(y)



def get_model(input_shape1, input_shape2, optimizer_name, learning_rate, loss_function, lstm_layers=None):
    model = Sequential()
    
    for i, layer_config in enumerate(lstm_layers):
        if i == len(lstm_layers) - 1:
            layer_config['return_sequences'] = False
        if i == 0:
            model.add(LSTM(
                units=layer_config['units'],
                return_sequences=layer_config['return_sequences'],
                activation=layer_config['activation'],
                recurrent_activation=layer_config['recurrent_activation'],
                dropout=layer_config['dropout'],
                recurrent_dropout=layer_config['recurrent_dropout'],
                input_shape=(input_shape1, input_shape2)
            ))
        else:
            model.add(LSTM(
                units=layer_config['units'],
                return_sequences=layer_config['return_sequences'],
                activation=layer_config['activation'],
                recurrent_activation=layer_config['recurrent_activation'],
                dropout=layer_config['dropout'],
                recurrent_dropout=layer_config['recurrent_dropout']
            ))
        
        if i < len(lstm_layers) - 1:
            model.add(Dropout(layer_config['dropout']))
    
    model.add(Dense(units=1))
    
    opt = get_optimizer(optimizer_name, learning_rate)
    model.compile(optimizer=opt, loss=loss_function)
    model.summary()
    
    return model


def inverse_scaller(predictions, df_org, scaler):
    temp_array = np.zeros((predictions.shape[0], df_org.shape[1]))
    temp_array[:, df_org.columns.get_loc("Close")] = predictions.flatten()
    predictions_original = scaler.inverse_transform(temp_array)
    predictions_close = predictions_original[:, df_org.columns.get_loc("Close")]
    return predictions_close




def get_X_test(X_test, model, df_org, days_to_train, days_in_future_to_predict):
    result = []
    last_sequence2 = X_test[0].copy()

    for _ in range(days_in_future_to_predict):
        pred2 = model.predict(
            last_sequence2.reshape(1, days_to_train, df_org.shape[1]), verbose=0
        )
        last_sequence2 = np.roll(last_sequence2, -1, axis=0)
        last_sequence2[-1, df_org.columns.get_loc("Close")] = pred2[0, 0]
        result.append(last_sequence2)
    return np.array(result)



def get_predicted_new_prices(
    df_org,
    days_in_future_to_predict,
    model,
    days_to_train,
    scaler,
    X_test,
    y_test,
    last_sequence=None,
):
    future_predictions = []
    if last_sequence is None:
        last_sequence = X_test[-1]
        last_sequence = np.roll(last_sequence, -1, axis=0)
        last_sequence[-1, df_org.columns.get_loc("Close")] = y_test[-1]

    for _ in range(days_in_future_to_predict):
        pred = model.predict(last_sequence.reshape(1, days_to_train, df_org.shape[1]))
        future_predictions.append(pred[0, 0])
        last_sequence = np.roll(last_sequence, -1, axis=0)
        last_sequence[-1, df_org.columns.get_loc("Close")] = pred[0, 0]

    future_predictions = np.array(future_predictions).reshape(-1, 1)
    future_predictions = inverse_scaller(future_predictions, df_org, scaler)
    return future_predictions



def get_optimizer(name, learning_rate):
    if name == "adam":
        return Adam(learning_rate=learning_rate)
    elif name == "rmsprop":
        return RMSprop(learning_rate=learning_rate)
    elif name == "sgd":
        return SGD(learning_rate=learning_rate)
    
def get_score(predicted, real, loss_function,scaler,df_org):
    real = inverse_scaller(real, df_org, scaler)
    if loss_function == "mse":
        score = mean_squared_error(predicted, real)
    elif loss_function == "mae":
        score = mean_absolute_error(predicted, real)
    return score
        