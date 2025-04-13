from sklearn.preprocessing import MinMaxScaler
import numpy as np
from tensorflow.keras.models import Sequential  # type: ignore
from tensorflow.keras.layers import LSTM, Dense, Dropout  # type: ignore


def predict_stock_prices(
    df_org, days_in_future_to_predict, days_to_train, epochs, stop_check
):
    # df_org=df_org[['Close']]
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df_org)
    X, y = create_sequences(scaled_data, days_to_train, df_org.columns.get_loc("Close"))

    X_train, X_test = X[:-days_in_future_to_predict], X[-days_in_future_to_predict:]
    y_train, y_test = y[:-days_in_future_to_predict], y[-days_in_future_to_predict:]

    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], df_org.shape[1]))
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], df_org.shape[1]))

    model = Sequential()

    model.add(
        LSTM(
            units=50,
            return_sequences=True,
            input_shape=(X_train.shape[1], df_org.shape[1]),
        )
    )
    model.add(Dropout(0.2))

    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dropout(0.2))

    model.add(Dense(units=1))

    model.compile(optimizer="adam", loss="mean_squared_error")

    model.summary()
    for epoch in range(epochs):
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
            batch_size=32,
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

    return (test_predictions, future_predictions)  # predictions


def create_sequences(df, days_to_train, close_index):
    X = []
    y = []
    for i in range(days_to_train, len(df)):
        X.append(df[i - days_to_train : i])
        y.append(df[i, close_index])
    return np.array(X), np.array(y)


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
