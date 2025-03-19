from sklearn.preprocessing import MinMaxScaler
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout


def predict_stock_prices(df_org, days_in_future_to_predict):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df_org)
    days_to_train = 60
    X, y = create_sequences(scaled_data, days_to_train, df_org.columns.get_loc("Close"))

    X_train, X_test = X[:-days_in_future_to_predict], X[days_in_future_to_predict:]
    y_train, y_test = y[:-days_in_future_to_predict], y[days_in_future_to_predict:]

    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], df_org.shape[1]))
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], df_org.shape[1]))

        
    model = Sequential()

    # Pierwsza warstwa LSTM
    model.add(
        LSTM(
            units=50,
            return_sequences=True,
            input_shape=(X_train.shape[1], df_org.shape[1]),
        )
    )
    model.add(Dropout(0.2))

    # Druga warstwa LSTM
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dropout(0.2))

    # Warstwa wyjściowa
    model.add(Dense(units=1))

    # Kompilacja modelu
    model.compile(optimizer="adam", loss="mean_squared_error")

    # Podsumowanie modelu
    model.summary()
    history = model.fit(
        X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test)
    )
    predictions = model.predict(X_test)

    predictions = inverse_scaller(predictions, df_org, scaler)

    y_test_original = inverse_scaller(y_test, df_org, scaler)
    future_predictions = []
    last_sequence = X_test[-1]
    for _ in range(30):
        pred = model.predict(last_sequence.reshape(1, days_to_train, df_org.shape[1]))
        future_predictions.append(pred[0, 0])
        # Aktualizuj sekwencję
        last_sequence = np.roll(last_sequence, -1, axis=0)  # Przesuń sekwencję wzdłuż osi czasu
        last_sequence[-1, df_org.columns.get_loc("Close")] = pred[0, 0]  # Zastąp 'Close' przewidywa





    # Skalowanie przewidywań z powrotem do oryginalnych wartości
    future_predictions = np.array(future_predictions).reshape(-1, 1)
    future_predictions = inverse_scaller(future_predictions, df_org, scaler)
    
    return predictions[-days_in_future_to_predict:], future_predictions, 60

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

    # Odwróć transformację
    predictions_original = scaler.inverse_transform(temp_array)

    # Wybierz tylko kolumnę 'Close'
    predictions_close = predictions_original[:, df_org.columns.get_loc("Close")]
    return predictions_close 

