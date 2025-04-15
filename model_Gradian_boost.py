import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

def predict_stock_prices(
    df: pd.DataFrame,
    days_in_future: int = 3,
    days_to_train: int = 10,
    model_params: dict = {"n_estimators": 100, "learning_rate": 0.1, "max_depth": 3},
    loss_function: str = "mse"
):
    required_cols = ["Open", "High", "Low", "Close"]
    if not all(col in df.columns for col in required_cols):
        raise ValueError("DataFrame musi zawierać kolumny: Open, High, Low, Close!")

    df = df.copy().reset_index(drop=True)

    # Tworzenie targetów
    for i in range(1, days_in_future + 1):
        df[f'target_{i}'] = df['Close'].shift(-i)

    df.dropna(inplace=True)
    
    # Tworzenie sekwencji danych wejściowych
    X = []
    y = {f'target_{i}': [] for i in range(1, days_in_future + 1)}
    
    for i in range(days_to_train, len(df)):
        X_seq = df[required_cols].iloc[i - days_to_train:i].values.flatten()
        for j in range(1, days_in_future + 1):
            y[f'target_{j}'].append(df[f'target_{j}'].iloc[i])
        X.append(X_seq)

    X = np.array(X)
    y = {k: np.array(v) for k, v in y.items()}

    # Podział na dane uczące i testowe (ostatni punkt to test)
    X_train, X_test = X[:-1], X[-1].reshape(1, -1)
    y_train = {k: v[:-1] for k, v in y.items()}
    y_test = {k: v[-1] for k, v in y.items()}

    models = {}
    test_preds = []
    future_preds = []

    for day in range(1, days_in_future + 1):
        model = XGBRegressor(**model_params)
        model.fit(X_train, y_train[f'target_{day}'])
        pred = model.predict(X_test)[0]
        test_preds.append(pred)
        future_preds.append(pred)
        models[f'model_day_{day}'] = model

    test_preds = np.array(test_preds)
    y_test_vals = np.array([y_test[f'target_{i}'] for i in range(1, days_in_future + 1)])

    score = get_score(test_preds, y_test_vals, loss_function)

    return test_preds, future_preds, score







def get_score(predicted, real, loss_function):
    if loss_function == "mse":
        score = mean_squared_error(predicted, real)
    elif loss_function == "mae":
        score = mean_absolute_error(predicted, real)
    return score
        