import numpy as np
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

def predict_stock_prices(
    df_org,
    days_in_future,
    days_to_train,
    model_params,
    loss_function,
    stop_check,
    progress_callback
):
    Status="OK"
    df = df_org.copy()
    required_cols = df.columns
    for i in range(1, days_in_future + 1):
        df[f'target_{i}'] = df['Close'].shift(-i)
    df.dropna(inplace=True)

    #get training and test data
    X = []
    y = {f'target_{i}': [] for i in range(1, days_in_future + 1)}
    
    for i in range(days_to_train, len(df)):
        X.append(df[required_cols].iloc[i - days_to_train:i].values.flatten())
        for j in range(1, days_in_future + 1):
            y[f'target_{j}'].append(df[f'target_{j}'].iloc[i])
    
    X = np.array(X)
    y = {k: np.array(v) for k, v in y.items()}

    X_train, X_test = X[:-1], X[-1].reshape(1, -1)
    y_train = {k: v[:-1] for k, v in y.items()}
    y_test = {k: v[-1] for k, v in y.items()}
    y_test = np.array([y_test[f'target_{i}'] for i in range(1, days_in_future + 1)])
    
    future_X = df_org[required_cols].values[-days_to_train:].flatten().reshape(1, -1)

    #train and predict 
    models = {}
    test_preds = []
    future_preds = []
    current_val_loss=0
    loss=0
    

    for day in range(1, days_in_future + 1):
        progress_callback("XGBRegressor", day, days_in_future,loss, current_val_loss)
        if stop_check():
            Status =(f" User clicked Stop Training.Stopped at day {day}")
            return None, None, None, Status
        model = XGBRegressor(**model_params)
        model.fit(X_train, y_train[f'target_{day}'])
        pred = model.predict(X_test)
        test_preds.append(pred)
        future_preds.append(model.predict(future_X))
        models[f'model_day_{day}'] = model

        train_pred = model.predict(X_train[-1].reshape(1, -1))
        loss = get_score(train_pred.reshape(1, -1), df_org['Close'].iloc[-days_in_future+day-2].reshape(1, -1), loss_function)

        current_val_loss = get_score(pred.reshape(1, -1), y_test[[-days_in_future+day-1]].reshape(1, -1), loss_function)
    test_preds = np.array(test_preds)
    

    score = get_score(test_preds, y_test[0:test_preds.shape[0]], loss_function)
    return test_preds, future_preds, score,Status


def get_score(predicted, real, loss_function):
    if loss_function == "mse":
        score = mean_squared_error(predicted, real)
    elif loss_function == "mae":
        score = mean_absolute_error(predicted, real)
    return score
        