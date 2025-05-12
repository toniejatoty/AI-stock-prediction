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

    X_train, y_train, X_test, y_test, future_X = get_split_data(df_org, days_to_train, days_in_future)
    
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
        model.fit(X_train, y_train[f'target_{day}'], eval_set=[(X_test, y_test[[-days_in_future+day-1]].reshape(1, -1))])
        pred = model.predict(X_test)
        test_preds.append(pred)
        future_preds.append(model.predict(future_X))
        models[f'model_day_{day}'] = model

        train_pred = model.predict(X_train[-1].reshape(1, -1))
        loss = get_score(train_pred.flatten(), df_org['Close'].iloc[-days_to_train-days_in_future+day-1].flatten(), loss_function)
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


def get_split_data(df_org, days_to_train, days_in_future):
    df = df_org.copy()
    required_cols = df.columns
    for i in range(1, days_in_future + 1):
        new_columns = {f'target_{i}': df['Close'].shift(-i)}
        df = df.assign(**new_columns)
    df.dropna(inplace=True)
    
    df_training = df.iloc[0:-days_to_train]
    df_test =df.iloc[-days_to_train:]
    X_train = []
    y_train = {f'target_{i}': [] for i in range(1, days_in_future + 1)}

    for i in range(days_to_train, len(df_training)):
        X_train.append(df_training[required_cols].iloc[i - days_to_train:i].values.flatten())
        for j in range(1, days_in_future + 1):
            y_train[f'target_{j}'].append(df_training[f'target_{j}'].iloc[i])
    
    X_train = np.array(X_train)
    y_train = {k: np.array(v) for k, v in y_train.items()}
    X_test = np.array(df_test[required_cols]).flatten().reshape(1,-1)
    y_test = np.array(df_org['Close'].iloc[-days_in_future:])
    
    future_X = df_org[required_cols].values[-days_to_train:].flatten().reshape(1, -1)
    return X_train, y_train, X_test, y_test, future_X