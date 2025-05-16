import math
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
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
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df_org)
    df_scaled = pd.DataFrame(scaled_data, columns=df_org.columns)
    X_train, y_train, X_test, y_test, future_X = get_split_data(df_scaled, days_to_train, days_in_future)
    
    #train and predict 
    models = {}
    test_preds = []
    future_preds = []
    current_val_loss=0
    loss=0
    
    for day in range(1, days_in_future + 1):
        progress_callback("XGBRegressor", day, days_in_future,loss, current_val_loss)
        if stop_check():
            Status =(f" User clicked Stop Training. Stopped at day {day}")
            return None, None, None, Status
        model = XGBRegressor(**model_params)

        model.fit(X_train, y_train[f'target_{day}'], eval_set=[(X_test, y_test[f'target_{day}'])], verbose=False)
        pred = model.predict(X_test)
        test_preds.append(pred[-1])
        future_preds.append(model.predict(future_X))
        models[f'model_day_{day}'] = model

        train_pred = model.predict(X_train[-1].reshape(1,-1))
        train_true_values= np.array([y_train[f'target_{day}'][-1]])
        loss = get_score(train_pred, train_true_values, loss_function)
        current_val_loss=0
        for day2 in range(0,X_test.shape[0]):
            current_val_loss = current_val_loss + get_score(pred[day2].reshape(1, -1), y_test[f'target_{day}'][day2].reshape(1, -1), loss_function)
        current_val_loss = current_val_loss / X_test.shape[0]
    test_preds = np.array(test_preds).reshape(-1,1)
    real_vals=np.array([y_test[f"target_{day}"][-1] for day in range(1,days_in_future+1)]).reshape(1,-1).flatten()
    test_preds=inverse_scaller(test_preds, df_org, scaler)
    real_vals=inverse_scaller(real_vals, df_org, scaler)
    score = get_score(test_preds,real_vals, loss_function)
    return test_preds, inverse_scaller(np.array(future_preds), df_org, scaler), score,Status


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

    X = []
    y = {f'target_{i}': [] for i in range(1, days_in_future + 1)}

    for i in range(days_to_train, df.shape[0]+1):
        X.append(df[required_cols].iloc[i - days_to_train:i].values.flatten())
        for j in range(1, days_in_future + 1):
            y[f'target_{j}'].append(df[f'target_{j}'].iloc[i-1])
    
    proc=0.8
    index_train = ((len(X)-2)-days_to_train+1) *proc   
    X = np.array(X)
    y = {k: np.array(v) for k, v in y.items()}

    index_train = math.ceil(index_train)
    index_test = index_train+days_to_train
    index_train=index_train+1
    X_train, X_test = X[:index_train], X[index_test:]
    y_train,y_test = {k: v[:index_train] for k, v in y.items()}, {k: v[index_test:] for k, v in y.items()}
    
    future_X = df_org[required_cols].values[-days_to_train:].flatten().reshape(1, -1)
    # print("XGBR")
    # print(f"X={X}")
    # print(f"len(X)={len(X)}")
    # print(f"day_train={days_to_train}")
    # print(f"future_days={days_in_future}")
    # print(f"df.shape[0]={df.shape[0]}")
    # print(f"index={index_train}")
    # print(f"X_train={X_train}")
    # print(f"X_test={X_test}")
    # print(f"y_train={y_train}")
    # print(f"y_test={y_test}")
    return X_train, y_train, X_test, y_test, future_X

def inverse_scaller(predictions, df, scaler):
    temp_array = np.zeros((predictions.shape[0], df.shape[1]))
    temp_array[:, df.columns.get_loc("Close")] = predictions.flatten()
    predictions_original = scaler.inverse_transform(temp_array)
    predictions_close = predictions_original[:, df.columns.get_loc("Close")]
    return predictions_close