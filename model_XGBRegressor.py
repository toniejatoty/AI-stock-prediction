import math
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
# it may be confusing but sometimes train_loss is higher that test_loss this is because
# old data is more volatile and harder to learn than new data 

def predict_stock_prices(
df_org,
days_in_future,
days_to_train,
model_params,
loss_function,
stop_check,
progress_callback):

    Status="OK"
    train_loss = 0
    test_loss = 0
    X_train, y_train, X_test, y_test, future_X = get_split_data(df_org, days_to_train, days_in_future)
    X_scaler = RobustScaler()
    y_scaler = RobustScaler()

    X_train = X_scaler.fit_transform(X_train)
    X_test = X_scaler.transform(X_test)
    future_X = X_scaler.transform(future_X)
    y_train_all_values = set()
    for i in range(1, days_in_future + 1):
        for j in range(0, y_train[f'target_{i}'].shape[0]):
            y_train_all_values.add(y_train[f'target_{i}'][j])
    y_scaler.fit(np.array(list(y_train_all_values)).reshape(-1, 1))
    for i in range(1, days_in_future + 1):
        y_train[f'target_{i}'] = y_scaler.transform(y_train[f'target_{i}'].reshape(-1, 1))
        y_test[f'target_{i}'] = y_scaler.transform(y_test[f'target_{i}'].reshape(-1, 1))


    #train and predict 
    #models = {}
    train_preds = []
    test_preds = []
    future_preds = []
    test_loss=0
    train_loss=0
    for day in range(1, days_in_future + 1):
        progress_callback("XGBRegressor", day, days_in_future,train_loss, test_loss)
        if stop_check():
            Status =(f" User clicked Stop Training. Stopped at day {day}")
            return None, None, None, Status
        model = XGBRegressor(**model_params)

        model.fit(X_train, y_train[f'target_{day}'], eval_set=[(X_test, y_test[f'target_{day}'])], verbose=False)
        pred_test = model.predict(X_test)
        test_preds.append(pred_test[-1])
        future_preds.append(model.predict(future_X))
        #models[f'model_day_{day}'] = model
        
        pred_train = model.predict(X_train)
        train_preds.append(pred_train[-1])
        
        test_loss=0
        for i_test in range(0,X_test.shape[0]):
            test_loss = test_loss + get_score(inverse_scaller(pred_test[i_test].reshape(1, -1), y_scaler), inverse_scaller(y_test[f'target_{day}'][i_test].reshape(1, -1), y_scaler), loss_function)
        test_loss = test_loss + test_loss
        test_loss = test_loss / X_test.shape[0]
        train_loss = 0
        for i_train in range(0,X_train.shape[0]):
            train_loss = train_loss + get_score(inverse_scaller(pred_train[i_train].reshape(1, -1), y_scaler), inverse_scaller(y_train[f'target_{day}'][i_train].reshape(1, -1), y_scaler), loss_function)
        train_loss = train_loss + train_loss
        train_loss = train_loss / X_train.shape[0]
   
    train_loss = train_loss / (X_train.shape[0]*days_in_future)
    test_loss = test_loss / (X_test.shape[0]*days_in_future)
    
    test_preds = np.array(test_preds).reshape(-1,1)
    real_vals_test=np.array([y_test[f"target_{day}"][-1] for day in range(1,days_in_future+1)]).reshape(1,-1).flatten()
    test_preds=inverse_scaller(test_preds, y_scaler)
    real_vals_test=inverse_scaller(real_vals_test, y_scaler)
    last_record_test_score = get_score(test_preds,real_vals_test, loss_function) #this will be displayed on the chart in legend
    
    train_preds = np.array(train_preds).reshape(-1,1)
    real_vals_train=np.array([y_train[f"target_{day}"][-1] for day in range(1,days_in_future+1)]).reshape(1,-1).flatten()
    train_preds=inverse_scaller(train_preds, y_scaler)
    real_vals_train=inverse_scaller(real_vals_train, y_scaler)
    last_record_train_score = get_score(train_preds,real_vals_train, loss_function) #this will be displayed on the chart in legend

    #print(train_loss, test_loss)
    return test_preds, inverse_scaller(np.array(future_preds), y_scaler), last_record_test_score,Status, train_loss, test_loss, last_record_train_score


def get_score(predicted, real, loss_function):
    if loss_function == "mse":
        score = mean_squared_error(predicted, real)
    elif loss_function == "mae":
        score = mean_absolute_error(predicted, real)
    return score


def get_split_data(df_org, days_to_train, days_in_future, proc=0.8):
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

    X = np.array(X)
    y = {k: np.array(v) for k, v in y.items()}

    index_train = (len(X) - days_to_train) * proc
    index_train = math.floor(index_train)
    index_test = index_train + days_to_train - 1
    X_train, X_test = X[:index_train], X[index_test:]
    y_train,y_test = {k: v[:index_train] for k, v in y.items()}, {k: v[index_test:] for k, v in y.items()}
    
    future_X = df_org[required_cols].values[-days_to_train:].flatten().reshape(1, -1)

    return X_train, y_train, X_test, y_test, future_X

def inverse_scaller(predictions, scaler):
    return scaler.inverse_transform(predictions.reshape(-1, 1)).flatten()