import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from model_XGBRegressor import (
        predict_stock_prices,
        get_score,
        get_split_data,
        inverse_scaller)

import pandas as pd
import numpy as np

def test_get_split_data(path):
    df = pd.read_csv(path)
    for _ in range(0,30):
        np.random.seed(42)
        days_to_train = np.random.randint(1, df.shape[0]//2 -1)
        np.random.seed(42)
        days_in_future = np.random.randint(1, df.shape[0]-2*days_to_train)
        proc=0.8
        X_train, y_train, X_test, y_test, future_X = get_split_data(df, days_to_train, days_in_future,proc=0.8)

        # Check the shapes of the returned arrays
        columns = df.columns

        assert future_X.shape[1] == days_to_train * len(columns)
        assert future_X.shape[0] == 1

        #print(len(X_train), len(X_test))
        assert X_train.shape[0]/(X_test.shape[0]) >=1
        assert X_train.shape[1] == days_to_train * len(columns)
        assert X_test.shape[1] == days_to_train * len(columns)
        assert len(y_train.keys()) == days_in_future
        assert len(y_test.keys()) == days_in_future

        
        for i in range(1, days_in_future + 1):
            assert f'target_{i}' in y_train.keys()
            assert f'target_{i}' in y_test.keys()
            assert y_train[f'target_{i}'].shape[0] == X_train.shape[0]
            assert y_test[f'target_{i}'].shape[0] == X_test.shape[0]

        X_train_indexes = set()
        for i in X_train:
            for j in range(0,days_to_train):
                X_train_indexes.add(i[j*len(columns)])
        assert len(X_train_indexes) == len(X_train)+days_to_train-1
        
        X_test_indexes = set()
        for i in X_test:
            for j in range(0,days_to_train):
                X_test_indexes.add(i[j*len(columns)])
        assert len(X_test_indexes) == len(X_test)+days_to_train-1

        assert X_train_indexes.isdisjoint(X_test_indexes)

        for j in range(1,days_in_future):
            for i in range(0, X_train.shape[0]-1):
                assert y_train[f'target_{j}'][i+1] == y_train[f'target_{j+1}'][i]

        for j in range(1,days_in_future):
            for i in range(0, X_test.shape[0]-1):
                assert y_test[f'target_{j}'][i+1] == y_test[f'target_{j+1}'][i]

    print("All tests passed!")

if __name__ == "__main__":
    test_get_split_data('C:\\Users\\konra\\Desktop\\wielki projekt\\pytest\\test_data_1110_records.csv')
    test_get_split_data('C:\\Users\\konra\\Desktop\\wielki projekt\\pytest\\test_data_100_records.csv')
    test_get_split_data('C:\\Users\\konra\\Desktop\\wielki projekt\\pytest\\test_data_20_records.csv')

