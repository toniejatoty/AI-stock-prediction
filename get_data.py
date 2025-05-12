import yfinance as yf
import pandas as pd

def get_stock_data(symbol, start_date):
    start_date = validade_params(symbol, start_date)
    history_prices = get_history_prices(symbol, start_date)
    income_statement = get_income_statement(symbol)
    if income_statement.empty:
        df_stock_full_info = history_prices
    else:
        df_stock_full_info_with_nan = merge(history_prices, income_statement)
        df_stock_full_info = delete_useless_col_in_dataframe(
            df_stock_full_info_with_nan
        )
    df_stock_full_info = drop_columns_with_only_nan_and_much_zeros(df_stock_full_info)
    #df_stock_full_info = df_stock_full_info.iloc[:-20]
    return df_stock_full_info


def validade_params(symbol, start_date):
    company = yf.Ticker(symbol)
    hist = company.history(period="max")
    if hist.empty:
        raise ValueError(f"Can't find data for that ticker: {symbol}")
    
    available_start = pd.to_datetime(hist.index[0]).tz_localize(None)
    try:
        start_date=pd.to_datetime(start_date)
    except:
        raise ValueError(f"You provided unapropriate start date. Format is YYYY-MM-DD")
    
    today = pd.to_datetime("today").normalize()
    business_days_diff = len(pd.bdate_range(start_date, today))
    if(business_days_diff <=0):
        raise ValueError(f"Please give past Start date")
    if(business_days_diff <=10):
        raise ValueError(f"Please give wider Start date")
    return max(start_date, available_start)


def get_history_prices(symbol, start_date):
    company = yf.Ticker(symbol)
    history_prices = company.history(start=start_date)
    return history_prices

def get_income_statement(symbol):
    company = yf.Ticker(symbol)
    quarterly_financials = company.quarterly_financials
    return quarterly_financials


def merge(df_stockdata, income_statement):
    df_stockdata["Quarter"] = pd.PeriodIndex(df_stockdata.index, freq="Q")
    income_statement = income_statement.T
    income_statement["Quarter"] = pd.PeriodIndex(income_statement.index, freq="Q")
    income_statement["Quarter"] = income_statement["Quarter"] + 1
    date_index = df_stockdata.index
    df_stockdata = df_stockdata.merge(income_statement, on="Quarter", how="left")
    df_stockdata.index = date_index
    df_stockdata.drop(columns="Quarter", inplace=True)
    df_stockdata = df_stockdata.astype(float)
    return df_stockdata


def delete_useless_col_in_dataframe(df_stockdata):
    close_corr = df_stockdata.corr()["Close"]
    filtered_corr = close_corr[(close_corr >= -0.55) & (close_corr <= 0.55)]
    df_stockdata = df_stockdata.drop(columns=filtered_corr.index)
    return df_stockdata


def drop_columns_with_only_nan_and_much_zeros(df_stockdata):
    proc=0.2
    df_stockdata = df_stockdata.dropna(axis=1, how="all")
    pd.set_option("future.no_silent_downcasting", True)
    df_stockdata = df_stockdata.fillna(0)
    for col in df_stockdata.columns:
        sum_zero=(df_stockdata[col]==0).sum()
        if sum_zero/df_stockdata.shape[1] > proc:
            df_stockdata=df_stockdata.drop(columns=col)
    return df_stockdata
