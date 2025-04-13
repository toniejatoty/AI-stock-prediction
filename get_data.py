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
    df_stock_full_info = drop_columns_with_only_zeros_and_nan(df_stock_full_info)
    df_stock_full_info = drop_very_historic_data(df_stock_full_info, 0)
    return df_stock_full_info


def validade_params(symbol, start_date):
    try:
        company = yf.Ticker(symbol)
        hist = company.history(period="max")

        if hist.empty:
            raise ValueError(f"Can't find data for that ticker: {symbol}")

        available_start = pd.to_datetime(hist.index[0]).tz_localize(None)

        input_date = pd.to_datetime(start_date)
        if pd.isna(input_date):
            return available_start

        if input_date < pd.Timestamp.min:
            return available_start

        input_date = input_date.tz_localize(None)

        return max(input_date, available_start)

    except Exception as e:
        raise ValueError(f"{str(e)}")



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
    useless_col = [
        "Tax Rate For Calcs",
        "Quarter",
        "Tax Effect Of Unusual Items",
        "Normalized EBITDA",
        "Normalized Income",
        "Net Interest Income",
        "Interest Expense",
        "Interest Income",
        "Interest Expense Non Operating",
        "Interest Income Non Operating",
        "Other Income Expense",
        "Other Non Operating Income Expenses",
        "Net Non Operating Interest Income Expense",
        "Diluted Average Shares",
        "Basic Average Shares",
        "Net Income From Continuing Operation Net Minority Interest",
        "Net Income From Continuing And Discontinued Operation",
        "Net Income Including Noncontrolling Interests",
        "Net Income Continuous Operations",
        "Diluted NI Availto Com Stockholders",
        "Net Income Common Stockholders",
        "Tax Provision",
        "Pretax Income",
        "Reconciled Depreciation",
        "Total Other Finance Cost",
        "Operating Expense",
    ]
    df_stockdata = df_stockdata.drop(columns=useless_col, errors="ignore")

    close_corr = df_stockdata.corr()["Close"]
    filtered_corr = close_corr[(close_corr >= -0.5) & (close_corr <= 0.5)]
    df_stockdata = df_stockdata.drop(columns=filtered_corr.index)
    return df_stockdata


def drop_columns_with_only_zeros_and_nan(df_stockdata):
    df_stockdata = df_stockdata.dropna(axis=1, how="all")
    pd.set_option("future.no_silent_downcasting", True)
    df_stockdata = df_stockdata.fillna(0)
    columns_to_drop = df_stockdata.columns[(df_stockdata == 0).all()]
    df_stockdata = df_stockdata.drop(columns=columns_to_drop)
    return df_stockdata


def drop_very_historic_data(df_stockdata, percentage_how_much_delete):
    how_much_days_delete = percentage_how_much_delete * df_stockdata.shape[0] / 100
    how_much_days_delete = int(how_much_days_delete)
    df_stockdata = df_stockdata.iloc[how_much_days_delete:]
    return df_stockdata
