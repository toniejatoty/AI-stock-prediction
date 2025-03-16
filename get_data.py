import yfinance as yf
import pandas as pd
import numpy as np

def get_stock_data(symbol):
    history_prices = get_history_prices(symbol)
    income_statement=get_income_statement(symbol)
    df_stock_full_info_with_nan = merge(history_prices, income_statement)
    df_stock_full_info = delete_Nan_and_useless_col_in_dataframe (df_stock_full_info_with_nan)
    return df_stock_full_info

def get_history_prices(symbol):
    company = yf.Ticker(symbol)
    history_prices = company.history(period="max")
    return history_prices

def get_income_statement(symbol):
    company = yf.Ticker(symbol)
    quarterly_financials = company.quarterly_financials
    return quarterly_financials

def merge(df_stockdata, income_statement):
    df_stockdata['Quarter'] = pd.PeriodIndex(df_stockdata.index, freq='Q')
    income_statement = income_statement.T
    income_statement['Quarter'] = pd.PeriodIndex(income_statement.index, freq='Q')
    income_statement['Quarter'] = income_statement['Quarter'] + 1
    date_index = df_stockdata.index
    df_stockdata=df_stockdata.merge(income_statement, on='Quarter', how='left')
    df_stockdata.index = date_index
    return df_stockdata

def delete_Nan_and_useless_col_in_dataframe(df_stockdata):
    useless_col = ["Tax Rate For Calcs","Quarter", "Tax Effect Of Unusual Items", "Normalized EBITDA", "Normalized Income","Net Interest Income", "Interest Expense",
                    "Interest Income", "Interest Expense Non Operating", "Interest Income Non Operating", "Other Income Expense", "Other Non Operating Income Expenses",
                    "Net Non Operating Interest Income Expense", "Diluted Average Shares", "Basic Average Shares","Net Income From Continuing Operation Net Minority Interest",
                    "Net Income From Continuing And Discontinued Operation", "Net Income Including Noncontrolling Interests", "Net Income Continuous Operations", 
                    "Diluted NI Availto Com Stockholders", "Net Income Common Stockholders", "Tax Provision", "Pretax Income"]
    df_stockdata = df_stockdata.drop(columns=useless_col)
    df_stockdata = df_stockdata.dropna(axis=1, how='all')
    pd.set_option('future.no_silent_downcasting', True)
    df_stockdata = df_stockdata.fillna(0)
    return df_stockdata