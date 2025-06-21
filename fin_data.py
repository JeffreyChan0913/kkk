import yfinance
from datetime import date
import pandas as pd

def get(ticker, PERIOD='max', close_to_return=True, return_on_close=False):
    if isinstance(ticker, list):
        package = {}
        for tick in ticker:
            package[tick] = get(tick)
        return package

    data         = yfinance.download(ticker, period=PERIOD, auto_adjust=False)
    data.columns = data.columns.get_level_values(0).to_list()
    data         = data.reset_index()

    if close_to_return:
        data['ret'] = data.Close.diff() / data.Close.shift(1) if return_on_close else data['Adj Close'].diff() / data['Adj Close'].shift(1)

    return data 

def get_yield(drop_features, date = date.today().strftime('%m_%d_%Y'), root = False):
    if root == False:
        ROOT = './'
    else:
        ROOT = "../../quant_data/" 

    new_yield = pd.read_csv(ROOT + date + "_daily-treasury-rates.csv")
    old_yield = pd.read_csv(ROOT + "yield-curve-rates-1990-2024.csv")
   
    DROP = set(drop_features) & set(new_yield.columns)
    if len(DROP):
        new_yield = new_yield.drop(columns=DROP)

    DROP = set(drop_features) & set(old_yield.columns)

    if len(DROP):
        old_yield = old_yield.drop(columns=DROP)

    package = pd.concat([new_yield, old_yield])

    package.Date = pd.to_datetime(package.Date, format='mixed')

    return package

def csv(tick):
    df = pd.read_csv(tick+'.csv', index_col = 0)
    df.Date = pd.to_datetime(df.Date, format='mixed')
    return df
        
