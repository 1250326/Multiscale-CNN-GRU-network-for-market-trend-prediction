import pandas as pd
import pymysql
import requests
import io
from retrying import retry
from fake_useragent import UserAgent
import time
import random

API_KEY = "<your_api_key>"
us_stock_url = f"https://fmpcloud.io/api/v3/stock-screener?apikey={API_KEY}&country=US"

SQL_CONNECTION_STR = "mysql+pymysql://root:@127.0.0.1:3306/us_stock"

get_yahoo_finance_url = lambda ticker: f"https://query1.finance.yahoo.com/v7/finance/download/{ticker}?period1=0&period2=1648101042&interval=1d&events=history&includeAdjustedClose=true"

# SYMBOL_LIST_SQL = "SELECT DISTINCT stock_info.symbol FROM stock_info WHERE stock_info.symbol not in (select DISTINCT symbol from daily_stock)"
SYMBOL_LIST_SQL = "SELECT DISTINCT symbol FROM stock_info"

conn = pymysql.connect(host = "127.0.0.1", user = "root", db = "us_stock")
cursor = conn.cursor()
cursor.execute(SYMBOL_LIST_SQL)
symbol_list = cursor.fetchall()
symbol_list = [i[0] for i in symbol_list]

header = {
    "accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9", 
    "accept-encoding": "gzip, deflate, br", 
    "accept-language": "zh-TW,zh;q=0.9,en-US;q=0.8,en;q=0.7,zh-CN;q=0.6,ja;q=0.5", 
    "cookie": "<your_cookie>", 
    "dnt": "1", 
    "referer": "https://www.google.com/", 
    "sec-ch-ua": '" Not A;Brand";v="99", "Chromium";v="99", "Google Chrome";v="99"', 
    "sec-ch-ua-mobile": "?0", 
    "sec-ch-ua-platform": '"Windows"', 
    "sec-fetch-dest": "document", 
    "sec-fetch-mode": "navigate", 
    "sec-fetch-site": "same-site", 
    "sec-fetch-user": "?1", 
    "upgrade-insecure-requests": "1", 
    "user-agent": UserAgent().random,
}

@retry(wait_random_min=5000, wait_random_max=15000, stop_max_attempt_number = 2)
def get_yahoo_data(symbol):
    try:
        data_io = io.StringIO(requests.get(get_yahoo_finance_url(symbol), headers=header).text)
        data = pd.read_csv(data_io)
        if data.empty:
            raise Exception()
    except Exception as e:
        print(f"Symbol: {symbol}: \tError in download, terminated.")
        raise
    data["symbol"] = symbol
    data = data.rename(columns={
        "Date" : "date",
        "Open" : "open",
        "High" : "high",
        "Low" : "low",
        "Close" : "close",
        "Adj Close" : "adj_close",
        "Volume" : "volume",
    }).drop_duplicates(subset = ["date"])

    try:
        data.to_sql(name = "daily_stock", con = SQL_CONNECTION_STR, if_exists = "append", index = False)
    except Exception as e:
        print(f"Symbol: {symbol}: \tError in insertion, terminated.")
        print(e)
        raise
    print(f"Symbol: {symbol}: \tData downloaded.")
    return

for sym in symbol_list:
    # if sym <"MRAGX": continue
    try:
        get_yahoo_data(sym)
    except Exception:
        pass
    finally:
        time.sleep(random.random()*0.2)
