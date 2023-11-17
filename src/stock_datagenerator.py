import yfinance as yf
import pandas as pd
import pandas_ta as ta
import os
from finta import TA

startDate = "2021-01-01"
endDate = "2023-11-01"


def get_sp500_tickers():
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    table = pd.read_html(url, header=0)[0]  # Read the first table found
    tickers = table["Symbol"].tolist()
    tickers.append("^GSPC")  # add in S&P500
    tickers = [ticker.replace(".", "-") for ticker in tickers]
    return tickers


def download_data(tickers, start_date, end_date):
    failed_tickers = []

    for ticker in tickers:
        try:
            ticker_data = yf.download(ticker, start=start_date, end=end_date)
            save_data(
                ticker_data,
                f"{ticker}.csv",
            )

        except Exception as e:
            print(f"Failed to download {ticker}: {e}")
            failed_tickers.append(ticker)

    if len(failed_tickers) > 0:
        print("Failed downloads:", failed_tickers)


def save_data(df, filename, directory="data/stocks"):
    """
    Saves the dataframe to a specified folder as a CSV file.
    :param df: DataFrame to save
    :param filename: Name of the file
    :param folder: Folder to save the file in
    """
    # Construct file path
    base_dir = os.path.dirname(
        __file__
    )  # gets the directory where the script is located
    file_path = os.path.join(base_dir, "..", directory, filename)

    # Create folder if it doesn't exist
    if not os.path.exists(os.path.join(base_dir, "..", directory)):
        os.makedirs(os.path.join(base_dir, "..", directory))

    # Save the file
    df.to_csv(file_path)

def LSTM_csv_generation(ticker):
    snp500_data = yf.download(ticker, start=startDate, end=endDate)
    snp500_data["RSI"] = ta.rsi(snp500_data.Close, length=15)
    snp500_data["EMAF"] = ta.ema(snp500_data.Close, length=20)
    snp500_data["EMAM"] = ta.ema(snp500_data.Close, length=100)
    snp500_data["EMAS"] = ta.ema(snp500_data.Close, length=150)
    snp500_data["Target"] = snp500_data["Adj Close"] - snp500_data.Open
    snp500_data["Target"] = snp500_data["Target"].shift(-1)
    snp500_data["TargetClass"] = [
        1 if snp500_data.Target[i] > 0 else 0 for i in range(len(snp500_data))
    ]
    snp500_data["TargetNextClose"] = snp500_data["Adj Close"].shift(-1)
    snp500_data.dropna(inplace=True)
    snp500_data.reset_index(inplace=True)
    snp500_data.drop(["Volume", "Close", "Date"], axis=1, inplace=True)
    snp500_data_set = snp500_data.iloc[:, 0:11]  # .values
    save_data(snp500_data_set, f"{ticker}_data_set_LSTM.csv", directory="data/indicators")

def XGBOOST_csv_generation(ticker):
    df = yf.download(ticker, '2017-01-01', '2021-12-20')
    df['SMA200'] = TA.SMA(df, 200)
    df['RSI'] = TA.RSI(df)
    df['ATR'] = TA.ATR(df)
    df['BBWidth'] = TA.BBWIDTH(df)
    df['Williams'] = TA.WILLIAMS(df)
    df = df.iloc[200:, :]
    df['target'] = df.Close.shift(-1)
    df.dropna(inplace=True)
    save_data(df, f"{ticker}_data_set_XGBOOST.csv", directory="data/indicators")

# Fetch data
# top500_stock_tickers = get_sp500_tickers()
# download_data(top500_stock_tickers, startDate, endDate)

# Generate indicator data
LSTM_csv_generation(ticker="^GSPC")
XGBOOST_csv_generation(ticker="^GSPC")




