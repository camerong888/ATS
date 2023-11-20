import yfinance as yf
import pandas as pd
import pandas_ta as ta
import os
from finta import TA

# Hyperparameters
ticker = '^GSPC'
startDate = "2000-01-01"
endDate = "2023-11-01"


def get_sp500_tickers():
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    table = pd.read_html(url, header=0)[0]  # Read the first table found
    tickers = table["Symbol"].tolist()
    tickers.append("^GSPC")  # add in S&P500
    tickers = [ticker.replace(".", "-") for ticker in tickers]
    return tickers

# CBOE Volatility Index (^VIX)
def addVIX(df):
    VIX_data = yf.download("^VIX", start=startDate, end=endDate)
    df["^VIX_Close"] = VIX_data["Close"]
    return df

# United States Oil Fund, LP (USO)
def addUSO(df):
    USO_data = yf.download("USO", start=startDate, end=endDate)
    df["USO_Close"] = USO_data["Close"]
    return df

# CBOE Interest Rate 10 Year T No (^TNX)
def addTNX(df):
    TNX_data = yf.download("^TNX", start=startDate, end=endDate)
    df["^TNX_Close"] = TNX_data["Close"]
    return df

# Energy Select Sector SPDR Fund (XLE)
def addXLE(df):
    XLE_data = yf.download("XLE", start=startDate, end=endDate)
    df["XLE_Close"] = XLE_data["Close"]
    return df

# CBOE Interest Rate 10 Year T No (^TNX)
def addSSE(df):
    SSE_data = yf.download("000001.SS", start=startDate, end=endDate)
    df["SSE_Close"] = SSE_data["Close"]
    return df


def download_data(tickers):
    failed_tickers = []

    for ticker in tickers:
        try:
            ticker_data = yf.download(ticker, start=startDate, end=endDate)
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
    df = yf.download(ticker, start=startDate, end=endDate)
    df['SMA10'] = TA.SMA(df, 10)
    df['SMA20'] = TA.SMA(df, 20)
    df['SMA30'] = TA.SMA(df, 30)
    df['SMA50'] = TA.SMA(df, 50)
    df['SMA200'] = TA.SMA(df, 200)
    df['SMA10_derivative'] = df['SMA10'].diff()
    df['SMA20_derivative'] = df['SMA20'].diff()
    df['SMA30_derivative'] = df['SMA30'].diff()
    df['SMA50_derivative'] = df['SMA50'].diff()
    df['SMA200_derivative'] = df['SMA200'].diff()
    df['EMA10'] = TA.EMA(df, 10)
    df['EMA20'] = TA.EMA(df, 20)
    df['EMA30'] = TA.EMA(df, 30)
    df['EMA50'] = TA.EMA(df, 50)
    df['EMA10_derivative'] = df['EMA10'].diff()
    df['EMA20_derivative'] = df['EMA20'].diff()
    df['EMA30_derivative'] = df['EMA30'].diff()
    df['EMA50_derivative'] = df['EMA50'].diff()
    df['RSI'] = TA.RSI(df)
    df['ATR'] = TA.ATR(df)
    df['BBWidth'] = TA.BBWIDTH(df)
    df['Williams'] = TA.WILLIAMS(df)
    df['MACD'] = TA.MACD(df)['MACD']
    df['VWAP'] = ta.vwap(df.High, df.Low, df.Close, df.Volume)
    df['StochasticOscillator'] = TA.STOCH(df)
    df['CCI'] = TA.CCI(df)
    df['OBV'] = TA.OBV(df)
    df['ParabolicSAR'] = TA.SAR(df)
    df["AO"] = ta.ao(df.High, df.Low)
    df["MOM"] = ta.mom(df.Close, length=16)
    df["BOP"] = ta.bop(df.Open, df.High, df.Low, df.Close, length=16)
    df["RVI"] = ta.rvi(df.Close)
    a = ta.dm(df.High, df.Low, length=16)
    df = df.join(a)
    a = ta.macd(df.Close)
    df = df.join(a)
    a = ta.stoch(df.High, df.Low, df.Close)
    df = df.join(a)
    a = ta.stochrsi(df.Close, length=16)
    df = df.join(a)

    df = addVIX(df)
    df = addTNX(df)
    df = addUSO(df)
    df = addXLE(df)
    df = addSSE(df)

    df = df.iloc[200:, :]
    df['Target'] = df.Close.shift(-1)
    df.dropna(inplace=True)
    save_data(df, f"{ticker}_data_set_XGBOOST.csv", directory="data/indicators")


# Fetch data
# top500_stock_tickers = get_sp500_tickers()
# download_data(top500_stock_tickers, startDate, endDate)
#LSTM_csv_generation(ticker=f"ticker")
XGBOOST_csv_generation(ticker)
