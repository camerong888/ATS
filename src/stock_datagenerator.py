import yfinance as yf
import pandas as pd
import os

startDate = '2021-01-01'
endDate = '2023-11-01'

def get_sp500_tickers():
    url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    table = pd.read_html(url, header=0)[0]  # Read the first table found
    tickers = table['Symbol'].tolist()
    tickers.append('^GSPC') #add in S&P500
    tickers = [ticker.replace('.', '-') for ticker in tickers]
    return tickers

def download_data(tickers, start_date, end_date):
    failed_tickers = []

    for ticker in tickers:
        try:
            ticker_data = yf.download(ticker, start=start_date, end=end_date)            
            save_data(ticker_data, f"{ticker}.csv", )
        
        except Exception as e:
            print(f"Failed to download {ticker}: {e}")
            failed_tickers.append(ticker)

    if len(failed_tickers) > 0:
        print("Failed downloads:", failed_tickers)

def save_data(df, filename, directory='data/stocks'):
    """
    Saves the dataframe to a specified folder as a CSV file.
    :param df: DataFrame to save
    :param filename: Name of the file
    :param folder: Folder to save the file in
    """
    # Construct file path
    base_dir = os.path.dirname(__file__)  # gets the directory where the script is located
    file_path = os.path.join(base_dir, '..', directory, filename)

    # Create folder if it doesn't exist
    if not os.path.exists(os.path.join(base_dir, '..', directory)):
        os.makedirs(os.path.join(base_dir, '..', directory))

    # Save the file
    df.to_csv(file_path)

# Fetch data
top500_stock_tickers = get_sp500_tickers()
download_data(top500_stock_tickers, startDate, endDate)


#recurrent neural network takes trends and more recent days into account

