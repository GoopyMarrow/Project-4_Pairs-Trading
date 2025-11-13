import yfinance as yf
import pandas as pd
from config import TICKER_LIST, DATA_START_DATE, DATA_END_DATE


def download_price_data(tickers: list = TICKER_LIST, start: str = DATA_START_DATE,
                        end: str = DATA_END_DATE) -> pd.DataFrame:
    print(f"Downloading data for {len(tickers)} tickers from {start} to {end}...")
    try:
        data = yf.download(tickers, start=start, end=end)

        if data.empty:
            print("Error: No data downloaded.")
            return pd.DataFrame()

        adj_close = data['Close']

        adj_close = adj_close.ffill().bfill()

        adj_close = adj_close.dropna(axis=1, how='any')

        print(f"Successfully downloaded data for {len(adj_close.columns)} tickers.")
        return adj_close

    except Exception as e:
        print(f"An error occurred during download: {e}")
        return pd.DataFrame()


def load_data(filepath: str = "data/stock_prices.csv") -> pd.DataFrame:
    try:
        data = pd.read_csv(filepath, index_col='Date', parse_dates=True)
        return data
    except FileNotFoundError:
        print("Data file not found. Downloading...")
        data = download_price_data()
        data.to_csv(filepath)
        return data


if __name__ == "__main__":
    data = download_price_data()
    if not data.empty:
        data.to_csv("data/stock_prices.csv")
        print("Data saved to data/stock_prices.csv")