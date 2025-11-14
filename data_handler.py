import yfinance as yf
import pandas as pd
from config import TICKER_LIST, DATA_START_DATE, DATA_END_DATE

"""
Data Handling Module

Responsible for downloading price data from yfinance and loading
it from a local CSV cache.
"""

def download_price_data(tickers: list = TICKER_LIST, start: str = DATA_START_DATE,
                        end: str = DATA_END_DATE) -> pd.DataFrame:
    """
    Downloads historical 'Close' price data from yfinance for a list of tickers.

    Args:
        tickers: A list of ticker symbols to download.
        start: The start date for the data.
        end: The end date for the data.

    Returns:
        A DataFrame with dates as the index and tickers as columns.
    """
    print(f"Downloading data for {len(tickers)} tickers from {start} to {end}...")
    try:
        data = yf.download(tickers, start=start, end=end)

        if data.empty:
            print("Error: No data downloaded.")
            return pd.DataFrame()

        # We only need the 'Close' prices
        adj_close = data['Close']

        # Fill missing values (e.g., holidays)
        adj_close = adj_close.ffill().bfill()

        # Drop any tickers that still have NaN values (no data)
        adj_close = adj_close.dropna(axis=1, how='any')

        print(f"Successfully downloaded data for {len(adj_close.columns)} tickers.")
        return adj_close

    except Exception as e:
        print(f"An error occurred during download: {e}")
        return pd.DataFrame()


def load_data(filepath: str = "data/stock_prices.csv") -> pd.DataFrame:
    """
    Loads price data from a local CSV file.
    If the file is not found, it triggers a new download.

    Args:
        filepath: The path to the local CSV cache.

    Returns:
        A DataFrame containing the price data.
    """
    try:
        # Attempt to load from cache
        data = pd.read_csv(filepath, index_col='Date', parse_dates=True)
        return data
    except FileNotFoundError:
        # If cache not found, download and create it
        print("Data file not found. Downloading...")
        data = download_price_data()
        data.to_csv(filepath)
        return data


if __name__ == "__main__":
    """Script entry point to manually refresh the data cache."""
    data = download_price_data()
    if not data.empty:
        data.to_csv("data/stock_prices.csv")
        print("Data saved to data/stock_prices.csv")