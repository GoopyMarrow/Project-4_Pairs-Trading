import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

plt.style.use('seaborn-v0_8-darkgrid')


def plot_pair_prices(data: pd.DataFrame, ticker1: str, ticker2: str):
    plt.figure(figsize=(14, 7))

    data[ticker1].plot(label=ticker1, color='blue', alpha=0.9)
    data[ticker2].plot(label=ticker2, color='orange', alpha=0.9)

    plt.title(f'Historical Prices ({ticker1} & {ticker2}) - Test Set')
    plt.ylabel('Price ($)')
    plt.xlabel('Date')
    plt.legend()
    plt.show()


def plot_normalized_prices(data: pd.DataFrame, ticker1: str, ticker2: str):
    plt.figure(figsize=(14, 7))

    # Normaliza los datos (Z-score) para ponerlos en la misma escala
    norm_data = (data - data.mean()) / data.std()

    norm_data[ticker1].plot(label=ticker1, color='blue', alpha=0.9)
    norm_data[ticker2].plot(label=ticker2, color='orange', alpha=0.9)

    plt.title(f'Normalized Historical Prices ({ticker1} & {ticker2}) - Test Set')
    plt.ylabel('Normalized Price (Z-Score)')
    plt.xlabel('Date')
    plt.legend()
    plt.show()


def plot_portfolio_value(portfolio_value: list, dates: pd.DatetimeIndex, ticker1: str, ticker2: str, threshold: float):
    plt.figure(figsize=(14, 7))
    pd.Series(portfolio_value, index=dates, name="Portfolio Value").plot(color='blue')
    plt.title(f'Portfolio Value Over Time ({ticker1} & {ticker2})\nStd: {threshold:.2f}')
    plt.ylabel('Portfolio Value ($)')
    plt.xlabel('Date')
    plt.show()


def plot_dynamic_spread(vecm_norm_history: list, dates: pd.DatetimeIndex, threshold: float):
    plt.figure(figsize=(14, 7))
    pd.Series(vecm_norm_history, index=dates, name="VECM Norm").plot(color='purple', label='Normalized VECM Signal')
    plt.axhline(threshold, color='red', linestyle='--', label=f'+{threshold:.2f} Std (Entry)')
    plt.axhline(-threshold, color='green', linestyle='--', label=f'-{threshold:.2f} Std (Entry)')
    plt.axhline(0, color='black', linestyle='-', linewidth=0.5)
    plt.title('Dynamic VECM Spread Signal and Trading Thresholds')
    plt.ylabel('Z-Score')
    plt.xlabel('Date')
    plt.legend()
    plt.show()


def plot_hedge_ratio(hedge_ratio_history: list, dates: pd.DatetimeIndex):
    plt.figure(figsize=(14, 7))
    pd.Series(hedge_ratio_history, index=dates, name="Hedge Ratio").plot(color='orange')
    plt.title('Dynamic Hedge Ratio (KF1) Over Time')
    plt.ylabel('Hedge Ratio (B1)')
    plt.xlabel('Date')
    plt.show()


def plot_trades_on_price(data: pd.DataFrame, ticker1: str, ticker2: str, all_positions: list):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

    data[ticker1].plot(ax=ax1, label='Price', color='gray')
    data[ticker2].plot(ax=ax2, label='Price', color='gray')

    ax1.set_title(f'Trades for {ticker1}')
    ax2.set_title(f'Trades for {ticker2}')
    ax1.set_ylabel('Price')
    ax2.set_ylabel('Price')

    for pos in all_positions:
        if pos.ticker == ticker1:
            ax = ax1
        elif pos.ticker == ticker2:
            ax = ax2
        else:
            continue

        if pos.position_type == 'long':
            ax.plot(pos.entry_date, pos.entry_price, '^', color='green', markersize=10, label='Long Entry')
            if pos.exit_date:
                ax.plot(pos.exit_date, pos.exit_price, 'x', color='green', markersize=10, label='Long Exit')
        elif pos.position_type == 'short':
            ax.plot(pos.entry_date, pos.entry_price, 'v', color='red', markersize=10, label='Short Entry')
            if pos.exit_date:
                ax.plot(pos.exit_date, pos.exit_price, 'x', color='red', markersize=10, label='Short Exit')

    handles, labels = ax1.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax1.legend(by_label.values(), by_label.keys())

    handles, labels = ax2.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax2.legend(by_label.values(), by_label.keys())

    plt.tight_layout()
    plt.show()


def plot_returns_distribution(all_positions: list):
    if not all_positions:
        print("No positions to plot returns distribution.")
        return

    returns = [pos.pnl / (pos.n_shares * pos.entry_price) for pos in all_positions if pos.n_shares > 0]

    plt.figure(figsize=(10, 6))
    sns.histplot(returns, bins=50, kde=True, color='blue')
    plt.title('Distribution of PnL per Trade')
    plt.xlabel('Return')
    plt.ylabel('Frequency')
    plt.axvline(0, color='black', linestyle='--')
    plt.show()


def plot_vecm_signals(observed_signal: list, filtered_signal: list, dates: pd.DatetimeIndex):
    plt.figure(figsize=(14, 7))

    observed_series = pd.Series(observed_signal, index=dates, name='Vecm (Observed)')
    filtered_series = pd.Series(filtered_signal, index=dates, name='Vecm Hat (Filtered)')

    observed_series.plot(color='gray', alpha=0.7, label='Vecm (Observed)')
    filtered_series.plot(color='blue', linewidth=2, label='Vecm Hat (Filtered)')

    plt.title('VECM Signal vs. Kalman-Filtered Signal (KF2)')
    plt.ylabel('Signal Value (Spread)')
    plt.xlabel('Date')
    plt.legend()
    plt.show()