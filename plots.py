import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

"""
Plotting Module

Contains all functions used to visualize the backtest results.
This module is configured with a custom 8-color theme.
"""

# --- Custom Plotting Theme & Color Palette ---
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = [12, 7]
plt.rcParams['axes.grid'] = True
plt.rcParams['axes.prop_cycle'] = plt.cycler(
    color=["skyblue", "mediumpurple", "royalblue", "mediumslateblue",
           "steelblue", "blueviolet", "navy", "indigo"]
)


def plot_pair_prices(data: pd.DataFrame, ticker1: str, ticker2: str):
    """Plots the raw historical prices of the selected pair."""
    plt.figure(figsize=(14, 7))

    data[ticker1].plot(label=ticker1, alpha=0.9)
    data[ticker2].plot(label=ticker2, alpha=0.9)

    plt.title(f'Historical Prices ({ticker1} & {ticker2}) - Test Set')
    plt.ylabel('Price ($)')
    plt.xlabel('Date')
    plt.legend()
    plt.show()


def plot_normalized_prices(data: pd.DataFrame, ticker1: str, ticker2: str):
    """Plots the Z-score normalized prices to visualize relative movements."""
    plt.figure(figsize=(14, 7))

    norm_data = (data - data.mean()) / data.std()

    norm_data[ticker1].plot(label=ticker1, alpha=0.9)
    norm_data[ticker2].plot(label=ticker2, alpha=0.9)

    plt.title(f'Normalized Historical Prices ({ticker1} & {ticker2}) - Test Set')
    plt.ylabel('Normalized Price (Z-Score)')
    plt.xlabel('Date')
    plt.legend()
    plt.show()


def plot_portfolio_value(portfolio_value: list, dates: pd.DatetimeIndex, ticker1: str, ticker2: str, threshold: float):
    """Plots the equity curve (portfolio value over time)."""
    plt.figure(figsize=(14, 7))
    pd.Series(portfolio_value, index=dates, name="Portfolio Value").plot()
    plt.title(f'Portfolio Value Over Time ({ticker1} & {ticker2})\nStd: {threshold:.2f}')
    plt.ylabel('Portfolio Value ($)')
    plt.xlabel('Date')
    plt.show()


def plot_dynamic_spread(vecm_norm_history: list, dates: pd.DatetimeIndex, threshold: float):
    """Plots the final Z-score trading signal (from KF2) and the entry thresholds."""
    plt.figure(figsize=(14, 7))
    pd.Series(vecm_norm_history, index=dates, name="VECM Norm").plot(label='Normalized VECM Signal')

    plt.axhline(threshold, color='red', linestyle='--', label=f'+{threshold:.2f} Std (Entry)')
    plt.axhline(-threshold, color='green', linestyle='--', label=f'-{threshold:.2f} Std (Entry)')
    plt.axhline(0, color='black', linestyle='-', linewidth=0.5)

    plt.title('Dynamic VECM Spread Signal and Trading Thresholds')
    plt.ylabel('Z-Score')
    plt.xlabel('Date')
    plt.legend()
    plt.show()


def plot_hedge_ratio(hedge_ratio_history: list, dates: pd.DatetimeIndex):
    """Plots the dynamic hedge ratio (B1) over time from KF1."""
    plt.figure(figsize=(14, 7))
    pd.Series(hedge_ratio_history, index=dates, name="Hedge Ratio").plot()
    plt.title('Dynamic Hedge Ratio (KF1) Over Time')
    plt.ylabel('Hedge Ratio (B1)')
    plt.xlabel('Date')
    plt.show()


def plot_trades_on_price(data: pd.DataFrame, ticker1: str, ticker2: str, all_positions: list):
    """
    Plots the price series for each asset and overlays the entry/exit
    points for all trades.
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

    data[ticker1].plot(ax=ax1, label='Price', color='gray')
    data[ticker2].plot(ax=ax2, label='Price', color='gray')

    ax1.set_title(f'Trades for {ticker1}')
    ax2.set_title(f'Trades for {ticker2}')
    ax1.set_ylabel('Price')
    ax2.set_ylabel('Price')

    # Plot trades
    for pos in all_positions:
        if pos.ticker == ticker1:
            ax = ax1
        elif pos.ticker == ticker2:
            ax = ax2
        else:
            continue

        # Semantic colors for long/short
        if pos.position_type == 'long':
            ax.plot(pos.entry_date, pos.entry_price, '^', color='green', markersize=10, label='Long Entry')
            if pos.exit_date:
                ax.plot(pos.exit_date, pos.exit_price, 'x', color='green', markersize=10, label='Long Exit')
        elif pos.position_type == 'short':
            ax.plot(pos.entry_date, pos.entry_price, 'v', color='red', markersize=10, label='Short Entry')
            if pos.exit_date:
                ax.plot(pos.exit_date, pos.exit_price, 'x', color='red', markersize=10, label='Short Exit')

    # Consolidate legends to avoid duplicates
    handles, labels = ax1.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax1.legend(by_label.values(), by_label.keys())

    handles, labels = ax2.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax2.legend(by_label.values(), by_label.keys())

    plt.tight_layout()
    plt.show()


def plot_returns_distribution(all_positions: list):
    """Plots a histogram of the PnL (as a % return) for all closed trades."""
    if not all_positions:
        print("No positions to plot returns distribution.")
        return

    returns = [pos.pnl / (pos.n_shares * pos.entry_price) for pos in all_positions if pos.n_shares > 0 and pos.pnl != 0]

    plt.figure(figsize=(10, 6))
    if returns:
        sns.histplot(returns, bins=50, kde=True)
        plt.title('Distribution of PnL per Trade (% Return)')
        plt.xlabel('Return')
    else:
        plt.title('No Completed Trades to Analyze')
        plt.xlabel('Return')

    plt.ylabel('Frequency')
    plt.axvline(0, color='black', linestyle='--')
    plt.show()


def plot_vecm_signals(observed_signal: list, filtered_signal: list, dates: pd.DatetimeIndex):
    """
    Plots the "observed" (noisy) VECM signal vs. the Kalman-Filtered (smoothed)
    VECM signal from KF2.
    """
    plt.figure(figsize=(14, 7))

    observed_series = pd.Series(observed_signal, index=dates, name='Vecm (Observed)')
    filtered_series = pd.Series(filtered_signal, index=dates, name='Vecm Hat (Filtered)')

    observed_series.plot(color='gray', alpha=0.7, label='Vecm (Observed)')
    filtered_series.plot(linewidth=2, label='Vecm Hat (Filtered)')

    plt.title('VECM Signal vs. Kalman-Filtered Signal (KF2)')
    plt.ylabel('Signal Value (Spread)')
    plt.xlabel('Date')
    plt.legend()
    plt.show()


def plot_dynamic_eigenvectors(e1_history: list, e2_history: list, dates: pd.DatetimeIndex):
    """
    Plots the evolution of the two state components [e1, e2] from the
    Kalman Filter 2 (KF2). Uses the first two colors from the cycler.
    """
    plt.figure(figsize=(14, 7))
    # Traducido:
    pd.Series(e1_history, index=dates, name="Component 1 (e1)").plot(label='Component 1 (e1)')
    pd.Series(e2_history, index=dates, name="Component 2 (e2)").plot(label='Component 2 (e2)')

    plt.title('Eigenvector Component Evolution (KF2)')
    plt.xlabel('Date')
    plt.ylabel('Component Value')
    plt.legend()
    plt.show()


def plot_kf1_spread(spread_history: list, dates: pd.DatetimeIndex, ticker1: str, ticker2: str):
    """
    Plots the raw (non-normalized) spread calculated from the
    Kalman Filter 1 (KF1) parameters.
    Spread = p2 - (B0 + B1*p1)
    """
    plt.figure(figsize=(14, 7))
    spread_series = pd.Series(spread_history, index=dates, name="Spread KF1")
    spread_series.plot(label='Spread (KF1)', alpha=0.9)

    mean_spread = spread_series.mean()
    std_spread = spread_series.std()

    # Traducido:
    plt.axhline(mean_spread, color='black', linestyle='--', linewidth=1.5,
                label='Spread Mean')
    plt.fill_between(
        spread_series.index,
        mean_spread + std_spread,
        mean_spread - std_spread,
        color='gray',
        alpha=0.15,
        label='±1 Std. Deviation'
    )

    plt.title(f'KF1 Spread Analysis - {ticker2} vs {ticker1}')
    plt.xlabel('Date')
    plt.ylabel('Spread Value (P2 - β_t * P1)')
    plt.legend()
    plt.show()