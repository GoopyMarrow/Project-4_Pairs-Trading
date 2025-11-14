import pandas as pd
import numpy as np
from backtest import PairsTradingBacktest
import config
import matplotlib.pyplot as plt

"""
Utilities Module

Contains helper functions for data splitting and parameter optimization.
"""


def split_data(data: pd.DataFrame, train_pct: float, test_pct: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Splits the data into training and testing sets based on percentages.

    Args:
        data: The full DataFrame of price data.
        train_pct: The percentage of data to use for training (e.g., 0.6).
        test_pct: The percentage of data to use for testing (e.g., 0.4).

    Returns:
        A tuple of (train_data, test_data).
    """
    train_end = int(len(data) * train_pct)

    train_data = data.iloc[:train_end]
    test_data = data.iloc[train_end:]

    print(f"Data split: Train ({len(train_data)}), Test ({len(test_data)})")
    return train_data, test_data


def optimize_std_threshold(data: pd.DataFrame, ticker1: str, ticker2: str, window_size: int,
                           set_name: str = "Train Set") -> tuple[float, dict]:
    """
    Iterates through a range of entry thresholds (theta/std) and runs a
    backtest for each one to find the best-performing parameter based on
    Calmar Ratio.

    Args:
        data: The price data (either train or test set).
        ticker1: The symbol for the first asset.
        ticker2: The symbol for the second asset.
        window_size: The rolling window size.
        set_name: A string label for the optimization (e.g., "Train Set").

    Returns:
        A tuple of (best_std, best_run_results):
        - best_std: The 'theta' value that produced the highest Calmar Ratio.
        - best_run_results: The full results dictionary from that best run.
    """
    print(f"Optimizing entry threshold on {set_name}...")
    # Generate the range of thetas (stds) to test
    std_thresholds = np.arange(
        config.OPTIMIZATION_MIN_STD,
        config.OPTIMIZATION_MAX_STD + config.OPTIMIZATION_STEP,
        config.OPTIMIZATION_STEP
    )

    best_calmar = -np.inf
    best_std = config.OPTIMIZATION_MIN_STD
    best_run_results = {}  # Stores the full results dict of the best run
    results_list = []

    # Run a backtest for each theta value
    for std in std_thresholds:

        # This creates a new, independent backtest for each simulation
        backtest_sim = PairsTradingBacktest(
            data=data,
            ticker1=ticker1,
            ticker2=ticker2,
            window_size=window_size,
            entry_threshold=std
        )

        sim_results = backtest_sim.run_backtest()

        # Extract metrics for comparison
        metrics = sim_results["metrics"]
        positions = sim_results["positions"]

        # Calculate trade stats
        num_trades = len(positions)
        if num_trades > 0:
            win_rate = np.mean([1 if pos.pnl > 0 else 0 for pos in positions])
            avg_pnl = np.mean([pos.pnl for pos in positions])
        else:
            win_rate = 0.0
            avg_pnl = 0.0

        # Log results for the summary table
        results_list.append({
            "Theta (Std)": std,
            "Calmar Ratio": metrics["Calmar Ratio"],
            "Sortino Ratio": metrics["Sortino Ratio"],
            "Sharpe Ratio": metrics["Sharpe Ratio"],
            "Max Drawdown": metrics["Max Drawdown"],
            "Final Value": metrics["Final Portfolio Value"],
            "Total Return": metrics["Total Return"],
            "Total Trades": num_trades,
            "Win Rate": win_rate,
            "Avg PnL": avg_pnl
        })

        calmar = metrics["Calmar Ratio"]

        # Check if this run is the new best
        if calmar > best_calmar:
            best_calmar = calmar
            best_std = std
            best_run_results = sim_results  # Save the results of this run

    # --- Print Optimization Summary Table ---
    print(f"\n--- Optimization Results ({set_name}) ---")
    results_df = pd.DataFrame(results_list)

    cols_order = [
        "Theta (Std)", "Calmar Ratio", "Sortino Ratio", "Sharpe Ratio", "Max Drawdown",
        "Final Value", "Total Return", "Total Trades", "Win Rate", "Avg PnL"
    ]
    results_df = results_df[cols_order]

    # Define formatting for the table
    formatters = {
        "Theta (Std)": "{:.2f}".format,
        "Calmar Ratio": "{:.4f}".format,
        "Sortino Ratio": "{:.4f}".format,
        "Sharpe Ratio": "{:.4f}".format,
        "Max Drawdown": "{:.2%}".format,
        "Final Value": "${:,.2f}".format,
        "Total Return": "{:.2%}".format,
        "Total Trades": "{:,.0f}".format,
        "Win Rate": "{:.2%}".format,
        "Avg PnL": "${:,.2f}".format
    }

    print(results_df.to_string(index=False, formatters=formatters))
    print("--------------------------------------------------\n")

    # --- Plot Optimization Results ---
    try:
        ax = results_df.set_index('Theta (Std)')['Calmar Ratio'].plot(kind='bar')
        plt.title(f'Calmar Ratio vs. Entry Threshold ({set_name})')
        plt.xlabel('Entry STD Threshold (Theta)')
        plt.ylabel('Calmar Ratio')

        # Clean up x-axis labels if there are too many
        if len(std_thresholds) > 20:
            n = int(len(std_thresholds) / 10)  # Show every Nth label
            ticks = ax.get_xticks()
            labels = [item.get_text() for item in ax.get_xticklabels()]
            ax.set_xticks(ticks[::n])
            ax.set_xticklabels(labels[::n])

        plt.tight_layout()
        plt.show()
    except Exception as e:
        print(f"Could not plot optimization results: {e}")

    # Return the best parameter and the full results from that run
    return best_std, best_run_results