import pandas as pd
import numpy as np
from backtest import PairsTradingBacktest
import config
import matplotlib.pyplot as plt


def split_data(data: pd.DataFrame, train_pct: float, test_pct: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_end = int(len(data) * train_pct)

    train_data = data.iloc[:train_end]
    test_data = data.iloc[train_end:]

    print(f"Data split: Train ({len(train_data)}), Test ({len(test_data)})")
    return train_data, test_data


def optimize_std_threshold(data: pd.DataFrame, ticker1: str, ticker2: str, window_size: int, set_name: str = "Train Set") -> tuple[float, dict]:
    """
    Runs optimization loop on the provided dataset and returns the best std and the results of that best run.
    """
    print(f"Optimizing entry threshold on {set_name}...")
    std_thresholds = np.arange(
        config.OPTIMIZATION_MIN_STD,
        config.OPTIMIZATION_MAX_STD + config.OPTIMIZATION_STEP,
        config.OPTIMIZATION_STEP
    )

    best_calmar = -np.inf
    best_std = config.OPTIMIZATION_MIN_STD
    best_run_results = {} # <-- To store the best simulation results

    results_list = []

    for std in std_thresholds:
        # Note: This assumes you will pre-fill the history for the test set.
        # If not, the first year of the test set will have 0.0 signals.
        # For this request, we assume each backtest is independent.
        backtest_sim = PairsTradingBacktest(
            data=data,
            ticker1=ticker1,
            ticker2=ticker2,
            window_size=window_size,
            entry_threshold=std
        )

        sim_results = backtest_sim.run_backtest()

        metrics = sim_results["metrics"]
        positions = sim_results["positions"]

        num_trades = len(positions)
        if num_trades > 0:
            win_rate = np.mean([1 if pos.pnl > 0 else 0 for pos in positions])
            avg_pnl = np.mean([pos.pnl for pos in positions])
        else:
            win_rate = 0.0
            avg_pnl = 0.0

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
        # Check if this run is better
        if calmar > best_calmar:
            best_calmar = calmar
            best_std = std
            best_run_results = sim_results # <-- Save the results dictionary

    print(f"\n--- Optimization Results ({set_name}) ---")

    results_df = pd.DataFrame(results_list)

    cols_order = [
        "Theta (Std)", "Calmar Ratio", "Sortino Ratio", "Sharpe Ratio", "Max Drawdown",
        "Final Value", "Total Return", "Total Trades", "Win Rate", "Avg PnL"
    ]
    results_df = results_df[cols_order]

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

    # print(f"Optimization complete. Best Calmar Ratio: {best_calmar:.4f} at Std: {best_std:.2f}")

    try:
        # Plot optimization results for this set
        ax = results_df.set_index('Theta (Std)')['Calmar Ratio'].plot(kind='bar')
        plt.title(f'Calmar Ratio vs. Entry Threshold ({set_name})')
        plt.xlabel('Entry STD Threshold (Theta)')
        plt.ylabel('Calmar Ratio')
        # Ensure xticks are not too crowded
        if len(std_thresholds) > 20:
             # Show every Nth label
             n = int(len(std_thresholds) / 10)
             ticks = ax.get_xticks()
             labels = [item.get_text() for item in ax.get_xticklabels()]
             ax.set_xticks(ticks[::n])
             ax.set_xticklabels(labels[::n])

        plt.tight_layout()
        plt.show()
    except Exception as e:
        print(f"Could not plot optimization results: {e}")

    # Return the best std *and* the results dictionary from that run
    return best_std, best_run_results