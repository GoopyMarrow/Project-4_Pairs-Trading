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


def optimize_std_threshold(train_data: pd.DataFrame, ticker1: str, ticker2: str, window_size: int) -> float:
    print("Optimizing entry threshold on training data...")
    std_thresholds = np.arange(
        config.OPTIMIZATION_MIN_STD,
        config.OPTIMIZATION_MAX_STD + config.OPTIMIZATION_STEP,
        config.OPTIMIZATION_STEP
    )

    best_calmar = -np.inf
    best_std = config.OPTIMIZATION_MIN_STD

    results_list = []

    for std in std_thresholds:
        backtest_sim = PairsTradingBacktest(
            data=train_data,
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
        if calmar > best_calmar:
            best_calmar = calmar
            best_std = std

    print("\n--- Optimization Results (Train Set) ---")

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

    print(f"Optimization complete. Best Calmar Ratio: {best_calmar:.4f} at Std: {best_std:.2f}")

    try:
        results_df.set_index('Theta (Std)')['Calmar Ratio'].plot(kind='bar')
        plt.title('Calmar Ratio vs. Entry Threshold (Train Set)')
        plt.xlabel('Entry STD Threshold (Theta)')
        plt.ylabel('Calmar Ratio')
        plt.show()
    except Exception as e:
        print(f"Could not plot optimization results: {e}")

    return best_std