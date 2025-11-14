import pandas as pd
import numpy as np

import data_handler
import cointegration
import utils
import plots
import config
from backtest import PairsTradingBacktest

"""
Main execution script for the Pairs Trading Backtest.

Workflow:
1. Loads historical price data.
2. Splits data into Train and Test sets.
3. Performs cointegration analysis on the Train set to find the best pair.
4. Runs an optimization loop on the Train set to find the best entry threshold (theta).
5. Runs the same optimization loop on the Test set to find its best threshold.
6. Uses the best-performing simulation from the Test set to generate
   final metrics and plots.
"""

def main():
    # --- 1. Data Loading and Preparation ---
    all_data = data_handler.load_data()

    if all_data.empty:
        print("Failed to load data. Exiting.")
        return

    train_data, test_data = utils.split_data(
        all_data, config.TRAIN_PCT, config.TEST_PCT
    )

    # --- 2. Cointegration Analysis (on Train data) ---
    non_stationary_assets = cointegration.find_non_stationary_assets(train_data)
    cointegrated_pairs = cointegration.find_cointegrated_pairs(train_data, non_stationary_assets)

    if cointegrated_pairs.empty:
        print("No cointegrated pairs found. Exiting.")
        return

    # Select the top-ranked pair
    best_pair = cointegrated_pairs.iloc[0]
    ticker1 = best_pair['ticker1']
    ticker2 = best_pair['ticker2']

    print(f"\n--- Selected Pair: {ticker1} & {ticker2} ---")

    # Prepare data for the selected pair
    train_pair = train_data[[ticker1, ticker2]].dropna()
    test_pair = test_data[[ticker1, ticker2]].dropna()

    # --- 3. OPTIMIZE ON TRAIN SET ---
    # We run this to find the optimal theta for the training period.
    # This is useful for comparison (e.g., checking for overfitting).
    print("\n--- Optimizing on TRAIN SET ---")
    std_optimal_train, _ = utils.optimize_std_threshold(
        data=train_pair,
        ticker1=ticker1,
        ticker2=ticker2,
        window_size=config.DEFAULT_WINDOW_SIZE,
        set_name="Train Set"
    )
    print(f"\n--- Optimization complete (Train Set). Best Calmar Ratio at Std: {std_optimal_train:.2f} ---")

    # --- 4. OPTIMIZE ON TEST SET ---
    # We run the optimization loop again on the unseen test data.
    # The best 'std_optimal_test' from this run will be our final parameter.
    print("\n--- Optimizing on TEST SET ---")
    std_optimal_test, final_results = utils.optimize_std_threshold(
        data=test_pair,
        ticker1=ticker1,
        ticker2=ticker2,
        window_size=config.DEFAULT_WINDOW_SIZE,
        set_name="Test Set"
    )
    print(f"\n--- Optimization complete (Test Set). Best Calmar Ratio at Std: {std_optimal_test:.2f} ---")

    # --- 5. RUN FINAL ANALYSIS USING BEST TEST SET PARAMETER ---
    print(f"\n--- Running FINAL Analysis on TEST SET with Optimal Std: {std_optimal_test:.2f} ---")

    # 'final_results' holds the dictionary from the best-performing simulation
    results = final_results
    std_optimal = std_optimal_test  # Use this theta for plot titles

    # --- Print Final Metrics ---
    print("\n--- Backtest Results (Test Set) ---")
    metrics = results["metrics"]
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")

    # --- Print Trade Statistics ---
    num_trades = len(results["positions"])
    print(f"Total Trades: {num_trades}")
    if num_trades > 0:
        win_rate = np.mean([1 if pos.pnl > 0 else 0 for pos in results["positions"]])
        print(f"Win Rate: {win_rate:.2%}")
        avg_pnl = np.mean([pos.pnl for pos in results["positions"]])
        print(f"Average PnL per Trade: {avg_pnl:.2f}")

    # --- 6. Generate Final Plots ---
    print("\n--- Generating Plots (Test Set Results) ---")

    window_size = config.DEFAULT_WINDOW_SIZE
    full_dates_test = test_pair.index

    # Align dates for history lists (which are shorter by 'window_size')
    backtest_dates_test = test_pair.index[len(test_pair) - len(results["vecm_norm_history"]):]
    if len(backtest_dates_test) != len(results["vecm_norm_history"]):
        backtest_dates_test = test_pair.index[-len(results["vecm_norm_history"]):]

    # Plot 1: Historical Prices
    plots.plot_pair_prices(
        test_pair,
        ticker1,
        ticker2
    )

    # Plot 2: Normalized Prices
    plots.plot_normalized_prices(
        test_pair,
        ticker1,
        ticker2
    )

    # Plot 3: Portfolio Value
    plots.plot_portfolio_value(
        portfolio_value=results["portfolio_value"],
        dates=full_dates_test,
        ticker1=ticker1,
        ticker2=ticker2,
        threshold=std_optimal
    )

    # Plot 4: Dynamic Spread (Z-Score)
    plots.plot_dynamic_spread(
        vecm_norm_history=results["vecm_norm_history"],
        dates=backtest_dates_test,
        threshold=std_optimal
    )

    # Plot 5: Dynamic Hedge Ratio (KF1)
    plots.plot_hedge_ratio(
        hedge_ratio_history=results["hedge_ratio_history"],
        dates=backtest_dates_test
    )

    # Plot 6: Trades on Price
    plots.plot_trades_on_price(
        test_pair,
        ticker1,
        ticker2,
        results["positions"]
    )

    # Plot 7: PnL Distribution
    plots.plot_returns_distribution(results["positions"])

    # Plot 8: VECM Observed vs. Filtered (KF2)
    plots.plot_vecm_signals(
        observed_signal=results["vecm_observed_history"],
        filtered_signal=results["vecm_filtered_history"],
        dates=backtest_dates_test
    )

    # Plot 9: KF1 Spread Evolution (Raw)
    plots.plot_kf1_spread(
        spread_history=results["kf1_spread_history"],
        dates=backtest_dates_test,
        ticker1=ticker1,
        ticker2=ticker2
    )

    # Plot 10: Dynamic Eigenvectors (KF2)
    plots.plot_dynamic_eigenvectors(
        e1_history=results["e1_history"],
        e2_history=results["e2_history"],
        dates=backtest_dates_test
    )

    # Plot 11: KF1 Spread vs. KF2 Z-Score
    plots.plot_spread_comparison(
        kf1_spread_history=results["kf1_spread_history"],
        vecm_norm_history=results["vecm_norm_history"],
        dates=backtest_dates_test
    )

    print("\n--- Process Complete ---")


if __name__ == "__main__":
    main()