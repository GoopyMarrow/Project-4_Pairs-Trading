import pandas as pd
import numpy as np

import data_handler
import cointegration
import utils
import plots
import config
from backtest import PairsTradingBacktest


def main():
    all_data = data_handler.load_data()

    if all_data.empty:
        print("Failed to load data. Exiting.")
        return

    train_data, test_data = utils.split_data(
        all_data, config.TRAIN_PCT, config.TEST_PCT
    )

    non_stationary_assets = cointegration.find_non_stationary_assets(train_data)
    cointegrated_pairs = cointegration.find_cointegrated_pairs(train_data, non_stationary_assets)

    if cointegrated_pairs.empty:
        print("No cointegrated pairs found. Exiting.")
        return

    print("\nTop 5 Cointegrated Pairs (from Train set):")
    print(cointegrated_pairs.head())

    best_pair = cointegrated_pairs.iloc[0]
    ticker1 = best_pair['ticker1']
    ticker2 = best_pair['ticker2']

    print(f"\n--- Selected Pair: {ticker1} & {ticker2} ---")

    train_pair = train_data[[ticker1, ticker2]].dropna()
    test_pair = test_data[[ticker1, ticker2]].dropna()

    std_optimal = utils.optimize_std_threshold(
        train_data=train_pair,
        ticker1=ticker1,
        ticker2=ticker2,
        window_size=config.DEFAULT_WINDOW_SIZE
    )

    print(f"\n--- Running FINAL Backtest on TEST SET with Optimal Std: {std_optimal:.2f} ---")

    final_backtest_run = PairsTradingBacktest(
        data=test_pair,
        ticker1=ticker1,
        ticker2=ticker2,
        window_size=config.DEFAULT_WINDOW_SIZE,
        entry_threshold=std_optimal
    )

    results = final_backtest_run.run_backtest()

    print("\n--- Backtest Results (Test Set) ---")
    metrics = results["metrics"]
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")

    num_trades = len(results["positions"])
    print(f"Total Trades: {num_trades}")
    if num_trades > 0:
        win_rate = np.mean([1 if pos.pnl > 0 else 0 for pos in results["positions"]])
        print(f"Win Rate: {win_rate:.2%}")
        avg_pnl = np.mean([pos.pnl for pos in results["positions"]])
        print(f"Average PnL per Trade: {avg_pnl:.2f}")

    print("\n--- Generating Plots (Test Set Results) ---")

    window_size = config.DEFAULT_WINDOW_SIZE

    full_dates_test = test_pair.index

    backtest_dates_test = test_pair.index[len(test_pair) - len(results["vecm_norm_history"]):]
    if len(backtest_dates_test) != len(results["vecm_norm_history"]):
        backtest_dates_test = test_pair.index[-len(results["vecm_norm_history"]):]

    # 1. Precios Históricos (Test Set)
    plots.plot_pair_prices(
        test_pair,
        ticker1,
        ticker2
    )

    # 2. Precios Normalizados
    plots.plot_normalized_prices(
        test_pair,
        ticker1,
        ticker2
    )

    # 3. Portfolio Value
    plots.plot_portfolio_value(
        portfolio_value=results["portfolio_value"],
        dates=full_dates_test,
        ticker1=ticker1,
        ticker2=ticker2,
        threshold=std_optimal
    )

    # 4. Dynamic Spread (VECM Norm)
    plots.plot_dynamic_spread(
        vecm_norm_history=results["vecm_norm_history"],
        dates=backtest_dates_test,
        threshold=std_optimal
    )

    # 5. Hedge Ratio
    plots.plot_hedge_ratio(
        hedge_ratio_history=results["hedge_ratio_history"],
        dates=backtest_dates_test
    )

    # 6. Trades en Precios
    plots.plot_trades_on_price(
        test_pair,
        ticker1,
        ticker2,
        results["positions"]
    )

    # 7. Distribución de Retornos
    plots.plot_returns_distribution(results["positions"])

    # 8. VECM vs VECM Hat
    plots.plot_vecm_signals(
        observed_signal=results["vecm_observed_history"],
        filtered_signal=results["vecm_filtered_history"],
        dates=backtest_dates_test
    )

    print("\n--- Process Complete ---")


if __name__ == "__main__":
    main()