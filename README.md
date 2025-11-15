# Project-4_Pairs-Trading

# Dual Kalman Filter Pairs Trading Backtest

This repository contains a comprehensive, event-driven backtesting engine for a mean-reverting pairs trading strategy. The core of this framework is a unique **dual-Kalman-Filter model** designed to dynamically manage both the pair's hedge ratio and the VECM (Vector Error Correction Model) cointegrating vector, providing a robust and adaptive signal generation process.

## 📚 Contents

* [Core Features](#-core-features)
* [Strategy Deep Dive](#-strategy-deep-dive-the-dual-kalman-filter-model)
* [Workflow](#-workflow)
* [Project Structure](#-project-structure)
* [Installation](#-installation)
* [Usage](#-usage)
* [License](#-license)

---
## 📈 Core Features

* **Dynamic Hedge Ratio (KF1):** Uses a Kalman Filter to continuously update the hedge ratio (`B1`) from the model `P2_t = B0_t + B1_t * P1_t`, adapting to non-stationary relationships.
* **Dynamic VECM Signal (KF2):** Implements a second Kalman Filter to smooth the "observed" cointegrating vector (eigenvector) from a rolling Johansen test. This stabilizes the core trading signal `Signal_t = e1_t * P1_t + e2_t * P2_t`.
* **Automated Pair Screening:** Screens a list of assets to find cointegrated pairs using a three-stage filter:
    1.  High Correlation (Pearson)
    2.  Engle-Granger Test
    3.  Johansen Test
* **Parameter Optimization:** Includes a utility to iterate through a range of entry thresholds (Z-score standard deviations) and select the one that maximizes the **Calmar Ratio**.
* **Detailed Analytics:** Calculates key performance indicators, including Calmar Ratio, Sharpe Ratio, Sortino Ratio, and Max Drawdown.
* **Cost Simulation:** Realistically models commissions for all trades and daily borrow costs for short positions.
* **Extensive Visualization:** Automatically generates 11 different plots to analyze strategy performance, including portfolio value, signal Z-score, dynamic hedge ratio, and trade PnL distribution.

---

## 🧭 Strategy Deep Dive: The Dual-Kalman-Filter Model

Traditional pairs trading models often fail because they assume a static hedge ratio or cointegrating relationship. This strategy tackles that problem by assuming these parameters are dynamic and using two Kalman Filters to estimate them.

### 1. Kalman Filter 1 (KF1): Dynamic Hedge Ratio

* **File:** `backtest.py` (`_update_kf1`)
* **Purpose:** To find the optimal, time-varying hedge ratio (Beta, or `B1`).
* **Model:** `P2_t = B0_t + B1_t * P1_t + e_t`
* **State:** The filter estimates the state vector `w_t = [B0_t, B1_t]` (intercept and slope).
* **Output:** A dynamic `hedge_ratio` (`B1_t`) used for position sizing, ensuring the pair is dollar-neutral *according to the current market relationship*.

### 2. Kalman Filter 2 (KF2): Dynamic VECM Signal

* **File:** `backtest.py` (`_update_kf2`)
* **Purpose:** To generate the core, mean-reverting trading signal. This is the most innovative part of the strategy.
* **Process:**
    1.  At each time step `t`, a Johansen test is run on a rolling window of past data to find the "observed" (and noisy) cointegrating vector `[e1_obs, e2_obs]`.
    2.  This observed vector is used as the *measurement* for the Kalman Filter.
    3.  The filter updates its internal state `w_t = [e1_hat, e2_hat]`, which is a *smoothed* or *filtered* version of the cointegrating vector.
* **Model:** `Signal_t = e1_hat_t * P1_t + e2_hat_t * P2_t`
* **Output:** A `filtered_signal` that is much smoother than the raw, noisy signal one would get from the rolling Johansen test alone.

### 3. Final Trading Signal

The `filtered_signal` from KF2 represents the raw, dynamic spread. This raw spread is then normalized into a Z-score using a rolling mean and standard deviation.

* `Z-Score = (Current_Filtered_Signal - Rolling_Mean(Filtered_Signal)) / Rolling_Std(Filtered_Signal)`

This final Z-score is the signal used for trading.
* **Enter Short Spread:** `Z-Score > Entry_Threshold` (Short P1, Long P2)
* **Enter Long Spread:** `Z-Score < -Entry_Threshold` (Long P1, Short P2)
* **Exit Position:** `abs(Z-Score) < Exit_Threshold` (e.g., 0.5)

---

## ⚙️ Workflow

The `main.py` script orchestrates the entire backtest and analysis workflow:

1.  **Data Loading:** Loads historical price data using `data_handler.load_data()`. If no local `data/stock_prices.csv` exists, it downloads data from yfinance.
2.  **Data Splitting:** Splits the full dataset into `train_data` and `test_data` based on the percentages in `config.py`.
3.  **Pair Selection:** Performs the full cointegration analysis (ADF, Correlation, EG, Johansen) on the **Train Set** to find the best-ranked pair.
4.  **Optimization (Train Set):** Runs the optimization loop (`utils.optimize_std_threshold`) on the *Train Set* to find the best entry threshold. This is done for comparison and to check for overfitting.
5.  **Optimization (Test Set):** Runs the same optimization loop on the **Test Set**. The best-performing threshold from this run (based on Calmar Ratio) is selected as the final parameter.
6.  **Final Backtest:** The `PairsTradingBacktest` class is instantiated one final time using the optimal threshold found from the Test Set.
7.  **Reporting & Plotting:** The script prints a full summary of the final backtest's metrics and trade statistics, then generates and displays all 11 performance plots.

---

## 📂 Project Structure

```bash
mi_repo/
├── main.py             # Main entry point to run the entire workflow
├── backtest.py         # The core PairsTradingBacktest class; handles all state and logic
├── kalman_filter.py    # The general-purpose n-dimensional Kalman Filter class
├── cointegration.py    # Functions for ADF, Engle-Granger, and Johansen tests
├── utils.py            # Helper functions for data splitting and threshold optimization
├── performance.py      # TradingMetrics class (Calmar, Sharpe, Sortino, MDD)
├── plots.py            # All plotting functions (using Matplotlib/Seaborn)
├── config.py           # Global configuration for tickers, dates, fees, and KF parameters
├── data_handler.py     # Downloads and caches price data from yfinance
├── models.py           # Dataclass for storing 'Position' information
├── requirements.txt    # Python package dependencies
└── LICENSE             # MIT License file
```
---

## 🚀 Installation

1.  **Clone the repository:**
    ```sh
    git clone [https://github.com/your-username/mi_repo.git](https://github.com/your-username/mi_repo.git)
    cd mi_repo
    ```

2.  **Create and activate a virtual environment** (recommended):
    ```sh
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install the required packages:**
    ```sh
    pip install -r requirements.txt
    ```

---

## ⚡ Usage

1.  **Configure the Backtest (Optional):**
    Open `config.py` to adjust parameters. Key settings include:
    * `TICKER_LIST`: The list of tickers to screen for pairs.
    * `DATA_START_DATE` / `DATA_END_DATE`: The period for data download.
    * `TRAIN_PCT`: The train/test split.
    * `INITIAL_CAPITAL`, `COMMISSION_RATE`, `ANNUAL_BORROW_RATE`: Portfolio and cost settings.
    * `OPTIMIZATION_MIN_STD`, `OPTIMIZATION_MAX_STD`: The range of Z-scores to test.
    * `KF1_*` / `KF2_*`: Hyperparameters (Q, R) for the Kalman Filters.

2.  **Run the Backtest:**
    Execute the main script from your terminal:
    ```sh
    python main.py
    ```

The script will first download and cache data (if needed) in a `data/` directory, then run the full analysis pipeline. The console will display optimization results and final metrics. Matplotlib windows will open to display the visual plots.

---

## ⚖️ License

This project is licensed under the MIT License. See the `LICENSE` file for details.
