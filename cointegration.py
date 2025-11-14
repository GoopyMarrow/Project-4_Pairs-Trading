import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.vector_ar.vecm import coint_johansen
from itertools import combinations
from config import ADF_P_VALUE_THRESHOLD, CORRELATION_THRESHOLD


def find_non_stationary_assets(data: pd.DataFrame, p_value_thresh: float = ADF_P_VALUE_THRESHOLD) -> list:
    """
    Performs an Augmented Dickey-Fuller (ADF) test on all assets to find
    non-stationary series (i.e., those with a unit root).

    H0 of ADF test: The series has a unit root (it is non-stationary).
    We *want* to fail to reject H0, so we look for p-value > threshold.

    Args:
        data: DataFrame of all asset prices.
        p_value_thresh: The significance level for the ADF test.

    Returns:
        A list of tickers that are non-stationary.
    """
    non_stationary_assets = []
    for ticker in data.columns:
        adf_result = adfuller(data[ticker])
        p_value = adf_result[1]
        # We are looking for assets that *are* non-stationary
        if p_value > p_value_thresh:
            non_stationary_assets.append(ticker)
    print(f"Found {len(non_stationary_assets)} non-stationary assets.")
    return non_stationary_assets


def run_engle_granger_test(series1: pd.Series, series2: pd.Series, p_value_thresh: float = ADF_P_VALUE_THRESHOLD) -> \
tuple[bool, float, float]:
    """
    Performs the Engle-Granger two-step cointegration test.

    Step 1: Run OLS regression (y = B0 + B1*x).
    Step 2: Run ADF test on the residuals of the regression.
    If residuals are stationary (p < threshold), the series are cointegrated.

    Args:
        series1: Price series for the independent variable (x).
        series2: Price series for the dependent variable (y).
        p_value_thresh: The significance level for the ADF test.

    Returns:
        A tuple of (is_cointegrated, p_value, hedge_ratio).
    """
    X = sm.add_constant(series1)
    y = series2

    model = sm.OLS(y, X).fit()
    residuals = model.resid
    hedge_ratio = model.params.iloc[1]

    # Test residuals for stationarity
    adf_result = adfuller(residuals)
    p_value = adf_result[1]

    is_cointegrated = p_value < p_value_thresh
    return is_cointegrated, p_value, hedge_ratio


def run_johansen_test(data: pd.DataFrame) -> tuple[bool, float, float, np.ndarray]:
    """
    Performs the Johansen cointegration test.

    This test checks for cointegrating relationships in a multivariate system.
    We check the trace statistic against the 95% critical value.

    Args:
        data: DataFrame with prices for the two assets.

    Returns:
        A tuple of (is_cointegrated, trace_stat, trace_crit_val_95, eigenvector).
    """
    try:
        johansen_result = coint_johansen(data, det_order=0, k_ar_diff=1)

        # Get the trace statistic for r=0 (H0: rank=0, no cointegration)
        trace_stat = johansen_result.lr1[0]
        # Get the 95% critical value for r=0
        trace_crit_val_95 = johansen_result.cvt[0, 1]

        # If trace_stat > critical_value, we reject H0 and conclude rank=1
        is_cointegrated = trace_stat > trace_crit_val_95
        eigenvector = johansen_result.evec[:, 0]

        return is_cointegrated, trace_stat, trace_crit_val_95, eigenvector

    except Exception as e:
        # Catch errors from Johansen test (e.g., singular matrix)
        return False, 0.0, 0.0, np.array([0.0, 0.0])


def find_cointegrated_pairs(data: pd.DataFrame, non_stationary_assets: list) -> pd.DataFrame:
    """
    Screens all combinations of non-stationary assets to find cointegrated pairs.

    A pair must pass three tests:
    1. High Correlation (config.CORRELATION_THRESHOLD)
    2. Engle-Granger Test (config.ADF_P_VALUE_THRESHOLD)
    3. Johansen Test

    Args:
        data: DataFrame of all asset prices.
        non_stationary_assets: A list of tickers to screen.

    Returns:
        A DataFrame of all pairs that passed the tests, sorted by strength.
    """
    print("Screening for cointegrated pairs...")
    tested_pairs = []

    # Iterate through all unique combinations of assets
    for ticker1, ticker2 in combinations(non_stationary_assets, 2):
        pair_data = data[[ticker1, ticker2]].dropna()

        # Ensure sufficient data
        if len(pair_data) < 252:
            continue

        # --- Test 1: Correlation ---
        correlation = pair_data.corr().iloc[0, 1]
        if correlation < CORRELATION_THRESHOLD:
            continue

        # --- Test 2: Engle-Granger ---
        eg_coint, eg_p_value, eg_hedge_ratio = run_engle_granger_test(pair_data[ticker1], pair_data[ticker2])
        if not eg_coint:
            continue

        # --- Test 3: Johansen ---
        jo_coint, jo_trace_stat, jo_crit_val, jo_eigenvector = run_johansen_test(pair_data)
        if not jo_coint:
            continue

        # --- Passed All Tests ---
        tested_pairs.append({
            "ticker1": ticker1,
            "ticker2": ticker2,
            "correlation": correlation,
            "eg_p_value": eg_p_value,
            "jo_trace_stat": jo_trace_stat,
            "jo_crit_val": jo_crit_val,
            "jo_coint": jo_coint,
            "jo_eigenvector": jo_eigenvector
        })

    pair_df = pd.DataFrame(tested_pairs)

    if not pair_df.empty:
        # Calculate "Strength" as a ratio of trace_stat to its critical value
        pair_df["Strength"] = pair_df["jo_trace_stat"] / pair_df["jo_crit_val"]
        # Sort by Engle-Granger p-value (lowest is best)
        pair_df = pair_df.sort_values(by="eg_p_value", ascending=True).reset_index(drop=True)

        print("\n--- Cointegrated Pairs Report (Passed All Tests) ---")
        pair_df["jo_eigenvector"] = pair_df["jo_eigenvector"].apply(lambda x: np.round(x, 6))

        cols_to_print = [
            "ticker1", "ticker2", "correlation", "eg_p_value",
            "jo_trace_stat", "jo_crit_val", "jo_coint",
            "Strength", "jo_eigenvector"
        ]

        print(pair_df[cols_to_print].to_string(index=False, float_format="%.6f"))
        print("--------------------------------------------------")

    print(f"Found {len(pair_df)} potential cointegrated pairs.")
    return pair_df