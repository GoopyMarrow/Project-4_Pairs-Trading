import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.vector_ar.vecm import coint_johansen
from itertools import combinations
from config import ADF_P_VALUE_THRESHOLD, CORRELATION_THRESHOLD


def find_non_stationary_assets(data: pd.DataFrame, p_value_thresh: float = ADF_P_VALUE_THRESHOLD) -> list:
    non_stationary_assets = []
    for ticker in data.columns:
        adf_result = adfuller(data[ticker])
        p_value = adf_result[1]
        if p_value > p_value_thresh:
            non_stationary_assets.append(ticker)
    print(f"Found {len(non_stationary_assets)} non-stationary assets.")
    return non_stationary_assets


def run_engle_granger_test(series1: pd.Series, series2: pd.Series, p_value_thresh: float = ADF_P_VALUE_THRESHOLD) -> \
tuple[bool, float, float]:
    X = sm.add_constant(series1)
    y = series2

    model = sm.OLS(y, X).fit()
    residuals = model.resid
    hedge_ratio = model.params.iloc[1]

    adf_result = adfuller(residuals)
    p_value = adf_result[1]

    is_cointegrated = p_value < p_value_thresh
    return is_cointegrated, p_value, hedge_ratio


def run_johansen_test(data: pd.DataFrame) -> tuple[bool, float, float, np.ndarray]:
    try:
        johansen_result = coint_johansen(data, det_order=0, k_ar_diff=1)

        trace_stat = johansen_result.lr1[0]
        trace_crit_val_95 = johansen_result.cvt[0, 1]

        is_cointegrated = trace_stat > trace_crit_val_95
        eigenvector = johansen_result.evec[:, 0]

        return is_cointegrated, trace_stat, trace_crit_val_95, eigenvector

    except Exception as e:
        # print(f"Johansen test failed: {e}")
        return False, 0.0, 0.0, np.array([0.0, 0.0])


def find_cointegrated_pairs(data: pd.DataFrame, non_stationary_assets: list) -> pd.DataFrame:
    print("Screening for cointegrated pairs...")
    tested_pairs = []

    for ticker1, ticker2 in combinations(non_stationary_assets, 2):
        pair_data = data[[ticker1, ticker2]].dropna()

        if len(pair_data) < 252:
            continue

        correlation = pair_data.corr().iloc[0, 1]

        if correlation < CORRELATION_THRESHOLD:
            # print(f"DEBUG: {ticker1}-{ticker2} FAILED Correlation: {correlation:.2f} (Threshold: {CORRELATION_THRESHOLD})")
            continue

        eg_coint, eg_p_value, eg_hedge_ratio = run_engle_granger_test(pair_data[ticker1], pair_data[ticker2])

        if not eg_coint:
            # print(f"DEBUG: {ticker1}-{ticker2} FAILED Engle-Granger: p-value={eg_p_value:.4f} (Threshold: {ADF_P_VALUE_THRESHOLD})")
            continue

        jo_coint, jo_trace_stat, jo_crit_val, jo_eigenvector = run_johansen_test(pair_data)

        if not jo_coint:
            # print(f"DEBUG: {ticker1}-{ticker2} FAILED Johansen: Trace={jo_trace_stat:.2f} (CritVal: {jo_crit_val:.2f})")
            continue

        # print(f"DEBUG: {ticker1}-{ticker2} PASSED ALL TESTS.")

        tested_pairs.append({
            "ticker1": ticker1,
            "ticker2": ticker2,
            "correlation": correlation,
            "eg_p_value": eg_p_value,
            "jo_trace_stat": jo_trace_stat,
            "jo_crit_val": jo_crit_val,
            "jo_coint": jo_coint,  # Siempre True si llega aquí
            "jo_eigenvector": jo_eigenvector
        })

    pair_df = pd.DataFrame(tested_pairs)

    if not pair_df.empty:
        pair_df["Strength"] = pair_df["jo_trace_stat"] / pair_df["jo_crit_val"]
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