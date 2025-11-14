import pandas as pd
import numpy as np
from statsmodels.tsa.vector_ar.vecm import coint_johansen

import config
from kalman_filter import KalmanFilter
from models import Position
from performance import TradingMetrics


class PairsTradingBacktest:
    """
    Encapsulates the entire pairs trading backtest logic.

    This class initializes two Kalman Filters (one for hedge ratio, one for
    the VECM signal), iterates through the provided data, updates the filters,
    and executes trades based on the normalized VECM signal.

    Attributes:
        data (pd.DataFrame): The price data for the two tickers.
        ticker1 (str): The symbol for the first asset (P1).
        ticker2 (str): The symbol for the second asset (P2).
        window_size (int): The rolling window for Johansen and normalization.
        entry_threshold (float): The Z-score threshold for opening a position.
        exit_threshold (float): The Z-score threshold for closing a position.
        capital (float): The current cash in the portfolio.
        portfolio_value (list[float]): A time series of the total portfolio value.
        all_positions (list[Position]): A log of all closed positions.
        kf_hedge (KalmanFilter): The filter for the hedge ratio (KF1).
        kf_vecm (KalmanFilter): The filter for the VECM signal (KF2).
    """

    def __init__(self, data: pd.DataFrame, ticker1: str, ticker2: str, window_size: int, entry_threshold: float):
        """Initializes the backtest environment and Kalman Filters."""
        self.data = data
        self.ticker1 = ticker1
        self.ticker2 = ticker2
        self.window_size = window_size
        self.entry_threshold = entry_threshold
        self.exit_threshold = config.EXIT_THRESHOLD

        # --- Portfolio and Cost Initialization ---
        self.capital = config.INITIAL_CAPITAL
        self.commission = config.COMMISSION_RATE
        self.borrow_rate = config.DAILY_BORROW_RATE

        # Pre-fill portfolio value for the initial window period
        self.portfolio_value = [self.capital] * window_size
        self.all_positions = []

        # --- Active Position Tracking ---
        self.active_long_p1 = None
        self.active_short_p1 = None
        self.active_long_p2 = None
        self.active_short_p2 = None

        # --- Kalman Filter Initialization ---
        self.kf_hedge = self._init_kf1()
        self.kf_vecm = self._init_kf2()

        # --- History Tracking Lists ---
        self.historical_filtered_signal = []
        self.historical_observed_signal = []
        self.historical_hedge_ratio = []
        self.historical_vecm_norm = []
        self.historical_kf1_spread = []
        self.historical_e1_hat = []
        self.historical_e2_hat = []

    def _init_kf1(self) -> KalmanFilter:
        """Initializes the Kalman Filter for the hedge ratio (KF1)."""
        return KalmanFilter(
            n_dim=config.KF1_N_DIM,
            R=config.KF1_R,
            Q=config.KF1_Q,
            P0=config.KF1_P0,
            w0=config.KF1_W0
        )

    def _init_kf2(self) -> KalmanFilter:
        """
        Initializes the Kalman Filter for the VECM signal (KF2).
        Uses the first 'window_size' data points to find the initial eigenvector.
        """
        initial_data = self.data.iloc[:self.window_size]
        try:
            # Run Johansen on the initial window to get a starting state
            initial_jo = coint_johansen(initial_data[[self.ticker1, self.ticker2]], 0, 1)
            initial_evec = initial_jo.evec[:, 0]
        except Exception:
            # Fallback to default if Johansen fails
            initial_evec = config.KF2_W0

        return KalmanFilter(
            n_dim=config.KF2_N_DIM,
            R=config.KF2_R,
            Q=config.KF2_Q,
            P0=config.KF2_P0,
            w0=initial_evec
        )

    def _update_kf1(self, p1: float, p2: float) -> float:
        """
        Performs one prediction and update step for the hedge ratio (KF1).

        Model: p2 = B0 + B1*p1 + error
        State: [B0, B1]

        Args:
            p1: Current price of ticker1.
            p2: Current price of ticker2.

        Returns:
            The new dynamic hedge ratio (B1).
        """
        # Observation matrix H_t = [1, p1]
        x_t_hedge = np.array([1, p1])
        # Observation value y_t = p2
        y_t_hedge = p2

        w_pred, P_pred = self.kf_hedge.predict()
        w_upd, P_upd = self.kf_hedge.update(x_t_hedge, y_t_hedge, w_pred, P_pred)

        hedge_ratio = w_upd[1]
        intercept = w_upd[0]

        # Calculate the non-normalized spread from KF1
        spread_kf1 = y_t_hedge - (intercept + hedge_ratio * p1)

        self.historical_hedge_ratio.append(hedge_ratio)
        self.historical_kf1_spread.append(spread_kf1)
        return hedge_ratio

    def _update_kf2(self, p1: float, p2: float, rolling_data: pd.DataFrame) -> float:
        """
        Performs one prediction and update step for the VECM signal (KF2).

        Model: VECM_Signal = e1*p1 + e2*p2 + error
        State: [e1, e2]

        This is the core of the trading signal generation.

        Args:
            p1: Current price of ticker1.
            p2: Current price of ticker2.
            rolling_data: DataFrame of the past 'window_size' prices.

        Returns:
            The new filtered (smoothed) VECM signal.
        """
        try:
            # 1. Get the "observed" eigenvector from the rolling window
            jo_result = coint_johansen(rolling_data, 0, 1)
            obs_evec = jo_result.evec[:, 0]

            # CRITICAL FIX: Stabilize eigenvector sign
            # Eigenvectors are unique only up to their sign (e.g., [1, -1] is
            # the same as [-1, 1]). This flip can cause massive instability in
            # the filter. We force the first element to be positive.
            if obs_evec[0] < 0:
                obs_evec = -obs_evec

        except Exception:
            # If Johansen fails (e.g., singular matrix), use the filter's last known state
            obs_evec = self.kf_vecm.w_state

        # 2. Define the observation matrix and value for KF2
        # Observation matrix H_t = [p1, p2]
        x_t_vecm_prices = np.array([p1, p2])
        # "Observed" value y_t = (obs_evec[0] * p1) + (obs_evec[1] * p2)
        y_t_vecm_signal = np.dot(obs_evec, x_t_vecm_prices)

        self.historical_observed_signal.append(y_t_vecm_signal)

        # 3. Predict and Update KF2
        e_pred, P_pred_e = self.kf_vecm.predict()
        e_upd, P_upd_e = self.kf_vecm.update(x_t_vecm_prices, y_t_vecm_signal, e_pred, P_pred_e)

        # Store the filtered state components [e1, e2]
        self.historical_e1_hat.append(e_upd[0])
        self.historical_e2_hat.append(e_upd[1])

        # 4. Calculate the new *filtered* signal
        filtered_signal = np.dot(e_upd, x_t_vecm_prices)
        self.historical_filtered_signal.append(filtered_signal)
        return filtered_signal

    def _normalize_signal(self) -> float:
        """
        Calculates the Z-score of the *filtered* VECM signal.
        This Z-score is the final trading signal (vecm_norm).

        Returns:
            The Z-score, or 0.0 if the history is not yet sufficient.
        """
        if len(self.historical_filtered_signal) < self.window_size:
            # Not enough data to normalize, return neutral signal
            self.historical_vecm_norm.append(0.0)
            return 0.0

        # Get the rolling window of the filtered signal
        signal_window = np.array(self.historical_filtered_signal[-self.window_size:])
        mean = np.mean(signal_window)
        std = np.std(signal_window)

        if std == 0:
            # Avoid division by zero if signal is flat
            norm_signal = 0.0
        else:
            current_signal = self.historical_filtered_signal[-1]
            norm_signal = (current_signal - mean) / std

        self.historical_vecm_norm.append(norm_signal)
        return norm_signal

    def _update_borrow_costs(self, row: pd.Series):
        """Applies daily borrow costs to any active short positions."""
        if self.active_short_p1:
            cost = self.active_short_p1.n_shares * row[self.ticker1] * self.borrow_rate
            self.capital -= cost
            self.active_short_p1.borrow_cost += cost

        if self.active_short_p2:
            cost = self.active_short_p2.n_shares * row[self.ticker2] * self.borrow_rate
            self.capital -= cost
            self.active_short_p2.borrow_cost += cost

    def _close_positions(self, row: pd.Series):
        """
        Closes all active positions, updates capital, and logs the trade.

        Args:
            row: The current data row (for exit prices).
        """
        if self.active_long_p1:
            price = row[self.ticker1]
            trade_cost = self.active_long_p1.n_shares * price * self.commission
            self.capital += self.active_long_p1.n_shares * price - trade_cost

            self.active_long_p1.exit_price = price
            self.active_long_p1.exit_date = row.name
            self.active_long_p1.commission += trade_cost
            self.active_long_p1.pnl = (
                                                  price - self.active_long_p1.entry_price) * self.active_long_p1.n_shares - self.active_long_p1.commission
            self.all_positions.append(self.active_long_p1)
            self.active_long_p1 = None

        if self.active_short_p1:
            price = row[self.ticker1]
            trade_cost = self.active_short_p1.n_shares * price * self.commission
            pnl = (self.active_short_p1.entry_price - price) * self.active_short_p1.n_shares

            # For shorts, PnL is credited, but we only "get back" the PnL minus costs
            self.capital += pnl - trade_cost

            self.active_short_p1.exit_price = price
            self.active_short_p1.exit_date = row.name
            self.active_short_p1.commission += trade_cost
            self.active_short_p1.pnl = pnl - self.active_short_p1.commission - self.active_short_p1.borrow_cost
            self.all_positions.append(self.active_short_p1)
            self.active_short_p1 = None

        if self.active_long_p2:
            price = row[self.ticker2]
            trade_cost = self.active_long_p2.n_shares * price * self.commission
            self.capital += self.active_long_p2.n_shares * price - trade_cost

            self.active_long_p2.exit_price = price
            self.active_long_p2.exit_date = row.name
            self.active_long_p2.commission += trade_cost
            self.active_long_p2.pnl = (
                                                  price - self.active_long_p2.entry_price) * self.active_long_p2.n_shares - self.active_long_p2.commission
            self.all_positions.append(self.active_long_p2)
            self.active_long_p2 = None

        if self.active_short_p2:
            price = row[self.ticker2]
            trade_cost = self.active_short_p2.n_shares * price * self.commission
            pnl = (self.active_short_p2.entry_price - price) * self.active_short_p2.n_shares

            self.capital += pnl - trade_cost

            self.active_short_p2.exit_price = price
            self.active_short_p2.exit_date = row.name
            self.active_short_p2.commission += trade_cost
            self.active_short_p2.pnl = pnl - self.active_short_p2.commission - self.active_short_p2.borrow_cost
            self.all_positions.append(self.active_short_p2)
            self.active_short_p2 = None

    def _open_short_pair(self, row: pd.Series, hedge_ratio: float, p1: float, p2: float):
        """
        Opens a "short spread" position: Long P1, Short P2.
        This is triggered when the VECM signal is HIGH (vecm_norm > threshold).
        """
        available_capital = self.capital * config.POSITION_SIZING_PCT

        # Hedge-ratio-weighted capital allocation
        # Total invested value = (B1*p1 * N) + (p2 * N)
        # We find the capital for each leg based on the dynamic hedge ratio
        capital_long_p1 = available_capital * (hedge_ratio * p1) / (hedge_ratio * p1 + p2)
        capital_short_p2 = available_capital - capital_long_p1

        n_shares_long_p1 = capital_long_p1 / p1
        n_shares_short_p2 = capital_short_p2 / p2

        # Calculate costs
        cost_long = n_shares_long_p1 * p1 * (1 + self.commission)
        cost_short = n_shares_short_p2 * p2 * self.commission

        if self.capital > (cost_long + cost_short):
            self.capital -= (cost_long + cost_short)

            # Log the two new positions
            self.active_long_p1 = Position(self.ticker1, 'long', n_shares_long_p1, p1, row.name, commission=cost_long)
            self.active_short_p2 = Position(self.ticker2, 'short', n_shares_short_p2, p2, row.name,
                                            commission=cost_short)

    def _open_long_pair(self, row: pd.Series, hedge_ratio: float, p1: float, p2: float):
        """
        Opens a "long spread" position: Short P1, Long P2.
        This is triggered when the VECM signal is LOW (vecm_norm < -threshold).
        """
        available_capital = self.capital * config.POSITION_SIZING_PCT

        # Hedge-ratio-weighted capital allocation
        capital_short_p1 = available_capital * (hedge_ratio * p1) / (hedge_ratio * p1 + p2)
        capital_long_p2 = available_capital - capital_short_p1

        n_shares_short_p1 = capital_short_p1 / p1
        n_shares_long_p2 = capital_long_p2 / p2

        # Calculate costs
        cost_long = n_shares_long_p2 * p2 * (1 + self.commission)
        cost_short = n_shares_short_p1 * p1 * self.commission

        if self.capital > (cost_long + cost_short):
            self.capital -= (cost_long + cost_short)

            # Log the two new positions
            self.active_short_p1 = Position(self.ticker1, 'short', n_shares_short_p1, p1, row.name,
                                            commission=cost_short)
            self.active_long_p2 = Position(self.ticker2, 'long', n_shares_long_p2, p2, row.name, commission=cost_long)

    def _get_current_portfolio_value(self, row: pd.Series) -> float:
        """
        Calculates the total mark-to-market value of the portfolio.
        Value = Cash + (Value of Longs) + (PnL of Shorts)
        """
        value = self.capital

        if self.active_long_p1:
            value += self.active_long_p1.n_shares * row[self.ticker1]
        if self.active_short_p1:
            # PnL of short = (entry - current) * shares
            pnl = (self.active_short_p1.entry_price - row[self.ticker1]) * self.active_short_p1.n_shares
            value += pnl

        if self.active_long_p2:
            value += self.active_long_p2.n_shares * row[self.ticker2]
        if self.active_short_p2:
            pnl = (self.active_short_p2.entry_price - row[self.ticker2]) * self.active_short_p2.n_shares
            value += pnl

        return value

    def run_backtest(self) -> dict:
        """
        Main backtest loop.
        Iterates from 'window_size' to the end of the data.

        Returns:
            A dictionary containing all simulation results and histories.
        """
        # print(f"Running backtest for {self.ticker1} & {self.ticker2} with entry std: {self.entry_threshold:.2f}")

        # Start loop after the initial window
        for i in range(self.window_size, len(self.data)):
            row = self.data.iloc[i]
            rolling_data = self.data.iloc[i - self.window_size: i][[self.ticker1, self.ticker2]]

            p1 = row[self.ticker1]
            p2 = row[self.ticker2]

            # Handle missing data
            if pd.isna(p1) or pd.isna(p2):
                self.portfolio_value.append(self.portfolio_value[-1])
                # We must append to all histories to maintain index alignment
                if len(self.historical_hedge_ratio) > 0:
                    self.historical_hedge_ratio.append(self.historical_hedge_ratio[-1])
                    self.historical_kf1_spread.append(self.historical_kf1_spread[-1])
                    self.historical_filtered_signal.append(self.historical_filtered_signal[-1])
                    self.historical_observed_signal.append(self.historical_observed_signal[-1])
                    self.historical_e1_hat.append(self.historical_e1_hat[-1])
                    self.historical_e2_hat.append(self.historical_e2_hat[-1])
                    self.historical_vecm_norm.append(self.historical_vecm_norm[-1])
                continue

            # --- 1. Update Filters ---
            hedge_ratio = self._update_kf1(p1, p2)
            _ = self._update_kf2(p1, p2, rolling_data)

            # --- 2. Generate Trading Signal ---
            vecm_norm = self._normalize_signal()

            # --- 3. Manage Costs & Positions ---
            self._update_borrow_costs(row)

            is_open = any([self.active_long_p1, self.active_short_p1, self.active_long_p2, self.active_short_p2])

            # --- 4. Check for Exit Signal ---
            if is_open and abs(vecm_norm) < self.exit_threshold:
                self._close_positions(row)

            # Re-check if still open (in case of partial close logic in future)
            is_open = any([self.active_long_p1, self.active_short_p1, self.active_long_p2, self.active_short_p2])

            # --- 5. Check for Entry Signal ---
            if not is_open:
                if vecm_norm > self.entry_threshold:
                    # Signal is high -> Short the spread
                    self._open_short_pair(row, hedge_ratio, p1, p2)
                elif vecm_norm < -self.entry_threshold:
                    # Signal is low -> Long the spread
                    self._open_long_pair(row, hedge_ratio, p1, p2)

            # --- 6. Log Portfolio Value ---
            current_value = self._get_current_portfolio_value(row)
            self.portfolio_value.append(current_value)

        # --- End of Loop: Close all positions ---
        self._close_positions(self.data.iloc[-1])

        # --- 7. Calculate Final Metrics ---
        metrics = TradingMetrics(self.portfolio_value)

        return {
            "metrics": metrics.get_metrics_summary(),
            "metrics_obj": metrics,
            "positions": self.all_positions,
            "portfolio_value": self.portfolio_value,
            "hedge_ratio_history": self.historical_hedge_ratio,
            "vecm_norm_history": self.historical_vecm_norm,
            "vecm_filtered_history": self.historical_filtered_signal,
            "vecm_observed_history": self.historical_observed_signal,
            "ticker1": self.ticker1,
            "ticker2": self.ticker2,
            "entry_threshold": self.entry_threshold,
            "kf1_spread_history": self.historical_kf1_spread,
            "e1_history": self.historical_e1_hat,
            "e2_history": self.historical_e2_hat
        }