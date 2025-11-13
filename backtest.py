import pandas as pd
import numpy as np
from statsmodels.tsa.vector_ar.vecm import coint_johansen

import config
from kalman_filter import KalmanFilter
from models import Position
from performance import TradingMetrics


class PairsTradingBacktest:
    def __init__(self, data: pd.DataFrame, ticker1: str, ticker2: str, window_size: int, entry_threshold: float):
        self.data = data
        self.ticker1 = ticker1
        self.ticker2 = ticker2
        self.window_size = window_size
        self.entry_threshold = entry_threshold
        self.exit_threshold = config.EXIT_THRESHOLD

        self.capital = config.INITIAL_CAPITAL
        self.commission = config.COMMISSION_RATE
        self.borrow_rate = config.DAILY_BORROW_RATE

        self.portfolio_value = [self.capital] * window_size
        self.all_positions = []
        self.active_long_p1 = None
        self.active_short_p1 = None
        self.active_long_p2 = None
        self.active_short_p2 = None

        self.kf_hedge = self._init_kf1()
        self.kf_vecm = self._init_kf2()

        self.historical_filtered_signal = []
        self.historical_observed_signal = []
        self.historical_hedge_ratio = []
        self.historical_vecm_norm = []

    def _init_kf1(self) -> KalmanFilter:
        return KalmanFilter(
            n_dim=config.KF1_N_DIM,
            R=config.KF1_R,
            Q=config.KF1_Q,
            P0=config.KF1_P0,
            w0=config.KF1_W0
        )

    def _init_kf2(self) -> KalmanFilter:
        initial_data = self.data.iloc[:self.window_size]
        try:
            initial_jo = coint_johansen(initial_data[[self.ticker1, self.ticker2]], 0, 1)
            initial_evec = initial_jo.evec[:, 0]
        except Exception:
            initial_evec = config.KF2_W0

        return KalmanFilter(
            n_dim=config.KF2_N_DIM,
            R=config.KF2_R,
            Q=config.KF2_Q,
            P0=config.KF2_P0,
            w0=initial_evec
        )

    def _update_kf1(self, p1: float, p2: float) -> float:
        x_t_hedge = np.array([1, p1])
        y_t_hedge = p2
        w_pred, P_pred = self.kf_hedge.predict()
        w_upd, P_upd = self.kf_hedge.update(x_t_hedge, y_t_hedge, w_pred, P_pred)
        hedge_ratio = w_upd[1]
        self.historical_hedge_ratio.append(hedge_ratio)
        return hedge_ratio

    def _update_kf2(self, p1: float, p2: float, rolling_data: pd.DataFrame) -> float:
        try:
            jo_result = coint_johansen(rolling_data, 0, 1)
            obs_evec = jo_result.evec[:, 0]
        except Exception:
            obs_evec = self.kf_vecm.w_state

        x_t_vecm_prices = np.array([p1, p2])
        y_t_vecm_signal = np.dot(obs_evec, x_t_vecm_prices)
        self.historical_observed_signal.append(y_t_vecm_signal)

        e_pred, P_pred_e = self.kf_vecm.predict()
        e_upd, P_upd_e = self.kf_vecm.update(x_t_vecm_prices, y_t_vecm_signal, e_pred, P_pred_e)

        filtered_signal = np.dot(e_upd, x_t_vecm_prices)
        self.historical_filtered_signal.append(filtered_signal)
        return filtered_signal

    def _normalize_signal(self) -> float:
        if len(self.historical_filtered_signal) < self.window_size:
            self.historical_vecm_norm.append(0.0)
            return 0.0

        signal_window = np.array(self.historical_filtered_signal[-self.window_size:])
        mean = np.mean(signal_window)
        std = np.std(signal_window)

        if std == 0:
            norm_signal = 0.0
        else:
            current_signal = self.historical_filtered_signal[-1]
            norm_signal = (current_signal - mean) / std

        self.historical_vecm_norm.append(norm_signal)
        return norm_signal

    def _update_borrow_costs(self, row: pd.Series):
        if self.active_short_p1:
            cost = self.active_short_p1.n_shares * row[self.ticker1] * self.borrow_rate
            self.capital -= cost
            self.active_short_p1.borrow_cost += cost

        if self.active_short_p2:
            cost = self.active_short_p2.n_shares * row[self.ticker2] * self.borrow_rate
            self.capital -= cost
            self.active_short_p2.borrow_cost += cost

    def _close_positions(self, row: pd.Series):
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
            return_capital = self.active_short_p1.n_shares * self.active_short_p1.entry_price
            pnl = (self.active_short_p1.entry_price - price) * self.active_short_p1.n_shares

            self.capital += return_capital + pnl - trade_cost

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
            return_capital = self.active_short_p2.n_shares * self.active_short_p2.entry_price
            pnl = (self.active_short_p2.entry_price - price) * self.active_short_p2.n_shares

            self.capital += return_capital + pnl - trade_cost

            self.active_short_p2.exit_price = price
            self.active_short_p2.exit_date = row.name
            self.active_short_p2.commission += trade_cost
            self.active_short_p2.pnl = pnl - self.active_short_p2.commission - self.active_short_p2.borrow_cost
            self.all_positions.append(self.active_short_p2)
            self.active_short_p2 = None

    def _open_short_pair(self, row: pd.Series, hedge_ratio: float, p1: float, p2: float):
        available_capital = self.capital * config.POSITION_SIZING_PCT

        # Short P2, Long P1
        # P2 = B0 + B1*P1 -> P2 - B1*P1 = Spread
        # If spread is high, sell spread: Short P2, Long P1

        capital_long_p1 = available_capital * (hedge_ratio * p1) / (hedge_ratio * p1 + p2)
        capital_short_p2 = available_capital - capital_long_p1

        n_shares_long_p1 = capital_long_p1 / p1
        n_shares_short_p2 = capital_short_p2 / p2

        cost_long = n_shares_long_p1 * p1 * (1 + self.commission)
        cost_short = n_shares_short_p2 * p2 * self.commission

        if self.capital > (cost_long + cost_short):
            self.capital -= (cost_long + cost_short)

            self.active_long_p1 = Position(self.ticker1, 'long', n_shares_long_p1, p1, row.name, commission=cost_long)

            self.active_short_p2 = Position(self.ticker2, 'short', n_shares_short_p2, p2, row.name,
                                            commission=cost_short)

    def _open_long_pair(self, row: pd.Series, hedge_ratio: float, p1: float, p2: float):
        available_capital = self.capital * config.POSITION_SIZING_PCT

        # Long P2, Short P1
        # If spread is low, buy spread: Long P2, Short P1

        capital_short_p1 = available_capital * (hedge_ratio * p1) / (hedge_ratio * p1 + p2)
        capital_long_p2 = available_capital - capital_short_p1

        n_shares_short_p1 = capital_short_p1 / p1
        n_shares_long_p2 = capital_long_p2 / p2

        cost_long = n_shares_long_p2 * p2 * (1 + self.commission)
        cost_short = n_shares_short_p1 * p1 * self.commission

        if self.capital > (cost_long + cost_short):
            self.capital -= (cost_long + cost_short)

            self.active_short_p1 = Position(self.ticker1, 'short', n_shares_short_p1, p1, row.name,
                                            commission=cost_short)

            self.active_long_p2 = Position(self.ticker2, 'long', n_shares_long_p2, p2, row.name, commission=cost_long)

    def _get_current_portfolio_value(self, row: pd.Series) -> float:
        value = self.capital

        if self.active_long_p1:
            value += self.active_long_p1.n_shares * row[self.ticker1]
        if self.active_short_p1:
            pnl = (self.active_short_p1.entry_price - row[self.ticker1]) * self.active_short_p1.n_shares
            value += (self.active_short_p1.entry_price * self.active_short_p1.n_shares) + pnl

        if self.active_long_p2:
            value += self.active_long_p2.n_shares * row[self.ticker2]
        if self.active_short_p2:
            pnl = (self.active_short_p2.entry_price - row[self.ticker2]) * self.active_short_p2.n_shares
            value += (self.active_short_p2.entry_price * self.active_short_p2.n_shares) + pnl

        return value

    def run_backtest(self) -> dict:
        print(f"Running backtest for {self.ticker1} & {self.ticker2} with entry std: {self.entry_threshold:.2f}")

        for i in range(self.window_size, len(self.data)):
            row = self.data.iloc[i]
            rolling_data = self.data.iloc[i - self.window_size: i][[self.ticker1, self.ticker2]]

            p1 = row[self.ticker1]
            p2 = row[self.ticker2]

            if pd.isna(p1) or pd.isna(p2):
                self.portfolio_value.append(self.portfolio_value[-1])
                continue

            hedge_ratio = self._update_kf1(p1, p2)

            _ = self._update_kf2(p1, p2, rolling_data)

            vecm_norm = self._normalize_signal()

            self._update_borrow_costs(row)

            is_open = any([self.active_long_p1, self.active_short_p1, self.active_long_p2, self.active_short_p2])

            if is_open and abs(vecm_norm) < self.exit_threshold:
                self._close_positions(row)

            is_open = any([self.active_long_p1, self.active_short_p1, self.active_long_p2, self.active_short_p2])

            if not is_open:
                if vecm_norm > self.entry_threshold:
                    self._open_short_pair(row, hedge_ratio, p1, p2)
                elif vecm_norm < -self.entry_threshold:
                    self._open_long_pair(row, hedge_ratio, p1, p2)

            current_value = self._get_current_portfolio_value(row)
            self.portfolio_value.append(current_value)

        self._close_positions(self.data.iloc[-1])

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
            "entry_threshold": self.entry_threshold
        }