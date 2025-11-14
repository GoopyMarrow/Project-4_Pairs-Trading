import numpy as np
import pandas as pd

"""
Performance Metrics Module

Contains the TradingMetrics class to calculate key performance indicators
from a portfolio value time series.
"""


class TradingMetrics:
    """
    Calculates key performance metrics from a portfolio value series.

    Attributes:
        portfolio_values (pd.Series): Time series of portfolio values.
        days_per_year (int): Trading days in a year (for annualization).
        returns (pd.Series): Daily percentage returns.
    """

    def __init__(self, portfolio_values: list[float], days_per_year: int = 252):
        """
        Initializes the metrics calculator.

        Args:
            portfolio_values: A list of portfolio values from the backtest.
            days_per_year: The number of trading days to use for annualization.
        """
        self.portfolio_values = pd.Series(portfolio_values, name="value").fillna(method="ffill")
        self.days_per_year = days_per_year
        self.returns = self.portfolio_values.pct_change().dropna()

    def calculate_sharpe_ratio(self, risk_free_rate: float = 0.0) -> float:
        """
        Calculates the annualized Sharpe Ratio.
        Formula: (Mean(Returns) - RiskFree) / Std(Returns) * Sqrt(Days)
        """
        if self.returns.std() == 0:
            return 0.0

        mean_return = self.returns.mean()
        std_return = self.returns.std()

        sharpe = (mean_return - (risk_free_rate / self.days_per_year)) / std_return
        annualized_sharpe = sharpe * np.sqrt(self.days_per_year)
        return annualized_sharpe

    def calculate_sortino_ratio(self, risk_free_rate: float = 0.0) -> float:
        """
        Calculates the annualized Sortino Ratio.
        Formula: (Mean(Returns) - Target) / Std(Downside Returns) * Sqrt(Days)
        """
        mean_return = self.returns.mean()
        target_return = risk_free_rate / self.days_per_year

        # Calculate downside deviation
        downside_returns = self.returns[self.returns < target_return]
        downside_std = downside_returns.std()

        if downside_std == 0 or pd.isna(downside_std):
            return 0.0

        sortino = (mean_return - target_return) / downside_std
        annualized_sortino = sortino * np.sqrt(self.days_per_year)
        return annualized_sortino

    def calculate_max_drawdown(self) -> float:
        """
        Calculates the Maximum Drawdown (MDD).
        The largest peak-to-trough drop in portfolio value.
        """
        # Calculate the cumulative maximum
        cumulative_max = self.portfolio_values.cummax()
        # Calculate the drawdown
        drawdown = (self.portfolio_values - cumulative_max) / cumulative_max
        # Return the minimum (most negative) drawdown
        return drawdown.min()

    def calculate_calmar_ratio(self) -> float:
        """
        Calculates the Calmar Ratio.
        Formula: Annualized Return / Abs(Max Drawdown)
        """
        # Calculate annualized return
        total_return = (self.portfolio_values.iloc[-1] / self.portfolio_values.iloc[0]) - 1
        num_years = len(self.portfolio_values) / self.days_per_year

        # Handle case where num_years is 0
        if num_years == 0:
            return 0.0

        annualized_return = (1 + total_return) ** (1 / num_years) - 1

        max_dd = self.calculate_max_drawdown()

        if max_dd == 0:
            return 0.0

        calmar = annualized_return / abs(max_dd)
        return calmar

    def get_metrics_summary(self) -> dict:
        """
        Generates a dictionary of all key performance metrics.

        Returns:
            A dictionary containing the calculated metrics.
        """
        return {
            "Sharpe Ratio": self.calculate_sharpe_ratio(),
            "Sortino Ratio": self.calculate_sortino_ratio(),
            "Max Drawdown": self.calculate_max_drawdown(),
            "Calmar Ratio": self.calculate_calmar_ratio(),
            "Final Portfolio Value": self.portfolio_values.iloc[-1],
            "Total Return": (self.portfolio_values.iloc[-1] / self.portfolio_values.iloc[0]) - 1
        }