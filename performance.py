import numpy as np
import pandas as pd


class TradingMetrics:
    def __init__(self, portfolio_values: list[float], days_per_year: int = 252):
        self.portfolio_values = pd.Series(portfolio_values, name="value").fillna(method="ffill")
        self.days_per_year = days_per_year
        self.returns = self.portfolio_values.pct_change().dropna()

    def calculate_sharpe_ratio(self, risk_free_rate: float = 0.0) -> float:
        if self.returns.std() == 0:
            return 0.0

        mean_return = self.returns.mean()
        std_return = self.returns.std()

        sharpe = (mean_return - (risk_free_rate / self.days_per_year)) / std_return
        annualized_sharpe = sharpe * np.sqrt(self.days_per_year)
        return annualized_sharpe

    def calculate_sortino_ratio(self, risk_free_rate: float = 0.0) -> float:
        mean_return = self.returns.mean()
        target_return = risk_free_rate / self.days_per_year

        downside_returns = self.returns[self.returns < target_return]
        downside_std = downside_returns.std()

        if downside_std == 0 or pd.isna(downside_std):
            return 0.0

        sortino = (mean_return - target_return) / downside_std
        annualized_sortino = sortino * np.sqrt(self.days_per_year)
        return annualized_sortino

    def calculate_max_drawdown(self) -> float:
        cumulative_max = self.portfolio_values.cummax()
        drawdown = (self.portfolio_values - cumulative_max) / cumulative_max
        return drawdown.min()

    def calculate_calmar_ratio(self) -> float:
        total_return = (self.portfolio_values.iloc[-1] / self.portfolio_values.iloc[0]) - 1
        num_years = len(self.portfolio_values) / self.days_per_year
        annualized_return = (1 + total_return) ** (1 / num_years) - 1

        max_dd = self.calculate_max_drawdown()

        if max_dd == 0:
            return 0.0

        calmar = annualized_return / abs(max_dd)
        return calmar

    def get_metrics_summary(self) -> dict:
        return {
            "Sharpe Ratio": self.calculate_sharpe_ratio(),
            "Sortino Ratio": self.calculate_sortino_ratio(),
            "Max Drawdown": self.calculate_max_drawdown(),
            "Calmar Ratio": self.calculate_calmar_ratio(),
            "Final Portfolio Value": self.portfolio_values.iloc[-1],
            "Total Return": (self.portfolio_values.iloc[-1] / self.portfolio_values.iloc[0]) - 1
        }