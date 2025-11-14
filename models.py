from dataclasses import dataclass
from datetime import datetime

"""
Data Models Module

Contains dataclasses used to structure data, such as trading positions.
"""

@dataclass
class Position:
    """
    A dataclass to store all information about a single trading position.

    Attributes:
        ticker (str): The ticker symbol.
        position_type (str): 'long' or 'short'.
        n_shares (float): The number of shares held.
        entry_price (float): The price at which the position was opened.
        entry_date (datetime): The date the position was opened.
        exit_price (float): The price at which the position was closed.
        exit_date (datetime): The date the position was closed.
        commission (float): Total commissions paid for this position (entry + exit).
        borrow_cost (float): Total borrow costs accrued for this position.
        pnl (float): The net profit or loss from the trade.
    """
    ticker: str
    position_type: str
    n_shares: float
    entry_price: float
    entry_date: datetime
    exit_price: float = 0.0
    exit_date: datetime = None
    commission: float = 0.0
    borrow_cost: float = 0.0
    pnl: float = 0.0

    def is_open(self) -> bool:
        """Helper method to check if the position is still open."""
        return self.exit_date is None