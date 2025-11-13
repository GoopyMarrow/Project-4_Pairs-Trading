from dataclasses import dataclass
from datetime import datetime

@dataclass
class Position:
    ticker: str
    position_type: str # 'long' or 'short'
    n_shares: float
    entry_price: float
    entry_date: datetime
    exit_price: float = 0.0
    exit_date: datetime = None
    commission: float = 0.0
    borrow_cost: float = 0.0
    pnl: float = 0.0

    def is_open(self) -> bool:
        return self.exit_date is None