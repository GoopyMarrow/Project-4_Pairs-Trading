import numpy as np
from datetime import datetime, timedelta

# Data Settings
DATA_END_DATE = datetime(2025, 11, 11)
DATA_START_DATE = DATA_END_DATE - timedelta(days=int(15 * 365))


# Tickers to screen for pairs
TICKER_LIST = ["XOM","CVX","COP","EOG","MPC","PSX","KMI","WMB","OKE","SLB",
           "VLO","DVN","OXY","ET","PXD","HES","FANG","BKR","HAL","ENPH"]

# Data Split
TRAIN_PCT = 0.6
TEST_PCT = 0.4

# Backtest Settings
INITIAL_CAPITAL = 1_000_000.0
COMMISSION_RATE = 0.00125  # 0.125%
ANNUAL_BORROW_RATE = 0.0025 # 0.25%
DAILY_BORROW_RATE = ANNUAL_BORROW_RATE / 252.0
POSITION_SIZING_PCT = 0.40

# Cointegration & KF Settings
DEFAULT_WINDOW_SIZE = 252
ADF_P_VALUE_THRESHOLD = 0.05
CORRELATION_THRESHOLD = 0.6
OPTIMIZATION_MIN_STD = 0.5
OPTIMIZATION_MAX_STD = 3.0
OPTIMIZATION_STEP = 0.1
ENTRY_THRESHOLD = None # To be optimized
EXIT_THRESHOLD = 0.05

# Kalman Filter 1 (Hedge Ratio) Settings
# Model: P2 = B0 + B1*P1 + e
# State: [B0, B1] (n=2)
KF1_N_DIM = 2
KF1_R = 0.01
KF1_Q = np.eye(KF1_N_DIM) * 0.01
KF1_P0 = np.eye(KF1_N_DIM) * 0.01
KF1_W0 = np.array([0.0, 1.0]) # Initial guess [intercept, slope]

# Kalman Filter 2 (VECM Signal) Settings
# Model: Spread = e1*P1 + e2*P2 + e
# State: [e1, e2] (n=2)
KF2_N_DIM = 2
KF2_R = 0.01
KF2_Q = np.eye(KF2_N_DIM) * 0.0001
KF2_P0 = np.eye(KF2_N_DIM) * 0.01
KF2_W0 = np.array([1.0, -1.0]) # Placeholder, will be overwritten by first Johansen test