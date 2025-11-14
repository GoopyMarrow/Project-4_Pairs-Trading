import numpy as np
from datetime import datetime, timedelta

"""
Global Configuration File

Contains all tunable parameters for the backtest, including API settings,
backtest parameters, and Kalman Filter hyperparameters.
"""

# --- Data Settings ---
# Set data range (15 years)
DATA_END_DATE = datetime(2025, 11, 11)
DATA_START_DATE = DATA_END_DATE - timedelta(days=int(15 * 365))

# Tickers to screen for pairs (Energy Sector)
TICKER_LIST = ["XOM","CVX","COP","EOG","MPC","PSX","KMI","WMB","OKE","SLB",
           "VLO","DVN","OXY","ET","PXD","HES","FANG","BKR","HAL","ENPH"]

# --- Data Split ---
# Defines the percentage split for training and testing data.
TRAIN_PCT = 0.6
TEST_PCT = 0.4 # The remainder after TRAIN_PCT

# --- Backtest Settings ---
INITIAL_CAPITAL = 1_000_000.0
COMMISSION_RATE = 0.00125    # 0.125% per trade
ANNUAL_BORROW_RATE = 0.0025  # 0.25% annualized
DAILY_BORROW_RATE = ANNUAL_BORROW_RATE / 252.0
POSITION_SIZING_PCT = 0.80   # Invest 80% of capital per pair trade

# --- Cointegration & Optimization Settings ---
DEFAULT_WINDOW_SIZE = 252    # 1 trading year for Johansen and Z-score
ADF_P_VALUE_THRESHOLD = 0.05 # Significance level for EG and ADF tests
CORRELATION_THRESHOLD = 0.6  # Minimum correlation to test a pair
OPTIMIZATION_MIN_STD = 0.5   # Start of theta optimization range
OPTIMIZATION_MAX_STD = 3.0   # End of theta optimization range
OPTIMIZATION_STEP = 0.1      # Step size for theta optimization
ENTRY_THRESHOLD = None       # To be set by optimization
EXIT_THRESHOLD = 0.50        # Z-score threshold to close a position (was 0.10)

# --- Kalman Filter 1 (Hedge Ratio) Settings ---
# Model: P2 = B0 + B1*P1 + e
# State: [B0, B1] (n=2)
# Q/R Ratio = 10 (Increased reactivity)
KF1_N_DIM = 2
KF1_R = 0.0001                   # Observation noise (trust prices)
KF1_Q = np.eye(KF1_N_DIM) * 0.001 # Process noise (hedge ratio is somewhat stable)
KF1_P0 = np.eye(KF1_N_DIM) * 0.01 # Initial covariance
KF1_W0 = np.array([0.0, 1.0])    # Initial guess [intercept, slope]

# --- Kalman Filter 2 (VECM Signal) Settings ---
# Model: Spread = e1*P1 + e2*P2 + e
# State: [e1, e2] (n=2)
# Q/R Ratio = 100 (High reactivity)
KF2_N_DIM = 2
KF2_R = 0.00001                  # Observation noise (trust Johansen)
KF2_Q = np.eye(KF2_N_DIM) * 0.001 # Process noise (eigenvector is somewhat stable)
KF2_P0 = np.eye(KF2_N_DIM) * 0.01 # Initial covariance
KF2_W0 = np.array([1.0, -1.0])   # Placeholder, overwritten by first Johansen test