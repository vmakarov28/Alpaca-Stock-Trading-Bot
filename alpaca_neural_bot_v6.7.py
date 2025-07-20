# +------------------------------------------------------------------------------+
# |                            Alpaca Neural Bot v7.0                            |
# +------------------------------------------------------------------------------+
# | Author: Vladimir Makarov                                                     |
# | Project Start Date: May 9, 2025                                              |
# | License: GNU Lesser General Public License v2.1                              |
# | Version: 7.0 (Hybrid Mean Reversion-Momentum Model)                          |
# |                                                                              |
# | Dependencies:                                                                |
# | - torch (Neural network framework)                                           |
# | - numpy (Numerical computations)                                             |
# | - pandas (Data manipulation)                                                 |
# | - alpaca-py (Alpaca integration, imports as 'alpaca')                        |
# | - transformers (Sentiment analysis)                                          |
# | - scikit-learn (Machine learning utilities)                                  |
# | - ta-lib (Technical analysis)                                                |
# | - tenacity (Retry logic)                                                     |
# | - smtplib (Email notifications)                                              |
# | - argparse (Command-line parsing)                                            |
# | - tqdm (Progress bars)                                                       |
# | - colorama (Console formatting)                                              |
# | Install using: pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128 |
# |                pip install alpaca-py transformers pandas numpy scikit-learn ta-lib tenacity tqdm colorama |
# |                pip install protobuf==5.28.3                                  |
# |                                                                              |
# | Double check before starting:                                                |
# | - Ensure Alpaca API keys are configured in CONFIG.                           |
# | - Requires stable internet for live trading and data fetching.               |
# | - GitHub: https://github.com/vmakarov28/Alpaca-Stock-Trading-Bot/tree/main   |
# |                                                                              |
# +------------------------------------------------------------------------------+
# Hybrid model incorporating mean reversion and momentum strategies from Ernest P. Chan
# Added regime detection, cointegration, momentum indicators, Kelly sizing, etc.

import os
import sys
import logging
import argparse
import importlib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from alpaca.data import StockHistoricalDataClient, TimeFrame, TimeFrameUnit
from alpaca.data.requests import StockBarsRequest
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
from transformers import pipeline
from sklearn.preprocessing import StandardScaler
import smtplib
from email.mime.text import MIMEText
from datetime import datetime, timedelta, timezone
import talib
import pickle
from typing import List, Tuple, Dict, Optional
import warnings
import time
from tenacity import retry, stop_after_attempt, wait_fixed, retry_if_exception_type
from tqdm import tqdm
from colorama import Fore, Style
import colorama
import multiprocessing as mp
import pywt
import optuna
from arch import arch_model
from imblearn.over_sampling import SMOTE
from statsmodels.tsa.stattools import coint
from scipy.stats import norm
from torch.cuda.amp import autocast, GradScaler

# Suppress PyTorch warnings
warnings.filterwarnings('ignore', category=UserWarning)

# Initialize colorama for colored console output
colorama.init()

CONFIG = {
    # Trading Parameters - Settings related to trading operations
    'SYMBOLS': ['SPY', 'MSFT', 'AAPL', 'AMZN', 'NVDA', 'META', 'TSLA', 'GOOGL'],  # List of stock symbols to trade
    'TIMEFRAME': TimeFrame(15, TimeFrameUnit.Minute),  # Time interval for data fetching
    'INITIAL_CASH': 100000.00,  # Starting cash for trading simulation
    'MIN_HOLDING_PERIOD_MINUTES': 45,  # Minimum holding period for trades
    'TRANSACTION_COST_PER_TRADE': 0.25,  # Cost per trade

    # Data Fetching and Caching - Parameters for data retrieval and storage
    'TRAIN_DATA_START_DATE': '2015-01-01',  # Start date for training data
    'BACKTEST_START_DATE': '2024-01-01',  # Start date for backtesting
    'SIMULATION_DAYS': 180,  # Number of days for simulation
    'MIN_DATA_POINTS': 100,  # Minimum data points required for processing
    'CACHE_DIR': './cache',  # Directory for caching data
    'CACHE_EXPIRY_SECONDS': 24 * 60 * 60,  # Expiry time for cached data in seconds
    'LIVE_DATA_BARS': 200,  # Number of bars to fetch for live data

    # Model Training - Settings for training the machine learning model
    'TRAIN_EPOCHS': 100,  # Number of epochs for training the model
    'BATCH_SIZE': 32,  # Batch size for training
    'TIMESTEPS': 30,  # Number of time steps for sequence data
    'EARLY_STOPPING_MONITOR': 'val_loss',  # Metric to monitor for early stopping
    'EARLY_STOPPING_PATIENCE': 15,  # Patience for early stopping
    'EARLY_STOPPING_MIN_DELTA': 0.0005,  # Minimum delta for early stopping

    # API and Authentication - Credentials for API access
    'ALPACA_API_KEY': 'PK1A36K33FUZKR7OAJC2',  # API key for Alpaca
    'ALPACA_SECRET_KEY': 'fid8r2QWmziK3zvN3HqgvuKJWux3HCUg6Jez39fY',  # Secret key for Alpaca

    # Email Notifications - Configuration for sending email alerts
    'EMAIL_SENDER': 'alpaca.ai.tradingbot@gmail.com',  # Email address for sending notifications
    'EMAIL_PASSWORD': 'hjdf sstp pyne rotq',  # Password for the email account
    'EMAIL_RECEIVER': ['aiplane.scientist@gmail.com', 'vmakarov28@students.d125.org', 'tchaikovskiy@hotmail.com'],  # List of email recipients
    'SMTP_SERVER': 'smtp.gmail.com',  # SMTP server for email
    'SMTP_PORT': 587,  # Port for SMTP server

    # Logging and Monitoring - Settings for tracking activities
    'LOG_FILE': 'trades.log',  # File for logging trades

    # Risk Management - Parameters to control trading risk
    'MAX_DRAWDOWN_LIMIT': 0.04,  # Maximum allowed drawdown
    'RISK_PERCENTAGE': 0.05,  # Percentage of cash to risk per trade
    'STOP_LOSS_ATR_MULTIPLIER': 1.0,  # Multiplier for ATR-based stop loss
    'TAKE_PROFIT_ATR_MULTIPLIER': 2.5,  # Multiplier for ATR-based take profit
    'TRAILING_STOP_PERCENTAGE': 0.015,  # Percentage for trailing stop
    'SECTOR_ALLOCATION_LIMIT': 0.15,  # Max allocation per sector

    # Strategy Thresholds - Thresholds for trading decisions
    'CONFIDENCE_THRESHOLD': 0.45,  # Threshold for prediction confidence
    'PREDICTION_THRESHOLD_BUY': 0.4,  # Threshold for buy signal
    'PREDICTION_THRESHOLD_SELL': 0.20,  # Threshold for sell signal
    'RSI_REVERSION_BUY': 30,
    'RSI_REVERSION_SELL': 70,
    'RSI_MOMENTUM_BUY': 55,
    'RSI_MOMENTUM_SELL': 45,
    'ADX_TREND_THRESHOLD': 16,  # Threshold for ADX trend strength
    'MAX_VOLATILITY': 4.1,  # Maximum allowed volatility
    'ANOMALY_VOL_MULTIPLIER': 3.0,  # Multiplier for anomaly detection

    # Sentiment Analysis - Settings for sentiment analysis
    'SENTIMENT_MODEL': 'distilbert-base-uncased-finetuned-sst-2-english',  # Model for sentiment analysis

    # API Retry Settings - Configuration for handling API failures
    'API_RETRY_ATTEMPTS': 3,  # Number of retry attempts for API calls
    'API_RETRY_DELAY': 1000,  # Delay between retry attempts in milliseconds

    # New Hybrid Parameters
    'MOMENTUM_LOOKBACK': 12,  # For ROC momentum
    'GAP_THRESHOLD': 1.0,  # Gap size in ATR multiples
    'COINT_WINDOW': 20,  # Rolling window for cointegration
    'HALF_LIFE_WINDOW': 20,  # Window for half-life calculation
    'REGIME_VOL_THRESHOLD': 1.5,  # GARCH volatility threshold for regime switch
    'ONLINE_RETRAIN_INTERVAL': 7,  # Days for online retrain
    'VOLUME_SPIKE_THRESHOLD': 1.5,  # Threshold for volume spike
    'EMA_SHORT': 50,  # Short EMA for trend
    'EMA_LONG': 200,  # Long EMA for trend
    'FIB_RETRACEMENT_LEVELS': [0.236, 0.382, 0.618],  # Fibonacci levels for mean reversion
    'ZSCORE_THRESHOLD': 2.0,  # Z-score threshold for arbitrage

    # Sectors for risk management
    'SECTORS': {
        'SPY': 'Index',
        'MSFT': 'Technology',
        'AAPL': 'Technology',
        'AMZN': 'Consumer Cyclical',
        'NVDA': 'Technology',
        'META': 'Communication Services',
        'TSLA': 'Consumer Cyclical',
        'GOOGL': 'Communication Services'
    }
}

# Configure logging
root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler = logging.FileHandler(CONFIG['LOG_FILE'])
file_handler.setFormatter(formatter)
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
root_logger.addHandler(file_handler)
root_logger.addHandler(stream_handler)

logger = logging.getLogger(__name__)

# Initialize sentiment analysis pipeline
sentiment_pipeline = pipeline("sentiment-analysis", model=CONFIG['SENTIMENT_MODEL'], framework="pt")

def check_dependencies() -> None:
    """Check for required Python modules."""
    required_modules = [
        'torch', 'numpy', 'pandas', 'alpaca', 'transformers',
        'sklearn', 'talib', 'tenacity', 'smtplib', 'argparse', 'tqdm', 'colorama',
        'pywt', 'optuna', 'statsmodels', 'arch', 'gplearn', 'imblearn'
    ]
    for module in required_modules:
        try:
            importlib.import_module(module)
        except ImportError:
            raise ImportError(f"Module '{module}' is required. Install it using: pip install {module}")

def validate_config(config: Dict) -> None:
    """Validate configuration parameters."""
    if not config['SYMBOLS']:
        raise ValueError("SYMBOLS list cannot be empty")
    if not isinstance(config['TIMEFRAME'], TimeFrame):
        raise ValueError("TIMEFRAME must be a valid TimeFrame object")
    for param in ['SIMULATION_DAYS', 'TRAIN_EPOCHS', 'BATCH_SIZE', 'TIMESTEPS', 'MIN_DATA_POINTS']:
        if not isinstance(config[param], int) or config[param] <= 0:
            raise ValueError(f"{param} must be a positive integer")
    for param in ['INITIAL_CASH', 'STOP_LOSS_ATR_MULTIPLIER', 'TAKE_PROFIT_ATR_MULTIPLIER', 'MAX_DRAWDOWN_LIMIT', 'RISK_PERCENTAGE']:
        if not isinstance(config[param], (int, float)) or config[param] <= 0:
            raise ValueError(f"{param} must be a positive number")

def create_cache_directory() -> None:
    """Create cache directory if it doesn't exist."""
    os.makedirs(CONFIG['CACHE_DIR'], exist_ok=True)

@retry(
    stop=stop_after_attempt(CONFIG['API_RETRY_ATTEMPTS']),
    wait=wait_fixed(CONFIG['API_RETRY_DELAY'] / 1000),
    retry=retry_if_exception_type(Exception)
)
def fetch_data(symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
    """Fetch historical bar data from Alpaca API in yearly increments."""
    try:
        client = StockHistoricalDataClient(CONFIG['ALPACA_API_KEY'], CONFIG['ALPACA_SECRET_KEY'])
        all_bars = []
        current_start = pd.Timestamp(start_date, tz='UTC')
        end_dt = pd.Timestamp(end_date, tz='UTC')
        increment = pd.DateOffset(years=1)

        while current_start < end_dt:
            current_end = min(current_start + increment, end_dt)
            logger.info(f"Fetching data for {symbol} from {current_start} to {current_end}")
            effective_symbol = 'FB' if symbol == 'META' and current_start < pd.Timestamp('2021-10-28', tz='UTC') else symbol
            request = StockBarsRequest(
                symbol_or_symbols=effective_symbol,
                timeframe=CONFIG['TIMEFRAME'],
                start=current_start,
                end=current_end
            )
            bars = client.get_stock_bars(request).df
            if not bars.empty:
                df_bars = bars.reset_index().rename(columns={'vwap': 'VWAP'})
                all_bars.append(df_bars)
                logger.info(f"Fetched {len(df_bars)} bars for {symbol} from {df_bars['timestamp'].min()} to {df_bars['timestamp'].max()}")
            else:
                logger.info(f"No data for {symbol} from {current_start} to {current_end}, skipping")
            current_start = current_end

        if all_bars:
            df = pd.concat(all_bars).sort_values('timestamp')
            df = df.drop_duplicates(subset='timestamp', keep='first')
            logger.info(f"Total fetched {len(df)} bars for {symbol}")
        else:
            df = pd.DataFrame(columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'VWAP'])

        if len(df) < CONFIG['MIN_DATA_POINTS']:
            raise ValueError(f"Insufficient data for {symbol}: got {len(df)} bars, need {CONFIG['MIN_DATA_POINTS']}")
        return df
    except Exception as e:
        logger.error(f"Error fetching data for {symbol}: {str(e)}")
        raise

def load_or_fetch_data(symbol: str, start_date: str, end_date: str) -> Tuple[pd.DataFrame, bool]:
    """Load historical data from cache or fetch from API."""
    cache_file = os.path.join(CONFIG['CACHE_DIR'], f"{symbol}_data.pkl")
    if os.path.exists(cache_file) and (time.time() - os.path.getmtime(cache_file)) < CONFIG['CACHE_EXPIRY_SECONDS']:
        with open(cache_file, 'rb') as f:
            df = pickle.load(f)
        logger.info(f"Loaded {len(df)} bars for {symbol} from cache")
        return df, True
    else:
        df = fetch_data(symbol, start_date, datetime.now().strftime('%Y-%m-%d'))
        with open(cache_file, 'wb') as f:
            pickle.dump(df, f)
        logger.info(f"Fetched and cached {len(df)} bars for {symbol}")
        return df, False

@retry(
    stop=stop_after_attempt(CONFIG['API_RETRY_ATTEMPTS']),
    wait=wait_fixed(CONFIG['API_RETRY_DELAY'] / 1000),
    retry=retry_if_exception_type(Exception)
)
def fetch_recent_data(symbol: str, num_bars: int) -> pd.DataFrame:
    """Fetch recent bars for live trading."""
    client = StockHistoricalDataClient(CONFIG['ALPACA_API_KEY'], CONFIG['ALPACA_SECRET_KEY'])
    end_date = datetime.now()
    start_date = end_date - timedelta(days=10)
    request = StockBarsRequest(
        symbol_or_symbols=symbol,
        timeframe=CONFIG['TIMEFRAME'],
        start=start_date,
        end=end_date,
        limit=num_bars
    )
    bars = client.get_stock_bars(request).df
    if bars.empty:
        raise ValueError(f"No recent data for {symbol}")
    df = bars.reset_index().rename(columns={'vwap': 'VWAP'})
    logger.info(f"Fetched {len(df)} recent bars for {symbol}")
    return df.sort_values('timestamp')

def load_news_sentiment(symbol: str) -> Tuple[float, bool]:
    """Compute real-time news sentiment using a pre-trained model or random for testing."""
    cache_file = os.path.join(CONFIG['CACHE_DIR'], f"{symbol}_news_sentiment.pkl")
    # Force new sentiment calculation by ignoring cache
    sentiment_score = np.random.uniform(-1.0, 1.0)  # Random sentiment for testing
    with open(cache_file, 'wb') as f:
        pickle.dump(sentiment_score, f)
    return sentiment_score, False

def calculate_half_life(series: pd.Series) -> float:
    """Calculate half-life of mean reversion using Ornstein-Uhlenbeck process."""
    lag = series.shift(1)
    ret = series - lag
    lag2 = lag.shift(1)
    ret2 = lag - lag2
    slope, intercept = np.polyfit(lag2.dropna(), ret2.dropna(), 1)
    half_life = -np.log(2) / slope if slope != 0 else np.inf
    return half_life

def detect_regime(returns: pd.Series) -> int:
    """Detect market regime using simple rolling volatility."""
    cond_vol = returns.rolling(20).std().iloc[-1]
    if cond_vol > CONFIG['REGIME_VOL_THRESHOLD']:
        return 1  # Momentum regime
    else:
        return 0  # Mean reversion regime

def calculate_indicators(df: pd.DataFrame, sentiment: float, pair_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """Calculate technical indicators, including hybrid features."""
    df = df.copy()
    required_cols = ['open', 'high', 'low', 'close', 'volume']
    if not all(col in df.columns for col in required_cols):
        raise ValueError(f"Missing required columns: {required_cols}")
    
    if 'VWAP' not in df.columns:
        df['VWAP'] = (df['volume'] * (df['high'] + df['low'] + df['close']) / 3).cumsum() / df['volume'].cumsum()
    
    df['MA20'] = talib.SMA(df['close'], timeperiod=20)
    df['MA50'] = talib.SMA(df['close'], timeperiod=50)
    df['RSI'] = talib.RSI(df['close'], timeperiod=14)
    df['MACD'], df['MACD_signal'], _ = talib.MACD(df['close'], fastperiod=12, slowperiod=26, signalperiod=9)
    df['MACD_cross'] = ((df['MACD'] > df['MACD_signal']) & (df['MACD'].shift(1) <= df['MACD_signal'].shift(1))).astype(int)
    df['OBV'] = talib.OBV(df['close'], df['volume'])
    df['ATR'] = talib.ATR(df['high'], df['low'], df['close'], timeperiod=14)
    df['CMF'] = talib.AD(df['high'], df['low'], df['close'], df['volume']) / df['volume'].rolling(20).sum()
    df['Close_ATR'] = df['close'] / df['ATR']
    df['MA20_ATR'] = df['MA20'] / df['ATR']
    df['Return_1d'] = df['close'].pct_change()
    df['Return_5d'] = df['close'].pct_change(periods=5)
    df['Volatility'] = df['close'].rolling(20).std()
    df['BB_upper'], df['BB_middle'], df['BB_lower'] = talib.BBANDS(
        df['close'], timeperiod=20, nbdevup=2, nbdevdn=2
    )
    df['Stoch_K'], df['Stoch_D'] = talib.STOCH(
        df['high'], df['low'], df['close'], fastk_period=14, slowk_period=3, slowd_period=3
    )
    df['ADX'] = talib.ADX(df['high'], df['low'], df['close'], timeperiod=14)
    df['Sentiment'] = sentiment
    df['Trend'] = np.where(df['close'] > df['MA20'], 1, 0)
    
    # New Hybrid Features
    df['ROC'] = talib.ROC(df['close'], timeperiod=CONFIG['MOMENTUM_LOOKBACK'])  # Momentum ROC
    df['Gap'] = (df['open'] - df['close'].shift(1)) / df['ATR'].shift(1)  # Gap in ATR
    df['Half_Life'] = df['close'].rolling(CONFIG['HALF_LIFE_WINDOW']).apply(calculate_half_life, raw=False)
    df['Regime'] = df['Return_1d'].rolling(20).apply(detect_regime, raw=False)
    df['Cointegration'] = 0.0
    if pair_df is not None and len(pair_df) >= CONFIG['COINT_WINDOW']:
        for i in range(CONFIG['COINT_WINDOW'] - 1, len(df)):
            _, pvalue, _ = coint(df['close'].iloc[i - CONFIG['COINT_WINDOW'] + 1:i + 1], pair_df['close'].iloc[i - CONFIG['COINT_WINDOW'] + 1:i + 1])
            df.loc[df.index[i], 'Cointegration'] = pvalue
    # Fama-French factors (placeholder; in practice, fetch from external source)
    df['Size'] = np.log(df['close'] * 1e9)  # Approximate market cap
    df['Value'] = 1 / (df['close'] / df['MA50'])  # Approximate book-to-market

    # Additional for Momentum and Trend
    df['EMA_SHORT'] = talib.EMA(df['close'], timeperiod=CONFIG['EMA_SHORT'])
    df['EMA_LONG'] = talib.EMA(df['close'], timeperiod=CONFIG['EMA_LONG'])
    df['Volume_Spike'] = (df['volume'] / df['volume'].rolling(20).mean()) > CONFIG['VOLUME_SPIKE_THRESHOLD']

    # For Arbitrage
    df['Spread_Zscore'] = 0.0
    if pair_df is not None:
        df['Spread'] = df['close'] - pair_df['close']
        spread_mean = df['Spread'].rolling(20).mean()
        spread_std = df['Spread'].rolling(20).std()
        df['Spread_Zscore'] = (df['Spread'] - spread_mean) / spread_std

    indicator_cols = [
        'MA20', 'MA50', 'RSI', 'MACD', 'MACD_signal', 'MACD_cross', 'OBV', 'VWAP', 'ATR',
        'CMF', 'Close_ATR', 'MA20_ATR', 'Return_1d', 'Return_5d', 'Volatility',
        'BB_upper', 'BB_middle', 'BB_lower', 'Stoch_K', 'Stoch_D', 'ADX', 'Sentiment', 'Trend',
        'ROC', 'Gap', 'Half_Life', 'Regime', 'Cointegration', 'Size', 'Value', 'EMA_SHORT', 'EMA_LONG', 'Volume_Spike', 'Spread_Zscore'
    ]
    df = df.dropna(subset=indicator_cols)
    return df

def validate_raw_data(df: pd.DataFrame, symbol: str) -> None:
    """Validate raw OHLCV data after fetching."""
    if df.empty:
        raise ValueError(f"Empty DataFrame for {symbol}")
    required_cols = ['open', 'high', 'low', 'close', 'volume', 'timestamp']
    if not all(col in df.columns for col in required_cols):
        missing = [col for col in required_cols if col not in df.columns]
        raise ValueError(f"Missing required columns for {symbol}: {missing}")
    if df[required_cols[:-1]].isna().any().any():
        raise ValueError(f"NaN values in OHLCV columns for {symbol}")

def validate_indicators(df: pd.DataFrame, symbol: str) -> None:
    """Validate data after calculating indicators."""
    required_cols = [
        'open', 'high', 'low', 'close', 'volume', 'timestamp',
        'MA20', 'MA50', 'RSI', 'MACD', 'MACD_signal', 'MACD_cross', 'OBV', 'VWAP', 'ATR',
        'CMF', 'Close_ATR', 'MA20_ATR', 'Return_1d', 'Return_5d', 'Volatility',
        'BB_upper', 'BB_lower', 'Stoch_K', 'Stoch_D', 'ADX', 'Sentiment', 'Trend',
        'ROC', 'Gap', 'Half_Life', 'Regime', 'Cointegration', 'Size', 'Value',
        'EMA_SHORT', 'EMA_LONG', 'Volume_Spike', 'Spread_Zscore'
    ]
    if not all(col in df.columns for col in required_cols):
        missing = [col for col in required_cols if col not in df.columns]
        raise ValueError(f"Missing indicator columns for {symbol}: {missing}")

def wavelet_denoise(series: pd.Series) -> pd.Series:
    """Apply wavelet transform for noise reduction."""
    coeffs = pywt.wavedec(series, 'db1', level=2)
    coeffs[1:] = [np.zeros_like(c) for c in coeffs[1:]]  # Zero out detail coeffs for denoising
    denoised = pywt.waverec(coeffs, 'db1')
    return pd.Series(denoised[:len(series)], index=series.index)

def preprocess_data(df: pd.DataFrame, timesteps: int, add_noise: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """Preprocess data into sequences with wavelet denoising and augmentation."""
    df = df.copy()
    # Denoise close and volume
    df['close'] = wavelet_denoise(df['close'])
    df['volume'] = wavelet_denoise(df['volume'])
    # Differencing for non-stationarity
    df['close'] = np.log(df['close']) - np.log(df['close'].shift(1))
    df['volume'] = np.log(df['volume']) - np.log(df['volume'].shift(1))
    df.ffill(inplace=True)
    df.fillna(0, inplace=True)
    
    features = [
        'close', 'high', 'low', 'volume', 'MA20', 'MA50', 'RSI',
        'MACD', 'MACD_signal', 'MACD_cross', 'OBV', 'VWAP', 'ATR', 'CMF', 'Close_ATR',
        'MA20_ATR', 'Return_1d', 'Return_5d', 'Volatility', 'BB_upper',
        'BB_lower', 'Stoch_K', 'Stoch_D', 'ADX', 'Sentiment', 'ROC', 'Gap',
        'Half_Life', 'Regime', 'Cointegration', 'Size', 'Value',
        'EMA_SHORT', 'EMA_LONG', 'Volume_Spike', 'Spread_Zscore'
    ]
    X = df[features].values
    y = df['Trend'].values
    
    if add_noise:
        X += np.random.normal(0, 0.02, X.shape)
    
    if X.shape[0] < timesteps:
        return np.zeros((0, timesteps, X.shape[1])), np.array([])
    X_seq = np.lib.stride_tricks.sliding_window_view(X, (timesteps, X.shape[1])).reshape(-1, timesteps, X.shape[1])
    y_seq = y[timesteps:]

    if len(X_seq) > len(y_seq):
        X_seq = X_seq[:len(y_seq)]
    elif len(y_seq) > len(X_seq):
        y_seq = y_seq[:len(X_seq)]

    # Data augmentation with SMOTE (for training only)
    if add_noise:
        smote = SMOTE()
        X_flat = X_seq.reshape(X_seq.shape[0], -1)
        X_aug, y_aug = smote.fit_resample(X_flat, y_seq)
        X_seq = X_aug.reshape(X_aug.shape[0], timesteps, X.shape[1])
        y_seq = y_aug

    return X_seq, y_seq

class SelfAttention(nn.Module):
    def __init__(self, hidden_size):
        super(SelfAttention, self).__init__()
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)
        attn = self.softmax(Q @ K.transpose(-2, -1) / np.sqrt(Q.size(-1)))
        return attn @ V

class TradingModel(nn.Module):
    def __init__(self, timesteps: int, features: int):
        super(TradingModel, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=features, out_channels=64, kernel_size=3)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3)
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.gru = nn.GRU(input_size=64, hidden_size=128, num_layers=1, batch_first=True, bidirectional=True)
        self.attention = SelfAttention(256)  # Bidirectional doubles hidden size
        self.dense1 = nn.Linear(256, 256)
        self.dense2 = nn.Linear(256, 128)
        self.dense3 = nn.Linear(128, 2)  # Dual output
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        nn.init.xavier_uniform_(self.dense1.weight)
        nn.init.xavier_uniform_(self.dense2.weight)
        nn.init.xavier_uniform_(self.dense3.weight)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = x.permute(0, 2, 1)
        x, _ = self.gru(x)
        x = self.attention(x)
        x = x[:, -1, :]
        x = self.relu(self.dense1(x))
        x = self.dropout(x)
        x = self.relu(self.dense2(x))
        x = self.dropout(x)
        x = self.dense3(x)
        return x

def build_model(timesteps: int, features: int) -> nn.Module:
    """Build a PyTorch neural network model with stronger regularization."""
    model = TradingModel(timesteps, features)
    return model

class TQDMCallback:
    """Custom callback to update tqdm progress bar after each epoch."""
    def __init__(self, progress_bar):
        self.progress_bar = progress_bar

    def on_epoch_end(self):
        self.progress_bar.update(1)

def objective(trial, symbol, X, y):
    lr = trial.suggest_float('lr', 1e-5, 1e-2, log=True)
    weight_decay = trial.suggest_float('weight_decay', 1e-5, 1e-1, log=True)
    batch_size = trial.suggest_int('batch_size', 32, 256)  # Expanded range for higher utilization
    accum_steps = trial.suggest_int('accum_steps', 1, 8)  # New param for gradient accumulation
    # Train and evaluate
    _, _, val_loss = train_model(symbol, X, y, CONFIG['TRAIN_EPOCHS'], batch_size, lr, weight_decay, accum_steps)
    return val_loss

def optimize_hyperparameters(symbol: str, X: np.ndarray, y: np.ndarray):
    study = optuna.create_study(direction='minimize')
    study.optimize(lambda trial: objective(trial, symbol, X, y), n_trials=50)
    return study.best_params

def train_model(symbol: str, X: np.ndarray, y: np.ndarray, epochs: int, batch_size: int, lr: float = 0.001, weight_decay: float = 0.005, accum_steps: int = 1) -> Tuple[nn.Module, StandardScaler, float]:
    """Train the PyTorch model with early stopping and logging."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Training model for {symbol} on device: {device}")
    train_size = int(0.8 * len(X))
    X_train, X_val = X[:train_size], X[train_size:]
    y_train, y_val = y[:train_size], y[train_size:]
    scaler = StandardScaler()
    X_scaled_train = scaler.fit_transform(X_train.reshape(-1, X.shape[-1])).reshape(X_train.shape)
    X_scaled_val = scaler.transform(X_val.reshape(-1, X.shape[-1])).reshape(X_val.shape)

    if len(X_scaled_train.shape) != 3:
        raise ValueError(f"Expected 3D input for X_scaled_train, got shape {X_scaled_train.shape}")
    if len(X_scaled_val.shape) != 3:
        raise ValueError(f"Expected 3D input for X_scaled_val, got shape {X_scaled_val.shape}")
    if len(y_train.shape) != 1:
        raise ValueError(f"Expected 1D input for y_train, got shape {y_train.shape}")
    if len(y_val.shape) != 1:
        raise ValueError(f"Expected 1D input for y_val, got shape {y_val.shape}")

    X_train_tensor = torch.tensor(X_scaled_train, dtype=torch.float32).to(device)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).reshape(-1, 1).to(device)
    X_val_tensor = torch.tensor(X_scaled_val, dtype=torch.float32).to(device)
    y_val_tensor = torch.tensor(y_val, dtype=torch.float32).reshape(-1, 1).to(device)

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=4, pin_memory=True)

    model = build_model(X.shape[1], X.shape[2]).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.BCEWithLogitsLoss()
    scaler_amp = GradScaler()  # For AMP

    best_val_loss = float('inf')
    patience_counter = 0
    patience = CONFIG['EARLY_STOPPING_PATIENCE']
    min_delta = CONFIG['EARLY_STOPPING_MIN_DELTA']
    best_model_state = None

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        optimizer.zero_grad()  # Zero at start of epoch or per accumulation cycle
        for i, (batch_x, batch_y) in enumerate(train_loader):
            with autocast():  # Enable mixed precision
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y.repeat(1, 2)) / accum_steps  # Normalize loss
            scaler_amp.scale(loss).backward()
            if (i + 1) % accum_steps == 0 or (i + 1) == len(train_loader):  # Update every accum_steps
                scaler_amp.step(optimizer)
                scaler_amp.update()
                optimizer.zero_grad()
            train_loss += loss.item() * batch_x.size(0) * accum_steps  # Adjust for normalization
        train_loss /= len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                with autocast():  # Enable mixed precision for validation
                    outputs = model(batch_x)
                    loss = criterion(outputs, batch_y.repeat(1, 2))
                val_loss += loss.item() * batch_x.size(0)
        val_loss /= len(val_loader.dataset)

        if val_loss < best_val_loss - min_delta:
            best_val_loss = val_loss
            best_model_state = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"Halted training with {symbol}: Training stopped after {epoch+1} epochs due to no improvement in val_loss.")
                break

    model.load_state_dict(best_model_state)
    model = model.cpu()
    torch.save(model.state_dict(), os.path.join(CONFIG['CACHE_DIR'], f"{symbol}_model.pt"))
    with open(os.path.join(CONFIG['CACHE_DIR'], f"{symbol}_scaler.pkl"), 'wb') as f:
        pickle.dump(scaler, f)

    # Clean up CUDA tensors and memory
    del X_train_tensor, y_train_tensor, X_val_tensor, y_val_tensor
    del train_dataset, val_dataset, train_loader, val_loader
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return model, scaler, best_val_loss


def load_model_and_scaler(symbol: str, expected_features: int, force_train: bool) -> Tuple[Optional[nn.Module], Optional[StandardScaler]]:
    """Load pre-trained PyTorch model and scaler."""
    if force_train:
        return None, None
    model_path = os.path.join(CONFIG['CACHE_DIR'], f"{symbol}_model.pt")
    scaler_path = os.path.join(CONFIG['CACHE_DIR'], f"{symbol}_scaler.pkl")
    if os.path.exists(model_path) and os.path.exists(scaler_path):
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        if scaler.n_features_in_ != expected_features:
            return None, None
        model = build_model(CONFIG['TIMESTEPS'], expected_features)
        model.load_state_dict(torch.load(model_path))
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        model.eval()
        return model, scaler
    return None, None

def train_symbol(symbol, expected_features, force_train):
    """Train or load model for a given symbol."""
    df, data_loaded = load_or_fetch_data(symbol, CONFIG['TRAIN_DATA_START_DATE'], CONFIG['BACKTEST_START_DATE'])
    validate_raw_data(df, symbol)
    sentiment, sentiment_loaded = load_news_sentiment(symbol)
    pair_symbol = 'SPY' if symbol != 'SPY' else None
    pair_df = load_or_fetch_data(pair_symbol, CONFIG['TRAIN_DATA_START_DATE'], CONFIG['BACKTEST_START_DATE'])[0] if pair_symbol else None
    df = calculate_indicators(df, sentiment, pair_df)
    validate_indicators(df, symbol)
    model, scaler = load_model_and_scaler(symbol, expected_features, force_train)
    model_loaded = model is not None and scaler is not None
    if not model_loaded:
        X, y = preprocess_data(df, CONFIG['TIMESTEPS'], add_noise=True)
        best_params = optimize_hyperparameters(symbol, X, y)
        model, scaler, _ = train_model(symbol, X, y, CONFIG['TRAIN_EPOCHS'], best_params['batch_size'], best_params['lr'], best_params['weight_decay'])
        del X, y
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return symbol, df, model, scaler, data_loaded, sentiment, sentiment_loaded, model_loaded

def backtest(
    symbol: str,
    model: nn.Module,
    scaler: StandardScaler,
    df: pd.DataFrame,
    initial_cash: float,
    stop_loss_atr_multiplier: float,
    take_profit_atr_multiplier: float,
    timesteps: int,
    buy_threshold: float,
    sell_threshold: float,
    min_holding_period_minutes: int,
    transaction_cost_per_trade: float,
    confidence_threshold: float = CONFIG['CONFIDENCE_THRESHOLD'],
    trailing_stop_percentage: float = CONFIG['TRAILING_STOP_PERCENTAGE'],
    max_volatility: float = CONFIG['MAX_VOLATILITY'],
    adx_trend_threshold: float = CONFIG['ADX_TREND_THRESHOLD'],
    risk_percentage: float = CONFIG['RISK_PERCENTAGE']
) -> Tuple[float, List[float], int, float]:
    """Run a backtest simulation with enhanced features: trailing stop-loss, volatility sizing, confidence threshold, no-trade zone, and ADX."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Running backtest for {symbol} on device: {device}")
    model.eval()
    model = model.to(device)
    X, _ = preprocess_data(df, timesteps)
    X_scaled = scaler.transform(X.reshape(-1, X.shape[-1])).reshape(X.shape)
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(device)
    
    predictions = []
    with torch.no_grad():
        for i in range(0, len(X_tensor), CONFIG['BATCH_SIZE']):
            batch = X_tensor[i:i + CONFIG['BATCH_SIZE']]
            outputs = model(batch)
            outputs = torch.softmax(outputs, dim=1)  # Softmax for probabilities
            predictions.extend(outputs.cpu().numpy())
            del outputs
    predictions = np.array(predictions)
    logger.info(f"Predictions for {symbol}: mean_reversion_mean={predictions[:,0].mean():.3f}, momentum_mean={predictions[:,1].mean():.3f}")
    
    # Clean up CUDA tensors
    del X_tensor
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    timestamps = df['timestamp'].iloc[timesteps:].reset_index(drop=True)
    sim_start = pd.Timestamp(CONFIG['BACKTEST_START_DATE'], tz='UTC')
    k_start = timestamps[timestamps >= sim_start].index[0]
    logger.info(f"Backtest for {symbol}: k_start={k_start}, len(predictions)={len(predictions)}")
    if k_start >= len(predictions):
        logger.warning(f"No data points for backtest of {symbol}")
        return initial_cash / len(CONFIG['SYMBOLS']), [], 0, 0.0
    cash = initial_cash / len(CONFIG['SYMBOLS'])
    position = 0
    entry_price = 0.0
    entry_time = None
    max_price = 0.0
    returns = []
    trade_count = 0
    winning_trades = 0
    atr = df['ATR'].iloc[timesteps:].values
    prices = df['close'].iloc[timesteps:].values
    rsi = df['RSI'].iloc[timesteps:].values
    adx = df['ADX'].iloc[timesteps:].values
    volatility = df['Volatility'].iloc[timesteps:].values
    regime = df['Regime'].iloc[timesteps:].values
    gap = df['Gap'].iloc[timesteps:].values
    half_life = df['Half_Life'].iloc[timesteps:].values
    ema_short = df['EMA_SHORT'].iloc[timesteps:].values
    ema_long = df['EMA_LONG'].iloc[timesteps:].values
    volume_spike = df['Volume_Spike'].iloc[timesteps:].values
    macd_cross = df['MACD_cross'].iloc[timesteps:].values
    bb_lower = df['BB_lower'].iloc[timesteps:].values
    bb_upper = df['BB_upper'].iloc[timesteps:].values
    spread_zscore = df['Spread_Zscore'].iloc[timesteps:].values
    vol_mean = df['Volatility'].rolling(20).mean().iloc[timesteps:].values
    sim_timestamps = timestamps.values

    for i in range(k_start, len(predictions)):
        pred_reversion, pred_momentum = predictions[i]
        price = prices[i]
        atr_val = atr[i]
        current_rsi = rsi[i]
        current_adx = adx[i]
        current_volatility = volatility[i]
        current_regime = regime[i]
        current_gap = gap[i]
        current_half_life = half_life[i]
        current_ema_short = ema_short[i]
        current_ema_long = ema_long[i]
        current_volume_spike = volume_spike[i]
        current_macd_cross = macd_cross[i]
        current_bb_lower = bb_lower[i]
        current_bb_upper = bb_upper[i]
        current_spread_zscore = spread_zscore[i]
        current_vol_mean = vol_mean[i]
        ts = pd.Timestamp(sim_timestamps[i])

        # Anomaly detection
        if current_volatility > CONFIG['ANOMALY_VOL_MULTIPLIER'] * current_vol_mean:
            continue

        if current_volatility > max_volatility or current_adx < adx_trend_threshold:
            continue

        # Hybrid logic: Use reversion if regime low_vol or half_life low, momentum if high_vol
        if current_regime == 0 or current_half_life < 10:  # Mean reversion mode
            if pred_reversion > confidence_threshold and current_gap < -CONFIG['GAP_THRESHOLD'] and current_rsi < CONFIG['RSI_REVERSION_BUY'] and price < current_bb_lower and current_spread_zscore < -CONFIG['ZSCORE_THRESHOLD']:
                # Buy on reversion signal
                qty = kelly_qty(cash, expected_return=pred_reversion, variance=current_volatility**2)
                qty = max(1, min(qty, int(cash * risk_percentage / price)))
                # VaR adjustment
                var = price * abs(norm.ppf(0.05)) * current_volatility
                if var > 0:
                    qty = min(qty, int(cash * 0.01 / var))
                cost = qty * price + transaction_cost_per_trade
                if qty > 0 and cost <= cash and position == 0:
                    position = qty
                    entry_price = price
                    max_price = price
                    entry_time = ts
                    cash -= cost
                    logger.info(f"{ts}: Bought {qty} shares of {symbol} at ${price:.2f} (reversion), cash: ${cash:.2f}")
            elif pred_reversion < sell_threshold and current_rsi > CONFIG['RSI_REVERSION_SELL'] and price > current_bb_upper and current_spread_zscore > CONFIG['ZSCORE_THRESHOLD'] and position > 0:
                cash += position * price - transaction_cost_per_trade
                ret = (price - entry_price) / entry_price
                returns.append(ret)
                trade_count += 1
                if ret > 0:
                    winning_trades += 1
                logger.info(f"{ts}: Sold {position} shares of {symbol} at ${price:.2f} (reversion), return: {ret:.3f}, cash: ${cash:.2f}")
                position = 0
                entry_time = None
                max_price = 0.0
        else:  # Momentum mode
            if pred_momentum > confidence_threshold and current_rsi > CONFIG['RSI_MOMENTUM_BUY'] and current_ema_short > current_ema_long and current_volume_spike and current_macd_cross and position == 0:
                # Buy on momentum signal
                qty = kelly_qty(cash, expected_return=pred_momentum, variance=current_volatility**2)
                qty = max(1, min(qty, int(cash * risk_percentage / price)))
                # VaR adjustment
                var = price * abs(norm.ppf(0.05)) * current_volatility
                if var > 0:
                    qty = min(qty, int(cash * 0.01 / var))
                cost = qty * price + transaction_cost_per_trade
                if qty > 0 and cost <= cash:
                    position = qty
                    entry_price = price
                    max_price = price
                    entry_time = ts
                    cash -= cost
                    logger.info(f"{ts}: Bought {qty} shares of {symbol} at ${price:.2f} (momentum), cash: ${cash:.2f}")
            elif pred_momentum < sell_threshold and current_rsi < CONFIG['RSI_MOMENTUM_SELL'] and position > 0:
                cash += position * price - transaction_cost_per_trade
                ret = (price - entry_price) / entry_price
                returns.append(ret)
                trade_count += 1
                if ret > 0:
                    winning_trades += 1
                logger.info(f"{ts}: Sold {position} shares of {symbol} at ${price:.2f} (momentum), return: {ret:.3f}, cash: ${cash:.2f}")
                position = 0
                entry_time = None
                max_price = 0.0

        # Common risk management
        if position > 0:
            if price > max_price:
                max_price = price
            trailing_stop = max_price * (1 - trailing_stop_percentage)
            stop_loss = entry_price - stop_loss_atr_multiplier * atr_val
            take_profit = entry_price + take_profit_atr_multiplier * atr_val
            time_held = (ts - entry_time).total_seconds() / 60 if entry_time else 0
            if time_held >= min_holding_period_minutes and (price <= trailing_stop or price <= stop_loss or price >= take_profit):
                cash += position * price - transaction_cost_per_trade
                ret = (price - entry_price) / entry_price
                returns.append(ret)
                trade_count += 1
                if ret > 0:
                    winning_trades += 1
                logger.info(f"{ts}: Sold {position} shares of {symbol} at ${price:.2f} (risk stop), return: {ret:.3f}, cash: ${cash:.2f}")
                position = 0
                entry_time = None
                max_price = 0.0

    if position > 0:
        cash += position * prices[-1] - transaction_cost_per_trade
        ret = (prices[-1] - entry_price) / entry_price
        returns.append(ret)
        trade_count += 1
        if ret > 0:
            winning_trades += 1
    win_rate = (winning_trades / trade_count * 100) if trade_count > 0 else 0.0
    return cash, returns, trade_count, win_rate

def kelly_qty(cash: float, expected_return: float, variance: float) -> int:
    """Kelly criterion for position sizing."""
    if variance == 0:
        return 0
    f = expected_return / variance
    qty = int(cash * f)
    return qty

def calculate_performance_metrics(returns: List[float], cash: float, initial_cash: float) -> Dict[str, float]:
    """Calculate performance metrics."""
    returns = np.array(returns)
    metrics = {
        'total_return': (cash - initial_cash) / initial_cash * 100,
        'sharpe_ratio': np.mean(returns) / np.std(returns) * np.sqrt(252) if len(returns) > 1 and np.std(returns) > 0 else 0.0,
        'max_drawdown': np.max((np.maximum.accumulate(np.cumprod(1 + returns)) - np.cumprod(1 + returns)) / np.maximum.accumulate(np.cumprod(1 + returns))) * 100 if returns.size > 0 else 0.0
    }
    return {k: round(v, 3) for k, v in metrics.items()}

def format_email_body(
    initial_cash: float,
    final_value: float,
    symbol_results: Dict[str, Dict[str, float]],
    trade_counts: Dict[str, int],
    win_rates: Dict[str, float]
) -> str:
    """Format the email body with backtest results."""
    body = [
        f"Backtest Results",
        f"Initial Cash: ${initial_cash:.2f}",
        f"Final Value: ${final_value:.2f}",
        f"Total Return: {(final_value - initial_cash) / initial_cash * 100:.2f}%",
        f"",
        f"Per-Symbol Performance:"
    ]
    for symbol, metrics in symbol_results.items():
        body.append(f"\n{symbol}:")
        for metric, value in metrics.items():
            body.append(f"  {metric.replace('_', ' ').title()}: {value:.3f}%")
        body.append(f"  Trades: {trade_counts.get(symbol, 0)}")
        body.append(f"  Win Rate: {win_rates.get(symbol, 0.0):.3f}%")
    return "\n".join(body)

def send_email(subject: str, body: str) -> None:
    """Send an email notification."""
    msg = MIMEText(body)
    msg['Subject'] = subject
    msg['From'] = CONFIG['EMAIL_SENDER']
    msg['To'] = ', '.join(CONFIG['EMAIL_RECEIVER'])
    with smtplib.SMTP(CONFIG['SMTP_SERVER'], CONFIG['SMTP_PORT']) as server:
        server.starttls()
        server.login(CONFIG['EMAIL_SENDER'], CONFIG['EMAIL_PASSWORD'])
        server.sendmail(CONFIG['EMAIL_SENDER'], CONFIG['EMAIL_RECEIVER'], msg.as_string())

def make_prediction(model: nn.Module, X_scaled: np.ndarray) -> Tuple[float, float]:
    """Make a prediction using the PyTorch model for reversion and momentum."""
    if X_scaled.size == 0 or X_scaled.shape[0] == 0:
        logger.error("Empty input data for prediction")
        raise ValueError("Input data for prediction is empty")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(device)
    with torch.no_grad():
        outputs = torch.softmax(model(X_tensor), dim=1).cpu().numpy()[0]
    if not np.all(np.isfinite(outputs)):
        logger.error("Invalid prediction value: non-finite")
        raise ValueError("Prediction resulted in non-finite value")
    return float(outputs[0]), float(outputs[1])

def get_api_keys(config: Dict) -> None:
    """Validate and prompt for Alpaca API keys if not set or invalid."""
    if config['ALPACA_API_KEY'] in [None, '', 'REPLACE ME'] or config['ALPACA_SECRET_KEY'] in [None, '', 'REPLACE ME']:
        logger.info("Alpaca API keys missing or invalid. Prompting for input.")
        config['ALPACA_API_KEY'] = input("Enter Alpaca API Key: ").strip()
        config['ALPACA_SECRET_KEY'] = input("Enter Alpaca Secret Key: ").strip()
        if not config['ALPACA_API_KEY'] or not config['ALPACA_SECRET_KEY']:
            raise ValueError("Alpaca API keys cannot be empty.")
    else:
        logger.info("Using hardcoded Alpaca API keys from CONFIG.")

def main(backtest_only: bool = False, force_train: bool = False) -> None:
    """Main function to execute the trading bot with portfolio-level risk management."""
    print(f"Device set to use {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}")
    if not backtest_only:
        logger.info("Bot started")
        stream_handler.setLevel(logging.CRITICAL)
    else:
        logger.info("Bot started in backtest mode")
    get_api_keys(CONFIG)
    
    check_dependencies()
    validate_config(CONFIG)
    create_cache_directory()
    trading_client = TradingClient(CONFIG['ALPACA_API_KEY'], CONFIG['ALPACA_SECRET_KEY'], paper=True)
    expected_features = 36  # Updated for new features (added MACD_cross, Spread_Zscore)
    models = {}
    scalers = {}
    dfs = {}
    stock_info = []
    total_epochs = len(CONFIG['SYMBOLS']) * CONFIG['TRAIN_EPOCHS']
    need_training = any(load_model_and_scaler(symbol, expected_features, force_train)[0] is None for symbol in CONFIG['SYMBOLS'])
    progress_bar = tqdm(total=total_epochs, desc="Training Progress", bar_format="{l_bar}\033[32m{bar}\033[0m{r_bar}") if need_training else None

    mp.set_start_method('forkserver', force=True)

    pool = mp.Pool(processes=min(mp.cpu_count(), len(CONFIG['SYMBOLS'])))
    results = [pool.apply_async(train_symbol, args=(symbol, expected_features, force_train)) for symbol in CONFIG['SYMBOLS']]
    pool.close()
    pool.join()

    # Get the results
    outputs = [res.get() for res in results]

    # Enhanced CUDA cleanup
    pool.terminate()  # Terminate pool to release resources
    del results  # Delete results reference
    if torch.cuda.is_available():
        torch.cuda.synchronize()  # Wait for all GPU operations to finish
        torch.cuda.empty_cache()  # Clear GPU memory
        time.sleep(2)  # Allow GPU to stabilize
        logger.info("CUDA memory cleared after multiprocessing.")

    for symbol, df, model, scaler, data_loaded, sentiment, sentiment_loaded, model_loaded in outputs:
        dfs[symbol] = df
        models[symbol] = model
        scalers[symbol] = scaler
        info = []
        info.append(f"{Fore.LIGHTBLUE_EX}{symbol}:{Style.RESET_ALL}")
        info.append(f"  {'Loaded cached model and scaler' if model_loaded else 'Trained model'} for {symbol}.")
        info.append(f"  {'Loaded' if data_loaded else 'Fetched'} {len(df)} bars for {symbol} {'from cache' if data_loaded else ''}.")
        info.append(f"  {'Loaded news data' if sentiment_loaded else 'Computed news sentiment'} for {symbol} {'from cache' if sentiment_loaded else ''}.")
        info.append(f"  Calculated sentiment score: {sentiment:.3f}")
        info.append(f"  Calculated stop-loss ATR multiplier: {CONFIG['STOP_LOSS_ATR_MULTIPLIER']:.2f}")
        try:
            position = trading_client.get_open_position(symbol)
            qty_owned = int(float(position.qty))
            info.append(f"  Current amount of stocks owned: {qty_owned}")
        except:
            qty_owned = 0
            info.append(f"  Current amount of stocks owned: {qty_owned}")
        if not model_loaded:
            info.append(f"  Saved model and scaler for {symbol}.")
        stock_info.append(info)
        if progress_bar and not model_loaded:
            progress_bar.update(CONFIG['TRAIN_EPOCHS'])

    if progress_bar:
        progress_bar.close()

    for info in stock_info:
        for line in info:
            print(line)
        print()

    if not backtest_only:
        portfolio_value = CONFIG['INITIAL_CASH']
        peak_value = portfolio_value
        max_prices = {symbol: 0.0 for symbol in CONFIG['SYMBOLS']}
        open_positions = trading_client.get_all_positions()
        for pos in open_positions:
            max_prices[pos.symbol] = float(pos.current_price)
        last_retrain = datetime.now()

        while True:
            clock = trading_client.get_clock()
            if clock.is_open:
                now = datetime.now()
                next_mark = now.replace(second=0, microsecond=0)
                minutes = now.minute
                if minutes % 15 != 0:
                    next_mark += timedelta(minutes=(15 - minutes % 15))
                else:
                    next_mark += timedelta(minutes=15)
                seconds_to_sleep = (next_mark - now).total_seconds()
                if seconds_to_sleep > 0:
                    time.sleep(seconds_to_sleep)
                
                # Online retrain
                if (now - last_retrain).days >= CONFIG['ONLINE_RETRAIN_INTERVAL']:
                    for symbol in CONFIG['SYMBOLS']:
                        df_recent = fetch_recent_data(symbol, 200)
                        sentiment = load_news_sentiment(symbol)[0]
                        pair_df = dfs.get('SPY' if symbol != 'SPY' else None)
                        df_recent = calculate_indicators(df_recent, sentiment, pair_df)
                        X_recent, y_recent = preprocess_data(df_recent, CONFIG['TIMESTEPS'], add_noise=False)
                        if len(X_recent) > 0:
                            model = models[symbol]
                            model.to(device)
                            optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.005)
                            criterion = nn.BCEWithLogitsLoss()
                            dataset = TensorDataset(torch.tensor(X_recent, dtype=torch.float32).to(device), torch.tensor(y_recent, dtype=torch.float32).reshape(-1, 1).to(device))
                            loader = DataLoader(dataset, batch_size=CONFIG['BATCH_SIZE'], shuffle=True)
                            for _ in range(5):  # Few epochs for fine-tuning
                                model.train()
                                train_loss = 0.0
                                for batch_x, batch_y in loader:
                                    optimizer.zero_grad()
                                    outputs = model(batch_x)
                                    loss = criterion(outputs, batch_y.repeat(1, 2))
                                    loss.backward()
                                    optimizer.step()
                                    train_loss += loss.item() * batch_x.size(0)
                                train_loss /= len(loader.dataset)
                            torch.save(model.state_dict(), os.path.join(CONFIG['CACHE_DIR'], f"{symbol}_model.pt"))
                            logger.info(f"Fine-tuned model for {symbol}")
                    last_retrain = now
                
                account = trading_client.get_account()
                portfolio_value = float(account.equity)
                peak_value = max(peak_value, portfolio_value)
                drawdown = (peak_value - portfolio_value) / peak_value
                if drawdown > CONFIG['MAX_DRAWDOWN_LIMIT']:
                    logger.warning(f"Portfolio drawdown exceeded {CONFIG['MAX_DRAWDOWN_LIMIT'] * 100}%. Pausing trading.")
                    send_email("Portfolio Drawdown Alert", f"Portfolio drawdown exceeded {CONFIG['MAX_DRAWDOWN_LIMIT'] * 100}%. Trading paused.")
                    break
                
                decisions = []
                open_positions = trading_client.get_all_positions()
                for symbol in CONFIG['SYMBOLS']:
                    if symbol in models:
                        df = fetch_recent_data(symbol, CONFIG['LIVE_DATA_BARS'])
                        sentiment = load_news_sentiment(symbol)[0]
                        pair_df = dfs.get('SPY' if symbol != 'SPY' else None)
                        df = calculate_indicators(df, sentiment, pair_df)
                        X, _ = preprocess_data(df, CONFIG['TIMESTEPS'], add_noise=False)
                        if X.shape[0] > 0:
                            last_sequence = X[-1].reshape(1, CONFIG['TIMESTEPS'], -1)
                            X_scaled = scalers[symbol].transform(last_sequence.reshape(-1, last_sequence.shape[-1])).reshape(last_sequence.shape)
                            pred_reversion, pred_momentum = make_prediction(models[symbol], X_scaled)
                            price = df['close'].iloc[-1]
                            current_rsi = df['RSI'].iloc[-1]
                            current_adx = df['ADX'].iloc[-1]
                            current_volatility = df['Volatility'].iloc[-1]
                            atr_val = df['ATR'].iloc[-1]
                            current_regime = df['Regime'].iloc[-1]
                            current_gap = df['Gap'].iloc[-1]
                            current_half_life = df['Half_Life'].iloc[-1]
                            current_ema_short = df['EMA_SHORT'].iloc[-1]
                            current_ema_long = df['EMA_LONG'].iloc[-1]
                            current_volume_spike = df['Volume_Spike'].iloc[-1]
                            current_macd_cross = df['MACD_cross'].iloc[-1]
                            current_bb_lower = df['BB_lower'].iloc[-1]
                            current_bb_upper = df['BB_upper'].iloc[-1]
                            current_spread_zscore = df['Spread_Zscore'].iloc[-1]
                            current_vol_mean = df['Volatility'].rolling(20).mean().iloc[-1]
                            qty_owned = 0
                            entry_price = 0.0
                            position = next((pos for pos in open_positions if pos.symbol == symbol), None)
                            if position:
                                qty_owned = int(float(position.qty))
                                entry_price = float(position.avg_entry_price)

                            decision = "Hold"
                            # Anomaly detection
                            if current_volatility > CONFIG['ANOMALY_VOL_MULTIPLIER'] * current_vol_mean:
                                decision = "Hold (Anomaly Detected)"
                                continue

                            if current_volatility <= CONFIG['MAX_VOLATILITY'] and current_adx >= CONFIG['ADX_TREND_THRESHOLD']:
                                if current_regime == 0 or current_half_life < 10:
                                    # Mean reversion mode
                                    if pred_reversion >= max(CONFIG['PREDICTION_THRESHOLD_BUY'], CONFIG['CONFIDENCE_THRESHOLD']) and current_gap < -CONFIG['GAP_THRESHOLD'] and current_rsi < CONFIG['RSI_REVERSION_BUY'] and price < current_bb_lower and current_spread_zscore < -CONFIG['ZSCORE_THRESHOLD'] and qty_owned == 0:
                                        decision = "Buy (Reversion)"
                                        cash = float(account.cash)
                                        expected_ret = pred_reversion
                                        variance = current_volatility**2
                                        qty = kelly_qty(cash, expected_ret, variance)
                                        qty = max(1, min(qty, int(cash * CONFIG['RISK_PERCENTAGE'] / price)))
                                        qty = min(qty, int(portfolio_value * 0.1 / price))
                                        # VaR adjustment
                                        var = price * abs(norm.ppf(0.05)) * current_volatility
                                        if var > 0:
                                            qty = min(qty, int(cash * 0.01 / var))
                                        # Sector allocation check
                                        sector = CONFIG['SECTORS'].get(symbol, 'Unknown')
                                        current_sector_alloc = sum(float(pos.market_value) for pos in open_positions if CONFIG['SECTORS'].get(pos.symbol, '') == sector)
                                        if current_sector_alloc + qty * price > portfolio_value * CONFIG['SECTOR_ALLOCATION_LIMIT']:
                                            decision = "Hold (Sector Limit Exceeded)"
                                            continue
                                        cost = qty * price + transaction_cost_per_trade
                                        if qty > 0 and cost <= cash:
                                            order = MarketOrderRequest(symbol=symbol, qty=qty, side=OrderSide.BUY, time_in_force=TimeInForce.GTC)
                                            try:
                                                trading_client.submit_order(order)
                                                email_body = f"""
Bought {qty} shares of {symbol} at ${price:.2f} (Reversion)
Prediction Reversion: {pred_reversion:.3f}, Momentum: {pred_momentum:.3f}
RSI: {current_rsi:.2f}
ADX: {current_adx:.2f}
Volatility: {current_volatility:.2f}
ATR: {atr_val:.2f}
Current Cash: ${cash:.2f}
Portfolio Value: ${portfolio_value:.2f}
"""
                                                send_email(f"Trade Update: {symbol}", email_body)
                                                max_prices[symbol] = price
                                            except Exception as e:
                                                logger.error(f"Failed to submit buy order for {symbol}: {str(e)}")
                                                decision = "Hold"
                                                email_body = f"""
Failed to buy {qty} shares of {symbol} at ${price:.2f}
Error: {str(e)}
Prediction Reversion: {pred_reversion:.3f}, Momentum: {pred_momentum:.3f}
RSI: {current_rsi:.2f}
ADX: {current_adx:.2f}
Volatility: {current_volatility:.2f}
ATR: {atr_val:.2f}
Current Cash: {cash:.2f}
Portfolio Value: ${portfolio_value:.2f}
"""
                                                send_email(f"Trade Failure: {symbol}", email_body)
                                        else:
                                            decision = "Hold"
                                            logger.info(f"Buy skipped for {symbol}: Insufficient qty ({qty}) or buying power (cost={qty * price:.2f}, buying_power={float(account.buying_power):.2f})")
                                    if qty_owned > 0 and pred_reversion <= CONFIG['PREDICTION_THRESHOLD_SELL'] and current_rsi > CONFIG['RSI_REVERSION_SELL'] and price > current_bb_upper and current_spread_zscore > CONFIG['ZSCORE_THRESHOLD']:
                                        decision = "Sell (Reversion)"
                                        order = MarketOrderRequest(symbol=symbol, qty=qty_owned, side=OrderSide.SELL, time_in_force=TimeInForce.GTC)
                                        try:
                                            trading_client.submit_order(order)
                                            email_body = f"""
Sold {qty_owned} shares of {symbol} at ${price:.2f} (Reversion)
Prediction Reversion: {pred_reversion:.3f}, Momentum: {pred_momentum:.3f}
RSI: {current_rsi:.2f}
ADX: {current_adx:.2f}
Volatility: {current_volatility:.2f}
ATR: {atr_val:.2f}
Current Cash: ${float(account.cash):.2f}
Portfolio Value: ${portfolio_value:.2f}
"""
                                            send_email(f"Trade Update: {symbol}", email_body)
                                            max_prices[symbol] = 0.0
                                        except Exception as e:
                                            logger.error(f"Failed to submit sell order for {symbol}: {str(e)}")
                                            decision = "Hold"
                                            email_body = f"""
Failed to sell {qty_owned} shares of {symbol} at ${price:.2f}
Error: {str(e)}
Prediction Reversion: {pred_reversion:.3f}, Momentum: {pred_momentum:.3f}
RSI: {current_rsi:.2f}
ADX: {current_adx:.2f}
Volatility: {current_volatility:.2f}
ATR: {atr_val:.2f}
Current Cash: {float(account.cash):.2f}
Portfolio Value: ${portfolio_value:.2f}
"""
                                            send_email(f"Trade Failure: {symbol}", email_body)
                                else:
                                    # Momentum mode
                                    if pred_momentum >= max(CONFIG['PREDICTION_THRESHOLD_BUY'], CONFIG['CONFIDENCE_THRESHOLD']) and current_rsi > CONFIG['RSI_MOMENTUM_BUY'] and current_ema_short > current_ema_long and current_volume_spike and current_macd_cross and qty_owned == 0:
                                        decision = "Buy (Momentum)"
                                        cash = float(account.cash)
                                        expected_ret = pred_momentum
                                        variance = current_volatility**2
                                        qty = kelly_qty(cash, expected_ret, variance)
                                        qty = max(1, min(qty, int(cash * CONFIG['RISK_PERCENTAGE'] / price)))
                                        qty = min(qty, int(portfolio_value * 0.1 / price))
                                        # VaR adjustment
                                        var = price * abs(norm.ppf(0.05)) * current_volatility
                                        if var > 0:
                                            qty = min(qty, int(cash * 0.01 / var))
                                        # Sector allocation check
                                        sector = CONFIG['SECTORS'].get(symbol, 'Unknown')
                                        current_sector_alloc = sum(float(pos.market_value) for pos in open_positions if CONFIG['SECTORS'].get(pos.symbol, '') == sector)
                                        if current_sector_alloc + qty * price > portfolio_value * CONFIG['SECTOR_ALLOCATION_LIMIT']:
                                            decision = "Hold (Sector Limit Exceeded)"
                                            continue
                                        cost = qty * price + transaction_cost_per_trade
                                        if qty > 0 and cost <= cash:
                                            order = MarketOrderRequest(symbol=symbol, qty=qty, side=OrderSide.BUY, time_in_force=TimeInForce.GTC)
                                            try:
                                                trading_client.submit_order(order)
                                                email_body = f"""
Bought {qty} shares of {symbol} at ${price:.2f} (Momentum)
Prediction Reversion: {pred_reversion:.3f}, Momentum: {pred_momentum:.3f}
RSI: {current_rsi:.2f}
ADX: {current_adx:.2f}
Volatility: {current_volatility:.2f}
ATR: {atr_val:.2f}
Current Cash: ${cash:.2f}
Portfolio Value: ${portfolio_value:.2f}
"""
                                                send_email(f"Trade Update: {symbol}", email_body)
                                                max_prices[symbol] = price
                                            except Exception as e:
                                                logger.error(f"Failed to submit buy order for {symbol}: {str(e)}")
                                                decision = "Hold"
                                                email_body = f"""
Failed to buy {qty} shares of {symbol} at ${price:.2f}
Error: {str(e)}
Prediction Reversion: {pred_reversion:.3f}, Momentum: {pred_momentum:.3f}
RSI: {current_rsi:.2f}
ADX: {current_adx:.2f}
Volatility: {current_volatility:.2f}
ATR: {atr_val:.2f}
Current Cash: {cash:.2f}
Portfolio Value: ${portfolio_value:.2f}
"""
                                                send_email(f"Trade Failure: {symbol}", email_body)
                                        else:
                                            decision = "Hold"
                                            logger.info(f"Buy skipped for {symbol}: Insufficient qty ({qty}) or buying power (cost={qty * price:.2f}, buying_power={float(account.buying_power):.2f})")
                                    if qty_owned > 0 and pred_momentum <= CONFIG['PREDICTION_THRESHOLD_SELL'] and current_rsi < CONFIG['RSI_MOMENTUM_SELL']:
                                        decision = "Sell (Momentum)"
                                        order = MarketOrderRequest(symbol=symbol, qty=qty_owned, side=OrderSide.SELL, time_in_force=TimeInForce.GTC)
                                        try:
                                            trading_client.submit_order(order)
                                            email_body = f"""
Sold {qty_owned} shares of {symbol} at ${price:.2f} (Momentum)
Prediction Reversion: {pred_reversion:.3f}, Momentum: {pred_momentum:.3f}
RSI: {current_rsi:.2f}
ADX: {current_adx:.2f}
Volatility: {current_volatility:.2f}
ATR: {atr_val:.2f}
Current Cash: ${float(account.cash):.2f}
Portfolio Value: ${portfolio_value:.2f}
"""
                                            send_email(f"Trade Update: {symbol}", email_body)
                                            max_prices[symbol] = 0.0
                                        except Exception as e:
                                            logger.error(f"Failed to submit sell order for {symbol}: {str(e)}")
                                            decision = "Hold"
                                            email_body = f"""
Failed to sell {qty_owned} shares of {symbol} at ${price:.2f}
Error: {str(e)}
Prediction Reversion: {pred_reversion:.3f}, Momentum: {pred_momentum:.3f}
RSI: {current_rsi:.2f}
ADX: {current_adx:.2f}
Volatility: {current_volatility:.2f}
ATR: {atr_val:.2f}
Current Cash: {float(account.cash):.2f}
Portfolio Value: {portfolio_value:.2f}
"""
                                            send_email(f"Trade Failure: {symbol}", email_body)
                            # Common risk
                            if qty_owned > 0:
                                max_prices[symbol] = max(max_prices[symbol], price)
                                trailing_stop = max_prices[symbol] * (1 - CONFIG['TRAILING_STOP_PERCENTAGE'])
                                stop_loss = entry_price - CONFIG['STOP_LOSS_ATR_MULTIPLIER'] * atr_val
                                take_profit = entry_price + CONFIG['TAKE_PROFIT_ATR_MULTIPLIER'] * atr_val
                                if price <= trailing_stop or price <= stop_loss or price >= take_profit:
                                    decision = "Sell (Risk)"
                                    order = MarketOrderRequest(symbol=symbol, qty=qty_owned, side=OrderSide.SELL, time_in_force=TimeInForce.GTC)
                                    try:
                                        trading_client.submit_order(order)
                                        email_body = f"""
Sold {qty_owned} shares of {symbol} at ${price:.2f} (Risk Stop)
Prediction Reversion: {pred_reversion:.3f}, Momentum: {pred_momentum:.3f}
RSI: {current_rsi:.2f}
ADX: {current_adx:.2f}
Volatility: {current_volatility:.2f}
ATR: {atr_val:.2f}
Current Cash: ${float(account.cash):.2f}
Portfolio Value: ${portfolio_value:.2f}
"""
                                        send_email(f"Trade Update: {symbol}", email_body)
                                        max_prices[symbol] = 0.0
                                    except Exception as e:
                                        logger.error(f"Failed to submit sell order for {symbol}: {str(e)}")
                                        decision = "Hold"
                                        email_body = f"""
Failed to sell {qty_owned} shares of {symbol} at ${price:.2f}
Error: {str(e)}
Prediction Reversion: {pred_reversion:.3f}, Momentum: {pred_momentum:.3f}
RSI: {current_rsi:.2f}
ADX: {current_adx:.2f}
Volatility: {current_volatility:.2f}
ATR: {atr_val:.2f}
Current Cash: {float(account.cash):.2f}
Portfolio Value: {portfolio_value:.2f}
"""
                                        send_email(f"Trade Failure: {symbol}", email_body)
                            decisions.append({
                                'symbol': symbol,
                                'decision': decision,
                                'confidence_reversion': pred_reversion,
                                'confidence_momentum': pred_momentum,
                                'rsi': current_rsi,
                                'adx': current_adx,
                                'volatility': current_volatility,
                                'price': price,
                                'owned': qty_owned
                            })
                summary_body = "Trading Summary (Hybrid):\n"
                for dec in decisions:
                    summary_body += f"{dec['symbol']}: {dec['decision']}, Reversion: {dec['confidence_reversion']:.3f}, Momentum: {dec['confidence_momentum']:.3f}, RSI: {dec['rsi']:.2f}, ADX: {dec['adx']:.2f}, Volatility: {dec['volatility']:.2f}, Price: ${dec['price']:.2f}, Owned: {dec['owned']}\n"
                summary_body += f"\nPortfolio Value: ${portfolio_value:.2f}\nNote: Actual trades are executed based on available cash and positions."
                send_email("Trading Summary", summary_body)
            else:
                next_open = clock.next_open
                while datetime.now(timezone.utc) < next_open:
                    time_left = next_open - datetime.now(timezone.utc)
                    hours, remainder = divmod(time_left.total_seconds(), 3600)
                    minutes, seconds = divmod(remainder, 60)
                    timer_str = f"{int(hours):02}:{int(minutes):02}:{int(seconds):02}"
                    print(f"\r{Fore.RED}Time until market opens: {timer_str}{Style.RESET_ALL}", end='')
                    time.sleep(1)
                print()

    else:
        initial_cash = CONFIG['INITIAL_CASH']
        final_value = initial_cash
        symbol_results = {}
        trade_counts = {}
        win_rates = {}
        for symbol in CONFIG['SYMBOLS']:
            if symbol in models:
                cash, returns, trade_count, win_rate = backtest(
                    symbol, models[symbol], scalers[symbol], dfs[symbol], CONFIG['INITIAL_CASH'],
                    CONFIG['STOP_LOSS_ATR_MULTIPLIER'], CONFIG['TAKE_PROFIT_ATR_MULTIPLIER'],
                    CONFIG['TIMESTEPS'], CONFIG['PREDICTION_THRESHOLD_BUY'], CONFIG['PREDICTION_THRESHOLD_SELL'],
                    CONFIG['MIN_HOLDING_PERIOD_MINUTES'], CONFIG['TRANSACTION_COST_PER_TRADE']
                )
                final_value += cash - (initial_cash / len(CONFIG['SYMBOLS']))
                symbol_results[symbol] = calculate_performance_metrics(returns, cash, initial_cash / len(CONFIG['SYMBOLS']))
                trade_counts[symbol] = trade_count
                win_rates[symbol] = win_rate
        email_body = format_email_body(initial_cash, final_value, symbol_results, trade_counts, win_rates)
        send_email("Backtest Completed", email_body)
        logger.info(f"Backtest completed: Final value: ${final_value:.2f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Trading bot with backtest mode")
    parser.add_argument('--backtest', action='store_true', help="Run in backtest-only mode")
    parser.add_argument('--force-train', action='store_true', help="Force retraining of models")
    args = parser.parse_args()
    main(backtest_only=args.backtest, force_train=args.force_train)
