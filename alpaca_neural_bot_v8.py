
# +------------------------------------------------------------------------------+
# |                            Alpaca Neural Bot v8.0                            |
# +------------------------------------------------------------------------------+
# | Author: Vladimir Makarov                                                     |
# | Project Start Date: May 9, 2025                                              |
# | License: GNU Lesser General Public License v2.1                              |
# | Version: 4 (Un-Released)                                                     |
# |                                                                              |
# | Dependencies:                                                                |
# | - torch (Neural network framework)                                           |
# | - numpy (Numerical computations)                                             |
# | - pandas (Data manipulation)                                                 |
# | - alpaca-py (Alpaca integration, imports as 'alpaca')                        |
# | - transformers (Sentiment analysis)  (REMOVED)                               |
# | - scikit-learn (Machine learning utilities)                                  |
# | - ta-lib (Technical analysis)                                                |
# | - tenacity (Retry logic)                                                     |
# | - smtplib (Email notifications)                                              |
# | - argparse (Command-line parsing)                                            |
# | - tqdm (Progress bars)                                                       |
# | - colorama (Console formatting)                                              |
# | Install using: pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128 |
|                pip install alpaca-py pandas numpy scikit-learn ta-lib tenacity tqdm colorama |
# |                pip install protobuf==5.28.3                                  |
# |                                                                              |
# | Double check before starting:                                                |
# | - Ensure Alpaca API keys are configured in CONFIG.                           |
# | - Requires stable internet for live trading and data fetching.               |
# | - GitHub: https://github.com/vmakarov28/Alpaca-Stock-Trading-Bot/tree/main   |
# |                                                                              |
# +------------------------------------------------------------------------------+
# Updated for LINUX with PyTorch for RTX 5080 GPU acceleration
#Activate virtu= environment: pyenv activate pytorch_env
#Run with: python /mnt/c/Users/aipla/Downloads/alpaca_neural_bot_v8.py --backtest --force-train

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
#from transformers import pipeline
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
    'BACKTEST_START_DATE': '2025-01-01',  # Start date for backtesting
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
    'ALPACA_API_KEY': 'PKEILCZUL5LIOG92HNTZ',  # API key for Alpaca
    'ALPACA_SECRET_KEY': 'hluGeGPYq9PpJFMpbf5cn0VGdD6NRsoEs21GNU8T',  # Secret key for Alpaca

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
    'RISK_PERCENTAGE': 0.02,  # Percentage of cash to risk per trade
    'STOP_LOSS_ATR_MULTIPLIER': 0.8,  # Multiplier for ATR-based stop loss
    'TAKE_PROFIT_ATR_MULTIPLIER': 2.5,  # Multiplier for ATR-based take profit
    'TRAILING_STOP_PERCENTAGE': 0.015,  # Percentage for trailing stop

    # Strategy Thresholds - Thresholds for trading decisions
    'CONFIDENCE_THRESHOLD': 0.5,  # Threshold for prediction confidence
    'PREDICTION_THRESHOLD_BUY': 0.55,  # Threshold for buy signal
    'PREDICTION_THRESHOLD_SELL': 0.45,  # Threshold for sell signal
    'RSI_BUY_THRESHOLD': 65,  # RSI threshold for buying 54 is optimal
    'RSI_SELL_THRESHOLD': 45,  # RSI threshold for selling^50
    'ADX_TREND_THRESHOLD': 25,  # Threshold for ADX trend strength
    'MAX_VOLATILITY': 10,  # Maximum allowed volatility

    # Sentiment Analysis - Settings for sentiment analysis
    #'SENTIMENT_MODEL': 'distilbert-base-uncased-finetuned-sst-2-english',  # Model for sentiment analysis

    # API Retry Settings - Configuration for handling API failures
    'API_RETRY_ATTEMPTS': 3,  # Number of retry attempts for API calls
    'API_RETRY_DELAY': 1000,  # Delay between retry attempts in milliseconds
    'DEBUG_MODE': True,  # Debug mode: True for verbose output, False for clean beginner-friendly UI
}
# transformers import pipeline
#pipe = pipeline('sentiment-analysis', device='cuda:0')  # Or the appropriate task name
#pipe = pipeline('sentiment-analysis', model='distilbert/distilbert-base-uncased-finetuned-sst-2-english', revision='714eb0f', device='cuda:0')
#sentiment_pipeline = pipeline("sentiment-analysis", model=CONFIG['SENTIMENT_MODEL'], framework="pt", device='cuda:0')

#pyenv activate pytorch_env
#python /mnt/c/Users/aipla/Downloads/alpaca_neural_bot_v8.py --backtest --force-train

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

if not CONFIG.get('DEBUG_MODE', True):
    stream_handler.setLevel(logging.WARNING)  # Suppress INFO logs in console (still written to file)

logger = logging.getLogger(__name__)

# Initialize sentiment analysis pipeline
#sentiment_pipeline = pipeline("sentiment-analysis", model=CONFIG['SENTIMENT_MODEL'], framework="pt", device='cuda:0') #Updated for Linux 24.04.03 Ubuntu with RTX 5080 GPU


def check_dependencies() -> None:
    """Check for required Python modules."""
    required_modules = [
        'torch', 'numpy', 'pandas', 'alpaca',
        'sklearn', 'talib', 'tenacity', 'smtplib', 'argparse', 'tqdm', 'colorama'
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
def backtest(symbol: str, model: nn.Module, scaler: StandardScaler, df: pd.DataFrame, initial_cash: float, stop_loss_multiplier: float, take_profit_multiplier: float, timesteps: int, buy_threshold: float, sell_threshold: float, min_holding_period: int, transaction_cost: float) -> Tuple[float, List[float], int, float]:
    """Backtest the trading strategy on historical data."""
    start_date = CONFIG['BACKTEST_START_DATE']
    # Set index to timestamp column, ensuring it's a timezone-aware DatetimeIndex
    df = df.set_index(pd.to_datetime(df['timestamp'], utc=True))
    backtest_df = df[df.index >= start_date]
    if len(backtest_df) < timesteps + 1:
        logger.warning(f"Insufficient data for backtest {symbol}: {len(backtest_df)} bars")
        return initial_cash, [], 0, 0

    X, y = preprocess_data(backtest_df, timesteps, add_noise=False)

    cash = initial_cash
    shares = 0
    entry_price = 0.0
    entry_time = None
    returns = []
    peak_value = initial_cash
    trade_count = 0
    wins = 0
    last_prediction = 0.5

    for i in tqdm(range(timesteps, len(backtest_df)), desc=f"Backtesting {symbol}", unit="step"):
        current_sequence = X[i - timesteps:i].reshape(1, timesteps, -1)
        prediction = make_prediction(model, scaler.transform(current_sequence.reshape(-1, current_sequence.shape[-1])).reshape(current_sequence.shape))
        price = backtest_df['close'].iloc[i]
        atr_val = backtest_df['ATR'].iloc[i]
        rsi = backtest_df['RSI'].iloc[i]
        adx = backtest_df['ADX'].iloc[i]
        volatility = backtest_df['Volatility'].iloc[i]
        timestamp = backtest_df.index[i]

        if shares == 0:
            qty = int((cash * CONFIG['RISK_PERCENTAGE']) / (price * stop_loss_multiplier * atr_val))
            if prediction > buy_threshold and rsi < CONFIG['RSI_BUY_THRESHOLD'] and adx > CONFIG['ADX_TREND_THRESHOLD'] and volatility < CONFIG['MAX_VOLATILITY']:
                if qty > 0:
                    shares = qty
                    entry_price = price
                    entry_time = timestamp
                    cash -= shares * price + transaction_cost
                    trade_count += 1
        else:
            holding_duration = (timestamp - entry_time).total_seconds() / 60
            if holding_duration >= min_holding_period:
                if prediction < sell_threshold or rsi > CONFIG['RSI_SELL_THRESHOLD'] or price <= entry_price - stop_loss_multiplier * atr_val or price >= entry_price + take_profit_multiplier * atr_val:
                    cash += shares * price - transaction_cost
                    return_val = (shares * (price - entry_price) - 2 * transaction_cost) / (shares * entry_price)
                    returns.append(return_val)
                    if return_val > 0:
                        wins += 1
                    shares = 0
                    entry_price = 0.0
                    entry_time = None

        portfolio_value = cash + shares * price
        peak_value = max(peak_value, portfolio_value)
        drawdown = (peak_value - portfolio_value) / peak_value
        if drawdown > CONFIG['MAX_DRAWDOWN_LIMIT']:
            # Liquidate if drawdown exceeds limit
            if shares > 0:
                cash += shares * price - transaction_cost
                shares = 0

        last_prediction = prediction

    return cash, returns, trade_count, (wins / trade_count * 100 if trade_count > 0 else 0)

def load_or_fetch_data(symbol: str, start_date: str, end_date: str) -> Tuple[pd.DataFrame, bool]:
    """Load historical data from cache or fetch from API."""
    cache_file = os.path.join(CONFIG['CACHE_DIR'], f"{symbol}_data.pkl")
    if os.path.exists(cache_file) and (time.time() - os.path.getmtime(cache_file)) < CONFIG['CACHE_EXPIRY_SECONDS']:
        with open(cache_file, 'rb') as f:
            df = pickle.load(f)
        logger.info(f"Loaded {len(df)} bars for {symbol} from cache")
        return df, True
    else:
        df = fetch_data(symbol, start_date, end_date)
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

@retry(
    stop=stop_after_attempt(CONFIG['API_RETRY_ATTEMPTS']),
    wait=wait_fixed(CONFIG['API_RETRY_DELAY'] / 1000),
    retry=retry_if_exception_type(Exception)
)
def fetch_data(symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
    """Fetch historical bars from Alpaca API."""
    client = StockHistoricalDataClient(CONFIG['ALPACA_API_KEY'], CONFIG['ALPACA_SECRET_KEY'])
    start_dt = datetime.strptime(start_date, '%Y-%m-%d').replace(tzinfo=timezone.utc)
    end_dt = datetime.strptime(end_date, '%Y-%m-%d').replace(tzinfo=timezone.utc)
    request = StockBarsRequest(
        symbol_or_symbols=symbol,
        timeframe=CONFIG['TIMEFRAME'],
        start=start_dt,
        end=end_dt
    )
    bars = client.get_stock_bars(request).df
    if bars.empty:
        raise ValueError(f"No data for {symbol} from {start_date} to {end_date}")
    df = bars.reset_index().rename(columns={'vwap': 'VWAP'})
    logger.info(f"Fetched {len(df)} bars for {symbol} from {start_date} to {end_date}")
    return df.sort_values('timestamp')


    bars = client.get_stock_bars(request).df
    if bars.empty:
        raise ValueError(f"No recent data for {symbol}")
    df = bars.reset_index().rename(columns={'vwap': 'VWAP'})
    logger.info(f"Fetched {len(df)} recent bars for {symbol}")
    return df.sort_values('timestamp')
"""
def load_news_sentiment(symbol: str) -> Tuple[float, bool]:
    #Compute real-time news sentiment using a pre-trained model or random for testing.
    cache_file = os.path.join(CONFIG['CACHE_DIR'], f"{symbol}_news_sentiment.pkl")
    # Force new sentiment calculation by ignoring cache
    sentiment_score = np.random.uniform(-1.0, 1.0)  # Random sentiment for testing
    with open(cache_file, 'wb') as f:
        pickle.dump(sentiment_score, f)
    return sentiment_score, False
"""



def calculate_indicators(df: pd.DataFrame, sentiment: float) -> pd.DataFrame:
    """Calculate technical indicators."""
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
    #df['Sentiment'] = sentiment
    df['Trend'] = np.where(df['close'] > df['MA20'], 1, 0)
    
    indicator_cols = [
        'MA20', 'MA50', 'RSI', 'MACD', 'MACD_signal', 'OBV', 'VWAP', 'ATR',
        'CMF', 'Close_ATR', 'MA20_ATR', 'Return_1d', 'Return_5d', 'Volatility',
        'BB_upper', 'BB_middle', 'BB_lower', 'Stoch_K', 'Stoch_D', 'ADX', 'Sentiment', 'Trend'
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
        'MA20', 'MA50', 'RSI', 'MACD', 'MACD_signal', 'OBV', 'VWAP', 'ATR',
        'CMF', 'Close_ATR', 'MA20_ATR', 'Return_1d', 'Return_5d', 'Volatility',
        'BB_upper', 'BB_lower', 'Stoch_K', 'Stoch_D', 'ADX', 'Sentiment', 'Trend'
    ]
    if not all(col in df.columns for col in required_cols):
        missing = [col for col in required_cols if col not in df.columns]
        raise ValueError(f"Missing indicator columns for {symbol}: {missing}")

def preprocess_data(df: pd.DataFrame, timesteps: int, add_noise: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """Preprocess data for model input."""
    features = ['open', 'high', 'low', 'close', 'volume', 'VWAP', 'EMA_12', 'EMA_26', 'MACD', 'MACD_signal', 'RSI', 'ATR', 'ADX', 'OBV', 'CCI', 'ROC', 'MOM', 'Bollinger_upper', 'Bollinger_middle', 'Bollinger_lower', 'Stochastic_K', 'Stochastic_D', 'Williams_R', 'Volatility']
    X = []
    y = []
    for i in range(timesteps, len(df)):
        X.append(df[features].iloc[i - timesteps:i].values)
        y.append(1 if df['close'].iloc[i] > df['close'].iloc[i - 1] else 0)
    X = np.array(X)
    y = np.array(y)
    if add_noise:
        noise = np.random.normal(0, 0.01, X.shape)
        X += noise
    return X, y

def test_predictions_shape():
    # Mock model output as (5,1) array
    mock_outputs = np.random.randn(5, 1)
    predictions = 1 / (1 + np.exp(-mock_outputs)).flatten()
    assert predictions.ndim == 1, "Predictions should be 1D"
    pred = predictions[0]
    assert isinstance(pred, np.float64), "Individual pred should be scalar"
    print(f"Formatted pred: {pred:.3f}")  # Should not raise error

class TradingModel(nn.Module):
    def __init__(self, timesteps: int, features: int):
        super(TradingModel, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=features, out_channels=64, kernel_size=3)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3)
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.lstm1 = nn.LSTM(input_size=64, hidden_size=128, num_layers=2, batch_first=True)
        self.lstm2 = nn.LSTM(input_size=128, hidden_size=128, num_layers=2, batch_first=True)
        self.dense1 = nn.Linear(128, 256)
        self.dense2 = nn.Linear(256, 128)
        self.dense3 = nn.Linear(128, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)  # Reduced dropout to allow better learning
        nn.init.xavier_uniform_(self.dense1.weight)
        nn.init.xavier_uniform_(self.dense2.weight)
        nn.init.xavier_uniform_(self.dense3.weight)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = x.permute(0, 2, 1)
        x, _ = self.lstm1(x)
        x, _ = self.lstm2(x)
        x = x[:, -1, :]
        x = self.relu(self.dense1(x))
        x = self.dropout(x)
        x = self.relu(self.dense2(x))
        x = self.dropout(x)
        x = self.dense3(x)  # Linear output
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

def train_model(symbol: str, X: np.ndarray, y: np.ndarray, epochs: int, batch_size: int) -> Tuple[nn.Module, StandardScaler]:
    """Train the PyTorch model with early stopping and logging."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Training model for {symbol} on device: {device}")
    train_size = int(0.8 * len(X))
    X_train, X_val = X[:train_size], X[train_size:]
    y_train, y_val = y[:train_size], y[train_size:]
    scaler = StandardScaler()
    X_scaled_train = scaler.fit_transform(X_train.reshape(-1, X.shape[-1])).reshape(X_train.shape)
    X_scaled_val = scaler.transform(X_val.reshape(-1, X.shape[-1])).reshape(X_val.shape)

        # Ensure X_scaled_train and X_scaled_val are 3D (n_samples, timesteps, features)
    if len(X_scaled_train.shape) != 3:
        raise ValueError(f"Expected 3D input for X_scaled_train, got shape {X_scaled_train.shape}")
    if len(X_scaled_val.shape) != 3:
        raise ValueError(f"Expected 3D input for X_scaled_val, got shape {X_scaled_val.shape}")
    # Ensure y_train and y_val are 1D (n_samples,)
    if len(y_train.shape) != 1:
        raise ValueError(f"Expected 1D input for y_train, got shape {y_train.shape}")
    if len(y_val.shape) != 1:
        raise ValueError(f"Expected 1D input for y_val, got shape {y_val.shape}")

    X_train_tensor = torch.tensor(X_scaled_train, dtype=torch.float32).to(device)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).reshape(-1, 1).to(device)
    X_val_tensor = torch.tensor(X_scaled_val, dtype=torch.float32).to(device)
    y_val_tensor = torch.tensor(y_val, dtype=torch.float32).reshape(-1, 1).to(device)

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=False)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=0, pin_memory=False)

    model = build_model(X.shape[1], X.shape[2]).to(device)

    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)  # Reduced LR and weight decay for better convergence
    criterion = nn.BCEWithLogitsLoss()

    best_val_loss = float('inf')
    patience_counter = 0
    patience = CONFIG['EARLY_STOPPING_PATIENCE']
    min_delta = CONFIG['EARLY_STOPPING_MIN_DELTA']
    best_model_state = None

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * batch_x.size(0)
        train_loss /= len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item() * batch_x.size(0)
        val_loss /= len(val_loader.dataset)

        logger.info(f"Epoch {epoch+1}/{epochs}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")  # Added logging for monitoring

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

    return model, scaler


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
    df, data_loaded = load_or_fetch_data(symbol, CONFIG['TRAIN_DATA_START_DATE'], datetime.now().strftime('%Y-%m-%d'))
    validate_raw_data(df, symbol)
    #sentiment, sentiment_loaded = load_news_sentiment(symbol)
    
    df = df.set_index(pd.to_datetime(df['timestamp'], utc=True))
    train_end = pd.to_datetime(CONFIG['BACKTEST_START_DATE'], utc=True)
    df_train = df[df.index < train_end].copy()
    #sentiment, sentiment_loaded = load_news_sentiment(symbol)
    df_train = calculate_indicators(df_train, sentiment)
    validate_indicators(df_train, symbol)
    X, y = preprocess_data(df_train, CONFIG['TIMESTEPS'], add_noise=True)
    
    df = calculate_indicators(df, sentiment)
    validate_indicators(df, symbol)
    train_end = pd.to_datetime(CONFIG['BACKTEST_START_DATE'], utc=True)
    df_train = df[df.index < train_end].copy()
    X, y = preprocess_data(df_train, CONFIG['TIMESTEPS'], add_noise=True)
    model, scaler = load_model_and_scaler(symbol, expected_features, force_train)
    model_loaded = model is not None and scaler is not None
    if not model_loaded:
        X, y = preprocess_data(df, CONFIG['TIMESTEPS'], add_noise=True)
        model, scaler = train_model(symbol, X, y, CONFIG['TRAIN_EPOCHS'], CONFIG['BATCH_SIZE'])
        del X, y
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return symbol, df, model, scaler, data_loaded, model_loaded

def backtest(symbol: str, model: nn.Module, scaler: StandardScaler, df: pd.DataFrame,
            initial_cash: float, stop_loss_atr_multiplier: float, take_profit_atr_multiplier: float,
            timesteps: int, threshold_buy: float, threshold_sell: float,
            min_holding_period_minutes: int, transaction_cost_per_trade: float) -> Tuple[float, List[float], int, float]:
    """Backtest the model on historical data."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()

    # Set index to timestamp column, ensuring it's a timezone-aware DatetimeIndex
    df = df.set_index(pd.to_datetime(df['timestamp'], utc=True))

    # Filter data for backtest period with timezone-aware datetime
    start_date = pd.to_datetime(CONFIG['BACKTEST_START_DATE'], utc=True)
    backtest_df = df[df.index >= start_date]
    if len(backtest_df) < CONFIG['MIN_DATA_POINTS']:
        logger.warning(f"Insufficient data for {symbol} backtest: {len(backtest_df)} points")
        return initial_cash, [], 0, 0.0

    #sentiment = load_news_sentiment(symbol)[0]
    #backtest_df = calculate_indicators(backtest_df, sentiment)
    X, _ = preprocess_data(backtest_df, timesteps, add_noise=False)
    if X.shape[0] == 0:
        logger.warning(f"No valid sequences for {symbol} backtest")
        return initial_cash, [], 0, 0.0

    X_scaled = scaler.transform(X.reshape(-1, X.shape[-1])).reshape(X.shape)
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(device)
    with torch.no_grad():
        outputs = model(X_tensor).cpu().numpy()
        predictions = 1 / (1 + np.exp(-outputs)).flatten()  # Sigmoid activation and flatten to 1D

    predictions = np.array(predictions)
    logger.info(f"Raw sigmoid predictions for {symbol}: min={predictions.min():.3f}, max={predictions.max():.3f}, mean={predictions.mean():.3f}")
    logger.info(f"Using raw predictions for {symbol} (no normalization applied)")
    
    k_start = 0  # Define starting index for backtest simulation
    
    pbar = None
    if not CONFIG['DEBUG_MODE']:
        pbar = tqdm(total=len(predictions) - k_start, desc=f"Backtesting {symbol}", leave=True)
    
    # Clean up CUDA tensors
    del X_tensor
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # Initialize backtest variables
    cash = initial_cash / len(CONFIG['SYMBOLS'])
    position = 0
    entry_price = 0.0
    entry_time = None
    max_price = 0.0
    returns = []
    trade_count = 0
    winning_trades = 0

    # Align prices and indicators with predictions (skip initial timesteps where indicators are NaN)
    prices = backtest_df['close'].iloc[timesteps:].values
    atr = backtest_df['ATR'].iloc[timesteps:].values
    rsi = backtest_df['RSI'].iloc[timesteps:].values
    adx = backtest_df['ADX'].iloc[timesteps:].values
    volatility = backtest_df['Volatility'].iloc[timesteps:].values
    timestamps = backtest_df.index[timesteps:]

    for i in range(k_start, len(predictions)):
        pred = predictions[i]
        price = prices[i]
        atr_val = atr[i]
        current_rsi = rsi[i]
        current_adx = adx[i]
        current_volatility = volatility[i]
        ts = timestamps[i]

        if position == 0:
            qty = max(1, int(cash * CONFIG['RISK_PERCENTAGE'] / (atr_val * stop_loss_atr_multiplier)))
            if current_volatility <= CONFIG['MAX_VOLATILITY'] and current_adx >= CONFIG['ADX_TREND_THRESHOLD'] and pred >= threshold_buy and current_rsi < CONFIG['RSI_BUY_THRESHOLD']:
                cost = qty * price + transaction_cost_per_trade
                if cost <= cash:
                    position = qty
                    entry_price = price
                    entry_time = ts
                    max_price = price
                    cash -= cost
                    logger.info(f"{ts}: Bought {qty} shares of {symbol} at ${price:.2f}, cash: ${cash:.2f}")
                    if CONFIG['DEBUG_MODE']:
                        print(f"{ts}: Bought {qty} shares of {symbol} at ${price:.2f}, cash: ${cash:.2f}")
                    if pbar and not CONFIG['DEBUG_MODE']:
                        pbar.set_postfix(status="Bought", cash=f"${cash:.2f}")  # Simplified postfix for clean UI
                else:
                    if CONFIG['DEBUG_MODE']:
                        logger.info(f"Insufficient cash to buy {qty} shares of {symbol}: cash={cash:.2f}, cost={cost:.2f}")
            else:
                if CONFIG['DEBUG_MODE']:
                    logger.info(f"Skipped buy for {symbol}: pred={pred:.3f}, rsi={current_rsi:.2f}, adx={current_adx:.2f}, volatility={current_volatility:.2f}, qty={qty}, cost={qty * price:.2f}, cash={cash:.2f}")
                if pbar and not CONFIG['DEBUG_MODE']:
                    pbar.set_postfix(status="Skipped")  # Simplified: Remove metrics to reduce clutter
        else:
            time_held = (ts - entry_time).total_seconds() / 60
            max_price = max(max_price, price)
            trailing_stop = max_price * (1 - CONFIG['TRAILING_STOP_PERCENTAGE'])
            stop_loss = entry_price - stop_loss_atr_multiplier * atr_val
            take_profit = entry_price + take_profit_atr_multiplier * atr_val
            if time_held >= min_holding_period_minutes and (price <= trailing_stop or price <= stop_loss or price >= take_profit or (pred <= threshold_sell and current_rsi > CONFIG['RSI_SELL_THRESHOLD'])):
                cash += position * price - transaction_cost_per_trade
                ret = (price - entry_price) / entry_price
                returns.append(ret)
                trade_count += 1
                if ret > 0:
                    winning_trades += 1
                logger.info(f"{ts}: Sold {position} shares of {symbol} at ${price:.2f}, return: {ret:.3f}, cash: ${cash:.2f}")
                if CONFIG['DEBUG_MODE']:
                    print(f"{ts}: Sold {position} shares of {symbol} at ${price:.2f}, return: {ret:.3f}, cash: ${cash:.2f}")
                if pbar and not CONFIG['DEBUG_MODE']:
                    pbar.set_postfix(status="Sold", cash=f"${cash:.2f}")  # Simplified postfix
                position = 0
                logger.info(f"After sell: position={position}, cash={cash:.2f}")
                entry_time = None
                max_price = 0.0
            else:
                if pbar and not CONFIG['DEBUG_MODE']:
                    pbar.set_postfix(status="Holding")  # Simplified: Remove detailed metrics for less clutter

        if pbar:
            pbar.update(1)

    if pbar:
        pbar.close()

    if position > 0:
        cash += position * prices[-1] - transaction_cost_per_trade
        ret = (prices[-1] - entry_price) / entry_price
        returns.append(ret)
        trade_count += 1
        if ret > 0:
            winning_trades += 1
        logger.info(f"Final close: Sold {position} shares of {symbol} at ${prices[-1]:.2f}, return: {ret:.3f}")
        if CONFIG['DEBUG_MODE']:
            print(f"Final close: Sold {position} shares of {symbol} at ${prices[-1]:.2f}, return: {ret:.3f}")
    win_rate = (winning_trades / trade_count * 100) if trade_count > 0 else 0.0
    return cash, returns, trade_count, win_rate

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

def make_prediction(model: nn.Module, X_scaled: np.ndarray) -> float:
    """Make a prediction using the PyTorch model."""
    if X_scaled.size == 0 or X_scaled.shape[0] == 0:
        logger.error("Empty input data for prediction")
        raise ValueError("Input data for prediction is empty")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(device)
    with torch.no_grad():
        prediction = torch.sigmoid(model(X_tensor)).cpu().numpy()[0][0]
    if not np.isfinite(prediction):
        logger.error("Invalid prediction value: non-finite")
        raise ValueError("Prediction resulted in non-finite value")
    return float(prediction)

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
    expected_features = 22
    models = {}
    scalers = {}
    dfs = {}
    stock_info = []
    total_epochs = len(CONFIG['SYMBOLS']) * CONFIG['TRAIN_EPOCHS']
    need_training = any(load_model_and_scaler(symbol, expected_features, force_train)[0] is None for symbol in CONFIG['SYMBOLS'])
    progress_bar = tqdm(total=total_epochs, desc="Training Progress", bar_format="{l_bar}\033[32m{bar}\033[0m{r_bar}") if need_training and force_train else None

    mp.set_start_method('spawn', force=True)
    import random
    np.random.seed(42)
    torch.manual_seed(42)
    random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    pool = mp.Pool(processes=min(mp.cpu_count(), len(CONFIG['SYMBOLS'])))
    results = [pool.apply_async(train_symbol, args=(symbol, expected_features, force_train)) for symbol in CONFIG['SYMBOLS']]
    pool.close()
    pool.join()

    # Get the results
    outputs = [res.get() for res in results]

    symbol_pbar = None
    if not CONFIG['DEBUG_MODE']:
        symbol_pbar = tqdm(total=len(CONFIG['SYMBOLS']), desc="Processing symbols", bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]")

    # Enhanced CUDA cleanup
    del results  # Delete results reference
    if torch.cuda.is_available():
        torch.cuda.synchronize()  # Wait for all GPU operations to finish
        torch.cuda.empty_cache()  # Clear GPU memory
        time.sleep(2)  # Allow GPU to stabilize
        logger.info("CUDA memory cleared after multiprocessing.")

        
        for symbol, df, model, scaler, data_loaded, model_loaded in outputs:
            dfs[symbol] = df
            models[symbol] = model
            scalers[symbol] = scaler
            info = []
            info.append(f"{Fore.LIGHTBLUE_EX}{symbol}:{Style.RESET_ALL}")
            info.append(f"  {'Loaded cached model and scaler' if model_loaded else 'Trained model'} for {symbol}.")
            info.append(f"  {'Loaded' if data_loaded else 'Fetched'} {len(df)} bars for {symbol} {'from cache' if data_loaded else ''}.")
            #info.append(f"  {'Loaded news data' if sentiment_loaded else 'Computed news sentiment'} for {symbol} {'from cache' if sentiment_loaded else ''}.")
            #info.append(f"  Calculated sentiment score: {sentiment:.3f}")
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
            if symbol_pbar:
                symbol_pbar.update(1)

    if progress_bar:
        progress_bar.close()
    if symbol_pbar:
        symbol_pbar.close()

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
                        #sentiment = load_news_sentiment(symbol)[0]
                        #df = calculate_indicators(df, sentiment)
                        X, _ = preprocess_data(df, CONFIG['TIMESTEPS'], add_noise=False)
                        #sentiment, sentiment_loaded = load_news_sentiment(symbol)
                        #df = calculate_indicators(df, sentiment)
                        validate_indicators(df, symbol)
                        df = df.set_index(pd.to_datetime(df['timestamp'], utc=True))
                        train_end = pd.to_datetime(CONFIG['BACKTEST_START_DATE'], utc=True)
                        df_train = df[df.index < train_end].copy()
                        X, y = preprocess_data(df_train, CONFIG['TIMESTEPS'], add_noise=True)
                        if X.shape[0] > 0:
                            last_sequence = X[-1].reshape(1, CONFIG['TIMESTEPS'], -1)
                            X_scaled = scalers[symbol].transform(last_sequence.reshape(-1, last_sequence.shape[-1])).reshape(last_sequence.shape)
                            prediction = make_prediction(models[symbol], X_scaled)
                            price = df['close'].iloc[-1]
                            current_rsi = df['RSI'].iloc[-1]
                            current_adx = df['ADX'].iloc[-1]
                            current_volatility = df['Volatility'].iloc[-1]
                            atr_val = df['ATR'].iloc[-1]
                            qty_owned = 0
                            entry_price = 0.0
                            position = next((pos for pos in open_positions if pos.symbol == symbol), None)
                            if position:
                                qty_owned = int(float(position.qty))
                                entry_price = float(position.avg_entry_price)

                            decision = "Hold"
                            if current_volatility <= CONFIG['MAX_VOLATILITY'] and current_adx >= CONFIG['ADX_TREND_THRESHOLD']:
                                if qty_owned == 0 and (prediction >= max(CONFIG['PREDICTION_THRESHOLD_BUY'], CONFIG['CONFIDENCE_THRESHOLD']) and current_rsi < CONFIG['RSI_BUY_THRESHOLD'] and current_adx > CONFIG['ADX_TREND_THRESHOLD']):
                                    decision = "Buy"
                                    cash = float(account.cash)
                                    max_qty = int(float(account.buying_power) / price)
                                    risk_amount = cash * CONFIG['RISK_PERCENTAGE']
                                    qty = max(1, int(min(risk_amount / (atr_val * CONFIG['STOP_LOSS_ATR_MULTIPLIER']), max_qty)))
                                    qty = min(qty, int(portfolio_value * 0.1 / price))
                                    logger.info(f"Buy attempt for {symbol}: qty={qty}, cash={cash:.2f}, buying_power={float(account.buying_power):.2f}, cost={qty * price:.2f}, atr_val={atr_val:.2f}")
                                    if qty > 0 and (qty * price) <= float(account.buying_power):
                                        order = MarketOrderRequest(symbol=symbol, qty=qty, side=OrderSide.BUY, time_in_force=TimeInForce.GTC)
                                        try:
                                            trading_client.submit_order(order)
                                            email_body = f"""
Bought {qty} shares of {symbol} at ${price:.2f}
Prediction Confidence: {prediction:.3f}
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
Prediction Confidence: {prediction:.3f}
RSI: {current_rsi:.2f}
ADX: {current_adx:.2f}
Volatility: {current_volatility:.2f}
ATR: {atr_val:.2f}
Current Cash: ${cash:.2f}
Portfolio Value: ${portfolio_value:.2f}
"""
                                            send_email(f"Trade Failure: {symbol}", email_body)
                                    else:
                                        decision = "Hold"
                                        logger.info(f"Buy skipped for {symbol}: Insufficient qty ({qty}) or buying power (cost={qty * price:.2f}, buying_power={float(account.buying_power):.2f})")
                                if qty_owned > 0:
                                    max_prices[symbol] = max(max_prices[symbol], price)
                                    trailing_stop = max_prices[symbol] * (1 - CONFIG['TRAILING_STOP_PERCENTAGE'])
                                    stop_loss = entry_price - CONFIG['STOP_LOSS_ATR_MULTIPLIER'] * atr_val
                                    take_profit = entry_price + CONFIG['TAKE_PROFIT_ATR_MULTIPLIER'] * atr_val
                                    if (price <= trailing_stop or price <= stop_loss or price >= take_profit or (prediction <= CONFIG['PREDICTION_THRESHOLD_SELL'] and current_rsi > CONFIG['RSI_SELL_THRESHOLD'])):
                                        decision = "Sell"
                                        order = MarketOrderRequest(symbol=symbol, qty=qty_owned, side=OrderSide.SELL, time_in_force=TimeInForce.GTC)
                                        try:
                                            trading_client.submit_order(order)
                                            email_body = f"""
Sold {qty_owned} shares of {symbol} at ${price:.2f}
Prediction Confidence: {prediction:.3f}
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
Prediction Confidence: {prediction:.3f}
RSI: {current_rsi:.2f}
ADX: {current_adx:.2f}
Volatility: {current_volatility:.2f}
ATR: {atr_val:.2f}
Current Cash: ${float(account.cash):.2f}
Portfolio Value: ${portfolio_value:.2f}
"""
                                            send_email(f"Trade Failure: {symbol}", email_body)
                            decisions.append({
                                'symbol': symbol,
                                'decision': decision,
                                'confidence': prediction,
                                'rsi': current_rsi,
                                'adx': current_adx,
                                'volatility': current_volatility,
                                'price': price,
                                'owned': qty_owned
                            })
                summary_body = "Trading Summary:\n"
                for dec in decisions:
                    summary_body += f"{dec['symbol']}: {dec['decision']}, Confidence: {dec['confidence']:.3f}, RSI: {dec['rsi']:.2f}, ADX: {dec['adx']:.2f}, Volatility: {dec['volatility']:.2f}, Price: ${dec['price']:.2f}, Owned: {dec['owned']}\n"
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
        if not CONFIG['DEBUG_MODE']:
            print("\nBacktest Summary:")
            print(f"Initial Cash: {Fore.CYAN}${initial_cash:.2f}{Style.RESET_ALL}")
            print("Per Symbol Performance:")
            for symbol in CONFIG['SYMBOLS']:
                if symbol in symbol_results:
                    metrics = symbol_results[symbol]
                    win_color = Fore.GREEN if win_rates.get(symbol, 0) > 50 else Fore.RED
                    sharpe_color = Fore.GREEN if metrics.get('sharpe_ratio', 0) > 0 else Fore.RED
                    drawdown = metrics.get('max_drawdown', 0)
                    draw_color = Fore.GREEN if drawdown < 10 else (Fore.YELLOW if drawdown < 20 else Fore.RED)
                    print(f"  {Fore.BLUE}{symbol}{Style.RESET_ALL}: Trades={trade_counts.get(symbol, 0)}, Win Rate={win_color}{win_rates.get(symbol, 0):.1f}%{Style.RESET_ALL}, Sharpe Ratio={sharpe_color}{metrics.get('sharpe_ratio', 0):.2f}{Style.RESET_ALL}, Max Drawdown={draw_color}{drawdown:.2f}%{Style.RESET_ALL}")
            final_color = Fore.GREEN if final_value > initial_cash else Fore.RED
            print(f"Final Portfolio Value: {final_color}${final_value:.2f}{Style.RESET_ALL}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Trading bot with backtest mode")
    parser.add_argument('--backtest', action='store_true', help="Run in backtest-only mode")
    parser.add_argument('--force-train', action='store_true', help="Force retraining of models")
    args = parser.parse_args()
    main(backtest_only=args.backtest, force_train=args.force_train)
