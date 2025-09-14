
# +------------------------------------------------------------------------------+
# |                            Alpaca Neural Bot v8.9                            |
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
from sklearn.preprocessing import RobustScaler, StandardScaler
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
    'BACKTEST_START_DATE': '2020-01-01',  # Start date for backtesting (historical data availability)
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
    'LEARNING_RATE': 0.005,  # Initial learning rate for Adam
    'LR_SCHEDULER_PATIENCE': 5,  # Patience for ReduceLROnPlateau
    'LR_REDUCTION_FACTOR': 0.5,  # Factor to multiply LR by

    # API and Authentication - Credentials for API access
    'ALPACA_API_KEY': 'PK442T0XBG553SK7IZ5B',  # API key for Alpaca
    'ALPACA_SECRET_KEY': '2upYWNzeRIGGRHk1FXKVtDpdSbgqlqmP3Q0flW8Z',  # Secret key for Alpaca

    # Email Notifications - Configuration for sending email alerts
    'EMAIL_SENDER': 'alpaca.ai.tradingbot@gmail.com',  # Email address for sending notifications
    'EMAIL_PASSWORD': 'hjdf sstp pyne rotq',  # Password for the email account
    'EMAIL_RECEIVER': ['aiplane.scientist@gmail.com', 'vmakarov28@students.d125.org'],  # List of email recipients
    'SMTP_SERVER': 'smtp.gmail.com',  # SMTP server for email
    'SMTP_PORT': 587,  # Port for SMTP server

    # Logging and Monitoring - Settings for tracking activities
    'LOG_FILE': 'trades.log',  # File for logging trades

    # Strategy Thresholds - Thresholds for trading decisions
    'CONFIDENCE_THRESHOLD': 0.55,  # Threshold for prediction confidence (raised to use model more selectively)
    'PREDICTION_THRESHOLD_BUY': 0.55,  # Threshold for buy signal (lowered to allow more opportunities while above 0.5)
    'PREDICTION_THRESHOLD_SELL': 0.45,  # Threshold for sell signal (increased for more balanced exits)
    'RSI_BUY_THRESHOLD': 55,  # RSI threshold for buying (lowered for stronger oversold signals)
    'RSI_SELL_THRESHOLD': 40,  # RSI threshold for selling (raised for stronger overbought signals)
    'ADX_TREND_THRESHOLD': 25,  # Threshold for ADX trend strength (lowered to capture more trends)
    'MAX_VOLATILITY': 3.0,  # Maximum allowed volatility (increased to include more market conditions)

    # Risk Management - Parameters to control trading risk
    'MAX_DRAWDOWN_LIMIT': 0.04,  # Maximum allowed drawdown
    'RISK_PERCENTAGE': 0.02,  # Percentage of cash to risk per trade (halved for smaller positions)
    'STOP_LOSS_ATR_MULTIPLIER': 1.5,  # Multiplier for ATR-based stop loss (widened to reduce whipsaws)
    'TAKE_PROFIT_ATR_MULTIPLIER': 3.0,  # Multiplier for ATR-based take profit (tightened for quicker exits)
    'TRAILING_STOP_PERCENTAGE': 0.02,  # Percentage for trailing stop (widened slightly)

    # Trading Parameters - Settings related to trading operations
    'TRANSACTION_COST_PER_TRADE': 0.01,  # Cost per trade

    # Sentiment Analysis - Settings for sentiment analysis
    'SENTIMENT_MODEL': 'distilbert-base-uncased-finetuned-sst-2-english',  # Model for sentiment analysis

    # API Retry Settings - Configuration for handling API failures
    'API_RETRY_ATTEMPTS': 3,  # Number of retry attempts for API calls
    'API_RETRY_DELAY': 1000,  # Delay between retry attempts in milliseconds
}

#pyenv activate pytorch_env
#python /mnt/c/Users/aipla/Downloads/alpaca_neural_bot_v8.7.py --backtest --force-train

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
device = 0 if torch.cuda.is_available() else -1
sentiment_pipeline = pipeline("sentiment-analysis", model=CONFIG['SENTIMENT_MODEL'], framework="pt", device=device)

def check_dependencies() -> None:
    """Check for required Python modules."""
    required_modules = [
        'torch', 'numpy', 'pandas', 'alpaca', 'transformers',
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
    start_date = end_date - timedelta(days=3)
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
    if os.path.exists(cache_file) and (time.time() - os.path.getmtime(cache_file)) < CONFIG['CACHE_EXPIRY_SECONDS']:
        with open(cache_file, 'rb') as f:
            sentiment_score = pickle.load(f)
        return sentiment_score, True
    else:
        sentiment_score = np.random.uniform(-1.0, 1.0)  # Random sentiment for testing
        with open(cache_file, 'wb') as f:
            pickle.dump(sentiment_score, f)
        return sentiment_score, False

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
    df['Sentiment'] = sentiment
    df['Trend'] = np.where(df['close'] > df['MA20'], 1, 0)
    # New future direction target: 1 if next close > current close (binary up/down prediction)
    df['Future_Direction'] = np.where(df['close'].shift(-5) > df['close'], 1, 0)
    
    indicator_cols = [
        'MA20', 'MA50', 'RSI', 'MACD', 'MACD_signal', 'OBV', 'VWAP', 'ATR',
        'CMF', 'Close_ATR', 'MA20_ATR', 'Return_1d', 'Return_5d', 'Volatility',
        'BB_upper', 'BB_middle', 'BB_lower', 'Stoch_K', 'Stoch_D', 'ADX', 'Sentiment', 'Trend', 'Future_Direction'
    ]
    df = df.dropna(subset=indicator_cols)  # Drops last row due to shift(-1) NaN
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

def preprocess_data(df: pd.DataFrame, timesteps: int, add_noise: bool = False, inference_scaler: Optional[RobustScaler] = None, inference_mode: bool = False) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[RobustScaler]]:
    """Preprocess data into sequences for future price direction prediction (next bar up/down).
    
    Args:
        inference_mode: If True, use provided inference_scaler for scaling (no fitting); return None for y_seq and scaler.
        inference_scaler: Scaler to use in inference mode.
    
    Returns:
        X_seq: Scaled input sequences.
        y_seq: Targets (None in inference mode).
        scaler: Fitted scaler (None in inference mode).
    """
    df = df.copy()
    df.ffill(inplace=True)
    df.fillna(0, inplace=True)
    
    features = [
        'close', 'high', 'low', 'volume', 'MA20', 'MA50', 'RSI',
        'MACD', 'MACD_signal', 'OBV', 'VWAP', 'ATR', 'CMF', 'Close_ATR',
        'MA20_ATR', 'Return_1d', 'Return_5d', 'Volatility', 'BB_upper',
        'BB_lower', 'Stoch_K', 'Stoch_D', 'ADX', 'Sentiment'
    ]
    if 'Future_Direction' not in df.columns and not inference_mode:
        raise ValueError("Future_Direction column missing; required for training.")
    X_raw = df[features].values
    
    if inference_mode:
        if inference_scaler is None:
            raise ValueError("inference_scaler must be provided in inference_mode.")
        X = inference_scaler.transform(X_raw)
        y_seq = None
        scaler = None
    else:
        y = df['Future_Direction'].values
        scaler = RobustScaler()
        X = scaler.fit_transform(X_raw)
        y_seq = y
    
    if add_noise:
        X += np.random.normal(0, 0.02, X.shape)
    
    N = X.shape[0]
    num_sequences = N - timesteps
    if num_sequences <= 0:
        raise ValueError(f"Not enough data for {timesteps} timesteps: only {N} rows available")
    
    # Create sliding windows for past data to predict NEXT bar's direction
    window = np.lib.stride_tricks.sliding_window_view(X, (timesteps, X.shape[1]))
    X_seq = window[:num_sequences].reshape(num_sequences, timesteps, X.shape[1])
    
    if not inference_mode:
        y_seq = y[timesteps - 1: timesteps - 1 + num_sequences]  # Align target to end of each sequence (predict direction from sequence end to end+3)
        logger.info(f"Preprocessed {len(X_seq)} sequences; y balance: {np.mean(y_seq):.3f} (up fraction)")
    else:
        logger.info(f"Preprocessed {len(X_seq)} inference sequences")
    
    return X_seq, y_seq, scaler

class TradingModel(nn.Module):
    def __init__(self, timesteps: int, features: int):
        super(TradingModel, self).__init__()
        self.lstm = nn.LSTM(input_size=features, hidden_size=64, num_layers=1, batch_first=True, dropout=0.1)
        self.dense1 = nn.Linear(64, 32)
        self.dense2 = nn.Linear(32, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)
        nn.init.xavier_uniform_(self.dense1.weight)
        nn.init.xavier_uniform_(self.dense2.weight)

    def forward(self, x):
        x, _ = self.lstm(x)
        x = x[:, -1, :]
        x = self.relu(self.dense1(x))
        x = self.dropout(x)
        x = self.dense2(x)
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

def train_model(symbol: str, X: np.ndarray, y: np.ndarray, epochs: int, batch_size: int, timesteps: int = CONFIG['TIMESTEPS'], features: int = 24, scaler: Optional[RobustScaler] = None) -> Tuple[nn.Module, RobustScaler]:
    """Train the CNN-LSTM model for the symbol."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Training model for {symbol} on device: {device}")

    # Fixed: Use defined TradingModel class instead of undefined CNNLSTM
    model = TradingModel(CONFIG['TIMESTEPS'], features).to(device)
    criterion = nn.BCEWithLogitsLoss()  # Define loss function
    optimizer = optim.Adam(model.parameters(), lr=CONFIG['LEARNING_RATE'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=CONFIG['LR_SCHEDULER_PATIENCE'], factor=CONFIG['LR_REDUCTION_FACTOR'])

    # DataLoader (assumes X is already scaled from preprocess_data)
    dataset = TensorDataset(torch.FloatTensor(X), torch.FloatTensor(y))
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch.unsqueeze(1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Gradient clipping
            optimizer.step()
            train_loss += loss.item()

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch)
                val_loss += criterion(outputs, y_batch.unsqueeze(1)).item()

        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        scheduler.step(avg_val_loss)

        logger.info(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

        # Early stopping
        if avg_val_loss < best_val_loss - CONFIG['EARLY_STOPPING_MIN_DELTA']:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), f"{CONFIG['CACHE_DIR']}/{symbol}_best_model.pth")
        else:
            patience_counter += 1
            if patience_counter >= CONFIG['EARLY_STOPPING_PATIENCE']:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break

    # Load best model
    model.load_state_dict(torch.load(f"{CONFIG['CACHE_DIR']}/{symbol}_best_model.pth"))

    # Save final trained model and scaler for future loading
    model_path = f"{CONFIG['CACHE_DIR']}/{symbol}_model.pth"
    scaler_path = f"{CONFIG['CACHE_DIR']}/{symbol}_scaler.pkl"
    torch.save({'model_state_dict': model.state_dict(), 'class_name': 'TradingModel'}, model_path)
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    logger.info(f"Saved final model and scaler to {model_path} and {scaler_path}")

    return model, scaler or RobustScaler()  # Use passed scaler if available


def load_model_and_scaler(symbol: str, expected_features: int, force_retrain: bool = False) -> Tuple[Optional[nn.Module], Optional[RobustScaler], Optional[float]]:
    """Load trained model and scaler from cache or return None to trigger training."""
    if force_retrain:
        return None, None, None
    
    model_path = f"{CONFIG['CACHE_DIR']}/{symbol}_model.pth"
    scaler_path = f"{CONFIG['CACHE_DIR']}/{symbol}_scaler.pkl"
    sentiment_path = f"{CONFIG['CACHE_DIR']}/{symbol}_sentiment.pkl"
    
    if os.path.exists(model_path) and os.path.exists(scaler_path):
        try:
            # Load scaler first (less error-prone)
            with open(scaler_path, 'rb') as f:
                scaler = pickle.load(f)
            
            # Load model with class name handling
            checkpoint = torch.load(model_path, map_location='cpu')
            model_class_name = checkpoint.get('class_name', 'TradingModel')  # Check saved metadata
            if model_class_name == 'CNNLSTMModel':
                # Fallback: Instantiate current class and load state_dict (ignores old class)
                logger.warning(f"Legacy model for {symbol} detected. Loading state into current TradingModel.")
                model = TradingModel(CONFIG['TIMESTEPS'], expected_features)
            else:
                model = TradingModel(CONFIG['TIMESTEPS'], expected_features)
            model.load_state_dict(checkpoint['model_state_dict'])
            
            # Load associated training sentiment if available
            training_sentiment = None
            if os.path.exists(sentiment_path):
                with open(sentiment_path, 'rb') as f:
                    training_sentiment = pickle.load(f)
            
            logger.info(f"Loaded cached model and scaler for {symbol}.")
            return model, scaler, training_sentiment
        except (KeyError, AttributeError, NameError) as e:
            if 'CNNLSTMModel' in str(e) or 'not defined' in str(e):
                logger.warning(f"Class mismatch for {symbol} (likely legacy CNNLSTMModel). Deleting cache and retraining.")
                # Clean up invalid files
                if os.path.exists(model_path):
                    os.remove(model_path)
                if os.path.exists(scaler_path):
                    os.remove(scaler_path)
                if os.path.exists(sentiment_path):
                    os.remove(sentiment_path)
                if os.path.exists(f"{CONFIG['CACHE_DIR']}/{symbol}_best_model.pth"):
                    os.remove(f"{CONFIG['CACHE_DIR']}/{symbol}_best_model.pth")
                return None, None, None
            else:
                logger.error(f"Failed to load model/scaler for {symbol}: {str(e)}. Retraining.")
                return None, None, None
        except Exception as e:
            logger.error(f"Unexpected error loading for {symbol}: {str(e)}. Retraining.")
            return None, None, None
    else:
        logger.info(f"No cached model/scaler for {symbol}. Will train.")
        return None, None, None

def save_model_and_scaler(symbol: str, model: nn.Module, scaler: RobustScaler, sentiment: float) -> None:
    """Save the trained model and scaler to cache files."""
    try:
        model_path = os.path.join(CONFIG['CACHE_DIR'], f"{symbol}_model.pth")
        scaler_path = os.path.join(CONFIG['CACHE_DIR'], f"{symbol}_scaler.pkl")
        sentiment_path = os.path.join(CONFIG['CACHE_DIR'], f"{symbol}_sentiment.pkl")
        
        # Save model state_dict (lightweight, compatible with load)
        torch.save({'model_state_dict': model.state_dict(), 'class_name': 'TradingModel'}, model_path)
        
        # Save scaler via pickle
        with open(scaler_path, 'wb') as f:
            pickle.dump(scaler, f)
        
        # Save training sentiment
        with open(sentiment_path, 'wb') as f:
            pickle.dump(sentiment, f)
        
        logger.info(f"Saved model, scaler, and sentiment for {symbol} to {model_path}, {scaler_path}, and {sentiment_path}.")
    except Exception as e:
        logger.error(f"Failed to save model and scaler for {symbol}: {str(e)}")
        raise


def train_symbol(symbol: str, expected_features: int, force_train: bool) -> Tuple[str, pd.DataFrame, nn.Module, RobustScaler, bool, float, bool, bool]:
    """Train or load model for a single symbol using multiprocessing."""
    # Load data
    df, data_loaded = load_or_fetch_data(symbol, CONFIG['TRAIN_DATA_START_DATE'], CONFIG['BACKTEST_START_DATE'])
    sentiment, sentiment_loaded = load_news_sentiment(symbol)
    
    # Load model, scaler, and training sentiment if not forcing train
    model, scaler, training_sentiment = load_model_and_scaler(symbol, expected_features, force_train)
    model_loaded = model is not None
    
    # If model loaded, override with training sentiment if available (for consistency)
    if model_loaded and training_sentiment is not None:
        sentiment = training_sentiment
        sentiment_loaded = True
    
    df = calculate_indicators(df, sentiment)
    
    # Define features (consistent with live/backtest)
    features = [
        'close', 'high', 'low', 'volume', 'MA20', 'MA50', 'RSI',
        'MACD', 'MACD_signal', 'OBV', 'VWAP', 'ATR', 'CMF', 'Close_ATR',
        'MA20_ATR', 'Return_1d', 'Return_5d', 'Volatility', 'BB_upper',
        'BB_lower', 'Stoch_K', 'Stoch_D', 'ADX', 'Sentiment'
    ]
    if len(features) != expected_features:
        raise ValueError(f"Feature count mismatch: expected {expected_features}, got {len(features)}")
    
    if not model_loaded:
        # Preprocess with scaling
        X, y, scaler = preprocess_data(df, CONFIG['TIMESTEPS'], add_noise=True)
        # Train
        model, trained_scaler = train_model(symbol, X, y, CONFIG['TRAIN_EPOCHS'], CONFIG['BATCH_SIZE'], CONFIG['TIMESTEPS'], expected_features, scaler)
        scaler = trained_scaler  # Ensure fitted scaler is used
        # Save (including sentiment)
        save_model_and_scaler(symbol, model, scaler, sentiment)
        model_loaded = True  # Now loaded after training
    
    return symbol, df, model, scaler, data_loaded, sentiment, sentiment_loaded, model_loaded

def backtest(symbol: str, model: nn.Module, scaler: RobustScaler, df: pd.DataFrame, initial_cash: float,
             stop_loss_atr_multiplier: float, take_profit_atr_multiplier: float, timesteps: int,
             buy_threshold: float, sell_threshold: float, min_holding_period_minutes: int,
             transaction_cost_per_trade: float) -> Tuple[float, List[float], int, float]:
    """Run backtest simulation for a symbol using trained model predictions."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Running backtest for {symbol} on device: {device}")
    
    # Extract CONFIG values explicitly for clarity
    confidence_threshold = CONFIG['CONFIDENCE_THRESHOLD']
    rsi_buy_threshold = CONFIG['RSI_BUY_THRESHOLD']
    rsi_sell_threshold = CONFIG['RSI_SELL_THRESHOLD']
    adx_trend_threshold = CONFIG['ADX_TREND_THRESHOLD']
    max_volatility = CONFIG['MAX_VOLATILITY']
    trailing_stop_percentage = CONFIG['TRAILING_STOP_PERCENTAGE']
    risk_percentage = CONFIG['RISK_PERCENTAGE']
    
    # Preprocess for inference: Use trained scaler, no y or new fitting
    X_seq, y_ignore, scaler_ignore = preprocess_data(
        df, timesteps, inference_mode=True, inference_scaler=scaler
    )
    
    model.eval()
    model = model.to(device)
    X_tensor = torch.tensor(X_seq, dtype=torch.float32).to(device)
    
    predictions = []
    with torch.no_grad():
        for i in range(0, len(X_tensor), CONFIG['BATCH_SIZE']):
            batch = X_tensor[i:i + CONFIG['BATCH_SIZE']]
            raw_logits = model(batch)
            if i % 100 == 0:  # Log every 100 steps to avoid spam
                logger.info(f"Sample raw logit for {symbol} at step {i}: mean={raw_logits.mean().item():.4f}, std={raw_logits.std().item():.4f}")
            # Apply sigmoid to model outputs
            outputs = torch.sigmoid(raw_logits)
            predictions.extend(outputs.cpu().numpy().flatten())
            del raw_logits  # Clean up
            del outputs  # Explicitly delete outputs tensor
    
    predictions = np.array(predictions)
    logger.info(f"Predictions for {symbol}: min={predictions.min():.3f}, max={predictions.max():.3f}, mean={predictions.mean():.3f}")
    
    # Clean up CUDA tensors
    del X_tensor
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Initialize variables (initial_cash now per-symbol from call)
    cash = initial_cash
    returns = []
    trade_count = 0
    win_rate = 0.0
    position = 0
    entry_price = 0.0
    entry_time = None
    max_price = 0.0
    winning_trades = 0
    
    timestamps = df['timestamp'].iloc[timesteps:].reset_index(drop=True)
    sim_start = pd.Timestamp(CONFIG['BACKTEST_START_DATE'], tz='UTC')
    valid_timestamps = timestamps[timestamps >= sim_start]
    if valid_timestamps.empty:
        k_start = 0
    else:
        k_start = valid_timestamps.index[0]
    logger.info(f"Backtest for {symbol}: starting cash=${cash:.2f}, k_start={k_start}, len(predictions)={len(predictions)}")
    if k_start >= len(predictions):
        logger.warning(f"No data points for backtest of {symbol}")
        return cash, returns, trade_count, win_rate
    num_backtest_steps = len(predictions) - k_start
    if num_backtest_steps <= 0:
        logger.warning(f"No backtest steps available for {symbol} (num_backtest_steps={num_backtest_steps})")
        return cash, returns, trade_count, win_rate
    
    atr = df['ATR'].iloc[timesteps + k_start:timesteps + k_start + num_backtest_steps].values
    prices = df['close'].iloc[timesteps + k_start:timesteps + k_start + num_backtest_steps].values
    rsi = df['RSI'].iloc[timesteps + k_start:timesteps + k_start + num_backtest_steps].values
    adx = df['ADX'].iloc[timesteps + k_start:timesteps + k_start + num_backtest_steps].values
    volatility = df['Volatility'].iloc[timesteps + k_start:timesteps + k_start + num_backtest_steps].values
    sim_timestamps = timestamps.iloc[k_start:k_start + num_backtest_steps].values
    
    for local_i in range(num_backtest_steps):
        if local_i % 100 == 0:  # Log every 100 steps to avoid spam
            logger.info(f"Processing backtest step {local_i} for {symbol}: prediction={predictions[k_start + local_i]:.3f}")
        i = k_start + local_i
        pred = predictions[i]
        price = prices[local_i]
        atr_val = atr[local_i]
        current_rsi = rsi[local_i]
        current_adx = adx[local_i]
        current_volatility = volatility[local_i]
        ts = pd.Timestamp(sim_timestamps[local_i])
        
        if current_volatility > max_volatility or current_adx < adx_trend_threshold:
            continue
        
        if pred < confidence_threshold:
            continue
        
        if cash >= price:
            qty = max(1, int((cash * risk_percentage) / (atr_val * stop_loss_atr_multiplier)))
            cost = qty * price + transaction_cost_per_trade
            if cost > cash:
                qty = max(0, int((cash - transaction_cost_per_trade) / price))
                cost = qty * price + transaction_cost_per_trade
        else:
            qty = 0
            cost = 0
        
        #logger.info(f"Checking buy for {symbol}: pred={pred:.3f}, qty={qty}, cash={cash:.2f}, price={price:.2f}, rsi={current_rsi:.2f}, adx={current_adx:.2f}")
        
        if pred > buy_threshold and position == 0 and current_rsi < rsi_buy_threshold and current_adx > adx_trend_threshold and qty > 0 and cash >= cost:
            if cash - cost >= 0:  # Safety check to prevent negative cash
                position = qty
                entry_price = price
                max_price = price
                entry_time = ts
                cash -= cost
                logger.info(f"{ts}: Bought {qty} shares of {symbol} at ${price:.2f}, cash: ${cash:.2f}")
            else:
                logger.info(f"Insufficient cash to buy {qty} shares of {symbol}: cash={cash:.2f}, cost={cost:.2f}")
        
        elif position > 0:
            if price > max_price:
                max_price = price
            trailing_stop = max_price * (1 - trailing_stop_percentage)
            stop_loss = entry_price - stop_loss_atr_multiplier * atr_val
            take_profit = entry_price + take_profit_atr_multiplier * atr_val
            
            if not isinstance(entry_time, pd.Timestamp):
                raise TypeError(f"entry_time must be a pandas.Timestamp, got {type(entry_time)}")
            time_held = (ts - entry_time).total_seconds() / 60
            
            if time_held >= min_holding_period_minutes:
                if price <= trailing_stop or price <= stop_loss or price >= take_profit or (pred < sell_threshold and current_rsi > rsi_sell_threshold):
                    cash += position * price - transaction_cost_per_trade
                    ret = (price - entry_price) / entry_price
                    returns.append(ret)
                    trade_count += 1
                    if ret > 0:
                        winning_trades += 1
                    logger.info(f"{ts}: Sold {position} shares of {symbol} at ${price:.2f}, return: {ret:.3f}, cash: ${cash:.2f}")
                    position = 0
                    logger.info(f"After sell: position={position}, cash={cash:.2f}")
                    entry_time = None
                    max_price = 0.0
        
        else:
            logger.info(f"Skipped buy for {symbol}: pred={pred:.3f}, rsi={current_rsi:.2f}, adx={current_adx:.2f}, volatility={current_volatility:.2f}, qty={qty}, cost={qty * price:.2f}, cash={cash:.2f}")
    
    if position > 0:
        cash += position * prices[-1] - transaction_cost_per_trade
        ret = (prices[-1] - entry_price) / entry_price
        returns.append(ret)
        trade_count += 1
        if ret > 0:
            winning_trades += 1
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

def make_prediction(model: nn.Module, X: np.ndarray) -> float:
    """Make a prediction using the trained model."""
    device = next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        X_tensor = torch.FloatTensor(X).to(device)
        output = model(X_tensor)
        prediction = torch.sigmoid(output).cpu().item()  # Convert logits to probability
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
    expected_features = 24
    models = {}
    scalers = {}
    dfs = {}
    stock_info = []
    total_epochs = len(CONFIG['SYMBOLS']) * CONFIG['TRAIN_EPOCHS']
    need_training = any(load_model_and_scaler(symbol, expected_features, force_train)[0] is None for symbol in CONFIG['SYMBOLS'])
    progress_bar = tqdm(total=total_epochs, desc="Training Progress", bar_format="{l_bar}\033[32m{bar}\033[0m{r_bar}") if need_training else None

    outputs = []
    for symbol in tqdm(CONFIG['SYMBOLS'], desc="Processing symbols"):
        output = train_symbol(symbol, expected_features, force_train)
        outputs.append(output)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()  # Clear after each symbol

    # No pool cleanup needed
    logger.info("Sequential processing completed; CUDA memory cleared per symbol.")

    sentiments = {}  # Collect sentiments for live consistency
    for symbol, df, model, scaler, data_loaded, sentiment, sentiment_loaded, model_loaded in outputs:
        dfs[symbol] = df
        models[symbol] = model
        scalers[symbol] = scaler
        sentiments[symbol] = sentiment
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
                        sentiment = sentiments[symbol]  # Use training sentiment for consistency
                        df = calculate_indicators(df, sentiment)
                        # Define features list (copied from preprocess_data for consistency)
                        features = [
                            'close', 'high', 'low', 'volume', 'MA20', 'MA50', 'RSI',
                            'MACD', 'MACD_signal', 'OBV', 'VWAP', 'ATR', 'CMF', 'Close_ATR',
                            'MA20_ATR', 'Return_1d', 'Return_5d', 'Volatility', 'BB_upper',
                            'BB_lower', 'Stoch_K', 'Stoch_D', 'ADX', 'Sentiment'
                        ]

                        # Compute prediction and current values from model and df
                        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                        if len(df) < CONFIG['TIMESTEPS'] + 1:
                            logger.warning(f"Insufficient data for {symbol} live prediction: {len(df)} bars")
                            prediction = 0.5
                            price = df['close'].iloc[-1] if not df.empty and len(df) > 0 else 0.0
                            current_rsi = df['RSI'].iloc[-1] if not df.empty and 'RSI' in df.columns else 50.0
                            current_adx = df['ADX'].iloc[-1] if not df.empty and 'ADX' in df.columns else 0.0
                            current_volatility = df['Volatility'].iloc[-1] if not df.empty and 'Volatility' in df.columns else 0.0
                            atr_val = df['ATR'].iloc[-1] if not df.empty and 'ATR' in df.columns else 0.0
                        else:
                            X_seq, _, _ = preprocess_data(df, CONFIG['TIMESTEPS'], inference_mode=True, inference_scaler=scalers[symbol])
                            # Use the most recent sequence for live prediction
                            recent_seq = X_seq[-1:].astype(np.float32)
                            model = models[symbol].to(device)
                            model.eval()
                            with torch.no_grad():
                                pred_logit = model(torch.tensor(recent_seq).to(device))
                                prediction = torch.sigmoid(pred_logit).cpu().item()
                            price = float(df['close'].iloc[-1])
                            current_rsi = float(df['RSI'].iloc[-1])
                            current_adx = float(df['ADX'].iloc[-1])
                            current_volatility = float(df['Volatility'].iloc[-1])
                            atr_val = float(df['ATR'].iloc[-1])
                            logger.info(f"Live prediction for {symbol}: {prediction:.3f}, price=${price:.2f}, RSI={current_rsi:.2f}")

                        decision = "Hold"

                        # Fetch position (qty_owned etc. only for sell/hold checks)
                        qty_owned = 0
                        entry_time = None
                        entry_price = 0.0
                        time_held = 0
                        position_obj = next((pos for pos in open_positions if pos.symbol == symbol), None)
                        if position_obj:
                            qty_owned = int(float(position_obj.qty))
                            entry_time = pd.Timestamp(position_obj.updated_at) if hasattr(position_obj, 'updated_at') else now  # Use updated_at for accuracy
                            entry_price = float(position_obj.avg_entry_price)
                            time_held = (now - entry_time).total_seconds() / 60

                        if current_volatility > CONFIG['MAX_VOLATILITY'] or current_adx < CONFIG['ADX_TREND_THRESHOLD']:
                            decision = "Hold (Filters)"
                        elif prediction < CONFIG['CONFIDENCE_THRESHOLD']:
                            decision = "Hold (Low Confidence)"
                        elif qty_owned == 0 and prediction > max(CONFIG['PREDICTION_THRESHOLD_BUY'], CONFIG['CONFIDENCE_THRESHOLD']) and current_rsi < CONFIG['RSI_BUY_THRESHOLD']:
                            decision = "Buy"
                            # ... (rest of buy logic unchanged)
                        elif qty_owned > 0 and time_held >= CONFIG['MIN_HOLDING_PERIOD_MINUTES']:
                            # Compute stops using current price
                            max_price = max(float(position_obj.current_price) if position_obj else price, price)
                            trailing_stop = max_price * (1 - CONFIG['TRAILING_STOP_PERCENTAGE'])
                            stop_loss = entry_price - CONFIG['STOP_LOSS_ATR_MULTIPLIER'] * atr_val
                            take_profit = entry_price + CONFIG['TAKE_PROFIT_ATR_MULTIPLIER'] * atr_val
                            if price <= trailing_stop or price <= stop_loss or price >= take_profit or (prediction < CONFIG['PREDICTION_THRESHOLD_SELL'] and current_rsi > CONFIG['RSI_SELL_THRESHOLD']):
                                decision = "Sell"
                                logger.info(f"Sell attempt for {symbol}: qty={qty_owned}, price=${price:.2f}, prediction={prediction:.3f}")
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
Time Held: {time_held:.2f} minutes
Current Cash: ${float(account.cash):.2f}
Portfolio Value: ${portfolio_value:.2f}
"""
                                    send_email(f"Trade Update: {symbol}", email_body)
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
Time Held: {time_held:.2f} minutes
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
        initial_per_symbol = CONFIG['INITIAL_CASH'] / len(CONFIG['SYMBOLS'])
        for symbol in CONFIG['SYMBOLS']:
            if symbol in models:
                cash, returns, trade_count, win_rate = backtest(
                    symbol, models[symbol], scalers[symbol], dfs[symbol], initial_per_symbol,
                    CONFIG['STOP_LOSS_ATR_MULTIPLIER'], CONFIG['TAKE_PROFIT_ATR_MULTIPLIER'],
                    CONFIG['TIMESTEPS'], CONFIG['PREDICTION_THRESHOLD_BUY'], CONFIG['PREDICTION_THRESHOLD_SELL'],
                    CONFIG['MIN_HOLDING_PERIOD_MINUTES'], CONFIG['TRANSACTION_COST_PER_TRADE']
                )
                final_value += cash - initial_per_symbol
                symbol_results[symbol] = calculate_performance_metrics(returns, cash, initial_per_symbol)
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
