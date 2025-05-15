# =============================================================================
# Alpaca Advanced Neural Bot with Email Notifications
# =============================================================================
#
# Description:
# ------------
# This script implements an advanced trading bot that uses deep learning (LSTM
# neural networks) to predict stock price movements and execute backtest
# simulations. It fetches historical stock data from the Alpaca API, calculates
# over 25 technical indicators, incorporates news sentiment using FinBERT
# (placeholder), trains LSTM models per stock, and performs backtesting with
# configurable trading logic. Results are logged to trades.log and emailed,
# including per-symbol performance metrics (total return, annualized return,
# Sharpe ratio, max drawdown, trade counts, win rates).
#
# Key Features:
# -------------
# - Configurable via CONFIG dictionary for symbols, timeframe, cash, thresholds.
# - Supports multiple stocks (e.g., TSLA, NVDA) with independent models.
# - Fetches 15-minute bar data (configurable) from Alpaca API with tenacity retries.
# - Calculates indicators (MA20, MA50, RSI, MACD, Bollinger Bands, etc.).
# - Trains LSTM models with error handling and StandardScaler normalization.
# - Backtests with buy (>0.6) and sell (<0.4 or ATR-based stop-loss/take-profit).
# - Supports --backtest mode using pre-trained models, with training fallback.
# - Logs detailed info (data fetching, training, trades) to trades.log.
# - Emails formatted results with per-symbol metrics and trade statistics.
# - Includes performance metrics (Sharpe ratio, max drawdown).
# - Handles errors, skipping failed symbols and logging issues.
# - Caches sentiment data with 24-hour expiry.
#
# Usage:
# ------
# 1. Set up a virtual environment (recommended to avoid dependency conflicts):
#    ```bash
#    python -m venv venv
#    source venv/bin/activate  # On Windows: venv\Scripts\activate
#    ```
# 2. Install dependencies in the active environment:
#    ```bash
#    python -m pip install tensorflow numpy pandas alpaca-py transformers scikit-learn ta-lib tenacity
#    ```
#    For ta-lib on Ubuntu:
#    ```bash
#    sudo apt-get install libta-lib0 libta-lib0-dev
#    python -m pip install ta-lib
#    ```
# 3. Configure API and email in CONFIG dictionary:
#    - Set ALPACA_API_KEY, ALPACA_SECRET_KEY (Alpaca paper trading).
#    - Set EMAIL_SENDER, EMAIL_PASSWORD, EMAIL_RECEIVER (e.g., Gmail App Password).
# 4. Modify CONFIG as needed (e.g., SYMBOLS=['AAPL', 'MSFT'], TIMEFRAME_INTERVAL='1H').
# 5. Run the script in the same environment:
#    - Full run (training + backtest):
#      ```bash
#      python alpaca_advanced_neural_bot_with_email_optimized.py
#      ```
#    - Backtest-only mode (requires pre-trained models in cache):
#      ```bash
#      python alpaca_advanced_neural_bot_with_email_optimized.py --backtest
#      ```
#
# Dependencies:
# -------------
# - tensorflow: Deep learning for LSTM models.
# - numpy, pandas: Data manipulation and analysis.
# - alpaca-py: Alpaca API for market data.
# - transformers: FinBERT for sentiment (placeholder).
# - scikit-learn: StandardScaler for normalization.
# - ta-lib: Technical indicators (MA, RSI, MACD, etc.).
# - tenacity: Retry logic for API calls.
# - smtplib: Email notifications.
# - argparse, time, warnings: Argument parsing, timing, warning suppression.
#
# Installation:
# ------------
# Ensure dependencies are installed in the same Python environment as the script:
# ```bash
# python -m pip install tensorflow numpy pandas alpaca-py transformers scikit-learn ta-lib tenacity
# ```
# If you encounter a ModuleNotFoundError, verify the Python environment:
# - Check Python version: `python --version` (must be 3.10+).
# - Check pip environment: `pip --version` (should match Python path).
# - Use `python -m pip install <package>` to install in the correct environment.
#
# Configuration:
# --------------
# Edit CONFIG dictionary:
# - SYMBOLS: List of stock symbols (e.g., ['TSLA', 'NVDA']).
# - TIMEFRAME: Alpaca TimeFrame (e.g., TimeFrame.Minute).
# - TIMEFRAME_INTERVAL: Interval for Minute (e.g., '15Min').
# - INITIAL_CASH: Starting cash (e.g., 1000.0).
# - TRAIN_EPOCHS, BATCH_SIZE: Training parameters (e.g., 15, 32).
# - STOP_LOSS_MULTIPLIER, TAKE_PROFIT_MULTIPLIER: ATR exits (e.g., 1.5, 2.0).
# - PREDICTION_THRESHOLD_BUY, PREDICTION_THRESHOLD_SELL: Trade thresholds (e.g., 0.6, 0.4).
# - CACHE_DIR: Model/sentiment cache directory (e.g., './cache').
# - CACHE_EXPIRY_SECONDS: Cache expiry (e.g., 24 hours).
#
# Notes:
# ------
# - Ensure valid Alpaca API credentials and internet connectivity.
# - Backtest-only mode requires pre-trained models (run without --backtest first).
# - FinBERT sentiment is a placeholder (0.2); replace with news processing for production.
# - Performance depends on hardware; GPU recommended for TensorFlow.
# - Logs saved to trades.log; check for errors or trade details.
# - Email requires correct SMTP settings; use Gmail App Password.
# - If ModuleNotFoundError persists, verify Python environment consistency.
#
# Author: Grok 3, built by xAI
# Date: May 14, 2025
# Version: 1.2
# =============================================================================

import os
import sys
import logging
import argparse
import importlib
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras import backend as K
from alpaca.data import StockHistoricalDataClient, TimeFrame
from alpaca.data.requests import StockBarsRequest
from transformers import pipeline
from sklearn.preprocessing import StandardScaler
import smtplib
from email.mime.text import MIMEText
from datetime import datetime, timedelta
import talib
import pickle
from typing import List, Tuple, Dict, Optional
import warnings
import time
from tenacity import retry, stop_after_attempt, wait_fixed, retry_if_exception_type

# Suppress TensorFlow warnings
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings('ignore', category=UserWarning)

# Configuration variables (modify these as needed)
CONFIG = {
    'SYMBOLS': ['TSLA', 'NVDA'],  # List of stock symbols to process
    'TIMEFRAME': TimeFrame.Minute,  # Alpaca TimeFrame (Minute, Hour, Day)
    'TIMEFRAME_INTERVAL': '15Min',  # Interval for Minute timeframe (e.g., '1Min', '5Min', '15Min')
    'NUM_BARS': 10000,  # Number of bars to fetch
    'TRAIN_EPOCHS': 25,  # Number of training epochs
    'BATCH_SIZE': 32,  # Training batch size
    'INITIAL_CASH': 100000.0,  # Initial cash for backtesting
    'STOP_LOSS_MULTIPLIER': 1.5,  # Stop-loss multiplier for ATR-based exits
    'TAKE_PROFIT_MULTIPLIER': 2.0,  # Take-profit multiplier for ATR-based exits
    'TIMESTEPS': 30,  # Number of timesteps for input sequences
    'MIN_DATA_POINTS': 100,  # Minimum number of bars required
    'CACHE_DIR': './cache',  # Directory for cached data/models
    'CACHE_EXPIRY_SECONDS': 24 * 60 * 60,  # Cache expiry (24 hours)
    'ALPACA_API_KEY': 'PKN7X4X2GEUK7LJTMVU8',  # Alpaca API key
    'ALPACA_SECRET_KEY': 'd8mEV3OqhfEYG1cT4CC6JJbjJ3sbimeglSKchBa0',  # Alpaca secret key
    'EMAIL_SENDER': 'alpaca.ai.tradingbot@gmail.com',  # Email sender address
    'EMAIL_PASSWORD': 'hjdf sstp pyne rotq',  # Email app password
    'EMAIL_RECEIVER': 'aiplane.scientist@gmail.com',  # Email receiver address
    'SMTP_SERVER': 'smtp.gmail.com',  # SMTP server
    'SMTP_PORT': 587,  # SMTP port
    'LOG_FILE': 'trades.log',  # Log file name
    'API_RETRY_ATTEMPTS': 3,  # Number of retries for API calls
    'API_RETRY_DELAY': 1000,  # Delay between retries in milliseconds
    'PREDICTION_THRESHOLD_BUY': 0.6,  # Prediction threshold for buy
    'PREDICTION_THRESHOLD_SELL': 0.4  # Prediction threshold for sell
}

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(CONFIG['LOG_FILE']),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def check_dependencies() -> None:
    """
    Check for required Python modules and raise an error if any are missing.
    
    Logs the Python executable and site-packages paths for debugging environment issues.
    
    Raises:
        ImportError: If a required module is not installed, with installation instructions.
    """
    try:
        # Log Python environment for debugging
        logger.info(f"Python executable: {sys.executable}")
        logger.info(f"Python version: {sys.version}")
        logger.info(f"Site-packages: {sys.path}")
        
        # Check Python version (3.10+ required)
        if sys.version_info < (3, 10):
            raise RuntimeError("Python 3.10 or higher is required")
        
        # List of required modules
        required_modules = [
            'tensorflow', 'numpy', 'pandas', 'alpaca_trade_api', 'transformers',
            'sklearn', 'talib', 'tenacity', 'smtplib', 'argparse'
        ]
        
        # Check each module
        for module in required_modules:
            try:
                importlib.import_module(module)
                logger.debug(f"Module {module} is installed")
            except ImportError:
                raise ImportError(
                    f"The '{module}' module is required. "
                    f"Please install it using: python -m pip install {module} "
                    f"in the same Python environment ({sys.executable})."
                )
        logger.info("All required dependencies are installed")
    except Exception as e:
        logger.error(f"Dependency check failed: {str(e)}")
        raise

def validate_config(config: Dict) -> None:
    """
    Validate configuration parameters to ensure correctness.
    
    Args:
        config (Dict): Configuration dictionary containing bot settings.
    
    Raises:
        ValueError: If any configuration parameter is invalid.
    """
    try:
        # Validate symbol list
        if not config['SYMBOLS']:
            raise ValueError("SYMBOLS list cannot be empty")
        
        # Validate timeframe
        if not isinstance(config['TIMEFRAME'], TimeFrame):
            raise ValueError("TIMEFRAME must be a valid TimeFrame enum (e.g., TimeFrame.Minute)")
        
        # Validate timeframe interval for Minute timeframe
        valid_intervals = ['1Min', '5Min', '15Min']
        if config['TIMEFRAME'] == TimeFrame.Minute and config['TIMEFRAME_INTERVAL'] not in valid_intervals:
            raise ValueError(
                f"TIMEFRAME_INTERVAL must be one of {valid_intervals} for TimeFrame.Minute"
            )
        
        # Validate integer parameters
        for param, value in [
            ('NUM_BARS', config['NUM_BARS']),
            ('TRAIN_EPOCHS', config['TRAIN_EPOCHS']),
            ('BATCH_SIZE', config['BATCH_SIZE']),
            ('TIMESTEPS', config['TIMESTEPS']),
            ('MIN_DATA_POINTS', config['MIN_DATA_POINTS']),
            ('API_RETRY_ATTEMPTS', config['API_RETRY_ATTEMPTS']),
            ('API_RETRY_DELAY', config['API_RETRY_DELAY']),
            ('CACHE_EXPIRY_SECONDS', config['CACHE_EXPIRY_SECONDS'])
        ]:
            if not isinstance(value, int) or value <= 0:
                raise ValueError(f"{param} must be a positive integer")
        
        # Validate float parameters
        for param, value in [
            ('INITIAL_CASH', config['INITIAL_CASH']),
            ('STOP_LOSS_MULTIPLIER', config['STOP_LOSS_MULTIPLIER']),
            ('TAKE_PROFIT_MULTIPLIER', config['TAKE_PROFIT_MULTIPLIER']),
            ('PREDICTION_THRESHOLD_BUY', config['PREDICTION_THRESHOLD_BUY']),
            ('PREDICTION_THRESHOLD_SELL', config['PREDICTION_THRESHOLD_SELL'])
        ]:
            if not isinstance(value, (int, float)) or value <= 0:
                raise ValueError(f"{param} must be a positive number")
        
        # Ensure sell threshold is less than buy threshold
        if config['PREDICTION_THRESHOLD_SELL'] >= config['PREDICTION_THRESHOLD_BUY']:
            raise ValueError("PREDICTION_THRESHOLD_SELL must be less than PREDICTION_THRESHOLD_BUY")
        
        # Validate symbol strings
        if not all(isinstance(s, str) for s in config['SYMBOLS']):
            raise ValueError("All SYMBOLS must be strings")
        
        # Validate timestep constraints
        if config['TIMESTEPS'] >= config['NUM_BARS']:
            raise ValueError("TIMESTEPS must be less than NUM_BARS")
        
        # Validate minimum data points
        if config['MIN_DATA_POINTS'] > config['NUM_BARS']:
            raise ValueError("MIN_DATA_POINTS must be less than or equal to NUM_BARS")
        
        logger.info("Configuration validated successfully")
    except Exception as e:
        logger.error(f"Configuration validation failed: {str(e)}")
        raise

def create_cache_directory() -> None:
    """
    Create cache directory if it doesn't exist.
    
    Raises:
        OSError: If directory creation fails.
    """
    try:
        os.makedirs(CONFIG['CACHE_DIR'], exist_ok=True)
        logger.info(f"Cache directory ensured: {CONFIG['CACHE_DIR']}")
    except OSError as e:
        logger.error(f"Failed to create cache directory: {str(e)}. Ensure write permissions.")
        raise

@retry(
    stop=stop_after_attempt(CONFIG['API_RETRY_ATTEMPTS']),
    wait=wait_fixed(CONFIG['API_RETRY_DELAY'] / 1000),
    retry=retry_if_exception_type(Exception)
)
def fetch_data(symbol: str, timeframe: TimeFrame, interval: str, num_bars: int) -> pd.DataFrame:
    """
    Fetch historical bar data from Alpaca API with retry logic.
    
    Args:
        symbol (str): Stock symbol (e.g., 'TSLA').
        timeframe (TimeFrame): Alpaca timeframe enum (e.g., TimeFrame.Minute).
        interval (str): Timeframe interval (e.g., '15Min' for Minute timeframe).
        num_bars (int): Number of bars to fetch.
    
    Returns:
        pd.DataFrame: DataFrame with historical bar data (OHLCV, timestamp).
    
    Raises:
        ValueError: If insufficient data is fetched.
        Exception: If API call fails after retries.
    """
    try:
        client = StockHistoricalDataClient(CONFIG['ALPACA_API_KEY'], CONFIG['ALPACA_SECRET_KEY'])
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365)
        request = StockBarsRequest(
            symbol_or_symbols=symbol,
            timeframe=timeframe,
            start=start_date,
            end=end_date,
            limit=num_bars
        )
        bars = client.get_stock_bars(request).df
        if len(bars) < CONFIG['MIN_DATA_POINTS']:
            raise ValueError(
                f"Insufficient data for {symbol}: got {len(bars)} bars, need {CONFIG['MIN_DATA_POINTS']}"
            )
        logger.info(f"Fetched {len(bars)} bars for {symbol} with timeframe {timeframe} ({interval})")
        return bars.reset_index()
    except Exception as e:
        logger.error(f"Error fetching data for {symbol}: {str(e)}")
        raise

def load_news_sentiment(symbol: str) -> float:
    """
    Load or compute news sentiment using FinBERT (placeholder).
    
    Args:
        symbol (str): Stock symbol.
    
    Returns:
        float: Sentiment score (0 to 1).
    
    Raises:
        Exception: If sentiment computation or caching fails.
    """
    cache_file = os.path.join(CONFIG['CACHE_DIR'], f"{symbol}_news_sentiment.pkl")
    try:
        # Check if cache exists and is not expired
        if os.path.exists(cache_file):
            file_mtime = os.path.getmtime(cache_file)
            if (time.time() - file_mtime) < CONFIG['CACHE_EXPIRY_SECONDS']:
                with open(cache_file, 'rb') as f:
                    sentiment = pickle.load(f)
                logger.info(f"Loaded news sentiment for {symbol} from cache: {sentiment:.3f}")
                return sentiment
        
        # Compute sentiment (placeholder)
        try:
            sentiment_analyzer = pipeline("sentiment-analysis", model="ProsusAI/finbert")
            # Assume news data is fetched; return dummy sentiment
            sentiment = 0.2
        except Exception as e:
            logger.warning(f"FinBERT failed for {symbol}: {str(e)}. Using default sentiment: 0.2")
            sentiment = 0.2
        
        # Cache sentiment
        with open(cache_file, 'wb') as f:
            pickle.dump(sentiment, f)
        logger.info(f"Computed and cached news sentiment for {symbol}: {sentiment:.3f}")
        return sentiment
    except Exception as e:
        logger.error(f"Error loading news sentiment for {symbol}: {str(e)}")
        return 0.0

def calculate_indicators(df: pd.DataFrame, sentiment: float) -> pd.DataFrame:
    """
    Calculate technical indicators and add sentiment score to the DataFrame.
    
    Args:
        df (pd.DataFrame): Input DataFrame with OHLCV data.
        sentiment (float): Sentiment score to include as a feature.
    
    Returns:
        pd.DataFrame: DataFrame with calculated indicators (MA20, RSI, etc.).
    
    Raises:
        ValueError: If required columns are missing.
        Exception: If indicator calculation fails.
    """
    try:
        df = df.copy()
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        if not all(col in df.columns for col in required_cols):
            raise ValueError(f"Missing required columns: {required_cols}")
        
        # Calculate basic indicators
        df['MA20'] = talib.SMA(df['close'], timeperiod=20)
        df['MA50'] = talib.SMA(df['close'], timeperiod=50)
        df['RSI'] = talib.RSI(df['close'], timeperiod=14)
        df['MACD'], df['MACD_signal'], _ = talib.MACD(df['close'], fastperiod=12, slowperiod=26, signalperiod=9)
        df['OBV'] = talib.OBV(df['close'], df['volume'])
        df['VWAP'] = (df['volume'] * (df['high'] + df['low'] + df['close']) / 3).cumsum() / df['volume'].cumsum()
        df['ATR'] = talib.ATR(df['high'], df['low'], df['close'], timeperiod=14)
        df['CMF'] = talib.AD(df['high'], df['low'], df['close'], df['volume']) / df['volume'].rolling(20).sum()
        
        # Calculate additional indicators
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
        
        # Log NaN counts for debugging
        nan_counts = df.isna().sum().to_dict()
        logger.info(f"Indicators calculated for shape: {df.shape}, NaN counts: {nan_counts}")
        return df
    except Exception as e:
        logger.error(f"Error calculating indicators: {str(e)}")
        raise

def validate_data(df: pd.DataFrame, symbol: str) -> None:
    """
    Validate DataFrame for NaNs, infinities, and sufficient data.
    
    Args:
        df (pd.DataFrame): Input DataFrame to validate.
        symbol (str): Stock symbol for error reporting.
    
    Raises:
        ValueError: If data validation fails (e.g., NaNs, insufficient rows).
    """
    try:
        if df.empty:
            raise ValueError(f"Empty DataFrame for {symbol}")
        required_cols = ['open', 'high', 'low', 'close', 'volume', 'timestamp']
        if not all(col in df.columns for col in required_cols):
            raise ValueError(f"Missing required columns for {symbol}: {required_cols}")
        if df[required_cols[:-1]].isna().any().any():
            raise ValueError(f"NaN values in OHLCV columns for {symbol}")
        if np.any(np.isinf(df[required_cols[:-1]].values)):
            raise ValueError(f"Infinite values in OHLCV columns for {symbol}")
        if len(df) < CONFIG['MIN_DATA_POINTS']:
            raise ValueError(
                f"Insufficient data for {symbol}: got {len(df)} bars, need {CONFIG['MIN_DATA_POINTS']}"
            )
        logger.info(f"Data validated for {symbol}: {len(df)} bars")
    except Exception as e:
        logger.error(f"Data validation failed for {symbol}: {str(e)}")
        raise

def preprocess_data(df: pd.DataFrame, timesteps: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Preprocess data for model input by creating sequences.
    
    Args:
        df (pd.DataFrame): Input DataFrame with indicators.
        timesteps (int): Number of timesteps for sequences.
    
    Returns:
        Tuple[np.ndarray, np.ndarray]: Features (X) and labels (y) as numpy arrays.
    
    Raises:
        ValueError: If preprocessing fails (e.g., missing features, invalid sequences).
    """
    try:
        df = df.copy()
        # Fill NaNs to ensure data integrity
        df.fillna(method='ffill', inplace=True)
        df.fillna(0, inplace=True)
        logger.info(f"After filling NaNs, data shape: {df.shape}")
        
        # Define features for model input
        features = [
            'close', 'high', 'low', 'volume', 'vwap', 'MA20', 'MA50', 'RSI',
            'MACD', 'MACD_signal', 'OBV', 'VWAP', 'ATR', 'CMF', 'Close_ATR',
            'MA20_ATR', 'Return_1d', 'Return_5d', 'Volatility', 'BB_upper',
            'BB_lower', 'Stoch_K', 'Stoch_D', 'ADX', 'Sentiment'
        ]
        if not all(f in df.columns for f in features):
            missing = [f for f in features if f not in df.columns]
            raise ValueError(f"Missing features for preprocessing: {missing}")
        
        # Extract feature values and labels
        X = df[features].values
        y = df['Trend'].values
        
        # Create sequences for LSTM input
        X_seq = []
        y_seq = []
        for i in range(timesteps, len(X)):
            X_seq.append(X[i-timesteps:i])
            y_seq.append(y[i])
        X_seq = np.array(X_seq)
        y_seq = np.array(y_seq)
        
        # Validate sequences
        if X_seq.shape[0] == 0:
            raise ValueError("No valid sequences generated after preprocessing")
        if np.any(np.isnan(X_seq)) or np.any(np.isinf(X_seq)):
            raise ValueError("NaN or Inf values detected in preprocessed data")
        
        logger.info(f"Preprocessed features: X shape: {X_seq.shape}, y shape: {y_seq.shape}")
        return X_seq, y_seq
    except Exception as e:
        logger.error(f"Error preprocessing data: {str(e)}")
        raise

def build_model(input_shape: Tuple[int, int]) -> tf.keras.Model:
    """
    Build and compile a neural network model for trading predictions.
    
    Args:
        input_shape (Tuple[int, int]): Shape of input data (timesteps, features).
    
    Returns:
        tf.keras.Model: Compiled Keras LSTM model.
    
    Raises:
        ValueError: If model creation fails (e.g., None model object).
        Exception: If compilation fails.
    """
    try:
        # Clear any existing TensorFlow session to prevent state issues
        K.clear_session()
        
        # Use explicit name scope to avoid Keras context errors
        with tf.name_scope("trading_model"):
            model = Sequential([
                LSTM(64, input_shape=input_shape, return_sequences=True),
                Dropout(0.2),
                LSTM(32),
                Dropout(0.2),
                Dense(16, activation='relu'),
                Dense(1, activation='sigmoid')
            ])
            
            # Verify model creation
            if model is None:
                raise ValueError("Model creation failed: Model object is None")
            
            # Compile model with adam optimizer and binary crossentropy loss
            model.compile(
                optimizer='adam',
                loss='binary_crossentropy',
                metrics=['accuracy'],
                jit_compile=False  # Disable XLA to avoid backend issues
            )
            
            logger.info(f"Model built successfully with input shape {input_shape}")
            return model
    except Exception as e:
        logger.error(f"Error building model: {str(e)}")
        raise
    finally:
        # Ensure session is cleared to prevent memory leaks
        tf.keras.backend.clear_session()

def train_model(symbol: str, X: np.ndarray, y: np.ndarray, epochs: int, batch_size: int) -> Tuple[tf.keras.Model, StandardScaler]:
    """
    Train and save the model for a given symbol.
    
    Args:
        symbol (str): Stock symbol.
        X (np.ndarray): Input features (sequences).
        y (np.ndarray): Target labels (trend).
        epochs (int): Number of training epochs.
        batch_size (int): Training batch size.
    
    Returns:
        Tuple[tf.keras.Model, StandardScaler]: Trained model and fitted scaler.
    
    Raises:
        Exception: If training or saving fails.
    """
    try:
        # Scale input features
        scaler = StandardScaler()
        X_flat = X.reshape(-1, X.shape[-1])
        X_scaled = scaler.fit_transform(X_flat).reshape(X.shape)
        
        # Build the LSTM model
        model = build_model(input_shape=(X.shape[1], X.shape[2]))
        
        # Train the model
        start_time = time.time()
        model.fit(
            X_scaled, y,
            epochs=epochs,
            batch_size=batch_size,
            verbose=0,
            callbacks=[tf.keras.callbacks.LambdaCallback(
                on_epoch_end=lambda epoch, logs: logger.info(
                    f"Epoch {epoch+1}/{epochs} for {symbol}, loss: {logs['loss']:.4f}, "
                    f"accuracy: {logs['accuracy']:.4f}, time: {(time.time() - start_time):.2f}s"
                )
            )]
        )
        
        # Save model and scaler to cache
        model_path = os.path.join(CONFIG['CACHE_DIR'], f"{symbol}_model.keras")
        scaler_path = os.path.join(CONFIG['CACHE_DIR'], f"{symbol}_scaler.pkl")
        model.save(model_path)
        with open(scaler_path, 'wb') as f:
            pickle.dump(scaler, f)
        
        logger.info(f"Saved model and scaler for {symbol}")
        return model, scaler
    except Exception as e:
        logger.error(f"Failed to train model for {symbol}: {str(e)}")
        raise

def load_model_and_scaler(symbol: str) -> Tuple[tf.keras.Model, StandardScaler]:
    """
    Load pre-trained model and scaler for a symbol.
    
    Args:
        symbol (str): Stock symbol.
    
    Returns:
        Tuple[tf.keras.Model, StandardScaler]: Loaded model and scaler.
    
    Raises:
        FileNotFoundError: If model or scaler file is missing.
        Exception: If loading fails.
    """
    try:
        model_path = os.path.join(CONFIG['CACHE_DIR'], f"{symbol}_model.keras")
        scaler_path = os.path.join(CONFIG['CACHE_DIR'], f"{symbol}_scaler.pkl")
        if not os.path.exists(model_path) or not os.path.exists(scaler_path):
            raise FileNotFoundError(
                f"Model or scaler not found for {symbol}. "
                f"Ensure {model_path} and {scaler_path} exist."
            )
        model = tf.keras.models.load_model(model_path)
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        logger.info(f"Loaded model and scaler for {symbol}")
        return model, scaler
    except Exception as e:
        logger.error(f"Error loading model/scaler for {symbol}: {str(e)}")
        raise

def calculate_performance_metrics(returns: List[float], cash: float, initial_cash: float) -> Dict[str, float]:
    """
    Calculate performance metrics for backtest results.
    
    Args:
        returns (List[float]): List of trade returns.
        cash (float): Final cash balance.
        initial_cash (float): Initial cash balance.
    
    Returns:
        Dict[str, float]: Performance metrics (total return, Sharpe ratio, etc.).
    
    Raises:
        Exception: If metric calculation fails.
    """
    try:
        returns = np.array(returns)
        metrics = {}
        
        # Calculate total return
        metrics['total_return'] = (cash - initial_cash) / initial_cash * 100
        
        # Calculate annualized return (assuming 252 trading days)
        if len(returns) > 0:
            metrics['annualized_return'] = (
                (1 + metrics['total_return'] / 100) ** (252 / len(returns)) - 1
            ) * 100
        else:
            metrics['annualized_return'] = 0.0
        
        # Calculate Sharpe ratio (assuming risk-free rate = 0)
        if len(returns) > 1 and np.std(returns) > 0:
            metrics['sharpe_ratio'] = np.mean(returns) / np.std(returns) * np.sqrt(252)
        else:
            metrics['sharpe_ratio'] = 0.0
        
        # Calculate maximum drawdown
        cumulative = np.cumprod(1 + np.array(returns))
        peak = np.maximum.accumulate(cumulative)
        drawdown = (peak - cumulative) / peak
        metrics['max_drawdown'] = np.max(drawdown) * 100 if len(drawdown) > 0 else 0.0
        
        # Round metrics to 3 decimal places
        return {k: round(v, 3) for k, v in metrics.items()}
    except Exception as e:
        logger.error(f"Error calculating performance metrics: {str(e)}")
        return {'total_return': 0.0, 'annualized_return': 0.0, 'sharpe_ratio': 0.0, 'max_drawdown': 0.0}

def backtest(
    symbol: str,
    model: tf.keras.Model,
    scaler: StandardScaler,
    df: pd.DataFrame,
    initial_cash: float,
    stop_loss_multiplier: float,
    take_profit_multiplier: float,
    timesteps: int,
    buy_threshold: float,
    sell_threshold: float
) -> Tuple[float, List[float], int, float]:
    """
    Run backtest simulation with trading logic.
    
    Args:
        symbol (str): Stock symbol.
        model (tf.keras.Model): Trained LSTM model.
        scaler (StandardScaler): Fitted scaler for features.
        df (pd.DataFrame): DataFrame with indicators.
        initial_cash (float): Initial cash per symbol.
        stop_loss_multiplier (float): ATR multiplier for stop-loss.
        take_profit_multiplier (float): ATR multiplier for take-profit.
        timesteps (int): Number of timesteps for sequences.
        buy_threshold (float): Prediction threshold for buying.
        sell_threshold (float): Prediction threshold for selling.
    
    Returns:
        Tuple[float, List[float], int, float]: Final cash, list of returns, trade count, win rate.
    
    Raises:
        Exception: If backtest fails.
    """
    try:
        # Preprocess data for predictions
        X, _ = preprocess_data(df, timesteps)
        X_flat = X.reshape(-1, X.shape[-1])
        X_scaled = scaler.transform(X_flat).reshape(X.shape)
        
        # Generate predictions
        predictions = model.predict(X_scaled, verbose=0).flatten()
        logger.info(
            f"Predictions for {symbol}: mean={predictions.mean():.3f}, "
            f"std={predictions.std():.3f}, min={predictions.min():.3f}, max={predictions.max():.3f}"
        )
        
        # Initialize trading variables
        cash = initial_cash / len(CONFIG['SYMBOLS'])
        position = 0
        entry_price = 0.0
        returns = []
        trade_count = 0
        winning_trades = 0
        atr = df['ATR'].iloc[-len(predictions):].values
        prices = df['close'].iloc[-len(predictions):].values
        timestamps = df['timestamp'].iloc[-len(predictions):].values
        
        # Execute trading logic
        for i, (pred, price, atr_val, ts) in enumerate(zip(predictions, prices, atr, timestamps)):
            qty = int(cash / price) if cash >= price else 0
            
            # Buy logic: Enter position if prediction exceeds buy threshold
            if pred > buy_threshold and position == 0 and qty > 0:
                position = qty
                entry_price = price
                cash -= qty * price
                logger.info(
                    f"{ts}: Bought {qty} shares of {symbol} at ${price:.2f}, "
                    f"cash: ${cash:.2f}, position: {position}"
                )
            
            # Sell logic: Exit position based on stop-loss, take-profit, or sell threshold
            elif position > 0:
                stop_loss = entry_price - stop_loss_multiplier * atr_val
                take_profit = entry_price + take_profit_multiplier * atr_val
                
                if price <= stop_loss or price >= take_profit or pred < sell_threshold:
                    cash += position * price
                    ret = (price - entry_price) / entry_price
                    returns.append(ret)
                    trade_count += 1
                    if ret > 0:
                        winning_trades += 1
                    logger.info(
                        f"{ts}: Sold {position} shares of {symbol} at ${price:.2f}, "
                        f"return: {ret:.3f}, cash: ${cash:.2f}, position: 0"
                    )
                    position = 0
                    entry_price = 0.0
                
                # Log zero quantity periodically to track insufficient cash
                if qty == 0 and i % 100 == 0:
                    logger.info(
                        f"{ts}: Zero quantity for {symbol}: cash=${cash:.2f}, price=${price:.2f}"
                    )
        
        # Close any open position at the end
        if position > 0:
            cash += position * prices[-1]
            ret = (prices[-1] - entry_price) / entry_price
            returns.append(ret)
            trade_count += 1
            if ret > 0:
                winning_trades += 1
            logger.info(
                f"{timestamps[-1]}: Closed position for {symbol} at ${prices[-1]:.2f}, "
                f"return: {ret:.3f}, final cash: ${cash:.2f}"
            )
        
        # Calculate win rate
        win_rate = (winning_trades / trade_count * 100) if trade_count > 0 else 0.0
        
        return round(cash, 2), returns, trade_count, round(win_rate, 3)
    except Exception as e:
        logger.error(f"Error in backtest for {symbol}: {str(e)}")
        raise

def format_email_body(
    initial_cash: float,
    final_value: float,
    symbol_results: Dict[str, Dict[str, float]],
    trade_counts: Dict[str, int],
    win_rates: Dict[str, float]
) -> str:
    """
    Format email body with backtest results, including trade statistics.
    
    Args:
        initial_cash (float): Initial cash amount.
        final_value (float): Final portfolio value.
        symbol_results (Dict[str, Dict[str, float]]): Per-symbol performance metrics.
        trade_counts (Dict[str, int]): Number of trades per symbol.
        win_rates (Dict[str, float]): Win rate per symbol.
    
    Returns:
        str: Formatted email body with results and statistics.
    
    Raises:
        Exception: If formatting fails.
    """
    try:
        body = [
            f"Backtest Results",
            f"=================",
            f"Initial Cash: ${initial_cash:.2f}",
            f"Final Value: ${final_value:.2f}",
            f"Profit/Loss: ${final_value - initial_cash:.2f}",
            f"Total Return: {(final_value - initial_cash) / initial_cash * 100:.2f}%",
            f"",
            f"Per-Symbol Performance:"
        ]
        
        for symbol, metrics in symbol_results.items():
            body.append(f"\n{symbol}:")
            for metric, value in metrics.items():
                body.append(f"  {metric.replace('_', ' ').title()}: {value:.3f}%")
            body.append(f"  Number of Trades: {trade_counts.get(symbol, 0)}")
            body.append(f"  Win Rate: {win_rates.get(symbol, 0.0):.3f}%")
        
        return "\n".join(body)
    except Exception as e:
        logger.error(f"Error formatting email body: {str(e)}")
        return f"Backtest Results\nError: {str(e)}"

def send_email(subject: str, body: str) -> None:
    """
    Send email notification with backtest results.
    
    Args:
        subject (str): Email subject line.
        body (str): Email body content.
    
    Raises:
        Exception: If email sending fails.
    """
    try:
        msg = MIMEText(body)
        msg['Subject'] = subject
        msg['From'] = CONFIG['EMAIL_SENDER']
        msg['To'] = CONFIG['EMAIL_RECEIVER']
        
        with smtplib.SMTP(CONFIG['SMTP_SERVER'], CONFIG['SMTP_PORT']) as server:
            server.starttls()
            server.login(CONFIG['EMAIL_SENDER'], CONFIG['EMAIL_PASSWORD'])
            server.send_message(msg)
        
        logger.info(f"Email sent: Subject: {subject} to {CONFIG['EMAIL_RECEIVER']}")
    except Exception as e:
        logger.error(
            f"Error sending email: {str(e)}. "
            f"Check SMTP settings (EMAIL_SENDER, EMAIL_PASSWORD, SMTP_SERVER, SMTP_PORT) in CONFIG."
        )
        raise

def main(backtest_only: bool = False) -> None:
    """
    Main function to run the trading bot, handling both training and backtesting.
    
    Args:
        backtest_only (bool): If True, run backtest using pre-trained models; otherwise, train models.
    
    Raises:
        Exception: If main loop fails (e.g., configuration error, API failure).
    """
    try:
        # Check for required dependencies
        check_dependencies()
        
        # Validate configuration to ensure all parameters are correct
        validate_config(CONFIG)
        
        # Create cache directory for storing models and data
        create_cache_directory()
        
        # Initialize portfolio variables
        initial_cash = CONFIG['INITIAL_CASH']
        final_value = initial_cash
        symbol_results = {}
        trade_counts = {}
        win_rates = {}
        
        # Process each symbol in the configuration
        for symbol in CONFIG['SYMBOLS']:
            try:
                logger.info(f"Processing {symbol}")
                
                # Fetch historical data
                df = fetch_data(
                    symbol, CONFIG['TIMEFRAME'], CONFIG['TIMEFRAME_INTERVAL'], CONFIG['NUM_BARS']
                )
                
                # Validate fetched data
                validate_data(df, symbol)
                
                # Load sentiment score
                sentiment = load_news_sentiment(symbol)
                
                # Calculate technical indicators
                df = calculate_indicators(df, sentiment)
                
                if backtest_only:
                    try:
                        # Attempt to load pre-trained model and scaler
                        model, scaler = load_model_and_scaler(symbol)
                    except FileNotFoundError as e:
                        logger.warning(
                            f"{str(e)}. Falling back to training for {symbol} in backtest-only mode"
                        )
                        X, y = preprocess_data(df, CONFIG['TIMESTEPS'])
                        model, scaler = train_model(
                            symbol, X, y, CONFIG['TRAIN_EPOCHS'], CONFIG['BATCH_SIZE']
                        )
                else:
                    # Clear cache to ensure fresh training
                    for file in [f"{symbol}_model.keras", f"{symbol}_scaler.pkl", f"{symbol}_data.pkl"]:
                        file_path = os.path.join(CONFIG['CACHE_DIR'], file)
                        if os.path.exists(file_path):
                            os.remove(file_path)
                            logger.info(f"Deleted cached {file} for {symbol}")
                    
                    # Preprocess data for training
                    X, y = preprocess_data(df, CONFIG['TIMESTEPS'])
                    
                    # Train the model
                    model, scaler = train_model(
                        symbol, X, y, CONFIG['TRAIN_EPOCHS'], CONFIG['BATCH_SIZE']
                    )
                
                # Run backtest with trading logic
                cash, returns, trade_count, win_rate = backtest(
                    symbol, model, scaler, df, CONFIG['INITIAL_CASH'],
                    CONFIG['STOP_LOSS_MULTIPLIER'], CONFIG['TAKE_PROFIT_MULTIPLIER'],
                    CONFIG['TIMESTEPS'], CONFIG['PREDICTION_THRESHOLD_BUY'],
                    CONFIG['PREDICTION_THRESHOLD_SELL']
                )
                
                # Update portfolio value
                final_value += cash - (initial_cash / len(CONFIG['SYMBOLS']))
                
                # Store performance metrics and trade statistics
                symbol_results[symbol] = calculate_performance_metrics(
                    returns, cash, initial_cash / len(CONFIG['SYMBOLS'])
                )
                trade_counts[symbol] = trade_count
                win_rates[symbol] = win_rate
                logger.info(f"Backtest for {symbol} completed: cash=${cash:.2f}, trades={trade_count}")
                
            except Exception as e:
                logger.error(f"Skipping {symbol} due to error: {str(e)}")
                continue
        
        # Format and send email with results
        email_body = format_email_body(initial_cash, final_value, symbol_results, trade_counts, win_rates)
        send_email("Backtest Completed", email_body)
        logger.info(f"Bot completed: Final value: ${final_value:.2f}")
    
    except Exception as e:
        logger.error(f"Main loop failed: {str(e)}")
        raise

if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Trading bot with backtest mode")
    parser.add_argument(
        '--backtest',
        action='store_true',
        help="Run in backtest-only mode using pre-trained models"
    )
    args = parser.parse_args()
    
    logger.info("Bot started")
    try:
        main(backtest_only=args.backtest)
    except Exception as e:
        logger.error(f"Bot failed: {str(e)}")
        raise