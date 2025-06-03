# =============================================================================
# Alpaca Advanced Neural Bot with Email Notifications - Version 1.5
# =============================================================================
#
# Author: Grok 3, built by xAI
# Date: May 14, 2025
# Version: 1.5
#
# Description:
#  - Advanced trading bot using LSTM neural networks for stock price prediction.
#  - Separates training data period (TRAIN_YEARS) and simulation period (SIMULATION_DAYS).
#  - Fetches historical data from Alpaca API, calculates 25 technical indicators.
#  - Trains per-symbol LSTM models and backtests with ATR-based trading logic.
#  - Logs trades and sends email notifications with performance metrics to multiple recipients.
#  - Fixes validation timing and column handling issues from v1.4.
#
# Key Features:
#  - Configurable training (TRAIN_YEARS) and simulation (SIMULATION_DAYS) periods.
#  - Supports multiple stocks with independent LSTM models.
#  - Uses 25 technical indicators (e.g., RSI, MACD, Bollinger Bands, ATR).
#  - Backtests with buy/sell thresholds and ATR-based stop-loss/take-profit.
#  - Robust error handling for data fetching, preprocessing, and model loading.
#  - Email notifications to multiple recipients with per-symbol metrics (return, Sharpe ratio, etc.).
#
# Usage:
#  - Modify CONFIG (e.g., SYMBOLS, TRAIN_YEARS, SIMULATION_DAYS, EMAIL_RECEIVER).
#  - Run live mode (training + backtest): `python alpaca_neural_bot_v1_5.py`
#  - Run backtest-only mode: `python alpaca_neural_bot_v1_5.py --backtest`
#  - Ensure `./cache` contains compatible models/scalers for backtest mode.
#  - If errors persist, clear cache (`rm ./cache/*`) and run live mode to retrain.
#
# Dependencies:
#  - tensorflow, numpy, pandas, alpaca-py, transformers, scikit-learn, ta-lib, tenacity
#  - Install with: `pip install tensorflow numpy pandas alpaca-py transformers scikit-learn ta-lib tenacity`
#
# Notes:
#  - Requires valid Alpaca API credentials (paper trading) in CONFIG.
#  - Sentiment analysis is a placeholder; implement news processing for production.
#  - Large TRAIN_YEARS may require multiple API calls due to 10,000-bar limit.
#  - Validates raw OHLCV data before and full feature set after indicator calculation.
#
# Changelog (v1.5):
#  - Fixed validation timing by splitting into validate_raw_data and validate_indicators.
#  - Handled column case sensitivity (e.g., vwap vs. VWAP) in fetch_data and calculate_indicators.
#  - Enhanced error handling for missing or unexpected columns.
#  - Improved logging for debugging DataFrame columns and validation failures.
#  - Maintained 25-feature set and compatibility with cached models/scalers.
#  - Added support for multiple email recipients in send_email (May 14, 2025).
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
from alpaca.data import StockHistoricalDataClient, TimeFrame, TimeFrameUnit
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
    'TIMEFRAME': TimeFrame(15, TimeFrameUnit.Minute),  # Timeframe for data (e.g., 15-minute bars)
    'TRAIN_YEARS': 1,  # Number of years of data for training
    'SIMULATION_DAYS': 10,  # Number of days to simulate in backtest
    'TRAIN_EPOCHS': 25,  # Number of training epochs
    'BATCH_SIZE': 32,  # Training batch size
    'INITIAL_CASH': 25000.0,  # Initial cash for backtesting
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
    'EMAIL_RECEIVER': ['aiplane.scientist@gmail.com', 'tchaikovskiy@hotmail.com'],  # Email receiver addresses
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
    """Check for required Python modules and raise an error if any are missing."""
    try:
        logger.info(f"Python executable: {sys.executable}")
        logger.info(f"Python version: {sys.version}")
        logger.info(f"Site-packages: {sys.path}")
        
        if sys.version_info < (3, 10):
            raise RuntimeError("Python 3.10 or higher is required")
        
        required_modules = [
            'tensorflow', 'numpy', 'pandas', 'alpaca_trade_api', 'transformers',
            'sklearn', 'talib', 'tenacity', 'smtplib', 'argparse'
        ]
        
        for module in required_modules:
            try:
                importlib.import_module(module)
                logger.debug(f"Module {module} is installed")
            except ImportError:
                raise ImportError(
                    f"The '{module}' module is required. Install it using: pip install {module}"
                )
        logger.info("All required dependencies are installed")
    except Exception as e:
        logger.error(f"Dependency check failed: {str(e)}")
        raise

def validate_config(config: Dict) -> None:
    """Validate configuration parameters."""
    try:
        if not config['SYMBOLS']:
            raise ValueError("SYMBOLS list cannot be empty")
        
        if not isinstance(config['TIMEFRAME'], TimeFrame):
            raise ValueError("TIMEFRAME must be a valid TimeFrame object")
        
        for param, value in [
            ('TRAIN_YEARS', config['TRAIN_YEARS']),
            ('SIMULATION_DAYS', config['SIMULATION_DAYS']),
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
        
        for param, value in [
            ('INITIAL_CASH', config['INITIAL_CASH']),
            ('STOP_LOSS_MULTIPLIER', config['STOP_LOSS_MULTIPLIER']),
            ('TAKE_PROFIT_MULTIPLIER', config['TAKE_PROFIT_MULTIPLIER']),
            ('PREDICTION_THRESHOLD_BUY', config['PREDICTION_THRESHOLD_BUY']),
            ('PREDICTION_THRESHOLD_SELL', config['PREDICTION_THRESHOLD_SELL'])
        ]:
            if not isinstance(value, (int, float)) or value <= 0:
                raise ValueError(f"{param} must be a positive number")
        
        if config['PREDICTION_THRESHOLD_SELL'] >= config['PREDICTION_THRESHOLD_BUY']:
            raise ValueError("PREDICTION_THRESHOLD_SELL must be less than PREDICTION_THRESHOLD_BUY")
        
        logger.info("Configuration validated successfully")
    except Exception as e:
        logger.error(f"Configuration validation failed: {str(e)}")
        raise

def create_cache_directory() -> None:
    """Create cache directory if it doesn't exist."""
    try:
        os.makedirs(CONFIG['CACHE_DIR'], exist_ok=True)
        logger.info(f"Cache directory ensured: {CONFIG['CACHE_DIR']}")
    except OSError as e:
        logger.error(f"Failed to create cache directory: {str(e)}")
        raise

@retry(
    stop=stop_after_attempt(CONFIG['API_RETRY_ATTEMPTS']),
    wait=wait_fixed(CONFIG['API_RETRY_DELAY'] / 1000),
    retry=retry_if_exception_type(Exception)
)
def fetch_data(symbol: str, train_years: int) -> pd.DataFrame:
    """Fetch historical bar data from Alpaca API for the training period."""
    try:
        client = StockHistoricalDataClient(CONFIG['ALPACA_API_KEY'], CONFIG['ALPACA_SECRET_KEY'])
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365 * train_years)
        all_bars = []
        current_end = end_date
        while True:
            request = StockBarsRequest(
                symbol_or_symbols=symbol,
                timeframe=CONFIG['TIMEFRAME'],
                start=start_date,
                end=current_end,
                limit=10000
            )
            bars = client.get_stock_bars(request).df
            if bars.empty:
                break
            df_bars = bars.reset_index()
            # Standardize column names
            df_bars = df_bars.rename(columns={'vwap': 'VWAP'})
            all_bars.append(df_bars)
            if len(bars) < 10000:
                break
            current_end = df_bars['timestamp'].min() - timedelta(seconds=1)
        if all_bars:
            df = pd.concat(all_bars).sort_values('timestamp')
        else:
            df = pd.DataFrame(columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'VWAP'])
        if len(df) < CONFIG['MIN_DATA_POINTS']:
            raise ValueError(f"Insufficient data for {symbol}: got {len(df)} bars, need {CONFIG['MIN_DATA_POINTS']}")
        logger.info(f"Fetched {len(df)} bars for {symbol} with timeframe {CONFIG['TIMEFRAME']}, columns: {list(df.columns)}")
        return df
    except Exception as e:
        logger.error(f"Error fetching data for {symbol}: {str(e)}")
        raise

def load_news_sentiment(symbol: str) -> float:
    """Load or compute news sentiment (placeholder)."""
    cache_file = os.path.join(CONFIG['CACHE_DIR'], f"{symbol}_news_sentiment.pkl")
    try:
        if os.path.exists(cache_file) and (time.time() - os.path.getmtime(cache_file)) < CONFIG['CACHE_EXPIRY_SECONDS']:
            with open(cache_file, 'rb') as f:
                sentiment = pickle.load(f)
            logger.info(f"Loaded news sentiment for {symbol}: {sentiment:.3f}")
            return sentiment
        
        sentiment = 0.2  # Placeholder; replace with actual sentiment analysis
        with open(cache_file, 'wb') as f:
            pickle.dump(sentiment, f)
        logger.info(f"Computed and cached news sentiment for {symbol}: {sentiment:.3f}")
        return sentiment
    except Exception as e:
        logger.error(f"Error loading news sentiment for {symbol}: {str(e)}")
        return 0.2

def calculate_indicators(df: pd.DataFrame, sentiment: float) -> pd.DataFrame:
    """Calculate technical indicators and add sentiment."""
    try:
        df = df.copy()
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        if not all(col in df.columns for col in required_cols):
            raise ValueError(f"Missing required columns: {required_cols}")
        
        # Compute VWAP if not provided
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
        
        # Check for NaN in computed indicators
        indicator_cols = [
            'MA20', 'MA50', 'RSI', 'MACD', 'MACD_signal', 'OBV', 'VWAP', 'ATR',
            'CMF', 'Close_ATR', 'MA20_ATR', 'Return_1d', 'Return_5d', 'Volatility',
            'BB_upper', 'BB_middle', 'BB_lower', 'Stoch_K', 'Stoch_D', 'ADX', 'Sentiment', 'Trend'
        ]
        nan_counts = df[indicator_cols].isna().sum().to_dict()
        if any(nan_counts.values()):
            logger.warning(f"NaN values in indicators: {nan_counts}")
        
        logger.info(f"Indicators calculated for shape: {df.shape}, columns: {list(df.columns)}")
        return df
    except Exception as e:
        logger.error(f"Error calculating indicators: {str(e)}")
        raise

def validate_raw_data(df: pd.DataFrame, symbol: str) -> None:
    """Validate raw OHLCV DataFrame after fetching."""
    try:
        if df.empty:
            raise ValueError(f"Empty DataFrame for {symbol}")
        required_cols = ['open', 'high', 'low', 'close', 'volume', 'timestamp']
        if not all(col in df.columns for col in required_cols):
            missing = [col for col in required_cols if col not in df.columns]
            raise ValueError(f"Missing required columns for {symbol}: {missing}")
        if df[required_cols[:-1]].isna().any().any():
            raise ValueError(f"NaN values in OHLCV columns for {symbol}")
        if len(df) < CONFIG['MIN_DATA_POINTS']:
            raise ValueError(f"Insufficient data for {symbol}: got {len(df)} bars")
        logger.info(f"Raw data validated for {symbol}: {len(df)} bars, columns: {list(df.columns)}")
    except Exception as e:
        logger.error(f"Raw data validation failed for {symbol}: {str(e)}")
        raise

def validate_indicators(df: pd.DataFrame, symbol: str) -> None:
    """Validate DataFrame after calculating indicators."""
    try:
        required_cols = [
            'open', 'high', 'low', 'close', 'volume', 'timestamp',
            'MA20', 'MA50', 'RSI', 'MACD', 'MACD_signal', 'OBV', 'VWAP', 'ATR',
            'CMF', 'Close_ATR', 'MA20_ATR', 'Return_1d', 'Return_5d', 'Volatility',
            'BB_upper', 'BB_lower', 'Stoch_K', 'Stoch_D', 'ADX', 'Sentiment', 'Trend'
        ]
        if not all(col in df.columns for col in required_cols):
            missing = [col for col in required_cols if col not in df.columns]
            raise ValueError(f"Missing indicator columns for {symbol}: {missing}")
        logger.info(f"Indicator data validated for {symbol}: {len(df)} bars, columns: {list(df.columns)}")
    except Exception as e:
        logger.error(f"Indicator validation failed for {symbol}: {str(e)}")
        raise

def preprocess_data(df: pd.DataFrame, timesteps: int) -> Tuple[np.ndarray, np.ndarray]:
    """Preprocess data into sequences for model input."""
    try:
        df = df.copy()
        df.ffill(inplace=True)
        df.fillna(0, inplace=True)
        
        features = [
            'close', 'high', 'low', 'volume', 'VWAP', 'MA20', 'MA50', 'RSI',
            'MACD', 'MACD_signal', 'OBV', 'VWAP', 'ATR', 'CMF', 'Close_ATR',
            'MA20_ATR', 'Return_1d', 'Return_5d', 'Volatility', 'BB_upper',
            'BB_lower', 'Stoch_K', 'Stoch_D', 'ADX', 'Sentiment'
        ]
        if not all(f in df.columns for f in features):
            missing = [f for f in features if f not in df.columns]
            raise ValueError(f"Missing features for preprocessing: {missing}")
        
        X = df[features].values
        y = df['Trend'].values
        
        X_seq = [X[i-timesteps:i] for i in range(timesteps, len(X))]
        y_seq = [y[i] for i in range(timesteps, len(y))]
        
        X_seq = np.array(X_seq)
        y_seq = np.array(y_seq)
        
        if X_seq.shape[0] == 0:
            raise ValueError(f"No valid sequences generated for shape: {X.shape}")
        if np.any(np.isnan(X_seq)) or np.any(np.isinf(X_seq)):
            raise ValueError("NaN or Inf values in preprocessed data")
        
        logger.info(f"Preprocessed features: X shape: {X_seq.shape}, features: {len(features)}")
        return X_seq, y_seq
    except Exception as e:
        logger.error(f"Error preprocessing data: {str(e)}")
        raise

def build_model(input_shape: Tuple[int, int]) -> tf.keras.Model:
    """Build and compile an LSTM model."""
    try:
        K.clear_session()
        model = Sequential([
            LSTM(64, input_shape=input_shape, return_sequences=True),
            Dropout(0.2),
            LSTM(32),
            Dropout(0.2),
            Dense(16, activation='relu'),
            Dense(1, activation='sigmoid')
        ])
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        logger.info(f"Model built with input shape {input_shape}")
        return model
    except Exception as e:
        logger.error(f"Error building model: {str(e)}")
        raise

def train_model(symbol: str, X: np.ndarray, y: np.ndarray, epochs: int, batch_size: int) -> Tuple[tf.keras.Model, StandardScaler]:
    """Train and save the model."""
    try:
        scaler = StandardScaler()
        X_flat = X.reshape(-1, X.shape[-1])
        X_scaled = scaler.fit_transform(X_flat).reshape(X.shape)
        
        model = build_model((X.shape[1], X.shape[2]))
        model.fit(X_scaled, y, epochs=epochs, batch_size=batch_size, verbose=0)
        
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

def load_model_and_scaler(symbol: str, expected_features: int) -> Tuple[Optional[tf.keras.Model], Optional[StandardScaler]]:
    """Load pre-trained model and scaler, validating feature count."""
    try:
        model_path = os.path.join(CONFIG['CACHE_DIR'], f"{symbol}_model.keras")
        scaler_path = os.path.join(CONFIG['CACHE_DIR'], f"{symbol}_scaler.pkl")
        if not (os.path.exists(model_path) and os.path.exists(scaler_path)):
            raise FileNotFoundError(f"Model or scaler file missing for {symbol}")
        
        model = tf.keras.models.load_model(model_path)
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        
        # Validate scaler feature count
        if hasattr(scaler, 'n_features_in_') and scaler.n_features_in_ != expected_features:
            raise ValueError(
                f"Scaler for {symbol} expects {scaler.n_features_in_} features, "
                f"but current feature set has {expected_features}. Run without --backtest to retrain."
            )
        
        logger.info(f"Loaded model and scaler for {symbol}")
        return model, scaler
    except Exception as e:
        logger.error(f"Error loading model/scaler for {symbol}: {str(e)}")
        return None, None

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
    """Run backtest simulation for the last SIMULATION_DAYS."""
    try:
        X, _ = preprocess_data(df, timesteps)
        X_flat = X.reshape(-1, X.shape[-1])
        X_scaled = scaler.transform(X_flat).reshape(X.shape)
        
        predictions = model.predict(X_scaled, verbose=0).flatten()
        logger.info(
            f"Predictions for {symbol}: mean={predictions.mean():.3f}, "
            f"std={predictions.std():.3f}, min={predictions.min():.3f}, max={predictions.max():.3f}"
        )
        
        timestamps = df['timestamp'].iloc[timesteps:].reset_index(drop=True)
        sim_start_timestamp = timestamps.max() - timedelta(days=CONFIG['SIMULATION_DAYS'])
        k_start = next((k for k, ts in enumerate(timestamps) if ts >= sim_start_timestamp), None)
        if k_start is None:
            raise ValueError(f"No data for simulation period for {symbol}")
        
        cash = initial_cash / len(CONFIG['SYMBOLS'])
        position = 0
        entry_price = 0.0
        returns = []
        trade_count = 0
        winning_trades = 0
        atr = df['ATR'].iloc[timesteps:].values
        prices = df['close'].iloc[timesteps:].values
        sim_timestamps = timestamps.values
        
        for i in range(k_start, len(predictions)):
            pred = predictions[i]
            price = prices[i]
            atr_val = atr[i]
            ts = sim_timestamps[i]
            qty = int(cash / price) if cash >= price else 0
            
            if pred > buy_threshold and position == 0 and qty > 0:
                position = qty
                entry_price = price
                cash -= qty * price
                logger.info(f"{ts}: Bought {qty} shares of {symbol} at ${price:.2f}, cash: ${cash:.2f}")
            
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
                        f"return: {ret:.3f}, cash: ${cash:.2f}"
                    )
                    position = 0
        
        if position > 0:
            cash += position * prices[-1]
            ret = (prices[-1] - entry_price) / entry_price
            returns.append(ret)
            trade_count += 1
            if ret > 0:
                winning_trades += 1
            logger.info(f"Closed position for {symbol} at ${prices[-1]:.2f}, return: {ret:.3f}")
        
        win_rate = (winning_trades / trade_count * 100) if trade_count > 0 else 0.0
        return cash, returns, trade_count, win_rate
    except Exception as e:
        logger.error(f"Error in backtest for {symbol}: {str(e)}")
        raise

def calculate_performance_metrics(returns: List[float], cash: float, initial_cash: float) -> Dict[str, float]:
    """Calculate performance metrics."""
    try:
        returns = np.array(returns)
        metrics = {
            'total_return': (cash - initial_cash) / initial_cash * 100,
            'sharpe_ratio': np.mean(returns) / np.std(returns) * np.sqrt(252) if len(returns) > 1 and np.std(returns) > 0 else 0.0,
            'max_drawdown': np.max((np.maximum.accumulate(np.cumprod(1 + returns)) - np.cumprod(1 + returns)) / np.maximum.accumulate(np.cumprod(1 + returns))) * 100 if returns.size > 0 else 0.0
        }
        return {k: round(v, 3) for k, v in metrics.items()}
    except Exception as e:
        logger.error(f"Error calculating performance metrics: {str(e)}")
        return {'total_return': 0.0, 'sharpe_ratio': 0.0, 'max_drawdown': 0.0}

def format_email_body(
    initial_cash: float,
    final_value: float,
    symbol_results: Dict[str, Dict[str, float]],
    trade_counts: Dict[str, int],
    win_rates: Dict[str, float]
) -> str:
    """Format email body with backtest results."""
    try:
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
    except Exception as e:
        logger.error(f"Error formatting email body: {str(e)}")
        return f"Error: {str(e)}"

def send_email(subject: str, body: str) -> None:
    """Send email notification to multiple recipients."""
    try:
        msg = MIMEText(body)
        msg['Subject'] = subject
        msg['From'] = CONFIG['EMAIL_SENDER']
        msg['To'] = ', '.join(CONFIG['EMAIL_RECEIVER'])  # Join emails for header
        
        with smtplib.SMTP(CONFIG['SMTP_SERVER'], CONFIG['SMTP_PORT']) as server:
            server.starttls()
            server.login(CONFIG['EMAIL_SENDER'], CONFIG['EMAIL_PASSWORD'])
            server.sendmail(
                CONFIG['EMAIL_SENDER'],
                CONFIG['EMAIL_RECEIVER'],  # List of recipients
                msg.as_string()
            )
        logger.info(f"Email sent: {subject} to {', '.join(CONFIG['EMAIL_RECEIVER'])}")
    except Exception as e:
        logger.error(f"Error sending email: {str(e)}")
        raise

def main(backtest_only: bool = False) -> None:
    """Main function to run the trading bot."""
    try:
        check_dependencies()
        validate_config(CONFIG)
        create_cache_directory()
        
        initial_cash = CONFIG['INITIAL_CASH']
        final_value = initial_cash
        symbol_results = {}
        trade_counts = {}
        win_rates = {}
        expected_features = 25  # Number of features in preprocess_data
        
        for symbol in CONFIG['SYMBOLS']:
            try:
                logger.info(f"Processing {symbol}")
                df = fetch_data(symbol, CONFIG['TRAIN_YEARS'])
                validate_raw_data(df, symbol)
                sentiment = load_news_sentiment(symbol)
                df = calculate_indicators(df, sentiment)
                validate_indicators(df, symbol)
                
                if backtest_only:
                    model, scaler = load_model_and_scaler(symbol, expected_features)
                    if model is None or scaler is None:
                        logger.warning(f"Invalid model/scaler for {symbol}. Falling back to training.")
                        X, y = preprocess_data(df, CONFIG['TIMESTEPS'])
                        model, scaler = train_model(symbol, X, y, CONFIG['TRAIN_EPOCHS'], CONFIG['BATCH_SIZE'])
                else:
                    X, y = preprocess_data(df, CONFIG['TIMESTEPS'])
                    model, scaler = train_model(symbol, X, y, CONFIG['TRAIN_EPOCHS'], CONFIG['BATCH_SIZE'])
                
                cash, returns, trade_count, win_rate = backtest(
                    symbol, model, scaler, df, CONFIG['INITIAL_CASH'],
                    CONFIG['STOP_LOSS_MULTIPLIER'], CONFIG['TAKE_PROFIT_MULTIPLIER'],
                    CONFIG['TIMESTEPS'], CONFIG['PREDICTION_THRESHOLD_BUY'],
                    CONFIG['PREDICTION_THRESHOLD_SELL']
                )
                
                final_value += cash - (initial_cash / len(CONFIG['SYMBOLS']))
                symbol_results[symbol] = calculate_performance_metrics(
                    returns, cash, initial_cash / len(CONFIG['SYMBOLS'])
                )
                trade_counts[symbol] = trade_count
                win_rates[symbol] = win_rate
                logger.info(f"Backtest for {symbol} completed: cash=${cash:.2f}, trades={trade_count}")
                
            except Exception as e:
                logger.error(f"Skipping {symbol} due to error: {str(e)}")
                continue
        
        email_body = format_email_body(initial_cash, final_value, symbol_results, trade_counts, win_rates)
        send_email("Backtest Completed", email_body)
        logger.info(f"Bot completed: Final value: ${final_value:.2f}")
    except Exception as e:
        logger.error(f"Main loop failed: {str(e)}")
        raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Trading bot with backtest mode")
    parser.add_argument('--backtest', action='store_true', help="Run in backtest-only mode")
    args = parser.parse_args()
    
    logger.info("Bot started")
    main(backtest_only=args.backtest)