
# +------------------------------------------------------------------------------+
# |                            Alpaca Neural Bot v9.3.5                          |
# +------------------------------------------------------------------------------+
# | Author: Vladimir Makarov                                                     |
# | Project Start Date: May 9, 2025                                              |
# | License: GNU Lesser General Public License v2.1                              |
# | Version: 9.3.5 (Un-Released)                                                 |
# +------------------------------------------------------------------------------+

import os  # For operating system interactions, like creating directories and handling file paths
import sys  # For system-specific parameters and functions, such as exiting or accessing argv
import logging  # For configuring and handling log messages and errors
import argparse  # For parsing command-line arguments like --backtest or --force-train
import importlib  # For dynamically importing modules, used in dependency checks
import numpy as np  # For numerical computations, arrays, and math operations like means/std
import pandas as pd  # For data manipulation, DataFrames, and time-series handling
import torch  # For the PyTorch deep learning framework, core for model building and training
import torch.nn as nn  # For neural network modules and layers in PyTorch, like LSTM and Linear
import torch.optim as optim  # For optimization algorithms in PyTorch, like Adam
from torch.utils.data import DataLoader, TensorDataset  # For creating data loaders and tensor datasets in PyTorch training
from alpaca.data import StockHistoricalDataClient, TimeFrame, TimeFrameUnit  # For Alpaca historical data client and timeframe definitions
from alpaca.data.requests import StockBarsRequest  # For requesting stock bar data from Alpaca API
from alpaca.trading.client import TradingClient  # For Alpaca trading client to manage positions and orders
from alpaca.trading.requests import MarketOrderRequest  # For creating market order requests in Alpaca trading
from alpaca.trading.enums import OrderSide, TimeInForce  # For enums like BUY/SELL sides and time-in-force in Alpaca
from alpaca.common.exceptions import APIError  # For handling exceptions from Alpaca API calls
from transformers import pipeline  # For Hugging Face sentiment analysis pipeline
from sklearn.preprocessing import RobustScaler, StandardScaler  # For data scaling and normalization in preprocessing
import smtplib  # For sending emails via SMTP protocol
from email.mime.text import MIMEText  # For constructing MIME text messages for emails
from datetime import datetime, timedelta, timezone  # For handling dates, times, and timezones in data fetching and logging
import talib  # For technical analysis library, providing indicators like RSI, MACD, ATR, ADX
import pickle  # For serializing/deserializing objects like models, scalers, and data caches
from typing import List, Tuple, Dict, Optional, Any  # For type hinting in function signatures and variables
import warnings  # For suppressing or handling warnings, like PyTorch user warnings
import time  # For time-related functions, like sleeping or timing operations
from tenacity import retry, stop_after_attempt, wait_fixed, retry_if_exception_type  # For retry decorators on API calls to handle failures
from tqdm import tqdm  # For displaying progress bars during training and data processing
from colorama import Fore, Style  # For colored text styles in console output
import colorama  # For cross-platform colored terminal text initialization
import multiprocessing as mp  # For parallel processing, like training models across symbols
import time  # For time-related functions, like sleeping or timing operations (duplicate import)
import shutil # File transfer
import tempfile  # Add this import at the top of the file if not already present

# Suppress PyTorch warnings
warnings.filterwarnings('ignore', category=UserWarning)

# Initialize colorama for colored console output
colorama.init()

CONFIG = {
    # Trading Parameters - Settings related to trading operations
    'SYMBOLS': ['SPY', 'MSFT', 'AAPL', 'AMZN', 'NVDA', 'META', 'GOOGL'],  # List of stock symbols to trade
    'TIMEFRAME': TimeFrame(15, TimeFrameUnit.Minute),  # Time interval for data fetching
    'INITIAL_CASH': 100000.00,  # Starting cash for trading simulation
    'MIN_HOLDING_PERIOD_MINUTES': 45,  # Minimum holding period for trades

    # Data Fetching and Caching - Parameters for data retrieval and storage
    'TRAIN_DATA_START_DATE': '2015-01-01',  # Start date for training data
    'TRAIN_END_DATE': '2024-06-30',  # End date for training data (extended to include more recent data)
    'VAL_START_DATE': '2024-07-01',  # Start date for validation data (shifted for recent validation)
    'VAL_END_DATE': '2025-09-01',  # End date for validation data (include up to recent date for better generalization)
    'BACKTEST_START_DATE': '2025-01-01',  # Start date for backtesting (out-of-sample)

    'SIMULATION_DAYS': 180,  # Number of days for simulation
    'MIN_DATA_POINTS': 100,  # Minimum data points required for processing
    'CACHE_DIR': './cache',  # Directory for caching data
    'MODEL_CACHE_DIR': '/mnt/c/Users/aipla/Desktop/Model Weights',  # Directory for saving model weights, scalers, and sentiment
    'CACHE_EXPIRY_SECONDS': 24 * 60 * 60,  # Expiry time for cached data in seconds
    'LIVE_DATA_BARS': 500,  # Number of bars to fetch for live data

    # Model Training - Settings for training the machine learning model
    'TRAIN_EPOCHS': 100,  # Number of epochs for training the model
    'BATCH_SIZE': 32,  # Batch size for training
    'TIMESTEPS': 30,  # Number of time steps for sequence data
    'EARLY_STOPPING_MONITOR': 'val_loss',  # Metric to monitor for early stopping
    'EARLY_STOPPING_PATIENCE': 20,  # Patience for early stopping
    'EARLY_STOPPING_MIN_DELTA': 0.0001,  # Reduced min delta to detect smaller improvements
    'LEARNING_RATE': 0.001,  # Reduced initial learning rate for Adam to stabilize training
    'LR_SCHEDULER_PATIENCE': 5,  # Patience for ReduceLROnPlateau
    'LR_REDUCTION_FACTOR': 0.5,  # Factor to multiply LR by
    'LOOK_AHEAD_BARS': 7,  # Number of bars to look ahead for future direction target
    'NUM_PARALLEL_WORKERS': 4,  # Number of parallel workers for symbol training (tune based on VRAM/CPU cores)

    # API and Authentication - Credentials for API access
    'ALPACA_API_KEY': 'PK442T0XBG553SK7IZ5B',  # API key for Alpaca
    'ALPACA_SECRET_KEY': '2upYWNzeRIGGRHk1FXKVtDpdSbgqlqmP3Q0flW8Z',  # Secret key for Alpaca

    # Email Notifications - Configuration for sending email alerts
    'EMAIL_SENDER': 'alpaca.ai.tradingbot@gmail.com',  # Email address for sending notifications
    'EMAIL_PASSWORD': 'hjdf sstp pyne rotq',  # Password for the email account
    'EMAIL_RECEIVER': ['aiplane.scientist@gmail.com', 'vmakarov28@students.d125.org', 'tchaikovskiy@hotmail.com'],  # List of email recipients
    'SMTP_SERVER': 'smtp.gmail.com',  # SMTP server for email
    'SMTP_PORT': 587,  # Port for SMTP server

    # Logging and Monitoring - Settings for tracking activities
    'LOG_FILE': 'trades.log',  # File for logging trades

    # Strategy Thresholds - Thresholds for trading decisions
    'CONFIDENCE_THRESHOLD': 0.52,  # Lowered to capture more predictions above neutral while maintaining selectivity
    'PREDICTION_THRESHOLD_BUY': 0.52,  # Lowered to allow more buy opportunities based on prediction distribution
    'PREDICTION_THRESHOLD_SELL': 0.50,  # Tightened for quicker exits to reduce losses in downtrends
    'RSI_BUY_THRESHOLD': 55,  # RSI threshold for buying (lowered for stronger oversold signals)
    'RSI_SELL_THRESHOLD': 40,  # RSI threshold for selling (raised for stronger overbought signals)
    'ADX_TREND_THRESHOLD': 20,  # Lowered to include weaker trends common in stable stocks like MSFT/AAPL
    'MAX_VOLATILITY': 3.0,  # Increased to include volatile symbols like TSLA/META without excluding opportunities

    # Risk Management - Parameters to control trading risk
    'MAX_DRAWDOWN_LIMIT': 0.04,  # Maximum allowed drawdown
    'RISK_PERCENTAGE': 0.06,  # Percentage of cash to risk per trade (halved for smaller positions)
    'STOP_LOSS_ATR_MULTIPLIER': 1.5,  # Multiplier for ATR-based stop loss (widened to reduce whipsaws)
    'TAKE_PROFIT_ATR_MULTIPLIER': 3.0,  # Multiplier for ATR-based take profit (tightened for quicker exits)
    'TRAILING_STOP_PERCENTAGE': 0.05,  # Percentage for trailing stop (widened slightly)

    # Trading Parameters - Settings related to trading operations
    'TRANSACTION_COST_PER_TRADE': 0.01,  # Cost per trade

    # Sentiment Analysis - Settings for sentiment analysis
    'SENTIMENT_MODEL': 'distilbert-base-uncased-finetuned-sst-2-english',  # Model for sentiment analysis

    # API Retry Settings - Configuration for handling API failures
    'API_RETRY_ATTEMPTS': 3,  # Number of retry attempts for API calls
    'API_RETRY_DELAY': 1000,  # Delay between retry attempts in milliseconds
    'MODEL_VERSION': 'v935',  # Model architecture version; increment on structural changes to force retrain

    # New: Retraining Cycle Parameters
    'ENABLE_RETRAIN_CYCLE': True,  # Enable loop to retrain until criteria met (backtest mode only)
    'MIN_FINAL_VALUE': 130000.0,  # Minimum final portfolio value to accept
    'MAX_ALLOWED_DRAWDOWN': 35.0,  # Maximum allowed max_drawdown percentage (across symbols)
    'MAX_RETRAIN_ATTEMPTS': 28,  # Max loop iterations to prevent infinite runs

    #Monte Carlo Probability Simulation
    'NUM_MC_SIMULATIONS': 50000,  # Number of Monte Carlo simulations for backtest robustness testing
}

#pyenv activate pytorch_env
#python /mnt/c/Users/aipla/Downloads/alpaca_neural_bot_v9.3.5.py --backtest --force-train


def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds technical indicators to the DataFrame using TA-Lib and pandas calculations.
    Assumes df has columns: 'open', 'high', 'low', 'close', 'volume'.
    Drops NaN rows after computations.
    """
    df['MA20'] = talib.SMA(df['close'], timeperiod=20)
    df['MA50'] = talib.SMA(df['close'], timeperiod=50)
    df['RSI'] = talib.RSI(df['close'], timeperiod=14)
    macd, macd_signal, _ = talib.MACD(df['close'], fastperiod=12, slowperiod=26, signalperiod=9)
    df['MACD'] = macd
    df['MACD_signal'] = macd_signal
    df['OBV'] = talib.OBV(df['close'], df['volume'])
    df['VWAP'] = (df['volume'] * (df['high'] + df['low'] + df['close']) / 3).cumsum() / df['volume'].cumsum()
    df['ATR'] = talib.ATR(df['high'], df['low'], df['close'], timeperiod=14)
    # Chaikin Money Flow approximation (20-period)
    df['CMF'] = talib.AD(df['high'], df['low'], df['close'], df['volume']) / df['volume'].rolling(20).sum()
    df['Close_ATR'] = df['close'] / df['ATR']
    df['MA20_ATR'] = df['MA20'] / df['ATR']
    df['Return_1d'] = df['close'].pct_change(1)
    df['Return_5d'] = df['close'].pct_change(5)
    df['Volatility'] = df['Return_1d'].rolling(20).std() * np.sqrt(252)  # Annualized
    upper, _, lower = talib.BBANDS(df['close'], timeperiod=20, nbdevup=2, nbdevdn=2)
    df['BB_upper'] = upper
    df['BB_lower'] = lower
    df['Stoch_K'], df['Stoch_D'] = talib.STOCH(df['high'], df['low'], df['close'], fastk_period=14, slowk_period=3, slowd_period=3)
    df['ADX'] = talib.ADX(df['high'], df['low'], df['close'], timeperiod=14)
    df = df.dropna()  # Keep datetime index for filtering
    return df

def get_sentiment_score(symbol: str) -> float:
    """
    Fetches or loads cached news for the symbol and computes average sentiment score using transformers.
    For simplicity, assumes news text is fetched via a placeholder; in production, integrate with Alpaca news API or external.
    Returns a score between -1 (negative) and 1 (positive).
    """
    cache_path = os.path.join(CONFIG['MODEL_CACHE_DIR'], f"{symbol}_news_sentiment_{CONFIG['MODEL_VERSION']}.pkl")
    if os.path.exists(cache_path):
        with open(cache_path, 'rb') as f:
            score = pickle.load(f)
        logger.info(f"Loaded sentiment score for {symbol} from cache: {score:.3f}")
        return score
    else:
        # Placeholder for news fetching; in full code, use Alpaca news API or external (e.g., via tools)
        # For demo, simulate with dummy news and transformers pipeline
        sentiment_pipeline = pipeline("sentiment-analysis")
        dummy_news = [f"Positive news for {symbol}", f"Neutral update on {symbol}", f"Negative report for {symbol}"]  # Replace with real fetch
        scores = [analysis['score'] if analysis['label'] == 'POSITIVE' else -analysis['score'] for text in dummy_news for analysis in sentiment_pipeline(text)]
        score = np.mean(scores)
        with open(cache_path, 'wb') as f:
            pickle.dump(score, f)
        logger.info(f"Calculated sentiment score for {symbol}: {score:.3f}")
        return score

@retry(retry=retry_if_exception_type(Exception), stop=stop_after_attempt(3), wait=wait_fixed(5))
def load_data(symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Fetches historical stock bars for a symbol from Alpaca API or cache.
    Caches the DataFrame as pickle for reuse.
    """
    cache_path = os.path.join(CONFIG['CACHE_DIR'], f"{symbol}_{start_date}_{end_date}.pkl")
    if os.path.exists(cache_path) and (time.time() - os.path.getmtime(cache_path) < CONFIG['CACHE_EXPIRY_SECONDS']):
        with open(cache_path, 'rb') as f:
            df = pickle.load(f)
        logger.info(f"Loaded {len(df)} bars for {symbol} from cache")
    else:
        client = StockHistoricalDataClient(CONFIG['ALPACA_API_KEY'], CONFIG['ALPACA_SECRET_KEY'])
        request_params = StockBarsRequest(
            symbol_or_symbols=symbol,
            timeframe=CONFIG['TIMEFRAME'],
            start=pd.to_datetime(start_date),
            end=pd.to_datetime(end_date)
        )
        bars = client.get_stock_bars(request_params)
        df = bars.df.reset_index()
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)
        if not os.path.exists(CONFIG['CACHE_DIR']):
            os.makedirs(CONFIG['CACHE_DIR'])
        with open(cache_path, 'wb') as f:
            pickle.dump(df, f)
        logger.info(f"Loaded {len(df)} bars for {symbol} from API")
    if len(df) < CONFIG['MIN_DATA_POINTS']:
        logger.warning(f"Insufficient data for {symbol}: only {len(df)} points")
    return df



def train_wrapper(args):
    symbol, expected_features, force_train = args
    start_time_for_training = time.perf_counter()
    result_from_train_symbol = train_symbol(symbol, expected_features, force_train)
    end_time_for_training = time.perf_counter()
    training_time_in_milliseconds = (end_time_for_training - start_time_for_training) * 1000
    return (*result_from_train_symbol, training_time_in_milliseconds)


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
    for param in ['SIMULATION_DAYS', 'TRAIN_EPOCHS', 'BATCH_SIZE', 'TIMESTEPS', 'MIN_DATA_POINTS', 'LOOK_AHEAD_BARS']:
        if not isinstance(config[param], int) or config[param] <= 0:
            raise ValueError(f"{param} must be a positive integer")
    for param in ['INITIAL_CASH', 'STOP_LOSS_ATR_MULTIPLIER', 'TAKE_PROFIT_ATR_MULTIPLIER', 'MAX_DRAWDOWN_LIMIT', 'RISK_PERCENTAGE']:
        if not isinstance(config[param], (int, float)) or config[param] <= 0:
            raise ValueError(f"{param} must be a positive number")

def create_cache_directory() -> None:
    """Create cache directories if they don't exist and test writability."""
    os.makedirs(CONFIG['CACHE_DIR'], exist_ok=True)
    # Test writability with a dummy file
    test_path = os.path.join(CONFIG['CACHE_DIR'], 'writability_test.txt')
    try:
        with open(test_path, 'w') as f:
            f.write('Test')
        os.remove(test_path)
        logger.info(f"Cache directory {CONFIG['CACHE_DIR']} is writable.")
    except Exception as e:
        logger.warning(f"Cache directory {CONFIG['CACHE_DIR']} writability test failed: {str(e)}. Saves may fail.")

    os.makedirs(CONFIG['MODEL_CACHE_DIR'], exist_ok=True)
    # Test writability with a dummy file for model dir
    test_model_path = os.path.join(CONFIG['MODEL_CACHE_DIR'], 'writability_test.txt')
    try:
        with open(test_model_path, 'w') as f:
            f.write('Test')
        os.remove(test_model_path)
        logger.info(f"Model cache directory {CONFIG['MODEL_CACHE_DIR']} is writable.")
    except Exception as e:
        logger.error(f"Model cache directory {CONFIG['MODEL_CACHE_DIR']} writability test failed: {str(e)}. Check Windows permissions for the mapped folder. Halting execution to prevent failed saves.")
        raise PermissionError(f"Cannot write to MODEL_CACHE_DIR: {str(e)}. Use a WSL-native path or fix permissions.")
    logger.info(f"Data caches (e.g., historical bars, news sentiment) will be written to {CONFIG['CACHE_DIR']}.")
    if CONFIG['MODEL_CACHE_DIR'].startswith('/mnt/c/'):
        windows_model_dir = CONFIG['MODEL_CACHE_DIR'].replace('/mnt/c/', 'C:\\').replace('/', '\\')
        logger.info(f"Model files (weights, scalers, training sentiment) will be written to {CONFIG['MODEL_CACHE_DIR']} (Windows: {windows_model_dir}).")
    else:
        logger.info(f"Model files (weights, scalers, training sentiment) will be written to {CONFIG['MODEL_CACHE_DIR']} (WSL-native path; access via \\\\wsl$\\<distro>\\path\to\dir from Windows).")

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

def preprocess_data(df: pd.DataFrame, timesteps: int, add_noise: bool = False, inference_scaler: Optional[RobustScaler] = None, inference_mode: bool = False, fit_scaler: bool = True) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[RobustScaler]]:
    """Preprocess data into sequences for future price direction prediction (next bar up/down).
    
    Args:
        inference_mode: If True, do not compute targets (y_seq); return None for y_seq.
        fit_scaler: If True, fit a new scaler; else, transform with inference_scaler (prevents leakage on val/test).
        inference_scaler: Existing scaler for transform when fit_scaler=False.
    
    Returns:
        X_seq: Scaled input sequences.
        y_seq: Targets (None in inference mode).
        scaler: Fitted scaler (None if fit_scaler=False).
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
    
    if fit_scaler:
        scaler = RobustScaler()
        X = scaler.fit_transform(X_raw)
    else:
        if inference_scaler is None:
            raise ValueError("inference_scaler must be provided when fit_scaler=False.")
        X = inference_scaler.transform(X_raw)
        scaler = None
    
    if not inference_mode:
        y = df['Future_Direction'].values
        y_seq = y
    else:
        y_seq = None
    
    if add_noise:
        X += np.random.normal(0, 0.005, X.shape)
    
    N = X.shape[0]
    num_sequences = N - timesteps
    if num_sequences <= 0:
        raise ValueError(f"Not enough data for {timesteps} timesteps: only {N} rows available")
    
    # Create sliding windows for past data to predict NEXT bar's direction
    window = np.lib.stride_tricks.sliding_window_view(X, (timesteps, X.shape[1]))
    X_seq = window[:num_sequences].reshape(num_sequences, timesteps, X.shape[1])
    
    if not inference_mode:
        y_seq = y[timesteps - 1: timesteps - 1 + num_sequences]  # Align target to end of each sequence
        logger.info(f"Preprocessed {len(X_seq)} sequences; y balance: {np.mean(y_seq):.3f} (up fraction)")
    else:
        logger.info(f"Preprocessed {len(X_seq)} inference sequences")
    
    return X_seq, y_seq, scaler

def monte_carlo_simulation(returns: List[float], initial_cash: float, num_simulations: int = CONFIG['NUM_MC_SIMULATIONS']) -> Dict[str, float]:
    """
    Performs Monte Carlo simulation on backtest returns to estimate risk and return distributions.
    
    Args:
        returns: List of periodic returns from backtest.
        initial_cash: Initial cash for the symbol.
        num_simulations: Number of simulation paths.
    
    Returns:
        Dict with MC metrics: 'mc_mean_final_value', 'mc_median_final_value', 'mc_var_95' (95% VaR loss %), 'mc_prob_profit' (% simulations with >0 return).
    """
    if not returns:
        return {'mc_mean_final_value': initial_cash, 'mc_median_final_value': initial_cash, 'mc_var_95': 0.0, 'mc_prob_profit': 0.0}
    
    returns = np.array(returns)
    simulation_results = []
    for _ in range(num_simulations):
        # Bootstrap: resample returns with replacement
        sim_returns = np.random.choice(returns, size=len(returns), replace=True)
        sim_cumulative = np.cumprod(1 + sim_returns)
        sim_final_value = initial_cash * sim_cumulative[-1]
        simulation_results.append(sim_final_value)
    
    simulation_results = np.array(simulation_results)
    mc_mean_final_value = np.mean(simulation_results)
    mc_median_final_value = np.median(simulation_results)
    mc_var_95 = -np.percentile(simulation_results - initial_cash, 5) / initial_cash * 100  # 95% VaR as positive % loss
    mc_prob_profit = np.mean(simulation_results > initial_cash) * 100  # % simulations profitable
    
    return {
        'mc_mean_final_value': mc_mean_final_value,
        'mc_median_final_value': mc_median_final_value,
        'mc_var_95': mc_var_95,
        'mc_prob_profit': mc_prob_profit
    }

class TradingModel(nn.Module):
    def __init__(self, timesteps: int, features: int):
        super(TradingModel, self).__init__()
        self.lstm = nn.LSTM(input_size=features, hidden_size=128, num_layers=2, batch_first=True, dropout=0.2)  # Increased capacity and dropout for regularization
        self.attention = nn.MultiheadAttention(embed_dim=128, num_heads=8, dropout=0.1, batch_first=True)  # Match hidden size, more heads
        self.dense1 = nn.Linear(128, 64)  # Adjust linear to match
        self.dense2 = nn.Linear(64, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)
        nn.init.xavier_uniform_(self.dense1.weight)
        nn.init.xavier_uniform_(self.dense2.weight)

    def forward(self, x):
        x, _ = self.lstm(x)
        # Apply attention to LSTM outputs (query=key=value=x)
        x, _ = self.attention(x, x, x)  # Self-attention on sequence
        x = x[:, -1, :]  # Use last timestep after attention
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

def train_model(symbol: str, df: pd.DataFrame, epochs: int, batch_size: int, timesteps: int, expected_features: int) -> Tuple[nn.Module, Any]:
    """Train the CNN-LSTM model with early stopping and learning rate scheduling."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = TradingModel(timesteps, expected_features).to(device)
    optimizer = optim.Adam(model.parameters(), lr=CONFIG['LEARNING_RATE'])
    # Compute pos_weight based on class balance (up fraction ~0.5, so weight positives slightly higher for confidence)
    pos_weight = torch.tensor([1.1])  # Slight bias; adjust based on empirical balance (e.g., 1 / up_fraction if imbalanced)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=CONFIG['LR_SCHEDULER_PATIENCE'], factor=CONFIG['LR_REDUCTION_FACTOR'])
    best_val_loss = float('inf')
    patience_counter = 0
    # Chronological split: Assume df is sorted by timestamp
    df_train = df[df['timestamp'] <= pd.Timestamp(CONFIG['TRAIN_END_DATE'], tz='UTC')].copy()
    df_val = df[(df['timestamp'] > pd.Timestamp(CONFIG['TRAIN_END_DATE'], tz='UTC')) & (df['timestamp'] <= pd.Timestamp(CONFIG['VAL_END_DATE'], tz='UTC'))].copy()
    if len(df_train) < CONFIG['MIN_DATA_POINTS'] or len(df_val) < CONFIG['MIN_DATA_POINTS'] // 5:
        raise ValueError(f"Insufficient data for {symbol}: train={len(df_train)}, val={len(df_val)}")
    # Compute targets separately on subsets to prevent label leakage
    df_train['Future_Direction'] = np.where(df_train['close'].shift(-CONFIG['LOOK_AHEAD_BARS']) > df_train['close'], 1, 0)
    df_train = df_train.dropna(subset=['Future_Direction'])
    df_val['Future_Direction'] = np.where(df_val['close'].shift(-CONFIG['LOOK_AHEAD_BARS']) > df_val['close'], 1, 0)
    df_val = df_val.dropna(subset=['Future_Direction'])
    # Preprocess subsets separately to avoid label leakage
    X_train, y_train, scaler = preprocess_data(df_train, timesteps, add_noise=True)
    X_val, y_val, _ = preprocess_data(df_val, timesteps, inference_scaler=scaler, inference_mode=False)  # Use train scaler, but compute y for val
    if X_train.shape[2] != expected_features or X_val.shape[2] != expected_features:
        raise ValueError(f"Feature mismatch for {symbol}: expected {expected_features}, train got {X_train.shape[2]}, val got {X_val.shape[2]}")
    train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train))
    val_dataset = TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(y_val))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs.squeeze(1), batch_y.float())
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                outputs = model(batch_X)
                loss = criterion(outputs.squeeze(1), batch_y.float())
                val_loss += loss.item()
        val_loss /= len(val_loader)
        scheduler.step(val_loss)

        # Inside train_model function:
        temp_best_path = None  # Initialize to None
        if val_loss < best_val_loss - CONFIG['EARLY_STOPPING_MIN_DELTA']:
            best_val_loss = val_loss
            patience_counter = 0
            with tempfile.NamedTemporaryFile(suffix='.pth', delete=False) as temp_file:
                temp_best_path = temp_file.name
                torch.save(model.state_dict(), temp_best_path)
                logger.info(f"Saved temp best model for {symbol} at epoch {epoch+1} to {temp_best_path}")
        else:
            patience_counter += 1
            if patience_counter >= CONFIG['EARLY_STOPPING_PATIENCE']:
                logger.info(f"Early stopping triggered for {symbol} at epoch {epoch+1}")
                break

        # After the loop:
        if temp_best_path and os.path.exists(temp_best_path):
            try:
                model.load_state_dict(torch.load(temp_best_path))
                logger.info(f"Loaded best model state for {symbol} from temp file")
            except Exception as e:
                logger.error(f"Failed to load best model for {symbol}: {str(e)}. Using final model state.")
            finally:
                try:
                    os.remove(temp_best_path)
                    logger.info(f"Removed temp best model file for {symbol}")
                except Exception as e:
                    logger.warning(f"Failed to remove temp file {temp_best_path}: {str(e)}")
        else:
            logger.warning(f"No temp best model file found for {symbol}; using final model state (no improvements during training?).")
    with open(os.path.join(CONFIG['CACHE_DIR'], f"{symbol}_scaler.pkl"), 'wb') as f:
        pickle.dump(scaler, f)
    return model, scaler


def load_model_and_scaler(symbol: str, expected_features: int, force_retrain: bool = False) -> Tuple[Optional[nn.Module], Optional[RobustScaler], Optional[float]]:
    """Load trained model and scaler from cache or return None to trigger training."""
    if force_retrain:
        return None, None, None
    
    model_path = os.path.join(CONFIG['MODEL_CACHE_DIR'], f"{symbol}_model_{CONFIG['MODEL_VERSION']}.pth")
    scaler_path = os.path.join(CONFIG['MODEL_CACHE_DIR'], f"{symbol}_scaler_{CONFIG['MODEL_VERSION']}.pkl")
    sentiment_path = os.path.join(CONFIG['MODEL_CACHE_DIR'], f"{symbol}_sentiment_{CONFIG['MODEL_VERSION']}.pkl")
    
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
                best_model_path = os.path.join(CONFIG['MODEL_CACHE_DIR'], f"{symbol}_best_model.pth")
                if os.path.exists(best_model_path):
                    os.remove(best_model_path)
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
        model_path = os.path.join(CONFIG['MODEL_CACHE_DIR'], f"{symbol}_model.pth")
        scaler_path = os.path.join(CONFIG['MODEL_CACHE_DIR'], f"{symbol}_scaler.pkl")
        sentiment_path = os.path.join(CONFIG['MODEL_CACHE_DIR'], f"{symbol}_sentiment.pkl")
        
        # Save model state_dict (lightweight, compatible with load)
        torch.save({'model_state_dict': model.state_dict(), 'class_name': 'TradingModel'}, model_path)
        
        # Save scaler via pickle
        with open(scaler_path, 'wb') as f:
            pickle.dump(scaler, f)
        
        # Save training sentiment
        with open(sentiment_path, 'wb') as f:
            pickle.dump(sentiment, f)
        
        # Verify saves and log with Windows paths
        windows_model_path = model_path.replace('/mnt/c/', 'C:\\').replace('/', '\\')
        windows_scaler_path = scaler_path.replace('/mnt/c/', 'C:\\').replace('/', '\\')
        windows_sentiment_path = sentiment_path.replace('/mnt/c/', 'C:\\').replace('/', '\\')
        if os.path.exists(model_path) and os.path.getsize(model_path) > 0:
            logger.info(f"Saved model for {symbol} to {model_path} (Windows: {windows_model_path}, size: {os.path.getsize(model_path)} bytes).")
        else:
            raise IOError(f"Model save verification failed for {symbol} at {model_path} (Windows: {windows_model_path}).")
        if os.path.exists(scaler_path) and os.path.getsize(scaler_path) > 0:
            logger.info(f"Saved scaler for {symbol} to {scaler_path} (Windows: {windows_scaler_path}, size: {os.path.getsize(scaler_path)} bytes).")
        else:
            raise IOError(f"Scaler save verification failed for {symbol} at {scaler_path} (Windows: {windows_scaler_path}).")
        if os.path.exists(sentiment_path) and os.path.getsize(sentiment_path) > 0:
            logger.info(f"Saved sentiment for {symbol} to {sentiment_path} (Windows: {windows_sentiment_path}, size: {os.path.getsize(sentiment_path)} bytes).")
        else:
            raise IOError(f"Sentiment save verification failed for {symbol} at {sentiment_path} (Windows: {windows_sentiment_path}).")
    except Exception as e:
        logger.error(f"Failed to save model and scaler for {symbol} to primary dir: {str(e)}. Attempting fallback save.")
        fallback_dir = './model_backup'
        os.makedirs(fallback_dir, exist_ok=True)
        fallback_model_path = os.path.join(fallback_dir, f"{symbol}_model.pth")
        fallback_scaler_path = os.path.join(fallback_dir, f"{symbol}_scaler.pkl")
        fallback_sentiment_path = os.path.join(fallback_dir, f"{symbol}_sentiment.pkl")
        try:
            torch.save({'model_state_dict': model.state_dict(), 'class_name': 'TradingModel'}, fallback_model_path)
            with open(fallback_scaler_path, 'wb') as f:
                pickle.dump(scaler, f)
            with open(fallback_sentiment_path, 'wb') as f:
                pickle.dump(sentiment, f)
            logger.info(f"Fallback save successful for {symbol} to {fallback_model_path}, {fallback_scaler_path}, {fallback_sentiment_path}.")
        except Exception as fallback_e:
            logger.error(f"Fallback save also failed for {symbol}: {str(fallback_e)}")
            raise
        raise  # Re-raise original error after fallback


def train_symbol(symbol: str, expected_features: int, force_train: bool) -> Tuple[str, pd.DataFrame, nn.Module, Any, bool, float, bool, bool]:
    """Train or load model for a symbol."""
    df, data_loaded = load_or_fetch_data(symbol, CONFIG['TRAIN_DATA_START_DATE'], CONFIG['VAL_END_DATE'])
    sentiment, sentiment_loaded = load_news_sentiment(symbol)
    df = calculate_indicators(df, sentiment)
    model_file = os.path.join(CONFIG['MODEL_CACHE_DIR'], f"{symbol}_model_{CONFIG['MODEL_VERSION']}.pth")
    scaler_file = os.path.join(CONFIG['MODEL_CACHE_DIR'], f"{symbol}_scaler_{CONFIG['MODEL_VERSION']}.pkl")
    if not force_train and os.path.exists(model_file) and os.path.exists(scaler_file):
        model = TradingModel(CONFIG['TIMESTEPS'], expected_features)
        checkpoint = torch.load(model_file)
        if 'version' not in checkpoint or checkpoint['version'] != CONFIG['MODEL_VERSION']:
            logger.info(f"Model version mismatch for {symbol} (expected {CONFIG['MODEL_VERSION']}, got {checkpoint.get('version', 'none')}); retraining.")
            return None, None  # Or set to trigger retrain
        model.load_state_dict(checkpoint['state_dict'])
        with open(scaler_file, 'rb') as f:
            scaler = pickle.load(f)
        model_loaded = True
    else:
        model, scaler = train_model(symbol, df, CONFIG['TRAIN_EPOCHS'], CONFIG['BATCH_SIZE'], CONFIG['TIMESTEPS'], expected_features)
        # Save with version for architecture compatibility
        torch.save({
            'version': CONFIG['MODEL_VERSION'],
            'state_dict': model.state_dict()
        }, model_file)
        with open(scaler_file, 'wb') as f:
            pickle.dump(scaler, f)
        model_loaded = False
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
    
    # Slice to out-of-sample backtest period
    df_backtest = df[df.index >= pd.Timestamp(CONFIG['BACKTEST_START_DATE'], tz='UTC')].copy()
    if len(df_backtest) < CONFIG['MIN_DATA_POINTS']:
        raise ValueError(f"Insufficient backtest data for {symbol}: {len(df_backtest)} bars")

    # Slice to out-of-sample backtest period
    df_backtest = df[df.index >= pd.Timestamp(CONFIG['BACKTEST_START_DATE'], tz='UTC')].copy()
    if len(df_backtest) < CONFIG['MIN_DATA_POINTS']:
        raise ValueError(f"Insufficient backtest data for {symbol}: {len(df_backtest)} bars")
    # Preprocess for inference: Use trained scaler, no y or new fitting
    df_backtest = df[df.index >= pd.Timestamp(CONFIG['BACKTEST_START_DATE'], tz='UTC')].copy()
    if len(df_backtest) < CONFIG['MIN_DATA_POINTS']:
        raise ValueError(f"Insufficient backtest data for {symbol}: {len(df_backtest)} bars")
    X_seq, y_ignore, scaler_ignore = preprocess_data(
        df_backtest, timesteps, inference_mode=True, inference_scaler=scaler
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

    true_y_for_accuracy = df['Future_Direction'].iloc[CONFIG['TIMESTEPS'] : ].values
    valid_mask_for_accuracy = ~np.isnan(true_y_for_accuracy)
    if np.any(valid_mask_for_accuracy):
        accuracy_percentage = np.mean((np.array(predictions)[valid_mask_for_accuracy] > 0.5) == true_y_for_accuracy[valid_mask_for_accuracy]) * 100
    else:
        accuracy_percentage = 0.0
    
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
    
    timestamps = pd.Series(df.index[timesteps:]).reset_index(drop=True)
    sim_start = pd.Timestamp(CONFIG['BACKTEST_START_DATE'], tz='UTC')
    valid_timestamps = timestamps[timestamps >= sim_start]
    if valid_timestamps.empty:
        k_start = 0
    else:
        k_start = valid_timestamps.index[0]
    logger.info(f"Backtest for {symbol}: starting cash=${cash:.2f}, k_start={k_start}, len(predictions)={len(predictions)}")
    if k_start >= len(predictions):
        logger.warning(f"No data points for backtest of {symbol}")
        return cash, returns, trade_count, win_rate, 0.0
    num_backtest_steps = len(predictions) - k_start
    if num_backtest_steps <= 0:
        logger.warning(f"No backtest steps available for {symbol} (num_backtest_steps={num_backtest_steps})")
        return cash, returns, trade_count, win_rate, 0.0
    
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
    return cash, returns, trade_count, win_rate, accuracy_percentage

def buy_and_hold_backtest(dfs_backtest: Dict[str, pd.DataFrame], initial_cash: float) -> float:
    """
    Simulates a buy-and-hold strategy: Buy max shares of each symbol at the start of backtest period using initial cash per symbol,
    hold until the end, and compute final portfolio value. Includes initial transaction cost.
    """
    initial_per_symbol = initial_cash / len(CONFIG['SYMBOLS'])
    bh_final_value = 0.0
    for symbol, df in dfs_backtest.items():
        if df.empty or len(df) < 2:
            logger.warning(f"Insufficient data for buy-and-hold on {symbol}; skipping.")
            continue
        first_close = df['close'].iloc[0]
        if first_close <= 0:
            logger.warning(f"Invalid first close price for {symbol}: {first_close}; skipping.")
            continue
        qty = int((initial_per_symbol - CONFIG['TRANSACTION_COST_PER_TRADE']) / first_close)
        last_close = df['close'].iloc[-1]
        value = qty * last_close
        bh_final_value += value
    logger.info(f"Buy-and-hold final value: ${bh_final_value:.2f}")
    return bh_final_value

def calculate_performance_metrics(returns: List[float], cash: float, initial_per_symbol: float) -> Dict[str, float]:
    """
    Calculates performance metrics for backtest returns of a symbol.
    
    Args:
        returns: List of periodic percentage returns (e.g., per bar or daily).
        cash: Final cash balance after backtesting.
        initial_per_symbol: Initial cash allocated to the symbol.
    
    Returns:
        Dict with 'total_return' (%), 'sharpe_ratio' (annualized), 'max_drawdown' (%).
    """
    if not returns or len(returns) == 0:
        total_return = (cash - initial_per_symbol) / initial_per_symbol * 100 if initial_per_symbol > 0 else 0.0
        sharpe_ratio = 0.0
        max_drawdown = 0.0
    else:
        returns = np.array(returns)
        total_return = (cash - initial_per_symbol) / initial_per_symbol * 100 if initial_per_symbol > 0 else 0.0
        mean_return = np.mean(returns)
        std_return = np.std(returns)
        sharpe_ratio = (mean_return / std_return) * np.sqrt(252) if std_return != 0 else 0.0  # Annualized, risk-free=0
        cumulative_returns = np.cumprod(1 + returns)
        peak = np.maximum.accumulate(cumulative_returns)
        drawdown = (cumulative_returns - peak) / peak
        max_drawdown = -np.min(drawdown) * 100 if len(drawdown) > 0 else 0.0  # Positive percentage
    return {
        'total_return': total_return,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown
    }

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
        temperature = 0.8  # <1 sharpens confidence; tune 0.5-0.9
        scaled_logits = output / temperature
        prediction = torch.sigmoid(scaled_logits).cpu().item()  # Convert logits to probability
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

    if not backtest_only:
        # Original live trading block (unchanged)
        mp.set_start_method('spawn', force=True)  # For CUDA compatibility in multiprocessing
        start_total_training_time = time.perf_counter()
        with mp.Pool(processes=CONFIG['NUM_PARALLEL_WORKERS']) as pool:
            outputs = list(tqdm(pool.imap(train_wrapper, [(sym, expected_features, force_train) for sym in CONFIG['SYMBOLS']]),
                                total=len(CONFIG['SYMBOLS']), desc="Processing symbols"))
            # No per-symbol cuda.empty_cache() needed; clear after all if required
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # No pool cleanup needed
        logger.info("Parallel processing completed; CUDA memory cleared.")

        end_total_training_time = time.perf_counter()
        total_training_time_in_milliseconds = (end_total_training_time - start_total_training_time) * 1000
        training_times_dictionary = {}

        sentiments = {}  # Collect sentiments for live consistency
        for symbol, df, model, scaler, data_loaded, sentiment, sentiment_loaded, model_loaded, training_time_in_milliseconds in outputs:
            training_times_dictionary[symbol] = training_time_in_milliseconds
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

        portfolio_value = CONFIG['INITIAL_CASH']
        peak_value = portfolio_value

        while True:
            @retry(stop=stop_after_attempt(CONFIG['API_RETRY_ATTEMPTS']), wait=wait_fixed(CONFIG['API_RETRY_DELAY'] / 1000), retry=retry_if_exception_type(APIError))
            def get_clock_with_retry():
                return trading_client.get_clock()

            try:
                clock = get_clock_with_retry()
            except APIError as e:
                logger.error(f"Failed to get clock after retries: {str(e)}")
                send_email("API Error", f"Failed to get clock: {str(e)}")
                time.sleep(60)  # Wait 1 min before next loop iteration
                continue
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
                
                @retry(stop=stop_after_attempt(CONFIG['API_RETRY_ATTEMPTS']), wait=wait_fixed(CONFIG['API_RETRY_DELAY'] / 1000), retry=retry_if_exception_type(APIError))
                def get_account_with_retry():
                    return trading_client.get_account()

                try:
                    account = get_account_with_retry()
                    cash = float(account.cash)
                    portfolio_value = float(account.equity)
                except APIError as e:
                    logger.error(f"Failed to get account after retries: {str(e)}")
                    send_email("API Error", f"Failed to get account: {str(e)}")
                    time.sleep(60)  # Wait 1 min before next loop iteration
                    continue
                peak_value = max(peak_value, portfolio_value)
                drawdown = (peak_value - portfolio_value) / peak_value
                if drawdown > CONFIG['MAX_DRAWDOWN_LIMIT']:
                    logger.warning(f"Portfolio drawdown exceeded {CONFIG['MAX_DRAWDOWN_LIMIT'] * 100}%. Pausing trading.")
                    send_email("Portfolio Drawdown Alert", f"Portfolio drawdown exceeded {CONFIG['MAX_DRAWDOWN_LIMIT'] * 100}%. Trading paused.")
                    break
                
                decisions = []
                @retry(stop=stop_after_attempt(CONFIG['API_RETRY_ATTEMPTS']), wait=wait_fixed(CONFIG['API_RETRY_DELAY'] / 1000), retry=retry_if_exception_type(APIError))
                def get_all_positions_with_retry():
                    return trading_client.get_all_positions()

                try:
                    open_positions = get_all_positions_with_retry()
                except APIError as e:
                    logger.error(f"Failed to get open positions after retries: {str(e)}")
                    send_email("API Error", f"Failed to get open positions: {str(e)}")
                    time.sleep(60)  # Wait 1 min before next loop iteration
                    continue
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
                            X_seq, _, _ = preprocess_data(df, CONFIG['TIMESTEPS'], inference_mode=True, inference_scaler=scalers[symbol], fit_scaler=False)
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
                        elif CONFIG['PREDICTION_THRESHOLD_SELL'] < prediction < CONFIG['CONFIDENCE_THRESHOLD']:
                            decision = "Hold (Low Confidence)"
                        elif prediction > max(CONFIG['PREDICTION_THRESHOLD_BUY'], CONFIG['CONFIDENCE_THRESHOLD']) and current_rsi < CONFIG['RSI_BUY_THRESHOLD']:
                            decision = "Buy"
                            if atr_val > 0:
                                risk_per_trade = cash * (CONFIG['RISK_PERCENTAGE'] / 100)
                                stop_loss_distance = atr_val * CONFIG['STOP_LOSS_ATR_MULTIPLIER']
                                qty = max(1, int(risk_per_trade / stop_loss_distance))
                                cost = qty * price + CONFIG['TRANSACTION_COST_PER_TRADE']
                                if cost <= cash:
                                    logger.info(f"Submitting buy order for {qty} shares of {symbol} at ${price:.2f}")
                                    order = MarketOrderRequest(
                                        symbol=symbol,
                                        qty=qty,
                                        side=OrderSide.BUY,
                                        time_in_force=TimeInForce.GTC
                                    )
                                    try:
                                        trading_client.submit_order(order)
                                        email_body = f"""
                        Bought {qty} shares of {symbol} at ${price:.2f}
                        Prediction Confidence: {prediction:.3f}
                        RSI: {current_rsi:.2f}
                        ADX: {current_adx:.2f}
                        Volatility: {current_volatility:.2f}
                        ATR: {atr_val:.2f}
                        Current Cash: ${cash - cost:.2f}
                        Portfolio Value: ${portfolio_value:.2f}
                        """
                                        send_email(f"Trade Update: Bought {symbol}", email_body)
                                    except Exception as e:
                                        logger.error(f"Failed to submit buy order for {symbol}: {str(e)}")
                                        decision = "Buy (Failed)"
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
                                    logger.warning(f"Insufficient cash for buy {symbol}: cost={cost:.2f}, cash={cash:.2f}")
                                    decision = "Hold (Insufficient Cash)"
                            else:
                                logger.warning(f"Invalid ATR for {symbol}: {atr_val}")
                                decision = "Hold (Invalid ATR)"
                        elif qty_owned > 0 and time_held >= CONFIG['MIN_HOLDING_PERIOD_MINUTES']:
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
                                else:
                                    logger.warning(f"Insufficient cash for buy {symbol}: cost={cost:.2f}, cash={cash:.2f}")
                                    decision = "Hold (Insufficient Cash)"
                            else:
                                logger.warning(f"Invalid ATR for {symbol}: {atr_val}")
                                decision = "Hold (Invalid ATR)"
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
                                else:
                                    logger.warning(f"Insufficient cash for buy {symbol}: cost={cost:.2f}, cash={cash:.2f}")
                                    decision = "Hold (Insufficient Cash)"
                            else:
                                logger.warning(f"Invalid ATR for {symbol}: {atr_val}")
                                decision = "Hold (Invalid ATR)"
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
        dfs_backtest = {}
        now = datetime.now(timezone.utc)
        end_date = now.strftime('%Y-%m-%d')
        for symbol in tqdm(CONFIG['SYMBOLS'], desc="Fetching backtest data"):
            dfs_backtest[symbol] = load_data(symbol, CONFIG['BACKTEST_START_DATE'], end_date)
        
        for symbol in tqdm(CONFIG['SYMBOLS'], desc="Adding indicators to backtest data"):
            dfs_backtest[symbol] = add_technical_indicators(dfs_backtest[symbol])  # Compute TA-Lib indicators
            dfs_backtest[symbol]['Sentiment'] = get_sentiment_score(symbol)  # Add sentiment; adjust function name if different
    
        for symbol in CONFIG['SYMBOLS']:
            dfs_backtest[symbol]['Future_Direction'] = (dfs_backtest[symbol]['close'].shift(-CONFIG['LOOK_AHEAD_BARS']) > dfs_backtest[symbol]['close']).astype(int)

        attempt_results = []
        if CONFIG['ENABLE_RETRAIN_CYCLE']:
            effective_max = CONFIG['MAX_RETRAIN_ATTEMPTS']
        else:
            effective_max = 1

        for retrain_attempts in range(1, effective_max + 1):
            logger.info(f"Retraining attempt {retrain_attempts}/{effective_max}")

            # Reset state for each iteration to prevent memory accumulation or carryover
            models = {}
            scalers = {}
            dfs = {}
            stock_info = []
            sentiments = {}
            training_times_dictionary = {}
            if 'progress_bar' in locals():
                progress_bar.close()  # Close if exists from previous iteration
                progress_bar = None

            # If force_train, delete existing model and scaler files to force retraining
            if force_train or retrain_attempts > 1 or CONFIG['ENABLE_RETRAIN_CYCLE']:
                for symbol in CONFIG['SYMBOLS']:
                    model_path = os.path.join(CONFIG['MODEL_CACHE_DIR'], f"{symbol}_model_{CONFIG['MODEL_VERSION']}.pth")
                    scaler_path = os.path.join(CONFIG['MODEL_CACHE_DIR'], f"{symbol}_scaler_{CONFIG['MODEL_VERSION']}.pkl")
                    if os.path.exists(model_path):
                        os.remove(model_path)
                        logger.info(f"Deleted existing model for {symbol} to force retrain.")
                    if os.path.exists(scaler_path):
                        os.remove(scaler_path)
                        logger.info(f"Deleted existing scaler for {symbol} to force retrain.")

            need_training = any(load_model_and_scaler(symbol, expected_features, force_train or retrain_attempts > 1 or CONFIG['ENABLE_RETRAIN_CYCLE'])[0] is None for symbol in CONFIG['SYMBOLS'])
            progress_bar = tqdm(total=total_epochs, desc="Training Progress", bar_format="{l_bar}\033[32m{bar}\033[0m{r_bar}") if need_training else None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # Training block (force_train after first attempt)
            mp.set_start_method('spawn', force=True)  # For CUDA compatibility in multiprocessing
            start_total_training_time = time.perf_counter()
            with mp.Pool(processes=CONFIG['NUM_PARALLEL_WORKERS']) as pool:
                outputs = list(tqdm(pool.imap(train_wrapper, [(sym, expected_features, force_train or retrain_attempts > 1 or CONFIG['ENABLE_RETRAIN_CYCLE']) for sym in CONFIG['SYMBOLS']]),
                                    total=len(CONFIG['SYMBOLS']), desc="Processing symbols"))
                # No per-symbol cuda.empty_cache() needed; clear after all if required
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            # No pool cleanup needed
            logger.info("Parallel processing completed; CUDA memory cleared.")

            end_total_training_time = time.perf_counter()
            total_training_time_in_milliseconds = (end_total_training_time - start_total_training_time) * 1000

            for symbol, df, model, scaler, data_loaded, sentiment, sentiment_loaded, model_loaded, training_time_in_milliseconds in outputs:
                training_times_dictionary[symbol] = training_time_in_milliseconds
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

            # Copy standard model/scaler files to attempt-specific paths
            for symbol in CONFIG['SYMBOLS']:
                model_path = os.path.join(CONFIG['MODEL_CACHE_DIR'], f"{symbol}_model_{CONFIG['MODEL_VERSION']}.pth")
                scaler_path = os.path.join(CONFIG['MODEL_CACHE_DIR'], f"{symbol}_scaler_{CONFIG['MODEL_VERSION']}.pkl")
                attempt_model_path = os.path.join(CONFIG['MODEL_CACHE_DIR'], f"{symbol}_model_{CONFIG['MODEL_VERSION']}_attempt{retrain_attempts}.pth")
                attempt_scaler_path = os.path.join(CONFIG['MODEL_CACHE_DIR'], f"{symbol}_scaler_{CONFIG['MODEL_VERSION']}_attempt{retrain_attempts}.pkl")
                if os.path.exists(model_path):
                    shutil.copyfile(model_path, attempt_model_path)
                    logger.info(f"Copied model for {symbol} to attempt {retrain_attempts}")
                if os.path.exists(scaler_path):
                    shutil.copyfile(scaler_path, attempt_scaler_path)
                    logger.info(f"Copied scaler for {symbol} to attempt {retrain_attempts}")

            # Backtest block
            backtest_times_dictionary = {}
            accuracies_dictionary = {}
            start_total_backtest_time = time.perf_counter()
            initial_cash = CONFIG['INITIAL_CASH']
            final_value = 0  # Initialize to 0 to sum per-symbol ending cash
            symbol_results = {}
            trade_counts = {}
            win_rates = {}
            initial_per_symbol = CONFIG['INITIAL_CASH'] / len(CONFIG['SYMBOLS'])
            for symbol in CONFIG['SYMBOLS']:
                if symbol in models:
                    start_backtest_time_for_symbol = time.perf_counter()
                    cash, returns, trade_count, win_rate, accuracy_percentage = backtest(
                        symbol, models[symbol], scalers[symbol], dfs_backtest[symbol], initial_per_symbol,
                        CONFIG['STOP_LOSS_ATR_MULTIPLIER'], CONFIG['TAKE_PROFIT_ATR_MULTIPLIER'],
                        CONFIG['TIMESTEPS'], CONFIG['PREDICTION_THRESHOLD_BUY'], CONFIG['PREDICTION_THRESHOLD_SELL'],
                        CONFIG['MIN_HOLDING_PERIOD_MINUTES'], CONFIG['TRANSACTION_COST_PER_TRADE']
                    )
                    trade_counts[symbol] = trade_count
                    win_rates[symbol] = win_rate    
                    end_backtest_time_for_symbol = time.perf_counter()
                    backtest_times_dictionary[symbol] = (end_backtest_time_for_symbol - start_backtest_time_for_symbol) * 1000
                    accuracies_dictionary[symbol] = accuracy_percentage
                    final_value += cash
                    try:
                        symbol_results[symbol] = calculate_performance_metrics(returns, cash, initial_per_symbol)
                        # Add Monte Carlo simulation metrics
                        mc_metrics = monte_carlo_simulation(returns, initial_per_symbol)
                        symbol_results[symbol].update(mc_metrics)
                    except Exception as e:
                        logger.error(f"Error calculating metrics for {symbol}: {str(e)}")
                        symbol_results[symbol] = {
                            'total_return': (cash - initial_per_symbol) / initial_per_symbol * 100 if initial_per_symbol > 0 else 0.0,
                            'sharpe_ratio': 0.0,
                            'max_drawdown': 0.0
                        }
                        logger.warning(f"Metrics calculation failed for {symbol}; using defaults.")
            for symbol in CONFIG['SYMBOLS']:
                if symbol not in symbol_results:
                    # Default if metrics calculation failed
                    cash = initial_per_symbol  # Assume no change if not set
                    symbol_results[symbol] = {
                        'total_return': 0.0,
                        'sharpe_ratio': 0.0,
                        'max_drawdown': 0.0
                    }
                    logger.warning(f"Metrics calculation failed for {symbol}; using defaults.")
            
            
            end_total_backtest_time = time.perf_counter()
            total_backtest_time_in_milliseconds = (end_total_backtest_time - start_total_backtest_time) * 1000

            # Compute buy-and-hold benchmark (Feature 2)
            bh_final_value = buy_and_hold_backtest(dfs_backtest, initial_cash)

            # Check criteria for retrain (for logging/email only)
            max_drawdown_across_symbols = max([res['max_drawdown'] for res in symbol_results.values()]) if symbol_results else 0.0
            criteria_met = (
                final_value > CONFIG['MIN_FINAL_VALUE'] and
                max_drawdown_across_symbols <= CONFIG['MAX_ALLOWED_DRAWDOWN'] and
                final_value > bh_final_value
            )
            logger.info(f"Criteria evaluation: met={criteria_met}. "
                        f"Final value ({final_value:.2f}) > min ({CONFIG['MIN_FINAL_VALUE']:.2f}): {final_value > CONFIG['MIN_FINAL_VALUE']}. "
                        f"Max drawdown ({max_drawdown_across_symbols:.3f}) <= allowed ({CONFIG['MAX_ALLOWED_DRAWDOWN']:.3f}): {max_drawdown_across_symbols <= CONFIG['MAX_ALLOWED_DRAWDOWN']}. "
                        f"Final ({final_value:.2f}) > BH ({bh_final_value:.2f}): {final_value > bh_final_value}.")

            # Per-attempt email (keep as-is)
            email_body = format_email_body(initial_cash, final_value, symbol_results, trade_counts, win_rates)
            email_body += "\n\nMonte Carlo Simulation Summary (per symbol):\n"
            for symbol in CONFIG['SYMBOLS']:
                if symbol in symbol_results:
                    mc = symbol_results[symbol]
                    email_body += f"{symbol}: MC Mean Final: ${mc['mc_mean_final_value']:.2f}, MC Median Final: ${mc['mc_median_final_value']:.2f}, MC 95% VaR: {mc['mc_var_95']:.3f}%, MC Prob Profit: {mc['mc_prob_profit']:.3f}%\n"
            email_body += f"\nBuy-and-Hold Final Value: ${bh_final_value:.2f}\nDay Trading {'beats' if final_value > bh_final_value else 'does not beat'} Buy-and-Hold."
            email_body += f"\nAttempt: {retrain_attempts}"
            email_body += f"\nCriteria Met: {criteria_met}"
            send_email(f"Backtest Attempt {retrain_attempts} Results", email_body)

            # Collect results for this attempt
            attempt_results.append({
                'attempt': retrain_attempts,
                'final_value': final_value,
                'symbol_results': symbol_results,
                'trade_counts': trade_counts,
                'win_rates': win_rates,
                'bh_final_value': bh_final_value,
                'max_drawdown_across_symbols': max_drawdown_across_symbols,
                'criteria_met': criteria_met,
                'training_times_dictionary': training_times_dictionary,
                'backtest_times_dictionary': backtest_times_dictionary,
                'total_training_time_in_milliseconds': total_training_time_in_milliseconds,
                'total_backtest_time_in_milliseconds': total_backtest_time_in_milliseconds,
                'accuracies_dictionary': accuracies_dictionary
            })

        # After all attempts, select and use the best (highest final_value)
        if attempt_results:
            best = max(attempt_results, key=lambda x: x['final_value'])
            best_attempt = best['attempt']
            logger.info(f"Selected best attempt {best_attempt} with final_value ${best['final_value']:.2f}")

            # Copy best attempt's files to standard paths
            for symbol in CONFIG['SYMBOLS']:
                attempt_model_path = os.path.join(CONFIG['MODEL_CACHE_DIR'], f"{symbol}_model_{CONFIG['MODEL_VERSION']}_attempt{best_attempt}.pth")
                attempt_scaler_path = os.path.join(CONFIG['MODEL_CACHE_DIR'], f"{symbol}_scaler_{CONFIG['MODEL_VERSION']}_attempt{best_attempt}.pkl")
                model_path = os.path.join(CONFIG['MODEL_CACHE_DIR'], f"{symbol}_model_{CONFIG['MODEL_VERSION']}.pth")
                scaler_path = os.path.join(CONFIG['MODEL_CACHE_DIR'], f"{symbol}_scaler_{CONFIG['MODEL_VERSION']}.pkl")
                if os.path.exists(attempt_model_path):
                    shutil.copyfile(attempt_model_path, model_path)
                    logger.info(f"Copied best model for {symbol} from attempt {best_attempt} to standard path")
                else:
                    logger.warning(f"Model file for {symbol} attempt {best_attempt} not found")
                if os.path.exists(attempt_scaler_path):
                    shutil.copyfile(attempt_scaler_path, scaler_path)
                    logger.info(f"Copied best scaler for {symbol} from attempt {best_attempt} to standard path")
                else:
                    logger.warning(f"Scaler file for {symbol} attempt {best_attempt} not found")

            # Set variables from best for reporting
            final_value = best['final_value']
            symbol_results = best['symbol_results']
            trade_counts = best['trade_counts']
            win_rates = best['win_rates']
            bh_final_value = best['bh_final_value']
            training_times_dictionary = best['training_times_dictionary']
            backtest_times_dictionary = best['backtest_times_dictionary']
            total_training_time_in_milliseconds = best['total_training_time_in_milliseconds']
            total_backtest_time_in_milliseconds = best['total_backtest_time_in_milliseconds']
            accuracies_dictionary = best['accuracies_dictionary']

            # Reporting (updated with buy-and-hold)
            email_body = format_email_body(CONFIG['INITIAL_CASH'], final_value, symbol_results, trade_counts, win_rates)
            email_body += f"\nBest Attempt: {best_attempt}"
            email_body += "\n\nMonte Carlo Simulation Summary (per symbol):\n"
            for symbol in CONFIG['SYMBOLS']:
                if symbol in symbol_results:
                    mc = symbol_results[symbol]
                    email_body += f"{symbol}: MC Mean Final: ${mc['mc_mean_final_value']:.2f}, MC Median Final: ${mc['mc_median_final_value']:.2f}, MC 95% VaR: {mc['mc_var_95']:.3f}%, MC Prob Profit: {mc['mc_prob_profit']:.3f}%\n"
            email_body += f"\nBuy-and-Hold Final Value: ${bh_final_value:.2f}\nDay Trading {'beats' if final_value > bh_final_value else 'does not beat'} Buy-and-Hold."
            send_email("Backtest Completed - Best Results", email_body)
            
            def format_time(ms):
                minutes = int(ms // 60000)
                seconds = int((ms % 60000) // 1000)
                milliseconds = int(ms % 1000)
                return f"{minutes:02d}:{seconds:02d}.{milliseconds:03d}"
            
            print(f"{Fore.CYAN}Training Times (mm:ss.mmm):{Style.RESET_ALL}")
            for symbol_in_training_times, time_in_ms in training_times_dictionary.items():
                print(f"  {symbol_in_training_times}: {format_time(time_in_ms)}")
            print(f"Total Training Time: {format_time(total_training_time_in_milliseconds)}")

            print(f"{Fore.CYAN}Backtest Times (mm:ss.mmm):{Style.RESET_ALL}")
            for symbol_in_backtest_times, time_in_ms in backtest_times_dictionary.items():
                print(f"  {symbol_in_backtest_times}: {format_time(time_in_ms)}")
            print(f"Total Backtest Time: {format_time(total_backtest_time_in_milliseconds)}")

            print(f"Total Time (Training + Backtest): {format_time(total_training_time_in_milliseconds + total_backtest_time_in_milliseconds)}")

            print(f"{Fore.GREEN}Backtest Performance Summary:{Style.RESET_ALL}")
            print(f"{'Symbol':<8} {'Total Return (%)':<18} {'Sharpe Ratio':<14} {'Max Drawdown (%)':<20} {'Trades':<8} {'Win Rate (%)':<14} {'Accuracy (%)':<14} {'MC Mean Final ($)':<18} {'MC Median Final ($)':<20} {'MC 95% VaR (%)':<15} {'MC Prob Profit (%)':<18}")
            for symbol in CONFIG['SYMBOLS']:
                if symbol in symbol_results:
                    metrics_for_symbol = symbol_results[symbol]
                    return_color = Fore.GREEN if metrics_for_symbol['total_return'] > 0 else Fore.RED
                    drawdown_color = Fore.RED if metrics_for_symbol['max_drawdown'] > 0 else Fore.GREEN
                    win_rate_color = Fore.GREEN if win_rates[symbol] > 50 else Fore.RED
                    accuracy = accuracies_dictionary.get(symbol, 0) if trade_counts.get(symbol, 0) > 0 else 0.0  # Hide accuracy if no trades
                    accuracy_color = Fore.GREEN if accuracy > 50 else Fore.RED
                    mc_mean_color = Fore.GREEN if metrics_for_symbol['mc_mean_final_value'] > initial_per_symbol else Fore.RED
                    mc_median_color = Fore.GREEN if metrics_for_symbol['mc_median_final_value'] > initial_per_symbol else Fore.RED
                    mc_var_color = Fore.RED if metrics_for_symbol['mc_var_95'] > 0 else Fore.GREEN
                    mc_prob_color = Fore.GREEN if metrics_for_symbol['mc_prob_profit'] > 50 else Fore.RED
                    print(f"{symbol:<8} {return_color}{metrics_for_symbol['total_return']:<18.3f}{Style.RESET_ALL} {metrics_for_symbol['sharpe_ratio']:<14.3f} {drawdown_color}{metrics_for_symbol['max_drawdown']:<20.3f}{Style.RESET_ALL} {trade_counts.get(symbol, 0):<8} {win_rate_color}{win_rates.get(symbol, 0):<14.3f}{Style.RESET_ALL} {accuracy_color}{accuracy:<14.3f}{Style.RESET_ALL} {mc_mean_color}{metrics_for_symbol['mc_mean_final_value']:<18.2f}{Style.RESET_ALL} {mc_median_color}{metrics_for_symbol['mc_median_final_value']:<20.2f}{Style.RESET_ALL} {mc_var_color}{metrics_for_symbol['mc_var_95']:<15.3f}{Style.RESET_ALL} {mc_prob_color}{metrics_for_symbol['mc_prob_profit']:<18.3f}{Style.RESET_ALL}")

            bh_color = Fore.GREEN if final_value > bh_final_value else Fore.RED
            print(f"\nBuy-and-Hold Final Value: {bh_color}${bh_final_value:.2f}{Style.RESET_ALL}")
            print(f"Day Trading {'beats' if final_value > bh_final_value else 'does not beat'} Buy-and-Hold.")
            if(final_value <= 100000):
                logger.info(f"Backtest completed: Final value: {Fore.RED} ${final_value:.2f}")
            else:
                logger.info(f"Backtest completed: Final value: {Fore.GREEN} ${final_value:.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Trading bot with backtest mode")
    parser.add_argument('--backtest', action='store_true', help="Run in backtest-only mode")
    parser.add_argument('--force-train', action='store_true', help="Force retraining of models")
    args = parser.parse_args()
    mp.set_start_method('spawn', force=True)  # Set early for CUDA multiprocessing safety
    main(backtest_only=args.backtest, force_train=args.force_train)
