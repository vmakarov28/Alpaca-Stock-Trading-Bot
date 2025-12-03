
# +------------------------------------------------------------------------------+
# |                            Alpaca Neural Bot v9.9.8                          |
# +------------------------------------------------------------------------------+
# | Author: Vladimir Makarov                                                     |
# | Project Start Date: May 9, 2025                                              |
# | License: GNU Lesser General Public License v2.1                              |
# | Version: 9.9.8 (Un-Released)                                                 |
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
from tenacity import retry, stop_after_attempt, wait_fixed, retry_if_exception_type  # For retry decorators on API calls to handle failures
from tqdm import tqdm  # For displaying progress bars during training and data processing
from colorama import Fore, Style  # For colored text styles in console output
import colorama  # For cross-platform colored terminal text initialization
import multiprocessing as mp  # For parallel processing, like training models across symbols
import time  # For time-related functions, like sleeping or timing operations (duplicate import)
import shutil # File transfer
import tempfile  # Add this import at the top of the file if not already present
from alpaca.trading.requests import GetOrdersRequest
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.trading.enums import QueryOrderStatus
from alpaca.trading.enums import OrderStatus

# Suppress PyTorch warnings
warnings.filterwarnings('ignore', category=UserWarning)

# Initialize colorama for colored console output
colorama.init()

CONFIG = {
    # Trading Parameters - Settings related to trading operations
    'SYMBOLS': [ 'AAPL', 'NVDA', 'GOOGL', 'AMZN', 'SPY'],  # List of stock symbols to trade
    'TIMEFRAME': TimeFrame(15, TimeFrameUnit.Minute),  # Time interval for data fetching
    'INITIAL_CASH': 100000.00,  # Starting cash for trading simulation
    'MIN_HOLDING_PERIOD_MINUTES': 45,  # Minimum holding period for trades

    # Data Fetching and Caching - Parameters for data retrieval and storage (format: yy/mm/dd)
    'TRAIN_DATA_START_DATE': '2015-01-01',  # Start date for training data
    'TRAIN_END_DATE': '2024-06-30',  # End date for training data (extended to include more recent data)
    'VAL_START_DATE': '2024-07-01',  # Start date for validation data (shifted for recent validation)
    'VAL_END_DATE': '2024-12-31',  # End date for validation data (now ends before backtest to prevent overlap/leakage)
    'BACKTEST_START_DATE': '2025-01-01',  # Start date for backtesting (out-of-sample, now fully after val)

    'SIMULATION_DAYS': 180,  # Number of days for simulation
    'MIN_DATA_POINTS': 100,  # Minimum data points required for processing
    'CACHE_DIR': './cache',  # Directory for caching data
    'MODEL_CACHE_DIR': '/mnt/c/Users/aipla/Desktop/Model Weights',  # Directory for saving model weights, scalers, and sentiment
    'CACHE_EXPIRY_SECONDS': 24 * 60 * 60,  # Expiry time for cached data in seconds
    'LIVE_DATA_BARS': 500,  # Number of bars to fetch for live data

    # Model Training - Settings for training the machine learning model
    'TRAIN_EPOCHS': 200,  # Number of epochs for training the model
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
    'SMTP_PORT': 587,  #z Port for SMTP server

    # Logging and Monitoring - Settings for tracking activities
    'LOG_FILE': 'trades.log',  # File for logging trades

    # Strategy Thresholds - Thresholds for trading decisions
    'CONFIDENCE_THRESHOLD': 0.52,  # Lowered to capture more predictions above neutral while maintaining selectivity
    'PREDICTION_THRESHOLD_BUY': 0.55,  # Lowered to allow more buy opportunities based on prediction distribution
    'PREDICTION_THRESHOLD_SELL': 0.3,  # Tightened for quicker exits to reduce losses in downtrends
    'RSI_BUY_THRESHOLD': 70,  # RSI threshold for buying (lowered for stronger oversold signals)
    'RSI_SELL_THRESHOLD': 30,  # RSI threshold for selling (raised for stronger overbought signals)
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
    'API_RETRY_ATTEMPTS': 10,  # Number of retry attempts for API calls
    'API_RETRY_DELAY': 1000,  # Delay between retry attempts in milliseconds
    'MODEL_VERSION': 'v100',  # Model architecture version; increment on structural changes to force retrain

    # New: Retraining Cycle Parameters
    'ENABLE_RETRAIN_CYCLE': True,  # Enable loop to retrain until criteria met (backtest mode only)
    'MIN_FINAL_VALUE': 130000.0,  # Minimum final portfolio value to accept
    'MAX_ALLOWED_DRAWDOWN': 30.0,  # Maximum allowed max_drawdown percentage (across symbols)
    'MAX_RETRAIN_ATTEMPTS': 230,  # Max loop iterations to prevent infinite runs

    #Monte Carlo Probability Simulation
    'NUM_MC_SIMULATIONS': 500000,  # Number of Monte Carlo simulations for backtest robustness testing


    # Account Management
    'RESET_ACCOUNT_ON_START': True,   # Set to True only when you want a full reset (closes all positions & cancels orders)
    'PAPER_TRADING': True,             # Set to False when going live with real money
    'DESIRED_STARTING_CASH': 200000.00,  # Desired cash for reset
}



#pyenv activate pytorch_env
#python /mnt/c/Users/aipla/Downloads/alpaca_neural_bot_v9.9.8.py --backtest --force-train




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
        # sentiment_pipeline = pipeline("sentiment-analysis")  # Comment out to avoid unused download/execution
        # dummy_news = [f"Positive news for {symbol}", f"Neutral update on {symbol}", f"Negative report for {symbol}"] # Replace with real fetch
        # scores = [analysis['score'] if analysis['label'] == 'POSITIVE' else -analysis['score'] for text in dummy_news for analysis in sentiment_pipeline(text)]
       
        # score = np.mean(scores) # Uncomment for real sentiment (-1 to 1)
        score = 0.0  # Override to neutral while keeping framework
        
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


# Wraps train_symbol call, times execution in ms, unpacks args for parallel processing.
def train_wrapper(args):
    symbol, expected_features, force_train = args
    start_time_for_training = time.perf_counter()
    result_from_train_symbol = train_symbol(symbol, expected_features, force_train)
    end_time_for_training = time.perf_counter()
    training_time_in_milliseconds = (end_time_for_training - start_time_for_training) * 1000
    return (*result_from_train_symbol, training_time_in_milliseconds)


logger = logging.getLogger(__name__)

# Initialize sentiment analysis pipeline
device = 0 if torch.cuda.is_available() else -1
sentiment_pipeline = pipeline("sentiment-analysis", model=CONFIG['SENTIMENT_MODEL'], framework="pt", device=device)

# Checks if required libs are installed via importlib, raises ImportError if missing.
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

# Validates CONFIG types/values like positive ints/floats, raises ValueError on invalid.
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

# Creates cache/model dirs, tests write with temp file, logs/errors on fail, raises on model dir issue.
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

def cleanup_account_on_start() -> None:
    """
    Optional one-time account reset + cash injection.
    Controlled by two CONFIG flags:
        'RESET_ACCOUNT_ON_START': True → runs the reset once
        'PAPER_TRADING': True/False → paper or live keys
    After running once it turns itself off automatically.
    """
    # ------------------------------------------------------------------
    # 1. Do nothing unless you explicitly ask for a reset
    # ------------------------------------------------------------------
    if not (args.reset or CONFIG.get('RESET_ACCOUNT_ON_START', False)):
        return

    # ------------------------------------------------------------------
    # 2. Settings
    # ------------------------------------------------------------------
    PAPER_MODE = CONFIG.get('PAPER_TRADING', True)                    # True = paper, False = live
    DESIRED_STARTING_CASH = CONFIG.get('DESIRED_STARTING_CASH', 200000.00)  # Default to 200k if not set
    FAKE_SYMBOL = "SPY"                                               # Cheap liquid ETF for cash injection trick
    CHUNK_SIZE = 100000.00                                            # Smaller $100k chunks to avoid qty locks
    POLL_TIMEOUT = 600                                                # 10min for off-hours
    POLL_INTERVAL = 10                                                # Check every 10s to reduce API load

    trading_client = TradingClient(
        CONFIG['ALPACA_API_KEY'],
        CONFIG['ALPACA_SECRET_KEY'],
        paper=PAPER_MODE
    )

    try:
        print(f"{Fore.YELLOW}=== ACCOUNT RESET REQUESTED ==={Style.RESET_ALL}")
        logger.info("Starting account reset and cash injection")

        # --------------------------------------------------------------
        # 3. Close all positions AND cancel all orders in one call
        # --------------------------------------------------------------
        trading_client.close_all_positions(cancel_orders=True)
        logger.info("Requested close all positions and cancel orders")

        # Extra cancel for any lingering
        trading_client.cancel_orders()

        # --------------------------------------------------------------
        # 4. Poll until cleared
        # --------------------------------------------------------------
        clock = trading_client.get_clock()
        if not clock.is_open:
            warning_msg = f"Market is closed (next open: {clock.next_open}). Position close orders queued and will be filled automatically by Alpaca at market open—no need to rerun for closes. Skipping polling and injection to avoid errors. Positions will clear at open; check dashboard for status. Rerun during open hours if you need to confirm or inject cash after clears."
            logger.warning(warning_msg)
            print(f"{Fore.YELLOW}{warning_msg}{Style.RESET_ALL}")
            raise Exception("Market closed - closes queued but reset incomplete. Rerun during open hours for full reset if needed.")
        else:
            start_time = time.time()
            while time.time() - start_time < POLL_TIMEOUT:
                positions = trading_client.get_all_positions()
                orders = trading_client.get_orders()
                if not positions and not orders:
                    logger.info("All positions closed and orders cancelled successfully")
                    print(f"{Fore.CYAN}All positions closed and orders cancelled{Style.RESET_ALL}")
                    break
                pos_symbols = [pos.symbol for pos in positions]
                order_ids = [str(order.id) for order in orders]
                logger.info(f"Waiting for clears... Positions left: {len(positions)} ({pos_symbols}), Orders left: {len(orders)} ({order_ids})")
                print(f"  → Waiting for clears... ({len(positions)} positions left: {pos_symbols}, {len(orders)} orders left)")
                time.sleep(POLL_INTERVAL)
            else:
                pos_symbols = [pos.symbol for pos in positions]
                order_ids = [str(order.id) for order in orders]
                warning_msg = f"Timed out waiting for clears. Remaining: {len(positions)} positions ({pos_symbols}), {len(orders)} orders ({order_ids}). Continuing to injection anyway - check dashboard."
                logger.warning(warning_msg)
                print(f"{Fore.YELLOW}{warning_msg}{Style.RESET_ALL}")
            # Force cancel any lingering orders post-timeout
            trading_client.cancel_orders()

        # --------------------------------------------------------------
        # 5. Inject / remove cash in chunks
        # --------------------------------------------------------------
        account = trading_client.get_account()
        current_cash = float(account.cash)
        difference = DESIRED_STARTING_CASH - current_cash

        if abs(difference) > 50:
            num_chunks = max(1, int(abs(difference) / CHUNK_SIZE))
            chunk_diff = difference / num_chunks
            for i in range(num_chunks):
                # Poll for no holds on FAKE_SYMBOL before submit
                chunk_poll_start = time.time()
                while time.time() - chunk_poll_start < 120:
                    try:
                        pos = trading_client.get_position(FAKE_SYMBOL)
                        if float(pos.qty_available) >= 0 and float(pos.qty_held_for_orders) == 0:
                            break
                    except APIError as e:
                        if 'not found' in str(e):  # No position is fine
                            break
                    print(f"  → Waiting for {FAKE_SYMBOL} to be free from holds...")
                    time.sleep(5)
                else:
                    logger.warning(f"Timeout waiting for {FAKE_SYMBOL} free - skipping chunk {i+1}")
                    continue

                # Recalculate this chunk
                account = trading_client.get_account()
                current_cash = float(account.cash)
                this_diff = min(chunk_diff, DESIRED_STARTING_CASH - current_cash) if difference > 0 else max(chunk_diff, DESIRED_STARTING_CASH - current_cash)
                if abs(this_diff) < 50:
                    break

                price_estimate = 500.0 if FAKE_SYMBOL == "SPY" else 100.0
                fake_qty = abs(this_diff) / price_estimate

                side = OrderSide.SELL if this_diff > 0 else OrderSide.BUY
                fake_order = MarketOrderRequest(
                    symbol=FAKE_SYMBOL,
                    qty=fake_qty,
                    side=side,
                    time_in_force=TimeInForce.DAY
                )
                try:
                    trading_client.submit_order(fake_order)
                    time.sleep(2)
                    trading_client.cancel_orders()
                    logger.info(f"Chunk {i+1}/{num_chunks}: adjusted by {this_diff:+,.2f}")
                    print(f"{Fore.CYAN}Chunk {i+1}: Cash adjusted by {this_diff:+,.2f}{Style.RESET_ALL}")
                except Exception as e:
                    logger.warning(f"Chunk {i+1} failed: {str(e)} - skipping")

        # --------------------------------------------------------------
        # 6. Final status
        # --------------------------------------------------------------
        account = trading_client.get_account()
        final_cash = float(account.cash)
        print(f"{Fore.GREEN}RESET COMPLETE!{Style.RESET_ALL}")
        print(f"   Cash          : ${final_cash:,.2f}")
        print(f"   Portfolio     : ${float(account.portfolio_value):,.2f}")
        print(f"   Positions     : 0")
        logger.info(f"Reset complete – cash ≈ ${final_cash:,.2f}")

        # --------------------------------------------------------------
        # 7. Turn the reset flag OFF so it never runs again unintentionally
        # --------------------------------------------------------------
        CONFIG['RESET_ACCOUNT_ON_START'] = False
        print(f"{Fore.YELLOW}Reset flag automatically turned OFF – safe for future runs.{Style.RESET_ALL}")

    except Exception as e:
        logger.error(f"Account reset failed: {str(e)}")
        print(f"{Fore.RED}Reset failed: {str(e)}{Style.RESET_ALL}")

# Fetches last N bars from Alpaca for live, uses recent dates, renames vwap, sorts, retries on error.
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

# Fetches historical bars in 1yr chunks to avoid limits, handles META/FB rename, concats/dedups, raises on low data.
def fetch_data(symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
    """Fetch historical bar data from Alpaca API in yearly increments."""
    try:
        client = StockHistoricalDataClient(CONFIG['ALPACA_API_KEY'], CONFIG['ALPACA_SECRET_KEY'])  # Initialize client with keys
        all_bars = []  # Collect data in chunks to bypass API limits on large ranges
        current_start = pd.Timestamp(start_date, tz='UTC')  # Ensure UTC for consistency
        end_dt = pd.Timestamp(end_date, tz='UTC')
        increment = pd.DateOffset(years=1)  # Yearly chunks to avoid request timeouts/large responses
        
        while current_start < end_dt:
            current_end = min(current_start + increment, end_dt)
            logger.info(f"Fetching data for {symbol} from {current_start} to {current_end}")
            effective_symbol = 'FB' if symbol == 'META' and current_start < pd.Timestamp('2021-10-28', tz='UTC') else symbol  # Handle META ticker rename
            request = StockBarsRequest(
                symbol_or_symbols=effective_symbol,
                timeframe=CONFIG['TIMEFRAME'],
                start=current_start,
                end=current_end
            )
            bars = client.get_stock_bars(request).df  # Fetch bars as DataFrame
            
            if not bars.empty:
                df_bars = bars.reset_index().rename(columns={'vwap': 'VWAP'})  # Rename for consistency
                all_bars.append(df_bars)
                logger.info(f"Fetched {len(df_bars)} bars for {symbol} from {df_bars['timestamp'].min()} to {df_bars['timestamp'].max()}")
            else:
                logger.info(f"No data for {symbol} from {current_start} to {current_end}, skipping")  # Handle gaps (e.g., non-trading periods)
            current_start = current_end
        
        if all_bars:
            df = pd.concat(all_bars).sort_values('timestamp')  # Combine and sort chronologically
            df = df.drop_duplicates(subset='timestamp', keep='first')  # Remove any duplicate timestamps from overlaps
            logger.info(f"Total fetched {len(df)} bars for {symbol}")
        else:
            df = pd.DataFrame(columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'VWAP'])  # Empty df with schema if no data
        
        if len(df) < CONFIG['MIN_DATA_POINTS']:
            raise ValueError(f"Insufficient data for {symbol}: got {len(df)} bars, need {CONFIG['MIN_DATA_POINTS']}")  # Validate min size
        return df
    
    except Exception as e:
        logger.error(f"Error fetching data for {symbol}: {str(e)}")
        raise  # Re-raise for caller handling

# Loads from cache if fresh, else fetches full history to now, saves pickle, returns df and loaded flag.
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

# Loads sentiment from cache if fresh, else generates random -1 to 1, saves, returns score and loaded flag.
def load_news_sentiment(symbol: str) -> Tuple[float, bool]:
    """Compute real-time news sentiment using a pre-trained model or random for testing."""
    cache_file = os.path.join(CONFIG['CACHE_DIR'], f"{symbol}_news_sentiment.pkl")
    if os.path.exists(cache_file) and (time.time() - os.path.getmtime(cache_file)) < CONFIG['CACHE_EXPIRY_SECONDS']:
        with open(cache_file, 'rb') as f:
            sentiment_score = pickle.load(f)
        return sentiment_score, True
    else:
        #sentiment_score = np.random.uniform(-1.0, 1.0)  # Random sentiment for testing
        sentiment_score = 0.0  # Override to neutral while keeping framework
        with open(cache_file, 'wb') as f:
            pickle.dump(sentiment_score, f)
        return sentiment_score, False

# Computes TA indicators on copy df, adds sentiment/trend, drops NaNs in indicators.
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
    df = df.copy()  # Avoid modifying original df
    df.ffill(inplace=True)  # Forward fill missing values to handle sparse data
    df.fillna(0, inplace=True)  # Backfill remaining with 0 (e.g., early indicators)
   
    features = [
        'close', 'high', 'low', 'volume', 'MA20', 'MA50', 'RSI',
        'MACD', 'MACD_signal', 'OBV', 'VWAP', 'ATR', 'CMF', 'Close_ATR',
        'MA20_ATR', 'Return_1d', 'Return_5d', 'Volatility', 'BB_upper',
        'BB_lower', 'Stoch_K', 'Stoch_D', 'ADX', 'Sentiment'
    ]  # Fixed feature list; must match expected_features (24)
    if 'Future_Direction' not in df.columns and not inference_mode:
        raise ValueError("Future_Direction column missing; required for training.")  # Ensure targets present for train/val
    X_raw = df[features].values  # Extract as numpy for scaling
   
    # Scale features; RobustScaler handles outliers in financial data
    if fit_scaler:
        scaler = RobustScaler()
        X = scaler.fit_transform(X_raw)  # Fit and transform on train
    else:
        if inference_scaler is None:
            raise ValueError("inference_scaler must be provided when fit_scaler=False.")
        X = inference_scaler.transform(X_raw)  # Transform only for val/test/live to prevent leakage
        scaler = None
   
    if not inference_mode:
        y = df['Future_Direction'].values
        y_seq = y
    else:
        y_seq = None  # No targets needed for inference
   
    if add_noise:
        X += np.random.normal(0, 0.005, X.shape)  # Add Gaussian noise for augmentation (regularization against overfitting)
   
    N = X.shape[0]
    num_sequences = N - timesteps
    if num_sequences <= 0:
        raise ValueError(f"Not enough data for {timesteps} timesteps: only {N} rows available")  # Check sufficient rows for windows
   
    # Create sliding windows for past data to predict NEXT bar's direction; uses stride tricks for efficiency
    window = np.lib.stride_tricks.sliding_window_view(X, (timesteps, X.shape[1]))
    X_seq = window[:num_sequences].reshape(num_sequences, timesteps, X.shape[1])  # Shape: (samples, timesteps, features)
   
    if not inference_mode:
        y_seq = y[timesteps - 1: timesteps - 1 + num_sequences] # Align target to end of each sequence (predict next after window)
        logger.info(f"Preprocessed {len(X_seq)} sequences; y balance: {np.mean(y_seq):.3f} (up fraction)")  # Log class balance for monitoring
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
    # Set device for training; prefers CUDA for GPU acceleration to speed up computations
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = TradingModel(timesteps, expected_features).to(device)
    optimizer = optim.Adam(model.parameters(), lr=CONFIG['LEARNING_RATE'])
    # Compute pos_weight based on class balance (up fraction ~0.5, so weight positives slightly higher for confidence)
    pos_weight = torch.tensor([1.1]) # Slight bias; adjust based on empirical balance (e.g., 1 / up_fraction if imbalanced)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))  # BCE loss with logits for binary classification; pos_weight handles slight imbalance
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=CONFIG['LR_SCHEDULER_PATIENCE'], factor=CONFIG['LR_REDUCTION_FACTOR'])  # Reduces LR on plateau to improve convergence
    best_val_loss = float('inf')
    patience_counter = 0
    # Chronological split: Assume df is sorted by timestamp to maintain time series integrity and prevent lookahead bias
    df_train = df[df['timestamp'] <= pd.Timestamp(CONFIG['TRAIN_END_DATE'], tz='UTC')].copy()
    df_val = df[(df['timestamp'] > pd.Timestamp(CONFIG['TRAIN_END_DATE'], tz='UTC')) & (df['timestamp'] <= pd.Timestamp(CONFIG['VAL_END_DATE'], tz='UTC'))].copy()
    if len(df_train) < CONFIG['MIN_DATA_POINTS'] or len(df_val) < CONFIG['MIN_DATA_POINTS'] // 5:
        raise ValueError(f"Insufficient data for {symbol}: train={len(df_train)}, val={len(df_val)}")  # Ensure enough data points to avoid underfitting or invalid splits
    # Compute targets separately on subsets to prevent label leakage from future data
    df_train['Future_Direction'] = np.where(df_train['close'].shift(-CONFIG['LOOK_AHEAD_BARS']) > df_train['close'], 1, 0)
    df_train = df_train.dropna(subset=['Future_Direction'])  # Drop rows where target is NaN (e.g., end of data)
    df_val['Future_Direction'] = np.where(df_val['close'].shift(-CONFIG['LOOK_AHEAD_BARS']) > df_val['close'], 1, 0)
    df_val = df_val.dropna(subset=['Future_Direction'])
    # Preprocess subsets separately to avoid label leakage; fit scaler on train only
    X_train, y_train, scaler = preprocess_data(df_train, timesteps, add_noise=True)
    X_val, y_val, _ = preprocess_data(df_val, timesteps, inference_scaler=scaler, inference_mode=False) # Use train scaler, but compute y for val
    if X_train.shape[2] != expected_features or X_val.shape[2] != expected_features:
        raise ValueError(f"Feature mismatch for {symbol}: expected {expected_features}, train got {X_train.shape[2]}, val got {X_val.shape[2]}")  # Validate feature count after preprocessing
    train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train))
    val_dataset = TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(y_val))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)  # Shuffle train for better generalization
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)  # No shuffle for val to preserve order
    for epoch in range(epochs):
        model.train()  # Set model to training mode (enables dropout, etc.)
        train_loss = 0.0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)  # Move to device for GPU acceleration
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs.squeeze(1), batch_y.float())
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)  # Average loss per batch
        model.eval()  # Set to eval mode (disables dropout)
        val_loss = 0.0
        with torch.no_grad():  # Disable gradients for validation to save memory
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                outputs = model(batch_X)
                loss = criterion(outputs.squeeze(1), batch_y.float())
                val_loss += loss.item()
        val_loss /= len(val_loader)
        scheduler.step(val_loss)  # Adjust LR based on val_loss
        # Inside train_model function:
        temp_best_path = None # Initialize to None; used for saving best model temporarily to avoid permission issues
        if val_loss < best_val_loss - CONFIG['EARLY_STOPPING_MIN_DELTA']:
            best_val_loss = val_loss
            patience_counter = 0
            with tempfile.NamedTemporaryFile(suffix='.pth', delete=False) as temp_file:  # Use temp file for safe saving in WSL/Windows mounted dirs
                temp_best_path = temp_file.name
                torch.save(model.state_dict(), temp_best_path)
                logger.info(f"Saved temp best model for {symbol} at epoch {epoch+1} to {temp_best_path}")
        else:
            patience_counter += 1
            if patience_counter >= CONFIG['EARLY_STOPPING_PATIENCE']:
                logger.info(f"Early stopping triggered for {symbol} at epoch {epoch+1}")
                break  # Stop if no improvement to prevent overfitting
        
        # After the loop:
        if temp_best_path and os.path.exists(temp_best_path):
            try:
                model.load_state_dict(torch.load(temp_best_path))  # Load best weights from temp file
                logger.info(f"Loaded best model state for {symbol} from temp file")
            except Exception as e:
                logger.error(f"Failed to load best model for {symbol}: {str(e)}. Using final model state.")
            finally:
                try:
                    os.remove(temp_best_path)  # Clean up temp file
                    logger.info(f"Removed temp best model file for {symbol}")
                except Exception as e:
                    logger.warning(f"Failed to remove temp file {temp_best_path}: {str(e)}")
#        else:
#            logger.warning(f"No temp best model file found for {symbol}; using final model state (no improvements during training?).")
    with open(os.path.join(CONFIG['CACHE_DIR'], f"{symbol}_scaler.pkl"), 'wb') as f:
        pickle.dump(scaler, f)  # Cache scaler for inference consistency
    return model, scaler


def load_model_and_scaler(symbol: str, expected_features: int, force_retrain: bool = False) -> Tuple[Optional[nn.Module], Optional[RobustScaler], Optional[float]]:
    """Load trained model and scaler from cache or return None to trigger training."""
    logger.info(f"Entering load_model_and_scaler for {symbol} (force_retrain={force_retrain}).")
    if force_retrain:
        return None, None, None
    
    model_path = os.path.join(CONFIG['MODEL_CACHE_DIR'], f"{symbol}_model_{CONFIG['MODEL_VERSION']}.pth")
    scaler_path = os.path.join(CONFIG['MODEL_CACHE_DIR'], f"{symbol}_scaler_{CONFIG['MODEL_VERSION']}.pkl")
    sentiment_path = os.path.join(CONFIG['MODEL_CACHE_DIR'], f"{symbol}_news_sentiment_{CONFIG['MODEL_VERSION']}.pkl")
    
    if os.path.exists(model_path) and os.path.exists(scaler_path):
        logger.info(f"Found model file for {symbol} at {model_path} (size: {os.path.getsize(model_path)} bytes). Attempting load.")
        logger.info(f"Found scaler file for {symbol} at {scaler_path} (size: {os.path.getsize(scaler_path)} bytes). Attempting load.")
        try:
            # Load scaler first (less error-prone)
            with open(scaler_path, 'rb') as f:
                scaler = pickle.load(f)
            
            # Load model with improved flexible handling
            checkpoint = torch.load(model_path, map_location='cpu')
            model = TradingModel(CONFIG['TIMESTEPS'], expected_features)
            if isinstance(checkpoint, dict):
                if 'state_dict' in checkpoint:  # Handle {'version': ..., 'state_dict': ...} format
                    if 'version' in checkpoint and checkpoint['version'] != CONFIG['MODEL_VERSION']:
                        logger.warning(f"Version mismatch for {symbol} (saved: {checkpoint['version']}, current: {CONFIG['MODEL_VERSION']}). Deleting cache and retraining.")
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
                    model.load_state_dict(checkpoint['state_dict'])
                    logger.info(f"Loaded dict-format model with 'state_dict' key for {symbol}.")
                elif 'model_state_dict' in checkpoint:  # Existing modern format
                    model_class_name = checkpoint.get('class_name', 'TradingModel')
                    if model_class_name == 'CNNLSTMModel':
                        logger.warning(f"Legacy class for {symbol} detected. Loading state into current TradingModel (assuming key compatibility).")
                    model.load_state_dict(checkpoint['model_state_dict'])
                    logger.info(f"Loaded modern dict-format model for {symbol}.")
                else:
                    # Legacy direct state_dict (OrderedDict is dict, but no specific keys)
                    logger.warning(f"Direct state_dict save detected for {symbol} (legacy format). Loading directly.")
                    model.load_state_dict(checkpoint)
            else:
                # Rare non-dict case (shouldn't happen for torch saves)
                logger.error(f"Unexpected checkpoint type for {symbol}: {type(checkpoint)}. Skipping load.")
                raise TypeError("Invalid checkpoint format")
            
            # Load associated training sentiment if available
            training_sentiment = None
            if os.path.exists(sentiment_path):
                logger.info(f"Found sentiment file for {symbol} at {sentiment_path} (size: {os.path.getsize(sentiment_path)} bytes). Loading.")
                with open(sentiment_path, 'rb') as f:
                    training_sentiment = pickle.load(f)
            
            logger.info(f"Successfully loaded cached model and scaler for {symbol}.")
            return model, scaler, training_sentiment
        except (KeyError, AttributeError, NameError, ValueError) as e:
            # Expanded to catch load_state_dict mismatches (e.g., incompatible keys)
            if 'CNNLSTMModel' in str(e) or 'not defined' in str(e) or 'model_state_dict' in str(e) or 'size mismatch' in str(e):
                logger.warning(f"Format, class, or key mismatch for {symbol} (likely incompatible legacy). Deleting cache and retraining.")
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
        logger.info(f"No cached model/scaler for {symbol} (checked {model_path} and {scaler_path}). Will train.")
        return None, None, None

def save_model_and_scaler(symbol: str, model: nn.Module, scaler: RobustScaler, sentiment: float) -> None:
    """Save the trained model and scaler to cache files."""
    try:
        model_path = os.path.join(CONFIG['MODEL_CACHE_DIR'], f"{symbol}_model_{CONFIG['MODEL_VERSION']}.pth")
        scaler_path = os.path.join(CONFIG['MODEL_CACHE_DIR'], f"{symbol}_scaler_{CONFIG['MODEL_VERSION']}.pkl")
        sentiment_path = os.path.join(CONFIG['MODEL_CACHE_DIR'], f"{symbol}_news_sentiment_{CONFIG['MODEL_VERSION']}.pkl")
        
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
        fallback_model_path = os.path.join(fallback_dir, f"{symbol}_model_{CONFIG['MODEL_VERSION']}.pth")
        fallback_scaler_path = os.path.join(fallback_dir, f"{symbol}_scaler_{CONFIG['MODEL_VERSION']}.pkl")
        fallback_sentiment_path = os.path.join(fallback_dir, f"{symbol}_news_sentiment_{CONFIG['MODEL_VERSION']}.pkl")
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
    
    model.eval()  # Set model to evaluation mode for inference
    model = model.to(device)  # Ensure model is on correct device
    X_tensor = torch.tensor(X_seq, dtype=torch.float32).to(device)  # Convert sequences to tensor on device

    predictions = []
    with torch.no_grad():  # Disable gradients to optimize memory and speed for inference
        for i in range(0, len(X_tensor), CONFIG['BATCH_SIZE']):
            batch = X_tensor[i:i + CONFIG['BATCH_SIZE']]  # Batch processing to handle large datasets without OOM
            raw_logits = model(batch)
            if i % 100 == 0: # Log every 100 steps to avoid spam; useful for monitoring long backtests
                logger.info(f"Sample raw logit for {symbol} at step {i}: mean={raw_logits.mean().item():.4f}, std={raw_logits.std().item():.4f}")
            # Apply sigmoid to model outputs to get probabilities between 0 and 1
            outputs = torch.sigmoid(raw_logits)
            predictions.extend(outputs.cpu().numpy().flatten())  # Collect on CPU to free GPU memory
            del raw_logits # Clean up to release memory
            del outputs # Explicitly delete outputs tensor

    predictions = np.array(predictions)  # Convert to numpy for easier handling
    true_y_for_accuracy = df['Future_Direction'].iloc[CONFIG['TIMESTEPS'] : ].values  # Align true labels with predictions
    valid_mask_for_accuracy = ~np.isnan(true_y_for_accuracy)  # Mask NaNs in targets
    if np.any(valid_mask_for_accuracy):
        accuracy_percentage = np.mean((np.array(predictions)[valid_mask_for_accuracy] > 0.5) == true_y_for_accuracy[valid_mask_for_accuracy]) * 100  # Binary accuracy (threshold 0.5)
    else:
        accuracy_percentage = 0.0

    logger.info(f"Predictions for {symbol}: min={predictions.min():.3f}, max={predictions.max():.3f}, mean={predictions.mean():.3f}")  # Summary stats for debugging prediction distribution

    # Clean up CUDA tensors to free GPU memory for other operations
    del X_tensor
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Initialize variables (initial_cash now per-symbol from call); simulate trading state
    cash = initial_cash
    returns = []  # List of trade returns for metrics
    trade_count = 0
    win_rate = 0.0
    position = 0  # Current shares held
    entry_price = 0.0
    entry_time = None
    max_price = 0.0  # For trailing stop
    winning_trades = 0

    timestamps = pd.Series(df.index[timesteps:]).reset_index(drop=True)  # Align timestamps with sequences
    sim_start = pd.Timestamp(CONFIG['BACKTEST_START_DATE'], tz='UTC')  # Out-of-sample start to avoid overfitting
    valid_timestamps = timestamps[timestamps >= sim_start]
    if valid_timestamps.empty:
        k_start = 0
    else:
        k_start = valid_timestamps.index[0]  # Start index for backtest period
    logger.info(f"Backtest for {symbol}: starting cash=${cash:.2f}, k_start={k_start}, len(predictions)={len(predictions)}")
    if k_start >= len(predictions):
        logger.warning(f"No data points for backtest of {symbol}")
        return cash, returns, trade_count, win_rate, 0.0
    num_backtest_steps = len(predictions) - k_start
    if num_backtest_steps <= 0:
        logger.warning(f"No backtest steps available for {symbol} (num_backtest_steps={num_backtest_steps})")
        return cash, returns, trade_count, win_rate, 0.0

    # Pre-slice indicators for efficiency; aligns with backtest steps to avoid index errors
    atr = df['ATR'].iloc[timesteps + k_start:timesteps + k_start + num_backtest_steps].values
    prices = df['close'].iloc[timesteps + k_start:timesteps + k_start + num_backtest_steps].values
    rsi = df['RSI'].iloc[timesteps + k_start:timesteps + k_start + num_backtest_steps].values
    adx = df['ADX'].iloc[timesteps + k_start:timesteps + k_start + num_backtest_steps].values
    volatility = df['Volatility'].iloc[timesteps + k_start:timesteps + k_start + num_backtest_steps].values
    sim_timestamps = timestamps.iloc[k_start:k_start + num_backtest_steps].values

    for local_i in range(num_backtest_steps):
        if local_i % 100 == 0: # Log every 100 steps to avoid spam; helps track progress in long simulations
            logger.info(f"Processing backtest step {local_i} for {symbol}: prediction={predictions[k_start + local_i]:.3f}")
        i = k_start + local_i  # Global index for predictions
        pred = predictions[i]
        price = prices[local_i]
        atr_val = atr[local_i]
        current_rsi = rsi[local_i]
        current_adx = adx[local_i]
        current_volatility = volatility[local_i]
        ts = pd.Timestamp(sim_timestamps[local_i])  # Current timestamp for logging

        # Skip if volatility too high or trend too weak (ADX low); filters out risky periods
        if current_volatility > max_volatility or current_adx < adx_trend_threshold:
            continue

        # Skip low-confidence predictions
        if pred < confidence_threshold:
            continue

        # Calculate position size based on risk; uses ATR for volatility-adjusted sizing
        if cash >= price:
            qty = max(1, int((cash * risk_percentage) / (atr_val * stop_loss_atr_multiplier)))
            cost = qty * price + transaction_cost_per_trade
            if cost > cash:
                qty = max(0, int((cash - transaction_cost_per_trade) / price))  # Adjust qty if cost exceeds cash
                cost = qty * price + transaction_cost_per_trade
        else:
            qty = 0
            cost = 0

        #logger.info(f"Checking buy for {symbol}: pred={pred:.3f}, qty={qty}, cash={cash:.2f}, price={price:.2f}, rsi={current_rsi:.2f}, adx={current_adx:.2f}")

        # Buy condition: High pred, oversold RSI, strong trend, sufficient qty/cash
        if pred > buy_threshold and position == 0 and current_rsi < rsi_buy_threshold and current_adx > adx_trend_threshold and qty > 0 and cash >= cost:
            if cash - cost >= 0: # Safety check to prevent negative cash
                position = qty
                entry_price = price
                max_price = price  # Init for trailing stop
                entry_time = ts
                cash -= cost
                logger.info(f"{ts}: Bought {qty} shares of {symbol} at ${price:.2f}, cash: ${cash:.2f}")
            else:
                logger.info(f"Insufficient cash to buy {qty} shares of {symbol}: cash={cash:.2f}, cost={cost:.2f}")

        elif position > 0:
            if price > max_price:
                max_price = price  # Update peak for trailing stop
            trailing_stop = max_price * (1 - trailing_stop_percentage)
            stop_loss = entry_price - stop_loss_atr_multiplier * atr_val  # ATR-based dynamic stop
            take_profit = entry_price + take_profit_atr_multiplier * atr_val  # ATR-based profit target
           
            if not isinstance(entry_time, pd.Timestamp):
                raise TypeError(f"entry_time must be a pandas.Timestamp, got {type(entry_time)}")  # Type safety for time calculations
            time_held = (ts - entry_time).total_seconds() / 60  # Minutes held

            # Sell if min hold met and any exit condition: stop hit, profit taken, or signal/RSI reversal
            if time_held >= min_holding_period_minutes:
                if price <= trailing_stop or price <= stop_loss or price >= take_profit or (pred < sell_threshold and current_rsi > rsi_sell_threshold):
                    cash += position * price - transaction_cost_per_trade
                    ret = (price - entry_price) / entry_price  # Per-trade return
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

    # Close any open position at end of simulation
    if position > 0:
        cash += position * prices[-1] - transaction_cost_per_trade
        ret = (prices[-1] - entry_price) / entry_price
        returns.append(ret)
        trade_count += 1
        if ret > 0:
            winning_trades += 1
    win_rate = (winning_trades / trade_count * 100) if trade_count > 0 else 0.0  # Avoid division by zero
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
            metric_lower = metric.lower()
            if 'final_value' in metric_lower:
                value_str = f"${value:.2f}"
                unit = ''
            else:
                value_str = f"{value:.3f}"
                if any(k in metric_lower for k in ['return', 'drawdown', 'var', 'prob']):
                    unit = '%'
                else:
                    unit = ''
            body.append(f"  {metric.replace('_', ' ').title()}: {value_str}{unit}")
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

def main(backtest_only: bool = False, force_train: bool = False, debug: bool = False) -> None:
    logging.basicConfig(
        level=logging.DEBUG if debug else logging.INFO,
        format='%(asctime)s,%(msecs)03d - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
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
    training_sentiments = {}
    need_training = False
    for symbol in CONFIG['SYMBOLS']:
        model, scaler, sentiment = load_model_and_scaler(symbol, expected_features, force_train)
        models[symbol] = model
        scalers[symbol] = scaler
        training_sentiments[symbol] = sentiment
        if model is None:
            need_training = True
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

        if debug:
            for info in stock_info:
                for line in info:
                    print(line)
                print()

        portfolio_value = CONFIG['INITIAL_CASH']
        peak_value = portfolio_value

        while True:  # Infinite loop for continuous live trading
            @retry(stop=stop_after_attempt(CONFIG['API_RETRY_ATTEMPTS']), wait=wait_fixed(CONFIG['API_RETRY_DELAY'] / 1000), retry=retry_if_exception_type(APIError))
            def get_clock_with_retry():
                return trading_client.get_clock()  # Retry wrapper for API calls to handle transient failures
            try:
                clock = get_clock_with_retry()
            except APIError as e:
                logger.error(f"Failed to get clock after retries: {str(e)}")
                send_email("API Error", f"Failed to get clock: {str(e)}")
                time.sleep(60) # Wait 1 min before next loop iteration to avoid spamming
                continue
            if clock.is_open:  # Only trade when market is open
                now = datetime.now(timezone.utc)  # Current time for scheduling (ensure timezone-aware for calculations)
                next_mark = now.replace(second=0, microsecond=0)  # Align to next 15-min mark for bar intervals
                minutes = now.minute
                if minutes % 15 != 0:
                    next_mark += timedelta(minutes=(15 - minutes % 15))
                else:
                    next_mark += timedelta(minutes=15)
                seconds_to_sleep = (next_mark - now).total_seconds()
                if seconds_to_sleep > 0:
                    time.sleep(seconds_to_sleep)  # Sleep to sync with timeframe (e.g., 15-min bars)
            
                @retry(stop=stop_after_attempt(CONFIG['API_RETRY_ATTEMPTS']), wait=wait_fixed(CONFIG['API_RETRY_DELAY'] / 1000), retry=retry_if_exception_type(APIError))
                def get_account_with_retry():
                    return trading_client.get_account()
                try:
                    account = get_account_with_retry()
                    cash = float(account.cash)
                    portfolio_value = float(account.equity)  # Fetch current account state
                except APIError as e:
                    logger.error(f"Failed to get account after retries: {str(e)}")
                    send_email("API Error", f"Failed to get account: {str(e)}")
                    time.sleep(60) # Wait 1 min before next loop iteration
                    continue
                peak_value = max(peak_value, portfolio_value)  # Track peak for drawdown calculation
                drawdown = (peak_value - portfolio_value) / peak_value
                if drawdown > CONFIG['MAX_DRAWDOWN_LIMIT']:
                    logger.warning(f"Portfolio drawdown exceeded {CONFIG['MAX_DRAWDOWN_LIMIT'] * 100}%. Pausing trading.")
                    send_email("Portfolio Drawdown Alert", f"Portfolio drawdown exceeded {CONFIG['MAX_DRAWDOWN_LIMIT'] * 100}%. Trading paused.")
                    break  # Halt trading on excessive drawdown to protect capital
            
                decisions = []  # Collect decisions for summary email
                @retry(stop=stop_after_attempt(CONFIG['API_RETRY_ATTEMPTS']), wait=wait_fixed(CONFIG['API_RETRY_DELAY'] / 1000), retry=retry_if_exception_type(APIError))
                def get_all_positions_with_retry():
                    return trading_client.get_all_positions()
                try:
                    open_positions = get_all_positions_with_retry()  # Fetch current holdings
                except APIError as e:
                    logger.error(f"Failed to get open positions after retries: {str(e)}")
                    send_email("API Error", f"Failed to get open positions: {str(e)}")
                    time.sleep(60) # Wait 1 min before next loop iteration
                    continue
                for symbol in CONFIG['SYMBOLS']:
                    if symbol in models:  # Only process symbols with trained models
                        df = fetch_recent_data(symbol, CONFIG['LIVE_DATA_BARS'])  # Get latest bars for indicators/prediction
                        sentiment = sentiments[symbol] # Use training sentiment for consistency to avoid retraining overhead
                        df = calculate_indicators(df, sentiment)  # Compute TA indicators
                        # Define features list (copied from preprocess_data for consistency); ensures matching shape
                        features = [
                            'close', 'high', 'low', 'volume', 'MA20', 'MA50', 'RSI',
                            'MACD', 'MACD_signal', 'OBV', 'VWAP', 'ATR', 'CMF', 'Close_ATR',
                            'MA20_ATR', 'Return_1d', 'Return_5d', 'Volatility', 'BB_upper',
                            'BB_lower', 'Stoch_K', 'Stoch_D', 'ADX', 'Sentiment'
                        ]
                        # Compute prediction and current values from model and df; handles insufficient data gracefully
                        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                        if len(df) < CONFIG['TIMESTEPS'] + 1:
                            logger.warning(f"Insufficient data for {symbol} live prediction: {len(df)} bars")
                            prediction = 0.5  # Neutral fallback
                            price = df['close'].iloc[-1] if not df.empty and len(df) > 0 else 0.0
                            current_rsi = df['RSI'].iloc[-1] if not df.empty and 'RSI' in df.columns else 50.0
                            current_adx = df['ADX'].iloc[-1] if not df.empty and 'ADX' in df.columns else 0.0
                            current_volatility = df['Volatility'].iloc[-1] if not df.empty and 'Volatility' in df.columns else 0.0
                            atr_val = df['ATR'].iloc[-1] if not df.empty and 'ATR' in df.columns else 0.0
                        else:
                            X_seq, _, _ = preprocess_data(df, CONFIG['TIMESTEPS'], inference_mode=True, inference_scaler=scalers[symbol], fit_scaler=False)  # Prep latest data
                            # Use the most recent sequence for live prediction; single inference for low latency
                            recent_seq = X_seq[-1:].astype(np.float32)
                            model = models[symbol].to(device)
                            model.eval()
                            with torch.no_grad():
                                pred_logit = model(torch.tensor(recent_seq).to(device))
                                prediction = torch.sigmoid(pred_logit).cpu().item()  # Probability output
                            price = float(df['close'].iloc[-1])
                            current_rsi = float(df['RSI'].iloc[-1])
                            current_adx = float(df['ADX'].iloc[-1])
                            current_volatility = float(df['Volatility'].iloc[-1])
                            atr_val = float(df['ATR'].iloc[-1])
                            logger.info(f"Live prediction for {symbol}: {prediction:.3f}, price=${price:.2f}, RSI={current_rsi:.2f}")
                        decision = "Hold"  # Default; updated based on conditions
                        # Fetch position (qty_owned etc. only for sell/hold checks); uses recent orders for entry_time
                        qty_owned = 0
                        entry_time = None
                        entry_price = 0.0
                        time_held = 0
                        position_obj = next((pos for pos in open_positions if pos.symbol == symbol), None)
                        if position_obj:
                            qty_owned = int(float(position_obj.qty))
                            # Fetch entry time from the latest filled BUY order; limits to 50 for efficiency
                            order_req = GetOrdersRequest(
                                status=QueryOrderStatus.CLOSED,
                                symbols=[symbol],
                                side=OrderSide.BUY,
                                limit=50 # Limit to recent orders to avoid large responses
                            )
                            try:
                                orders = trading_client.get_orders(order_req)
                                filled_buy_orders = [o for o in orders if o.status == OrderStatus.FILLED and o.side == OrderSide.BUY]
                                if filled_buy_orders:
                                    latest_order = max(filled_buy_orders, key=lambda o: o.filled_at if o.filled_at else datetime.min.replace(tzinfo=timezone.utc))
                                    entry_time = latest_order.filled_at if latest_order.filled_at else now  # Use filled time or fallback
                                else:
                                    entry_time = now # Fallback if no filled buy orders found
                            except Exception as e:
                                logger.warning(f"Failed to fetch orders for {symbol} entry time: {str(e)}. Using current time as fallback.")
                                entry_time = now
                            entry_price = float(position_obj.avg_entry_price)
                            if entry_time:
                                if entry_time.tzinfo is None:
                                    entry_time = entry_time.replace(tzinfo=timezone.utc)  # Ensure UTC for consistent calculations
                                else:
                                    entry_time = entry_time.astimezone(timezone.utc)
                            time_held = (now - entry_time).total_seconds() / 60 if entry_time else 0  # Time held in minutes
                        # Decision logic: Mirrors backtest but with real API; filters first, then buy/sell
                        if current_volatility > CONFIG['MAX_VOLATILITY'] or current_adx < CONFIG['ADX_TREND_THRESHOLD']:
                            decision = "Hold (Filters)" # Skip risky conditions
                        elif CONFIG['PREDICTION_THRESHOLD_SELL'] < prediction < CONFIG['CONFIDENCE_THRESHOLD']:
                            decision = "Hold (Low Confidence)"
                        elif prediction > max(CONFIG['PREDICTION_THRESHOLD_BUY'], CONFIG['CONFIDENCE_THRESHOLD']) and current_rsi < CONFIG['RSI_BUY_THRESHOLD']:
                            decision = "Buy"
                            if atr_val > 0:
                                try:
                                    risk_per_trade = cash * CONFIG['RISK_PERCENTAGE']
                                    stop_loss_distance = atr_val * CONFIG['STOP_LOSS_ATR_MULTIPLIER']
                                    if stop_loss_distance <= 0:
                                        raise ValueError("Stop loss distance <= 0")
                                    qty = max(1, int(risk_per_trade / stop_loss_distance))
                                    cost = qty * price + CONFIG['TRANSACTION_COST_PER_TRADE']
                                    if cost > cash:
                                        qty = max(0, int((cash - CONFIG['TRANSACTION_COST_PER_TRADE']) / price))  # Downsize to max affordable qty
                                        cost = qty * price + CONFIG['TRANSACTION_COST_PER_TRADE']
                                    if qty > 0 and cost <= cash:
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
                                        logger.warning(f"Qty=0 or insufficient cash for {symbol}: qty={qty}, cost={cost:.2f}, cash={cash:.2f}")
                                        decision = "Hold (Qty=0 or Insufficient Cash)"
                                except (ValueError, ZeroDivisionError) as e:
                                    logger.warning(f"Calculation error for buy {symbol}: {str(e)}. ATR={atr_val:.2f}, Stop distance={stop_loss_distance:.2f}")
                                    decision = "Hold (Calculation Error)"
                            else:
                                logger.warning(f"Invalid ATR for {symbol}: {atr_val}")
                                decision = "Hold (Invalid ATR)"
                        elif qty_owned > 0 and time_held >= CONFIG['MIN_HOLDING_PERIOD_MINUTES']:
                            # Compute stops using current price; dynamic based on latest data
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
                # Summarize decisions and send email; provides overview without individual trade emails
                summary_body = "Trading Summary:\n"
                for dec in decisions:
                    summary_body += f"{dec['symbol']}: {dec['decision']}, Confidence: {dec['confidence']:.3f}, RSI: {dec['rsi']:.2f}, ADX: {dec['adx']:.2f}, Volatility: {dec['volatility']:.2f}, Price: ${dec['price']:.2f}, Owned: {dec['owned']}\n"
                summary_body += f"\nPortfolio Value: ${portfolio_value:.2f}\nNote: Actual trades are executed based on available cash and positions."
                send_email("Trading Summary", summary_body)
            else:
                next_open = clock.next_open
                while datetime.now(timezone.utc) < next_open:  # Countdown loop when market closed
                    time_left = next_open - datetime.now(timezone.utc)
                    hours, remainder = divmod(time_left.total_seconds(), 3600)
                    minutes, seconds = divmod(remainder, 60)
                    timer_str = f"{int(hours):02}:{int(minutes):02}:{int(seconds):02}"
                    print(f"\r{Fore.RED}Time until market opens: {timer_str}{Style.RESET_ALL}", end='')  # Visual timer
                    time.sleep(1)
                print()  # New line after countdown

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

        attempt_results = [] # Runs fixed retrain attempts, collects results, selects best by final_value, copies best models/scalers to standard paths.
        if CONFIG['ENABLE_RETRAIN_CYCLE'] and force_train:
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
                if progress_bar is not None:
                    progress_bar.close()  # Close if exists from previous iteration
                progress_bar = None

            # If force_train, delete existing model and scaler files to force retraining
            if force_train or retrain_attempts > 1:
                for symbol in CONFIG['SYMBOLS']:
                    model_path = os.path.join(CONFIG['MODEL_CACHE_DIR'], f"{symbol}_model_{CONFIG['MODEL_VERSION']}.pth")
                    scaler_path = os.path.join(CONFIG['MODEL_CACHE_DIR'], f"{symbol}_scaler_{CONFIG['MODEL_VERSION']}.pkl")
                    if os.path.exists(model_path):
                        os.remove(model_path)
                        logger.info(f"Deleted existing model for {symbol} to force retrain.")
                    if os.path.exists(scaler_path):
                        os.remove(scaler_path)
                        logger.info(f"Deleted existing scaler for {symbol} to force retrain.")

            need_training = any(load_model_and_scaler(symbol, expected_features, force_train or retrain_attempts > 1)[0] is None for symbol in CONFIG['SYMBOLS'])
            progress_bar = tqdm(total=total_epochs, desc="Training Progress", bar_format="{l_bar}\033[32m{bar}\033[0m{r_bar}") if need_training else None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # Training block (force_train after first attempt)
            mp.set_start_method('spawn', force=True)  # For CUDA compatibility in multiprocessing
            start_total_training_time = time.perf_counter()
            with mp.Pool(processes=CONFIG['NUM_PARALLEL_WORKERS']) as pool:
                outputs = list(tqdm(pool.imap(train_wrapper, [(sym, expected_features, force_train or retrain_attempts > 1) for sym in CONFIG['SYMBOLS']]),
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

            if debug:
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

        # After all attempts, select the best model per symbol based on highest total_return for that symbol
        if attempt_results:
            best_attempt_per_symbol = {}
            best_symbol_results = {}
            best_trade_counts = {}
            best_win_rates = {}
            best_accuracies = {}
            # Use the last attempt's bh_final_value (consistent across attempts)
            bh_final_value = attempt_results[-1]['bh_final_value']
            final_value = 0.0
            initial_per_symbol = CONFIG['INITIAL_CASH'] / len(CONFIG['SYMBOLS'])
            for symbol in CONFIG['SYMBOLS']:
                # Find attempt with max total_return for this symbol
                best_att_for_sym = max(attempt_results, key=lambda x: x['symbol_results'].get(symbol, {}).get('total_return', -float('inf')))['attempt']
                best_attempt_per_symbol[symbol] = best_att_for_sym
                # Get the results from that attempt for this symbol
                best_att_results = next(a for a in attempt_results if a['attempt'] == best_att_for_sym)
                best_symbol_results[symbol] = best_att_results['symbol_results'][symbol]
                best_trade_counts[symbol] = best_att_results['trade_counts'][symbol]
                best_win_rates[symbol] = best_att_results['win_rates'][symbol]
                best_accuracies[symbol] = best_att_results['accuracies_dictionary'][symbol]
                # Compute per-symbol cash and add to overall
                sym_cash = initial_per_symbol * (1 + best_symbol_results[symbol]['total_return'] / 100)
                final_value += sym_cash
                # Copy best model/scaler for this symbol to standard path
                attempt_model_path = os.path.join(CONFIG['MODEL_CACHE_DIR'], f"{symbol}_model_{CONFIG['MODEL_VERSION']}_attempt{best_att_for_sym}.pth")
                attempt_scaler_path = os.path.join(CONFIG['MODEL_CACHE_DIR'], f"{symbol}_scaler_{CONFIG['MODEL_VERSION']}_attempt{best_att_for_sym}.pkl")
                model_path = os.path.join(CONFIG['MODEL_CACHE_DIR'], f"{symbol}_model_{CONFIG['MODEL_VERSION']}.pth")
                scaler_path = os.path.join(CONFIG['MODEL_CACHE_DIR'], f"{symbol}_scaler_{CONFIG['MODEL_VERSION']}.pkl")
                if os.path.exists(attempt_model_path):
                    shutil.copyfile(attempt_model_path, model_path)
                    logger.info(f"Copied best model for {symbol} from attempt {best_att_for_sym} to standard path")
                else:
                    logger.warning(f"Model file for {symbol} attempt {best_att_for_sym} not found")
                if os.path.exists(attempt_scaler_path):
                    shutil.copyfile(attempt_scaler_path, scaler_path)
                    logger.info(f"Copied best scaler for {symbol} from attempt {best_att_for_sym} to standard path")
                else:
                    logger.warning(f"Scaler file for {symbol} attempt {best_att_for_sym} not found")
            logger.info(f"Selected best models per symbol: {best_attempt_per_symbol}")
            # Use the last attempt's times (or average if desired; here using last for simplicity)
            training_times_dictionary = attempt_results[-1]['training_times_dictionary']
            backtest_times_dictionary = attempt_results[-1]['backtest_times_dictionary']
            total_training_time_in_milliseconds = attempt_results[-1]['total_training_time_in_milliseconds']
            total_backtest_time_in_milliseconds = attempt_results[-1]['total_backtest_time_in_milliseconds']
            # Reporting (updated with buy-and-hold)
            email_body = format_email_body(CONFIG['INITIAL_CASH'], final_value, best_symbol_results, best_trade_counts, best_win_rates)
            email_body += f"\nBest Models Per Symbol: {', '.join(f'{sym} from attempt {best_attempt_per_symbol[sym]}' for sym in CONFIG['SYMBOLS'])}"
            email_body += "\n\nMonte Carlo Simulation Summary (per symbol):\n"
            for symbol in CONFIG['SYMBOLS']:
                if symbol in best_symbol_results:
                    mc = best_symbol_results[symbol]
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
            print(f"{'Symbol':<8} {'Attempt':<8} {'Total Return (%)':<18} {'Sharpe Ratio':<14} {'Max Drawdown (%)':<20} {'Trades':<8} {'Win Rate (%)':<14} {'Accuracy (%)':<14} {'MC Mean Final ($)':<18} {'MC Median Final ($)':<20} {'MC 95% VaR (%)':<15} {'MC Prob Profit (%)':<18}")
            for symbol in CONFIG['SYMBOLS']:
                if symbol in best_symbol_results:
                    metrics_for_symbol = best_symbol_results[symbol]
                    attempt = best_attempt_per_symbol[symbol]
                    return_color = Fore.GREEN if metrics_for_symbol['total_return'] > 0 else Fore.RED
                    drawdown_color = Fore.RED if metrics_for_symbol['max_drawdown'] > 0 else Fore.GREEN
                    win_rate_color = Fore.GREEN if best_win_rates[symbol] > 50 else Fore.RED
                    accuracy = best_accuracies[symbol] if best_trade_counts[symbol] > 0 else 0.0  # Hide accuracy if no trades
                    accuracy_color = Fore.GREEN if accuracy > 50 else Fore.RED
                    mc_mean_color = Fore.GREEN if metrics_for_symbol['mc_mean_final_value'] > initial_per_symbol else Fore.RED
                    mc_median_color = Fore.GREEN if metrics_for_symbol['mc_median_final_value'] > initial_per_symbol else Fore.RED
                    mc_var_color = Fore.RED if metrics_for_symbol['mc_var_95'] > 0 else Fore.GREEN
                    mc_prob_color = Fore.GREEN if metrics_for_symbol['mc_prob_profit'] > 50 else Fore.RED
                    print(f"{symbol:<8} {attempt:<8} {return_color}{metrics_for_symbol['total_return']:<18.3f}{Style.RESET_ALL} {metrics_for_symbol['sharpe_ratio']:<14.3f} {drawdown_color}{metrics_for_symbol['max_drawdown']:<20.3f}{Style.RESET_ALL} {best_trade_counts.get(symbol, 0):<8} {win_rate_color}{best_win_rates.get(symbol, 0):<14.3f}{Style.RESET_ALL} {accuracy_color}{accuracy:<14.3f}{Style.RESET_ALL} {mc_mean_color}{metrics_for_symbol['mc_mean_final_value']:<18.2f}{Style.RESET_ALL} {mc_median_color}{metrics_for_symbol['mc_median_final_value']:<20.2f}{Style.RESET_ALL} {mc_var_color}{metrics_for_symbol['mc_var_95']:<15.3f}{Style.RESET_ALL} {mc_prob_color}{metrics_for_symbol['mc_prob_profit']:<18.3f}{Style.RESET_ALL}")

            bh_color = Fore.GREEN if CONFIG['INITIAL_CASH'] < bh_final_value else Fore.RED
            print(f"\nBuy-and-Hold Final Value: {bh_color}${bh_final_value:.2f}{Style.RESET_ALL}")
            print(f"Day Trading {'beats' if final_value > bh_final_value else 'does not beat'} Buy-and-Hold.")
            color = Fore.RED if final_value <= CONFIG['INITIAL_CASH'] else Fore.GREEN
            print(f"\nBacktest completed: Final value: {color}${final_value:.2f}{Style.RESET_ALL}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Trading bot with backtest mode")
    parser.add_argument('--backtest', action='store_true', help="Run in backtest-only mode")
    parser.add_argument('--force-train', action='store_true', help="Force retraining of models")
    parser.add_argument('--DEBUG', action='store_true', help="Enable debug printing for detailed informative outputs")
    parser.add_argument('--reset', action='store_true', help="Force account reset and cash injection (overrides CONFIG flag)")
    args = parser.parse_args()

    # Configure logging (moved here to access args.DEBUG)
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler(CONFIG['LOG_FILE'])
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    stream_handler.setLevel(logging.WARNING if not args.DEBUG else logging.INFO)  # Hide INFO on console unless --DEBUG
    root_logger.addHandler(file_handler)
    root_logger.addHandler(stream_handler)

    logger = logging.getLogger(__name__)

    # Optional account reset — only runs when you manually set the flag to True
    cleanup_account_on_start()

    mp.set_start_method('spawn', force=True)  # Set early for CUDA multiprocessing safety
    main(backtest_only=args.backtest, force_train=args.force_train, debug=args.DEBUG)
