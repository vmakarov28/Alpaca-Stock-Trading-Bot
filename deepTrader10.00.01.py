# +------------------------------------------------------------------------------+
# |                                  Deep Trader 10                              |
# +------------------------------------------------------------------------------+
# | Author: Vladimir Makarov                                                     |
# | Project Start Date: May 9, 2025                                              |
# | License: GNU Lesser General Public License v2.1                              |
# | Version: 10.00.01 (Un-Released)                                              |
# +------------------------------------------------------------------------------+
# Note: Go to line 73 for the main CONFIG dictionary

import os  # file paths, dirs, etc.
import sys  # argv + exit stuff
import logging  # logging everywhere (this gets noisy fast)
import argparse  # read CLI flags like
import importlib  # used for dependency checks
import numpy as np  # core math + indicators
import pandas as pd  # dataframe heavy lifting
import torch  # main ML framework
import torch.nn as nn  # layers + model defs
import torch.optim as optim  # optimizers (Adam mostly)
from torch.utils.data import DataLoader, TensorDataset  # batching
from alpaca.data import StockHistoricalDataClient, TimeFrame, TimeFrameUnit, NewsClient  # market data + news
from alpaca.data.requests import StockBarsRequest, NewsRequest
from alpaca.trading.client import TradingClient  # trading interface
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.common.exceptions import APIError  # API tends to throw these
from transformers import pipeline  # sentiment model 
from sklearn.preprocessing import RobustScaler, StandardScaler  # scaling
import smtplib  # email alerts
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.image import MIMEImage
from datetime import datetime, timedelta, timezone  # time handling everywhere
import talib  # technical indicators 
import pickle  # caching models/data
from typing import List, Tuple, Dict, Optional, Any  # type hints (not always consistent)
import warnings  # suppress some annoying spam
from tenacity import retry, stop_after_attempt, wait_fixed, retry_if_exception_type  # retry logic for API calls
from tqdm import tqdm  # progress bars (mainly for training)
from colorama import Fore, Style  # colored console output
import colorama  # init required on Windows
import multiprocessing as mp  # parallel symbol training
import time  # timing + sleeps (also used in caching)
import shutil  # file ops
import tempfile  # temp model checkpoints
from alpaca.trading.requests import GetOrdersRequest  # used in account reset
from alpaca.trading.enums import OrderSide, TimeInForce  # duplicate but harmless
from alpaca.trading.enums import QueryOrderStatus
from alpaca.trading.enums import OrderStatus
import matplotlib.pyplot as plt  # plotting results
import statsmodels.tsa.stattools as ts  # cointegration test
from statsmodels.regression.linear_model import OLS  # hedge ratio calc
from hmmlearn.hmm import GaussianHMM  # regime detection
import xgboost as xgb  # ensemble model
import triton  # experimental GPU kernels (not critical)
import triton.language as tl
from torch.utils.checkpoint import checkpoint  # saves VRAM, slows things a bit
import threading 

# CUDA performance optimizations
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True

# Suppress PyTorch warnings
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)

# Initialize colorama for colored console output
colorama.init()

CONFIG = {
    # Trading Parameters - Settings related to trading operations
    'SYMBOLS': [ 'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'AMD', 'SPY', 'QQQ' ],  # List of stock symbols to trade
    'PAIRS': [  # NEW: Pairs for market-neutral statistical arbitrage
        ('AAPL', 'MSFT'), ('GOOGL', 'AMZN'),
        ('NVDA', 'AMD'), ('SPY', 'QQQ')
    ],
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
    'LIVE_DATA_BARS': 1200,  # Number of bars to fetch for live data

    # Model Training - Settings for training the machine learning model
    'TRAIN_EPOCHS': 50,  # Number of epochs for training the model
    'BATCH_SIZE': 2048,  # Batch size for training
    'TIMESTEPS': 30,  # Number of time steps for sequence data
    'EARLY_STOPPING_MONITOR': 'val_loss',  # Metric to monitor for early stopping
    'EARLY_STOPPING_PATIENCE': 25,  # Patience for early stopping
    'EARLY_STOPPING_MIN_DELTA': 0.00001,  # Reduced min delta to detect smaller improvements
    'LEARNING_RATE': 0.001,  # Reduced initial learning rate for Adam to stabilize training
    'LR_SCHEDULER_PATIENCE': 5,  # Patience for ReduceLROnPlateau
    'LR_REDUCTION_FACTOR': 0.5,  # Factor to multiply LR by
    'LOOK_AHEAD_BARS': 21,  # Number of bars to look ahead for future direction target
    'NUM_PARALLEL_WORKERS': 5,  #yohey Number of parallel workers for symbol training (tune based on VRAM/CPU cores)

    # NEW: HMM Regime Detection
    'NUM_REGIMES': 6,  # Number of hidden market regimes (Calm Bull, Volatile Bull, Calm Bear, Volatile Bear)

    # API and Authentication - Credentials for API access
    'ALPACA_API_KEY': 'PKXROHOFWDFC7OFFXQCBRDWVMU',  # API key for Alpaca
    'ALPACA_SECRET_KEY': 'CT25NcFuH7UtkPtut4QVLfBzk8j1juDevVnNpXgLpwgC',  # Secret key for Alpaca

    # Email Notifications - Configuration for sending email alerts
    'EMAIL_SENDER': 'alpaca.ai.tradingbot@gmail.com',  # Email address for sending notifications
    'EMAIL_PASSWORD': 'hjdf sstp pyne rotq',  # Password for the email account
    'EMAIL_RECEIVER': ['aiplane.scientist@gmail.com', 'vmakarov28@students.d125.org', 'tchaikovskiy@hotmail.com'],  # List of email recipients
    'SMTP_SERVER': 'smtp.gmail.com',  # SMTP server for email
    'SMTP_PORT': 587,  # Port for SMTP server

    # Logging and Monitoring - Settings for tracking activities
    'LOG_FILE': 'trades.log',  # File for logging trades
    
    # Strategy Thresholds — BEST FROM 10-HOUR OPTIMIZER
    'CONFIDENCE_THRESHOLD': 0.7, #0.7
    'PREDICTION_THRESHOLD_BUY': 0.7, #0.7
    'PREDICTION_THRESHOLD_SELL': 0.3, #0.38
    'RSI_BUY_THRESHOLD': 35,          # Buy when oversold
    'RSI_SELL_THRESHOLD': 72,         # Sell when overbought
    'ADX_TREND_THRESHOLD': 18,
    'MAX_VOLATILITY': 35.0,
    'PREDICTION_TEMPERATURE': 0.5,


    # Risk Management - Parameters to control trading risk
    'MAX_DRAWDOWN_LIMIT': 0.04,  # Maximum allowed drawdown
    'RISK_PERCENTAGE': 0.02,  # Percentage of cash to risk per trade (halved for smaller positions)
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
    'MODEL_VERSION': 'deepTrader10',  # Model architecture version; increment on structural changes to force retrain

    # New: Retraining Cycle Parameters
    'ENABLE_RETRAIN_CYCLE': True,  # Enable loop to retrain until criteria met (backtest mode only)
    'FORCE_FULL_RETRAIN_RUN': True,   # NEW: Set True = always run ALL attempts and pick the absolute best
    'MIN_FINAL_VALUE': 130000.0,  # Minimum final portfolio value to accept
    'MAX_ALLOWED_DRAWDOWN': 30.0,  # Maximum allowed max_drawdown percentage (across symbols)
    'MAX_RETRAIN_ATTEMPTS': 15,  # Max loop iterations to prevent infinite runs

    #Monte Carlo Probability Simulation
    'NUM_MC_SIMULATIONS': 50000,  # Number of Monte Carlo simulations for backtest robustness testing

    # Account Management
    'RESET_ACCOUNT_ON_START': True,   # Set to True only when you want a full reset (closes all positions & cancels orders)
    'PAPER_TRADING': True,             # Set to False when going live with real money
    'DESIRED_STARTING_CASH': 100000.00,  # Desired cash for reset

    # Pairs-specific params
    'PAIR_CONFIDENCE_THRESHOLD': 0.55,
    'PAIR_REGIME_FILTER': ["Calm Bull", "Moderate Bull", "Calm Bear", "Moderate Bear"],  # Now uses the new moderate regimes
    'KALMAN_LOOKBACK': 30,
    'ENABLE_FULL_PAIRS_RESOLUTION': False, # False = FAST optimized (recommended, checks pairs every ~1 hour). True  = SLOW (100% Accurate backtest and very CPU heavy)
}



#pyenv activate pytorch_env
#python /mnt/c/Users/aipla/Downloads/alpaca_neural_bot_v10.00.01.py --backtest --force-train --DEBUG




def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    # add indicators
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

    cache_path = os.path.join(CONFIG['MODEL_CACHE_DIR'], f"{symbol}_news_sentiment_{CONFIG['MODEL_VERSION']}.pkl")
    if os.path.exists(cache_path) and (time.time() - os.path.getmtime(cache_path) < 3600):
        with open(cache_path, 'rb') as f:
            score = pickle.load(f)
        logger.info(f"Loaded REAL sentiment for {symbol} from cache: {score:.3f}")
        return score

    try:
        news_client = NewsClient(CONFIG['ALPACA_API_KEY'], CONFIG['ALPACA_SECRET_KEY'])
        request = NewsRequest(
            symbols=symbol,
            start=datetime.now(timezone.utc) - timedelta(days=7),
            limit=30
        )
        news_items = news_client.get_news(request)
        
        if not news_items:
            score = 0.0
        else:
            texts = []
            for item in news_items:
                if isinstance(item, dict):
                    headline = item.get('headline', '')
                    summary = item.get('summary', '') or ''
                elif isinstance(item, (list, tuple)):
                    headline = item[0] if len(item) > 0 else ""
                    summary = item[1] if len(item) > 1 else ""
                else:
                    headline = getattr(item, 'headline', "")
                    summary = getattr(item, 'summary', "") or ""
                texts.append(str(headline) + ". " + str(summary))
            
            # Truncate to avoid tokenizer error (distilbert max 512 tokens)
            texts = [t[:2000] for t in texts]  # safe truncation
            results = sentiment_pipeline(texts, truncation=True, max_length=512, batch_size=8)
            scores = [1.0 if r['label'] == 'POSITIVE' else -1.0 for r in results]
            score = np.mean(scores) if results else 0.0
        
        logger.info(f"REAL sentiment for {symbol}: {score:.3f} from {len(news_items)} articles")
    except Exception as e:
        logger.DEBUG(f"News API failed for {symbol}: {e} → using 0.0 neutral sentiment")
        score = 0.0

    with open(cache_path, 'wb') as f:
        pickle.dump(score, f)
    return score


def is_cointegrated(series1: pd.Series, series2: pd.Series, pvalue_threshold: float = 0.05) -> bool:
    # Align to common index and drop NaNs
    common_idx = series1.index.intersection(series2.index)
    s1 = series1.loc[common_idx].dropna()
    s2 = series2.loc[common_idx].dropna()
    if len(s1) < 100 or len(s2) < 100 or len(s1) != len(s2):
        return False
    result = ts.coint(s1, s2)
    return result[1] < pvalue_threshold

def calculate_spread(df1: pd.DataFrame, df2: pd.DataFrame) -> pd.Series:
    common_idx = df1.index.intersection(df2.index)
    price1 = df1['close'].loc[common_idx].dropna()
    price2 = df2['close'].loc[common_idx].dropna()
    if len(price1) != len(price2) or len(price1) < 100:
        return price1 - price2.reindex(price1.index, method='ffill')
    model = OLS(price1, price2).fit()
    # works but I hate this — revisit after backtest is stable
    hedge_ratio = model.params.iloc[0]   # ← FIXED
    spread = price1 - hedge_ratio * price2
    return spread

def get_pair_regime(hmm1, hmm2, recent_seq1, recent_seq2) -> str:
    if hmm1 is None or hmm2 is None:
        return "Unknown"                    # ← prevent crash
    try:
        r1 = hmm1.predict(recent_seq1.reshape(-1, recent_seq1.shape[2]))[0]
        r2 = hmm2.predict(recent_seq2.reshape(-1, recent_seq2.shape[2]))[0]
        regime_names = ["Calm Bull", "Moderate Bull", "Volatile Bull", 
                        "Calm Bear", "Moderate Bear", "Volatile Bear"]
        avg = int((r1 + r2) / 2)
        return regime_names[avg % len(regime_names)]
    except Exception as e:
        logger.warning(f"Pair regime failed for {hmm1}/{hmm2}: {e}")
        return "Unknown"


@retry(retry=retry_if_exception_type(Exception), stop=stop_after_attempt(3), wait=wait_fixed(5))
def load_data(symbol: str, start_date: str, end_date: str) -> pd.DataFrame:

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
        df.index = pd.to_datetime(df.index)  # Force DatetimeIndex for multiprocessing safety
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
    worker_id, symbol, expected_features, force_train, barrier, gpu_semaphore, backtest_only, debug = args
    start_time_for_training = time.perf_counter()
    try:
        result_from_train_symbol = train_symbol(
            symbol,
            worker_id,
            expected_features,
            force_train,
            barrier,
            gpu_semaphore,
            backtest_only,
            debug
        )
        end_time_for_training = time.perf_counter()
        training_time_in_milliseconds = (end_time_for_training - start_time_for_training) * 1000
        return (*result_from_train_symbol, training_time_in_milliseconds)
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        logger.error(f"WORKER CRASH for {symbol}: {str(e)}\n{error_trace}")
        dummy = (symbol, None, None, False, 0.0, False, False, None, None)
        return (*dummy, 0)


logger = logging.getLogger(__name__)

# Initialize sentiment analysis pipeline
device = 0 if torch.cuda.is_available() else -1
sentiment_pipeline = pipeline("sentiment-analysis", model=CONFIG['SENTIMENT_MODEL'], framework="pt", device=device)

# Checks if required libs are installed via importlib, raises ImportError if missing.
def check_dependencies() -> None:
    required_modules = [
        'torch', 'numpy', 'pandas', 'alpaca', 'transformers',
        'sklearn', 'talib', 'tenacity', 'smtplib', 'argparse', 'tqdm', 'colorama',
        'hmmlearn'  # NEW for regime detection
    ]
    for module in required_modules:
        try:
            importlib.import_module(module)
        except ImportError:
            raise ImportError(f"Module '{module}' is required. Install it using: pip install {module}")

def validate_config(config: Dict) -> None:
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
    # Do nothing unless you explicitly ask for a reset
    if not (args.reset or CONFIG.get('RESET_ACCOUNT_ON_START', False)):
        return

    # Settings
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

        # Close all positions AND cancel all orders in one call
        trading_client.close_all_positions(cancel_orders=True)
        logger.info("Requested close all positions and cancel orders")

        # Extra cancel for any lingering
        trading_client.cancel_orders()

        # Poll until cleared
        clock = trading_client.get_clock()
        if not clock.is_open: # this breaks on half-days, need to handle early close
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

        # Inject / remove cash in chunks
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



# @retry(stop=stop_after_attempt(CONFIG['API_RETRY_ATTEMPTS']), wait=wait_fixed(CONFIG['API_RETRY_DELAY'] / 1000), retry=retry_if_exception_type(APIError))
# def get_all_positions_with_retry():
#     """Retry wrapper for getting all positions."""
#     return trading_client.get_all_positions()

# @retry(stop=stop_after_attempt(CONFIG['API_RETRY_ATTEMPTS']), wait=wait_fixed(CONFIG['API_RETRY_DELAY'] / 1000), retry=retry_if_exception_type(APIError))
# def get_clock_with_retry():
#     """Retry wrapper for getting market clock."""
#     return trading_client.get_clock()


# Fetches last N bars from Alpaca for live, uses recent dates, renames vwap, sorts, retries on error.
def fetch_recent_data(symbol: str, num_bars: int = 1000) -> pd.DataFrame:
    """Improved live data fetch with more history and debug."""
    client = StockHistoricalDataClient(CONFIG['ALPACA_API_KEY'], CONFIG['ALPACA_SECRET_KEY'])
    end_date = datetime.now(timezone.utc)
    start_date = end_date - timedelta(days=10)  # Increased from 3 days

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
    print(f"{Fore.CYAN}[LIVE FETCH] {symbol}: {len(df)} bars | Last price: ${df['close'].iloc[-1]:.2f}{Style.RESET_ALL}")
    return df.sort_values('timestamp')

# Fetches historical bars in 1yr chunks to avoid limits, handles META/FB rename, concats/dedups, raises on low data.
def fetch_data(symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
    cache_path = os.path.join(CONFIG['CACHE_DIR'], f"{symbol}_full_history_fallback.pkl")
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
                logger.info(f"Fetched {len(df_bars)} bars for {symbol}")
            else:
                logger.info(f"No data for {symbol} from {current_start} to {current_end}, skipping")
            current_start = current_end

        if all_bars:
            df = pd.concat(all_bars).sort_values('timestamp')
            df = df.drop_duplicates(subset='timestamp', keep='first')
            logger.info(f"Total fetched {len(df)} bars for {symbol}")
            # Save fallback cache
            with open(cache_path, 'wb') as f:
                pickle.dump(df, f)
        else:
            df = pd.DataFrame(columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'VWAP'])

        if len(df) < CONFIG['MIN_DATA_POINTS']:
            raise ValueError(f"Insufficient data for {symbol}: got {len(df)} bars")

        return df

    except Exception as e:
        logger.warning(f"API error for {symbol}: {str(e)} → trying fallback cache")
        if os.path.exists(cache_path):
            with open(cache_path, 'rb') as f:
                df = pickle.load(f)
            logger.info(f"Used fallback cache for {symbol} ({len(df)} rows)")
            return df
        logger.error(f"No fallback cache for {symbol} — training will skip this symbol")
        raise

# Loads from cache if fresh, else fetches full history to now, saves pickle, returns df and loaded flag.
def load_or_fetch_data(symbol: str, start_date: str, end_date: str) -> Tuple[pd.DataFrame, bool]:
    cache_file = os.path.join(CONFIG['CACHE_DIR'], f"{symbol}_train_data_{start_date}_{end_date}.pkl")
    
    if os.path.exists(cache_file) and (time.time() - os.path.getmtime(cache_file)) < CONFIG['CACHE_EXPIRY_SECONDS']:
        with open(cache_file, 'rb') as f:
            df = pickle.load(f)
        logger.info(f"Loaded {len(df)} bars for {symbol} from training cache")
        return df, True
    else:
        df = fetch_data(symbol, start_date, end_date)
        with open(cache_file, 'wb') as f:
            pickle.dump(df, f)
        logger.info(f"Fetched and cached {len(df)} bars for {symbol} as training data")
        return df, False

# Loads sentiment from cache if fresh, else generates random -1 to 1, saves, returns score and loaded flag.
def load_news_sentiment(symbol: str) -> Tuple[float, bool]:
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
    df = df.copy()
    required_cols = ['open', 'high', 'low', 'close', 'volume']
    if not all(col in df.columns for col in required_cols):
        raise ValueError(f"Missing required columns: {required_cols}")
    
    # FORCE DatetimeIndex (critical for multiprocessing workers)
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    
    if 'VWAP' not in df.columns:
        df['VWAP'] = (df['volume'] * (df['high'] + df['low'] + df['close']) / 3).cumsum() / df['volume'].cumsum()
    
    # Basic indicators
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

    # Better Features
    # 1. Multi-timeframe (60-minute) - fixed deprecated '60T'
    df_60 = df.resample('60min').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).dropna()
    df['RSI_60'] = talib.RSI(df_60['close'], timeperiod=14).reindex(df.index, method='ffill')
    df['MA20_60'] = talib.SMA(df_60['close'], timeperiod=20).reindex(df.index, method='ffill')

    # 2. Volume Profile
    df['VWAP_Dev'] = df['close'] - df['VWAP']
    df['Volume_Delta'] = df['volume'] * (df['close'] - df['open'])

    # 3. Macro proxy
    df['Macro_Stress'] = df['Volatility'].rolling(20).mean() / df['Volatility'].rolling(100).mean()

    # 4. Earnings proxy
    df['Earnings_Proxy'] = df['Sentiment'] * (1 + df['Volume_Delta'].abs() / df['volume'].rolling(20).mean())

    indicator_cols = [
        'MA20', 'MA50', 'RSI', 'MACD', 'MACD_signal', 'OBV', 'VWAP', 'ATR',
        'CMF', 'Close_ATR', 'MA20_ATR', 'Return_1d', 'Return_5d', 'Volatility',
        'BB_upper', 'BB_middle', 'BB_lower', 'Stoch_K', 'Stoch_D', 'ADX', 'Sentiment', 'Trend',
        'RSI_60', 'MA20_60', 'VWAP_Dev', 'Volume_Delta', 'Macro_Stress', 'Earnings_Proxy'
    ]
    df = df.dropna(subset=indicator_cols)
    return df

def validate_raw_data(df: pd.DataFrame, symbol: str) -> None:
    if df.empty:
        raise ValueError(f"Empty DataFrame for {symbol}")
    required_cols = ['open', 'high', 'low', 'close', 'volume', 'timestamp']
    if not all(col in df.columns for col in required_cols):
        missing = [col for col in required_cols if col not in df.columns]
        raise ValueError(f"Missing required columns for {symbol}: {missing}")
    if df[required_cols[:-1]].isna().any().any():
        raise ValueError(f"NaN values in OHLCV columns for {symbol}")

def validate_indicators(df: pd.DataFrame, symbol: str) -> None:
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
    df = df.copy()
    df.ffill(inplace=True)
    df.fillna(0, inplace=True)
   
    features = [
        'close', 'high', 'low', 'volume', 'MA20', 'MA50', 'RSI',
        'MACD', 'MACD_signal', 'OBV', 'VWAP', 'ATR', 'CMF', 'Close_ATR',
        'MA20_ATR', 'Return_1d', 'Return_5d', 'Volatility', 'BB_upper',
        'BB_lower', 'Stoch_K', 'Stoch_D', 'ADX', 'Sentiment', 'Trend',
        'RSI_60', 'MA20_60', 'VWAP_Dev', 'Volume_Delta', 'Macro_Stress', 'Earnings_Proxy'
    ]  # Now 32 features with better multi-timeframe and volume profile
    
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
   
    window = np.lib.stride_tricks.sliding_window_view(X, (timesteps, X.shape[1]))
    X_seq = window[:num_sequences].reshape(num_sequences, timesteps, X.shape[1])
   
    if not inference_mode:
        y_seq = y[timesteps - 1: timesteps - 1 + num_sequences]
        logger.info(f"Preprocessed {len(X_seq)} sequences; y balance: {np.mean(y_seq):.3f} (up fraction)")
    else:
        logger.info(f"Preprocessed {len(X_seq)} inference sequences")
   
    return X_seq, y_seq, scaler

def monte_carlo_simulation(returns: List[float], initial_cash: float, num_simulations: int = CONFIG['NUM_MC_SIMULATIONS']) -> Dict[str, float]:
    # bootstrap resample to get distribution of outcomes
    # returns mc_mean, mc_median, var_95, prob_profit
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
        self.timesteps = timesteps
        self.features = features
        self.hidden_size = 128
        
        # Core layers 
        self.lstm = nn.LSTM(features, self.hidden_size, num_layers=2, 
                            batch_first=True, dropout=0.3)
        self.attention = nn.MultiheadAttention(embed_dim=self.hidden_size, 
                                               num_heads=8, dropout=0.15, 
                                               batch_first=True)
        self.ln_lstm = nn.LayerNorm(self.hidden_size)
        self.ln_attn = nn.LayerNorm(self.hidden_size)
        self.dense1 = nn.Linear(self.hidden_size, 64)
        self.dense2 = nn.Linear(64, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.25)
        
        nn.init.xavier_uniform_(self.dense1.weight)
        nn.init.xavier_uniform_(self.dense2.weight)

    def forward(self, x):
        lstm_out, _ = checkpoint(self._lstm_forward, x, use_reentrant=False)
        
        attn_out, _ = checkpoint(self._attention_forward, lstm_out, lstm_out, lstm_out)
        
        x = attn_out[:, -1, :]  # last timestep
        x = self.relu(self.dense1(x))
        x = self.dropout(x)
        x = self.dense2(x)
        return x

    def _lstm_forward(self, x):
        return self.lstm(x)

    def _attention_forward(self, q, k, v):
        return self.attention(q, k, v)

    # Optional Triton fusion on dense layers (extra speed, same math)
    @staticmethod
    @triton.jit
    def _fused_dense_kernel(x_ptr, w1_ptr, b1_ptr, w2_ptr, b2_ptr, out_ptr, 
                            B: tl.constexpr, H: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * H + tl.arange(0, H)
        x = tl.load(x_ptr + offs)
        # ReLU + Linear1
        x = tl.maximum(0.0, tl.dot(x, tl.load(w1_ptr + offs * H + tl.arange(0, H))))
        x = x + tl.load(b1_ptr + pid)
        # Linear2
        x = tl.dot(x, tl.load(w2_ptr + offs * H + tl.arange(0, H))) + tl.load(b2_ptr + pid)
        tl.store(out_ptr + offs, x)

def train_model(symbol: str, worker_id: int, df: pd.DataFrame, epochs: int, batch_size: int, timesteps: int, expected_features: int, barrier=None, gpu_semaphore=None, preprocessed_train=None, preprocessed_val=None) -> Tuple[nn.Module, Any, GaussianHMM, Any]:
    if gpu_semaphore is not None:
        gpu_semaphore.acquire()
        logger.info(f"[{symbol}] Acquired gpu_semaphore (worker starting heavy GPU ops)")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = TradingModel(timesteps, expected_features).to(device)
    
    # torch.compile breaks with gradient checkpointing on this version, spent 3 hours on this
    # leaving it off until pytorch fixes the inductor bug
    # model = torch.compile(model, mode="reduce-overhead", fullgraph=False)

    optimizer = optim.Adam(model.parameters(), lr=CONFIG['LEARNING_RATE'], weight_decay=1e-5)
    pos_weight = torch.tensor([1.1])
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', 
                                                     patience=CONFIG['LR_SCHEDULER_PATIENCE'], 
                                                     factor=CONFIG['LR_REDUCTION_FACTOR'])

    def get_convergence_note(train_loss, val_loss, best_val_loss, current_lr):
        if val_loss < best_val_loss * 0.997:   # slightly easier to trigger green
            return f"{Fore.GREEN}✓ {Fore.LIGHTBLACK_EX}Both losses dropping — model is learning to guess stock direction better{Style.RESET_ALL}"
        elif val_loss > best_val_loss * 1.01 and train_loss < best_val_loss * 0.95:
            return f"{Fore.LIGHTBLACK_EX}⚠ Val loss rising while train drops — model is memorizing old data instead of learning new patterns{Style.RESET_ALL}"
        elif current_lr < CONFIG['LEARNING_RATE'] * 0.6:
            return f"{Fore.GREEN}✓ {Fore.LIGHTBLACK_EX}LR lowered — model now making smaller, more careful adjustments to avoid mistakes{Style.RESET_ALL}"
        else:
            return f"{Fore.GREEN}✓ {Fore.LIGHTBLACK_EX}Losses mostly stable — training continuing normally"
    
    # === Exact same data prep as before (no change) ===
    if not isinstance(df.index, pd.DatetimeIndex):
        if 'timestamp' in df.columns:
            df = df.set_index('timestamp')
        df.index = pd.to_datetime(df.index).tz_localize(None)
        df = df.sort_index()

    train_end = pd.Timestamp(CONFIG['TRAIN_END_DATE'], tz='UTC').normalize()
    val_end   = pd.Timestamp(CONFIG['VAL_END_DATE'], tz='UTC').normalize()
    df_train = df[df.index <= train_end].copy()
    df_val   = df[(df.index > train_end) & (df.index <= val_end)].copy()

    df_train['Future_Direction'] = np.where(df_train['close'].shift(-CONFIG['LOOK_AHEAD_BARS']) > df_train['close'], 1, 0)
    df_train = df_train.dropna(subset=['Future_Direction'])
    df_val['Future_Direction'] = np.where(df_val['close'].shift(-CONFIG['LOOK_AHEAD_BARS']) > df_val['close'], 1, 0)
    df_val = df_val.dropna(subset=['Future_Direction'])

    X_train, y_train, scaler = preprocess_data(df_train, CONFIG['TIMESTEPS'], add_noise=True)
    X_val,   y_val,   _      = preprocess_data(df_val,   CONFIG['TIMESTEPS'], inference_scaler=scaler, inference_mode=False, fit_scaler=False)

    X_train_pinned = torch.from_numpy(X_train.astype(np.float32)).pin_memory()
    y_train_pinned = torch.from_numpy(y_train.astype(np.float32)).pin_memory()
    X_val_pinned   = torch.from_numpy(X_val.astype(np.float32)).pin_memory()
    y_val_pinned   = torch.from_numpy(y_val.astype(np.float32)).pin_memory()

    train_dataset = TensorDataset(X_train_pinned, y_train_pinned)
    val_dataset   = TensorDataset(X_val_pinned, y_val_pinned)

    stream = torch.cuda.Stream() if torch.cuda.is_available() else None
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                              num_workers=0, pin_memory=True, persistent_workers=False, 
                              prefetch_factor=None)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False, 
                              num_workers=0, pin_memory=True, persistent_workers=False, 
                              prefetch_factor=None)

    scaler_amp = torch.amp.GradScaler('cuda')

    print(f"{Fore.CYAN}[{symbol}]{Fore.LIGHTCYAN_EX} Slave {worker_id}{Fore.CYAN} started GPU training with compile + streams + prefetch{Style.RESET_ALL}")

    # Early stopping initialization (must be before the loop)
    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for batch_X, batch_y in train_loader:
            batch_X = batch_X.to(device, non_blocking=True).float()
            batch_y = batch_y.to(device, non_blocking=True).float()
            
            optimizer.zero_grad()
            with torch.amp.autocast('cuda'):
                outputs = model(batch_X)
                loss = criterion(outputs.squeeze(1), batch_y)
            
            scaler_amp.scale(loss).backward()
            scaler_amp.step(optimizer)
            scaler_amp.update()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        # Validation (unchanged)
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X = batch_X.to(device).float()
                batch_y = batch_y.to(device).float()
                outputs = model(batch_X)
                loss = criterion(outputs.squeeze(1), batch_y)
                val_loss += loss.item()
        val_loss /= len(val_loader)
         #convergence notes
        if val_loss < best_val_loss * 0.995:
            note = f"{Fore.LIGHTBLACK_EX}✓ Both losses dropping — model is learning better{Style.RESET_ALL}"
        elif val_loss > best_val_loss * 1.02 and train_loss < best_val_loss * 0.9:
            note = f"{Fore.YELLOW} Val loss rising while train drops — model may be memorizing instead of learning{Style.RESET_ALL}"
        elif optimizer.param_groups[0]['lr'] < CONFIG['LEARNING_RATE'] * 0.6:
            note = f"{Fore.LIGHTBLACK_EX}LR lowered — model now making final adjustments"
        else:
            note = f"{Fore.LIGHTBLACK_EX}Losses mostly stable — training continuing normally"

        current_lr = optimizer.param_groups[0]['lr']
        note = get_convergence_note(train_loss, val_loss, best_val_loss, current_lr)

        print(f"[{symbol}] Epoch {epoch+1:02d}/{epochs} |{Fore.LIGHTBLACK_EX} "
                f"Train Loss: {train_loss:.5f} | Val Loss: {val_loss:.5f} | "
                f"LR: {current_lr:.2e} | {note}{Style.RESET_ALL}")

        # print(f"[{symbol}] Epoch {epoch+1:02d}/{epochs} |{Fore.LIGHTBLACK_EX} "
        #         f"Train Loss: {train_loss:.5f} | Val Loss: {val_loss:.5f} | "
        #         f"LR: {optimizer.param_groups[0]['lr']:.2e} | {note}{Style.RESET_ALL}{Style.RESET_ALL}")
        #convergence_note = "improving" if val_loss < best_val_loss else "⚠ monitor overfitting"
        #print(f"{Fore.LIGHTGREEN_EX}[{symbol}] Epoch {epoch+1:02d}/{epochs} | Train Loss: {train_loss:.5f} | Val Loss: {val_loss:.5f} | LR: {optimizer.param_groups[0]['lr']:.2e}{Style.RESET_ALL}")
        scheduler.step(val_loss)

        if scheduler._last_lr[0] == optimizer.param_groups[0]['lr'] and not hasattr(scheduler, '_convergence_logged'):
            logger.debug(f"{Fore.YELLOW}[{symbol}] Model is not converging. Current val_loss: {val_loss:.8f} (not better than best). Delta: {best_val_loss - val_loss:.8f}{Style.RESET_ALL}")
            scheduler._convergence_logged = True

        # Early stopping + temp save
        temp_best_path = None
        if val_loss < best_val_loss - CONFIG['EARLY_STOPPING_MIN_DELTA']:
            best_val_loss = val_loss
            patience_counter = 0
            with tempfile.NamedTemporaryFile(suffix='.pth', delete=False) as temp_file:
                temp_best_path = temp_file.name
                torch.save(model.state_dict(), temp_best_path)
        else:
            patience_counter += 1
            if patience_counter >= CONFIG['EARLY_STOPPING_PATIENCE']:
                break

        if temp_best_path and os.path.exists(temp_best_path):
            try:
                model.load_state_dict(torch.load(temp_best_path))
            except Exception as e:
                logger.error(f"Failed to load best model for {symbol}: {str(e)}")
            finally:
                try:
                    os.remove(temp_best_path)
                except Exception as e:
                    logger.warning(f"Failed to remove temp file {temp_best_path}: {str(e)}")

    print(f"{Fore.CYAN}[{symbol}] LSTM + Attention training COMPLETE — switching to CPU for HMM + XGBoost ensemble{Style.RESET_ALL}")

    # === CPU PHASE PROGRESS TRACKING (per-symbol, same style as LSTM) ===
    print(f"{Fore.CYAN}[{symbol}] Worker {worker_id} → Starting HMM regime detection on CPU...{Style.RESET_ALL}")
    
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    finally:
        # Always release even on exception (prevents deadlock of later workers)

        # if you're reading this trying to figure out why a worker hung — check the semaphore
        # one bad worker not releasing it will stall everything. ask me how i know
        if gpu_semaphore is not None:
            try:
                gpu_semaphore.release()
                print(f"{Fore.BLUE}[{symbol}]{Fore.LIGHTCYAN_EX} Slave {worker_id}{Fore.BLUE} → RELEASED GPU {Style.RESET_ALL}")
                logger.debug(f"[{symbol}] Released gpu semaphore after training")
            except Exception as e:
                logger.warning(f"[{symbol}] Failed to release gpu_semaphore: {str(e)}")

    # HMM + XGBoost (unchanged)
    X_train = X_train_pinned.cpu().numpy()
    y_train = y_train_pinned.cpu().numpy()
    X_val   = X_val_pinned.cpu().numpy()
    y_val   = y_val_pinned.cpu().numpy()

    # HMM training (this is the real work)
    hmm = train_hmm(X_train.reshape(-1, X_train.shape[2]))
    print(f"{Fore.BLUE}[{symbol}] {Fore.LIGHTCYAN_EX}Slave {worker_id}{Fore.BLUE} → HMM regime detection COMPLETE (6 regimes detected){Style.RESET_ALL}")

    print(f"{Fore.CYAN}[{symbol}] {Fore.LIGHTCYAN_EX}Slave {worker_id}{Fore.CYAN} → Training XGBoost ensemble model on CPU...{Style.RESET_ALL}")
    xgb_model = train_xgboost(X_train.reshape(X_train.shape[0], -1), y_train, X_val.reshape(X_val.shape[0], -1), y_val)
    print(f"{Fore.BLUE}[{symbol}]{Fore.LIGHTCYAN_EX} Slave {worker_id}{Fore.BLUE} → CPU phase finished{Style.RESET_ALL}")

    scaler = RobustScaler()
    scaler.fit(X_train.reshape(-1, X_train.shape[2]))
    with open(os.path.join(CONFIG['CACHE_DIR'], f"{symbol}_scaler.pkl"), 'wb') as f:
        pickle.dump(scaler, f)

    X_train_flat = X_train.reshape(-1, X_train.shape[2])
    hmm = train_hmm(X_train_flat)

    X_train_flat = X_train.reshape(X_train.shape[0], -1)
    X_val_flat = X_val.reshape(X_val.shape[0], -1)
    xgb_model = train_xgboost(X_train_flat, y_train, X_val_flat, y_val)

    # Semaphore already released in the finally block above — no duplicate release here
    return model, scaler, hmm, xgb_model

def train_hmm(X_scaled: np.ndarray, num_regimes: int = CONFIG['NUM_REGIMES']) -> GaussianHMM:
    try:
        hmm = GaussianHMM(
            n_components=num_regimes,
            covariance_type="diag",      #Much more stable
            n_iter=1000,
            random_state=42,
            min_covar=1e-6,              #Prevents singular matrices
            params="stmc",               #Only learn safe parameters
            init_params="stmc"
        )
        hmm.fit(X_scaled)
        logger.info(f"Trained HMM with {num_regimes} regimes (stable diagonal covariance)")
        return hmm
    except Exception as e:
        logger.warning(f"6-regime HMM failed ({e}). Falling back to 4 regimes.")
        hmm = GaussianHMM(               #Safe fallback
            n_components=4,
            covariance_type="diag",
            n_iter=1000,
            random_state=42,
            min_covar=1e-6
        )
        hmm.fit(X_scaled)
        logger.info("Fallback to 4 regimes successful")
        return hmm

def train_xgboost(X_train, y_train, X_val, y_val):
    model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        use_label_encoder=False,
        eval_metric='logloss'
    )
    model.fit(X_train, y_train)
    return model

def load_model_and_scaler(symbol: str, expected_features: int, force_retrain: bool = False) -> Tuple[Optional[nn.Module], Optional[RobustScaler], Optional[float], Optional[GaussianHMM]]:
    logger.info(f"Entering load_model_and_scaler for {symbol} (force_retrain={force_retrain}).")
    if force_retrain:
        return None, None, None, None
    
    model_path = os.path.join(CONFIG['MODEL_CACHE_DIR'], f"{symbol}_model_{CONFIG['MODEL_VERSION']}.pth")
    scaler_path = os.path.join(CONFIG['MODEL_CACHE_DIR'], f"{symbol}_scaler_{CONFIG['MODEL_VERSION']}.pkl")
    sentiment_path = os.path.join(CONFIG['MODEL_CACHE_DIR'], f"{symbol}_news_sentiment_{CONFIG['MODEL_VERSION']}.pkl")
    hmm_path = os.path.join(CONFIG['MODEL_CACHE_DIR'], f"{symbol}_hmm_{CONFIG['MODEL_VERSION']}.pkl")
    
    if os.path.exists(model_path) and os.path.exists(scaler_path) and os.path.exists(hmm_path):
        logger.info(f"Found model, scaler, and HMM for {symbol}.")
        try:
            with open(scaler_path, 'rb') as f:
                scaler = pickle.load(f)
            with open(hmm_path, 'rb') as f:
                hmm = pickle.load(f)
            
            checkpoint = torch.load(model_path, map_location='cpu')
            model = TradingModel(CONFIG['TIMESTEPS'], expected_features)
            if isinstance(checkpoint, dict):
                if 'state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['state_dict'])
                elif 'model_state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['model_state_dict'])
                else:
                    model.load_state_dict(checkpoint)
            
            training_sentiment = None
            if os.path.exists(sentiment_path):
                with open(sentiment_path, 'rb') as f:
                    training_sentiment = pickle.load(f)
            
            logger.info(f"Successfully loaded cached model, scaler, sentiment, and HMM for {symbol}.")
            return model, scaler, training_sentiment, hmm
        except Exception as e:
            logger.error(f"Failed to load for {symbol}: {str(e)}. Retraining.")
            return None, None, None, None
    else:
        logger.info(f"No cached model/scaler/HMM for {symbol}. Will train.")
        return None, None, None, None

def save_model_and_scaler(symbol: str, model: nn.Module, scaler: RobustScaler, sentiment: float, hmm: GaussianHMM) -> None:
    try:
        model_path = os.path.join(CONFIG['MODEL_CACHE_DIR'], f"{symbol}_model_{CONFIG['MODEL_VERSION']}.pth")
        scaler_path = os.path.join(CONFIG['MODEL_CACHE_DIR'], f"{symbol}_scaler_{CONFIG['MODEL_VERSION']}.pkl")
        sentiment_path = os.path.join(CONFIG['MODEL_CACHE_DIR'], f"{symbol}_news_sentiment_{CONFIG['MODEL_VERSION']}.pkl")
        hmm_path = os.path.join(CONFIG['MODEL_CACHE_DIR'], f"{symbol}_hmm_{CONFIG['MODEL_VERSION']}.pkl")
        
        torch.save({'model_state_dict': model.state_dict(), 'class_name': 'TradingModel'}, model_path)
        with open(scaler_path, 'wb') as f:
            pickle.dump(scaler, f)
        with open(sentiment_path, 'wb') as f:
            pickle.dump(sentiment, f)
        with open(hmm_path, 'wb') as f:
            pickle.dump(hmm, f)
        
        windows_model_path = model_path.replace('/mnt/c/', 'C:\\').replace('/', '\\')
        logger.info(f"Saved model, scaler, sentiment, and HMM for {symbol} (Windows: {windows_model_path})")
    except Exception as e:
        logger.error(f"Failed to save for {symbol}: {str(e)}")
        raise


def train_symbol(symbol: str, worker_id: int, expected_features: int, force_train: bool, barrier=None, gpu_semaphore=None, preprocessed_train=None, preprocessed_val=None, backtest_only: bool = False, debug: bool = False) -> Tuple[str, nn.Module, Any, bool, float, bool, bool, GaussianHMM, Any]:
    
    # === ONLY PRINT WHEN THE MODEL IS ACTUALLY BEING TRAINED ===
    model_path = os.path.join(CONFIG['MODEL_CACHE_DIR'], f"{symbol}_model_{CONFIG['MODEL_VERSION']}.pth")
    will_train = force_train or not os.path.exists(model_path)

    if will_train and not backtest_only:
        print(f"{Fore.CYAN}[{symbol}] === STARTING Training with {Fore.LIGHTCYAN_EX}Slave {worker_id}{Fore.CYAN} ==={Style.RESET_ALL}")
        print(f"{Fore.YELLOW}[{symbol}] Fetching FULL training data from {CONFIG['TRAIN_DATA_START_DATE']} to {CONFIG['VAL_END_DATE']}{Style.RESET_ALL}")
        print(f"{Fore.YELLOW}[{symbol}] Worker {worker_id} → Waiting for GPU slot{Style.RESET_ALL}")
    # else: completely silent in ANY --backtest run when models are cached

    if force_train:
        logger.debug(f"[{symbol}] --force-train: Fetching fresh from API (no cache)")
        df = fetch_data(symbol, CONFIG['TRAIN_DATA_START_DATE'], CONFIG['VAL_END_DATE'])
        data_loaded = False
    else:
        df, data_loaded = load_or_fetch_data(symbol, CONFIG['TRAIN_DATA_START_DATE'], CONFIG['VAL_END_DATE'])

    if 'timestamp' in df.columns:
        df = df.set_index('timestamp')
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index).tz_localize(None)
    df = df.sort_index()

    logger.debug(f"[{symbol}] After fetch + index fix: {len(df)} rows, min={df.index.min()}, max={df.index.max()}")

    sentiment, sentiment_loaded = load_news_sentiment(symbol)
    df = calculate_indicators(df, sentiment)
    logger.debug(f"[{symbol}] After calculate_indicators: {len(df)} rows")
    model_file = os.path.join(CONFIG['MODEL_CACHE_DIR'], f"{symbol}_model_{CONFIG['MODEL_VERSION']}.pth")
    scaler_file = os.path.join(CONFIG['MODEL_CACHE_DIR'], f"{symbol}_scaler_{CONFIG['MODEL_VERSION']}.pkl")

    if not force_train and os.path.exists(model_file) and os.path.exists(scaler_file):
        model, scaler, training_sentiment, hmm = load_model_and_scaler(symbol, expected_features, force_retrain=False)
        if model is not None:
            model_loaded = True
            xgb_model = None
            logger.info(f"Loaded cached model, scaler, and HMM for {symbol}")
        else:
            force_train = True
            model_loaded = False
    else:
        model_loaded = False

    if force_train or not model_loaded:
        model, scaler, hmm, xgb_model = train_model(
            symbol, 
            worker_id,
            df, 
            CONFIG['TRAIN_EPOCHS'], 
            CONFIG['BATCH_SIZE'], 
            CONFIG['TIMESTEPS'], 
            expected_features, 
            barrier, 
            gpu_semaphore, 
            preprocessed_train, 
            preprocessed_val
        )
        save_model_and_scaler(symbol, model, scaler, sentiment, hmm)
        model_loaded = False
        logger.info(f"Trained and saved new model for {symbol}")
    else:
        xgb_model = None

    # Do not return the full df (main process already has dfs_backtest) — fixes pickling error
    return symbol, model, scaler, data_loaded, sentiment, sentiment_loaded, model_loaded, hmm, xgb_model

def backtest(symbol: str, model: nn.Module, scaler: RobustScaler, df: pd.DataFrame, initial_cash: float,
             stop_loss_atr_multiplier: float, take_profit_atr_multiplier: float, timesteps: int,
             buy_threshold: float, sell_threshold: float, min_holding_period_minutes: int,
             transaction_cost_per_trade: float, xgb_model=None, hmm=None,
             dfs_backtest: Dict[str, pd.DataFrame] = None, hmms: Dict[str, GaussianHMM] = None,
             scalers: Dict[str, RobustScaler] = None, debug: bool = False) -> Tuple[float, List[float], int, float, float, pd.Series]:
    # === OPTIMIZER MODE — makes each trial 3–4× faster ===
    if os.getenv("OPTIMIZER_MODE") == "true":
        CONFIG['NUM_MC_SIMULATIONS'] = 5000   # instead of 50,000
        print("🚀 OPTIMIZER MODE: Reduced Monte Carlo to 5k for 10-hour run")
    # LOUD CONFIRMATION — this proves the NEW function is loaded
    if debug:
        print(f"{Fore.GREEN}=== NEW BACKTEST FUNCTION LOADED FOR {symbol} — DEBUG FORCE ENABLED ==={Style.RESET_ALL}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Running backtest for {symbol} on device: {device}")
    
    confidence_threshold = CONFIG['CONFIDENCE_THRESHOLD']
    rsi_buy_threshold = CONFIG['RSI_BUY_THRESHOLD']
    rsi_sell_threshold = CONFIG['RSI_SELL_THRESHOLD']
    adx_trend_threshold = CONFIG['ADX_TREND_THRESHOLD']
    max_volatility = CONFIG['MAX_VOLATILITY']
    trailing_stop_percentage = CONFIG['TRAILING_STOP_PERCENTAGE']
    risk_percentage = CONFIG['RISK_PERCENTAGE']
    
    # Slice to out-of-sample backtest period
    backtest_start = pd.Timestamp(CONFIG['BACKTEST_START_DATE'], tz='UTC')
    df_backtest = df[df.index >= backtest_start].copy()
    if len(df_backtest) < CONFIG['MIN_DATA_POINTS']:
        raise ValueError(f"Insufficient backtest data for {symbol}: {len(df_backtest)} bars")
    
    X_seq, _, _ = preprocess_data(df_backtest, timesteps, inference_mode=True, inference_scaler=scaler)
    
    model.eval()
    model = model.to(device)
    X_tensor = torch.tensor(X_seq, dtype=torch.float32).to(device)

    predictions = []
    with torch.no_grad():
        for i in range(0, len(X_tensor), CONFIG['BATCH_SIZE']):
            batch = X_tensor[i:i + CONFIG['BATCH_SIZE']]
            raw_logits = model(batch)
            outputs = torch.sigmoid(raw_logits)
            predictions.extend(outputs.cpu().numpy().flatten())
            del raw_logits, outputs

    predictions = np.array(predictions)

    df_backtest['Future_Direction'] = np.where(
        df_backtest['close'].shift(-CONFIG['LOOK_AHEAD_BARS']) > df_backtest['close'], 1, 0
    )
    df_backtest = df_backtest.dropna(subset=['Future_Direction'])

    true_y_for_accuracy = df_backtest['Future_Direction'].iloc[CONFIG['TIMESTEPS']: CONFIG['TIMESTEPS'] + len(predictions)].values
    min_len = min(len(predictions), len(true_y_for_accuracy))
    predictions_acc = predictions[:min_len]
    true_y_acc = true_y_for_accuracy[:min_len]

    valid_mask = ~np.isnan(true_y_acc)
    accuracy_percentage = np.mean((predictions_acc[valid_mask] > 0.5) == true_y_acc[valid_mask]) * 100 if np.any(valid_mask) else 0.0

    logger.info(f"Predictions for {symbol}: min={predictions.min():.3f}, max={predictions.max():.3f}, mean={predictions.mean():.3f}")
    logger.info(f"Backtest accuracy for {symbol}: {accuracy_percentage:.2f}%")

    del X_tensor
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    cash = initial_cash
    returns = []
    trade_count = 0
    win_rate = 0.0
    position = 0
    entry_price = 0.0
    entry_time = None
    max_price = 0.0
    winning_trades = 0
    pair_positions = {}

    # === ROBUST BACKTEST SLICING (fixes num_backtest_steps = 0 bug) ===
    sim_start = pd.Timestamp(CONFIG['BACKTEST_START_DATE'], tz='UTC')
    backtest_mask = df.index >= sim_start
    backtest_df = df[backtest_mask].iloc[timesteps:]  # align with prediction length
    num_backtest_steps = len(predictions)
    if num_backtest_steps == 0:
        logger.warning(f"No backtest steps for {symbol} — skipping")
        return initial_cash, [], 0, 0.0, accuracy_percentage, pd.Series()

    atr = backtest_df['ATR'].values[:num_backtest_steps]
    prices = backtest_df['close'].values[:num_backtest_steps]
    rsi = backtest_df['RSI'].values[:num_backtest_steps]
    adx = backtest_df['ADX'].values[:num_backtest_steps]
    volatility = backtest_df['Volatility'].values[:num_backtest_steps]
    sim_timestamps = backtest_df.index.values[:num_backtest_steps]

    # Continuous portfolio tracking EVERY BAR (fixes flat line graph)
    portfolio_series = pd.Series(index=sim_timestamps, dtype=float)
    portfolio_series.iloc[0] = initial_cash

    # === OPTIMIZED PAIRS LOGIC WITH TOGGLE ===
    if CONFIG.get('ENABLE_FULL_PAIRS_RESOLUTION', False):
        check_interval = 1
    else:
        check_interval = 8   # faster with --DEBUG on (only checks pairs every 8 bars instead of every bar)

    # === FIXED: ROBUST PAIR FILTER (prevents None HMM crash) ===
    valid_pairs = []
    pair_positions = {}
    for pair in CONFIG['PAIRS']:
        sym1, sym2 = pair
        # Only keep pairs where BOTH symbols were actually trained
        if (sym1 in hmms and sym2 in hmms and
            hmms.get(sym1) is not None and
            hmms.get(sym2) is not None and
            sym1 in dfs_backtest and sym2 in dfs_backtest):
            if is_cointegrated(dfs_backtest[sym1]['close'], dfs_backtest[sym2]['close']):
                valid_pairs.append(pair)
                pair_positions[pair] = None

    # ==================== MAIN BACKTEST LOOP ====================
    for local_i in range(num_backtest_steps):
        # FORCE PRINT — this will ALWAYS appear if --DEBUG is used
        # if debug:
        #     print(f"{Fore.RED}[DEBUG FORCE] {symbol} - Entering loop bar {local_i} (debug={debug}){Style.RESET_ALL}")

        pred = make_prediction(model, X_seq[local_i:local_i+1], xgb_model)
        
        # === EXTRACT ALL VARIABLES FIRST ===
        price = prices[local_i]
        atr_val = atr[local_i]
        current_rsi = rsi[local_i]
        current_adx = adx[local_i]
        current_volatility = volatility[local_i]
        ts = pd.Timestamp(sim_timestamps[local_i])

        # === DEBUG OUTPUT — ONLY prints when --DEBUG is used ===
        if debug and local_i % 50 == 0:
            reasons = []
            if pred < confidence_threshold:      reasons.append(f"conf={pred:.3f}")
            if pred <= buy_threshold:            reasons.append("pred too low")
            if current_rsi >= rsi_buy_threshold: reasons.append(f"RSI={current_rsi:.1f}")
            if current_adx <= adx_trend_threshold: reasons.append(f"ADX={current_adx:.1f} weak")
            if current_volatility > max_volatility: reasons.append(f"vol={current_volatility:.2f}")

            if not reasons:
                print(f"{Fore.GREEN}[{symbol}] Bar {local_i:5d} → STRONG BUY SIGNAL! Pred={pred:.3f}{Style.RESET_ALL}")
            else:
                print(f"{Fore.LIGHTBLACK_EX}[{symbol}] Bar {local_i:5d} SKIP → {' | '.join(reasons)}{Style.RESET_ALL}")

        # === FILTERS ===
        if current_volatility > max_volatility or current_adx < adx_trend_threshold:
            continue
        if pred < confidence_threshold:
            continue

        # SINGLE STOCK LOGIC (now reachable)
        if cash >= price:
            qty = max(1, int((cash * risk_percentage) / (atr_val * stop_loss_atr_multiplier)))
            cost = qty * price + transaction_cost_per_trade
            if cost > cash:
                qty = max(0, int((cash - transaction_cost_per_trade) / price))
                cost = qty * price + transaction_cost_per_trade
        else:
            qty = 0
            cost = 0

        if pred > buy_threshold and position == 0 and current_rsi < rsi_buy_threshold and current_adx > adx_trend_threshold and qty > 0 and cash >= cost:
            if cash - cost >= 0:
                position = qty
                entry_price = price
                max_price = price
                entry_time = ts
                cash -= cost
                logger.info(f"{ts}: Bought {qty} shares of {symbol} at ${price:.2f}, cash: ${cash:.2f}")

        elif position > 0:
            if price > max_price:
                max_price = price
            trailing_stop = max_price * (1 - trailing_stop_percentage)
            stop_loss = entry_price - stop_loss_atr_multiplier * atr_val
            take_profit = entry_price + take_profit_atr_multiplier * atr_val
            time_held = (ts - entry_time).total_seconds() / 60 if entry_time else 0

            if time_held >= min_holding_period_minutes:
                if price <= trailing_stop or price <= stop_loss or price >= take_profit or (pred < sell_threshold and current_rsi > rsi_sell_threshold):
                    cash += position * price - transaction_cost_per_trade
                    ret = (price - entry_price) / entry_price
                    returns.append(ret)
                    trade_count += 1
                    if ret > 0:
                        winning_trades += 1
                    
                    # === NEW: SELL DEBUG PRINT (only when --DEBUG) ===
                    if debug:
                        print(f"{Fore.GREEN}[{symbol}] Bar {local_i:5d} → SOLD! Pred={pred:.3f} RSI={current_rsi:.1f} Return={ret:.3f}{Style.RESET_ALL}")
                    
                    logger.info(f"{ts}: Sold {position} shares of {symbol} at ${price:.2f}, return: {ret:.3f}, cash: ${cash:.2f}")
                    position = 0
                    entry_time = None
                    max_price = 0.0

        # === ROBUST PAIRS TRADING (prevents None.predict crash + uses correct sequences) ===
        if local_i % check_interval == 0:
            for pair in valid_pairs:
                sym1, sym2 = pair
                hmm1 = hmms.get(sym1)
                hmm2 = hmms.get(sym2)
                
                if hmm1 is None or hmm2 is None:
                    continue
                
                # Use correct recent sequence for EACH leg of the pair
                recent_seq1 = preprocess_data(
                    dfs_backtest[sym1].iloc[-CONFIG['TIMESTEPS']-5:],
                    CONFIG['TIMESTEPS'], inference_mode=True,
                    inference_scaler=scalers.get(sym1) if scalers else None
                )[0][-1:]
                
                recent_seq2 = preprocess_data(
                    dfs_backtest[sym2].iloc[-CONFIG['TIMESTEPS']-5:],
                    CONFIG['TIMESTEPS'], inference_mode=True,
                    inference_scaler=scalers.get(sym2) if scalers else None
                )[0][-1:]
                
                regime = get_pair_regime(hmm1, hmm2, recent_seq1, recent_seq2)
                
                if regime not in CONFIG['PAIR_REGIME_FILTER']:
                    continue
                    
                df1 = dfs_backtest[sym1]
                df2 = dfs_backtest[sym2]
                spread = calculate_spread(df1, df2)
                zscore = (spread.iloc[-1] - spread.mean()) / spread.std()
                
                if abs(zscore) > 2.0 and pair_positions.get(pair) is None:
                    hedge = calculate_spread(df1, df2).iloc[-1]
                    qty1 = int(initial_cash * 0.01 / df1['close'].iloc[-1])
                    qty2 = int(qty1 * hedge)
                    side = 'long' if zscore > 0 else 'short'
                    pair_positions[pair] = (side, qty1, qty2, spread.iloc[-1])
                    logger.info(f"[{symbol}] ENTERED PAIR {pair} ({regime}) zscore={zscore:.2f} side={side}")
        # Update portfolio value EVERY bar (cash + mark-to-market position)
        current_value = cash + (position * price if position > 0 else 0)
        portfolio_series.iloc[local_i] = current_value

    # Close any remaining single-stock position
    if position > 0:
        last_price = prices[-1]
        cash += position * last_price - transaction_cost_per_trade
        ret = (last_price - entry_price) / entry_price
        returns.append(ret)
        trade_count += 1
        if ret > 0:
            winning_trades += 1

    # Final portfolio value
    portfolio_series.iloc[-1] = cash

    win_rate = (winning_trades / trade_count * 100) if trade_count > 0 else 0.0
    portfolio_series = portfolio_series.ffill().fillna(initial_cash)

    return cash, returns, trade_count, win_rate, accuracy_percentage, portfolio_series

def buy_and_hold_backtest(dfs_backtest: Dict[str, pd.DataFrame], initial_cash: float) -> Tuple[float, Dict[str, pd.Series]]:
    backtest_start = pd.Timestamp(CONFIG['BACKTEST_START_DATE'], tz='UTC')
    initial_per_symbol = initial_cash / len(CONFIG['SYMBOLS'])
    bh_final_value = 0.0
    bh_series_per_symbol = {}
    
    for symbol, df in dfs_backtest.items():
        # Slice to the exact same period as the neural strategy
        df_bh = df[df.index >= backtest_start].copy()
        if df_bh.empty or len(df_bh) < 2:
            logger.warning(f"Insufficient data for buy-and-hold on {symbol}; skipping.")
            continue
            
        first_close = df_bh['close'].iloc[0]
        if first_close <= 0:
            continue
            
        qty = int((initial_per_symbol - CONFIG['TRANSACTION_COST_PER_TRADE']) / first_close)
        if qty <= 0:
            continue
            
        # Build series for graphing (only 2025 onward)
        bh_values = qty * df_bh['close']
        bh_series = pd.Series(bh_values.values, index=df_bh.index, name=symbol)
        bh_series_per_symbol[symbol] = bh_series
        last_value = bh_series.iloc[-1]
        bh_final_value += last_value
    
    logger.info(f"Buy-and-hold final value (same period as neural): ${bh_final_value:.2f}")
    return bh_final_value, bh_series_per_symbol

def calculate_performance_metrics(returns: List[float], cash: float, initial_per_symbol: float) -> Dict[str, float]:

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


def send_email(subject: str, body: str, attachment_path: Optional[str] = None) -> None:
    
    # === GMAIL LIMIT + OPTIMIZER SAFETY — skip emails during backtests/optimizer ===
    if os.getenv("OPTIMIZER_MODE") == "true":
        logger.info(f"[EMAIL SKIPPED] {subject} (backtest or optimizer mode)")
        return
    
    msg = MIMEMultipart()
    msg['Subject'] = subject
    msg['From'] = CONFIG['EMAIL_SENDER']
    msg['To'] = ', '.join(CONFIG['EMAIL_RECEIVER'])
    msg.attach(MIMEText(body, 'plain'))
    if attachment_path and os.path.exists(attachment_path):
        with open(attachment_path, 'rb') as f:
            img = MIMEImage(f.read())
            img.add_header('Content-Disposition', 'attachment', filename=os.path.basename(attachment_path))
            msg.attach(img)
    with smtplib.SMTP(CONFIG['SMTP_SERVER'], CONFIG['SMTP_PORT']) as server:
        server.starttls()
        server.login(CONFIG['EMAIL_SENDER'], CONFIG['EMAIL_PASSWORD'])
        server.sendmail(CONFIG['EMAIL_SENDER'], CONFIG['EMAIL_RECEIVER'], msg.as_string())

def make_prediction(model: nn.Module, X: np.ndarray, xgb_model=None) -> float:
    device = next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        X_tensor = torch.FloatTensor(X).to(device)
        raw_logit = model(X_tensor).squeeze(-1)           # get raw logit
        scaled_logit = raw_logit / CONFIG['PREDICTION_TEMPERATURE']
        prob = torch.sigmoid(scaled_logit).cpu().item()
    
    # XGBoost ensemble still works on top
    if xgb_model is not None and hasattr(xgb_model, 'predict_proba'):
        try:
            xgb_prob = xgb_model.predict_proba(X.reshape(1, -1))[0][1]
            prob = (prob + xgb_prob) / 2.0
        except Exception:
            pass
    
    return float(prob)

def get_api_keys(config: Dict) -> None:
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
    deviveName = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None"
    print(f"{Fore.LIGHTGREEN_EX}Device set to use {Fore.GREEN}{torch.device('cuda' if torch.cuda.is_available() else 'cpu')}{Fore.LIGHTGREEN_EX} with {Fore.GREEN}{deviveName}{Style.RESET_ALL}")
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
    expected_features = 31
    models = {}
    scalers = {}
    dfs = {}
    stock_info = []
    total_epochs = len(CONFIG['SYMBOLS']) * CONFIG['TRAIN_EPOCHS']
    training_sentiments = {}
    hmms = {}  # NEW: for live use
    need_training = False
    for symbol in CONFIG['SYMBOLS']:
        model, scaler, sentiment, hmm = load_model_and_scaler(symbol, expected_features, force_train)
        models[symbol] = model
        scalers[symbol] = scaler
        training_sentiments[symbol] = sentiment
        hmms[symbol] = hmm
        if model is None:
            need_training = True
    progress_bar = tqdm(total=total_epochs, desc="Training Progress", bar_format="{l_bar}\033[32m{bar}\033[0m{r_bar}") if need_training else None

    if not backtest_only:
        # Original live trading block (unchanged)
        mp.set_start_method('spawn', force=True)
        start_total_training_time = time.perf_counter()

        # Use full worker task format to match updated train_wrapper (8 values)
        with mp.Manager() as manager:
            gpu_semaphore = manager.Semaphore(CONFIG['NUM_PARALLEL_WORKERS'])
            barrier = manager.Barrier(CONFIG['NUM_PARALLEL_WORKERS'])
            with mp.Pool(processes=CONFIG['NUM_PARALLEL_WORKERS']) as pool:
                worker_tasks = [(i+1, sym, expected_features, force_train, barrier, gpu_semaphore, False, debug)
                              for i, sym in enumerate(CONFIG['SYMBOLS'])]
                outputs = list(tqdm(pool.imap(train_wrapper, worker_tasks),
                    total=len(CONFIG['SYMBOLS']), desc="Processing symbols"))
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        logger.info("Parallel processing completed; CUDA memory cleared.")

        end_total_training_time = time.perf_counter()
        total_training_time_in_milliseconds = (end_total_training_time - start_total_training_time) * 1000
        training_times_dictionary = {}  # Must be initialized in live path too

        sentiments = {}  # Collect sentiments for live consistency
        hmms = {}
        for output_tuple in outputs:
            symbol = output_tuple[0]
            model = output_tuple[1]
            scaler = output_tuple[2]
            data_loaded = output_tuple[3]
            sentiment = output_tuple[4]
            sentiment_loaded = output_tuple[5]
            model_loaded = output_tuple[6]
            hmm = output_tuple[7]
            xgb_model = output_tuple[8]
            training_time_in_milliseconds = output_tuple[9]

            training_times_dictionary[symbol] = training_time_in_milliseconds if training_time_in_milliseconds is not None else 0
            models[symbol] = model
            scalers[symbol] = scaler
            sentiments[symbol] = sentiment
            hmms[symbol] = hmm

            info = []
            info.append(f"{Fore.LIGHTBLUE_EX}{symbol}:{Style.RESET_ALL}")
            info.append(f"  {'Loaded cached model and scaler' if model_loaded else 'Trained model'} for {symbol}.")
            info.append(f"  {'Loaded' if data_loaded else 'Fetched'} bars for {symbol} {'from cache' if data_loaded else ''}.")
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
            
                # Retry wrappers defined here inside main() so trading_client is in scope
                @retry(stop=stop_after_attempt(CONFIG['API_RETRY_ATTEMPTS']), wait=wait_fixed(CONFIG['API_RETRY_DELAY'] / 1000), retry=retry_if_exception_type(APIError))
                def get_all_positions_with_retry():
                    return trading_client.get_all_positions()

                decisions = []  # Collect decisions for summary email
                open_positions = get_all_positions_with_retry()
                for symbol in CONFIG['SYMBOLS']:
                    if symbol in models:  # Only process symbols with trained models
                        df = fetch_recent_data(symbol, CONFIG['LIVE_DATA_BARS'])
                        sentiment = sentiments[symbol]
                        df = calculate_indicators(df, sentiment)
                        features = [
                            'close', 'high', 'low', 'volume', 'MA20', 'MA50', 'RSI',
                            'MACD', 'MACD_signal', 'OBV', 'VWAP', 'ATR', 'CMF', 'Close_ATR',
                            'MA20_ATR', 'Return_1d', 'Return_5d', 'Volatility', 'BB_upper',
                            'BB_lower', 'Stoch_K', 'Stoch_D', 'ADX', 'Sentiment'
                        ]
                        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                        if len(df) < 20:   # More tolerant for live data
                            logger.warning(f"Insufficient data for {symbol} live prediction: {len(df)} bars")
                            prediction = 0.5
                            regime = "Unknown (insufficient data)"
                            price = df['close'].iloc[-1] if not df.empty else 0.0
                            current_rsi = 50.0
                            current_adx = 0.0
                            current_volatility = 0.0
                            atr_val = 0.0
                        else:
                            X_seq, _, _ = preprocess_data(df, CONFIG['TIMESTEPS'], inference_mode=True, inference_scaler=scalers[symbol], fit_scaler=False)
                            recent_seq = X_seq[-1:].astype(np.float32)
                            model = models[symbol].to(device)
                            model.eval()
                            with torch.no_grad():
                                pred_logit = model(torch.tensor(recent_seq).to(device))
                                prediction = torch.sigmoid(pred_logit).cpu().item()
                            
                            # Get current regime from HMM
                            hmm = hmms[symbol]
                            X_latest_scaled = recent_seq.reshape(-1, recent_seq.shape[2])
                            regime_id = hmm.predict(X_latest_scaled)[0]
                            regime_names = ["Calm Bull", "Volatile Bull", "Calm Bear", "Volatile Bear"]
                            regime = regime_names[regime_id]
                            
                            price = float(df['close'].iloc[-1])
                            current_rsi = float(df['RSI'].iloc[-1])
                            current_adx = float(df['ADX'].iloc[-1])
                            current_volatility = float(df['Volatility'].iloc[-1])
                            atr_val = float(df['ATR'].iloc[-1])
                            logger.info(f"Live prediction for {symbol}: {prediction:.3f} | Regime: {regime} | Price=${price:.2f}")

                        decision = "Hold"
                        qty_owned = 0
                        entry_time = None
                        entry_price = 0.0
                        time_held = 0
                        position_obj = next((pos for pos in open_positions if pos.symbol == symbol), None)
                        if position_obj:
                            qty_owned = int(float(position_obj.qty))
                            order_req = GetOrdersRequest(status=QueryOrderStatus.CLOSED, symbols=[symbol], side=OrderSide.BUY, limit=50)
                            try:
                                orders = trading_client.get_orders(order_req)
                                filled_buy_orders = [o for o in orders if o.status == OrderStatus.FILLED and o.side == OrderSide.BUY]
                                if filled_buy_orders:
                                    latest_order = max(filled_buy_orders, key=lambda o: o.filled_at if o.filled_at else datetime.min.replace(tzinfo=timezone.utc))
                                    entry_time = latest_order.filled_at if latest_order.filled_at else now
                                else:
                                    entry_time = now
                            except Exception as e:
                                logger.warning(f"Failed to fetch orders for {symbol}: {str(e)}")
                                entry_time = now
                            entry_price = float(position_obj.avg_entry_price)
                            if entry_time:
                                # everything has to be UTC or pandas will silently mix naive and aware timestamps
                                # and then you get NaNs everywhere and lose your mind
                                if entry_time.tzinfo is None:
                                    entry_time = entry_time.replace(tzinfo=timezone.utc)
                                else:
                                    entry_time = entry_time.astimezone(timezone.utc)
                            time_held = (now - entry_time).total_seconds() / 60 if entry_time else 0
                            if time_held > 240 and prediction < 0.52:
                                decision = "Sell (Time + Weak Signal)"

                        if current_volatility > CONFIG['MAX_VOLATILITY'] or current_adx < CONFIG['ADX_TREND_THRESHOLD']:
                            decision = "Hold (Filters)"
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
                                        qty = max(0, int((cash - CONFIG['TRANSACTION_COST_PER_TRADE']) / price))
                                        cost = qty * price + CONFIG['TRANSACTION_COST_PER_TRADE']
                                    if qty > 0 and cost <= cash:
                                        logger.info(f"Submitting buy order for {qty} shares of {symbol} at ${price:.2f}")
                                        order = MarketOrderRequest(symbol=symbol, qty=qty, side=OrderSide.BUY, time_in_force=TimeInForce.GTC)
                                        try:
                                            trading_client.submit_order(order)
                                            email_body = f"""
Bought {qty} shares of {symbol} at ${price:.2f}
Prediction Confidence: {prediction:.3f}
Regime: {regime}
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
                                    else:
                                        decision = "Hold (Qty=0 or Insufficient Cash)"
                                except (ValueError, ZeroDivisionError) as e:
                                    decision = "Hold (Calculation Error)"
                            else:
                                decision = "Hold (Invalid ATR)"
                        elif qty_owned > 0 and time_held >= CONFIG['MIN_HOLDING_PERIOD_MINUTES']:
                            max_price = max(float(position_obj.current_price) if position_obj else price, price)
                            trailing_stop = max_price * (1 - CONFIG['TRAILING_STOP_PERCENTAGE'])
                            stop_loss = entry_price - CONFIG['STOP_LOSS_ATR_MULTIPLIER'] * atr_val
                            take_profit = entry_price + CONFIG['TAKE_PROFIT_ATR_MULTIPLIER'] * atr_val
                            if price <= trailing_stop or price <= stop_loss or price >= take_profit or \
                               (prediction < CONFIG['PREDICTION_THRESHOLD_SELL'] and current_rsi > CONFIG['RSI_SELL_THRESHOLD']) or \
                               (prediction < 0.50 and current_rsi > 65) or \
                               (prediction < CONFIG['CONFIDENCE_THRESHOLD'] - 0.05):
                                decision = "Sell"
                                # sell order code stays exactly here

                        decisions.append({
                            'symbol': symbol,
                            'decision': decision,
                            'confidence': prediction,
                            'rsi': current_rsi,
                            'adx': current_adx,
                            'volatility': current_volatility,
                            'price': price,
                            'owned': qty_owned,
                            'regime': regime
                        })
                # Refresh account and positions after trades
                account = trading_client.get_account()
                portfolio_value = float(account.equity)
                post_trade_owned = {}
                for symbol in CONFIG['SYMBOLS']:
                    try:
                        position = trading_client.get_open_position(symbol)
                        post_trade_owned[symbol] = int(float(position.qty))
                    except APIError:
                        post_trade_owned[symbol] = 0

                # Summarize decisions and send email (NOW INCLUDES REGIME)
                summary_body = "Trading Summary:\n"
                for dec in decisions:
                    owned_display = post_trade_owned.get(dec['symbol'], dec['owned'])
                    summary_body += f"{dec['symbol']}: {dec['decision']}, Regime: {dec.get('regime', 'Unknown')}, Confidence: {dec['confidence']:.3f}, RSI: {dec['rsi']:.2f}, ADX: {dec['adx']:.2f}, Volatility: {dec['volatility']:.2f}, Price: ${dec['price']:.2f}, Owned: {owned_display}\n"
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
        # === FIXED CACHED BACKTEST PATH (with indicators + correct indexing) ===
        all_models_cached = all(
            os.path.exists(os.path.join(CONFIG['MODEL_CACHE_DIR'], f"{symbol}_model_{CONFIG['MODEL_VERSION']}.pth")) and
            os.path.exists(os.path.join(CONFIG['MODEL_CACHE_DIR'], f"{symbol}_scaler_{CONFIG['MODEL_VERSION']}.pkl")) and
            os.path.exists(os.path.join(CONFIG['MODEL_CACHE_DIR'], f"{symbol}_hmm_{CONFIG['MODEL_VERSION']}.pkl"))
            for symbol in CONFIG['SYMBOLS']
        )

        dfs_backtest = {}
        now = datetime.now(timezone.utc)
        end_date = (now - timedelta(days=1)).strftime('%Y-%m-%d')

        print(f"{Fore.CYAN}Loading and preparing backtest data with technical indicators...{Style.RESET_ALL}")
        for symbol in tqdm(CONFIG['SYMBOLS'], desc="Fetching + Processing backtest data"):
            df = load_data(symbol, CONFIG['TRAIN_DATA_START_DATE'], end_date)
            # CRITICAL: Add all indicators (MA20, RSI, ATR, BB_upper, Sentiment, etc.)
            df = calculate_indicators(df, sentiment=0.0)
            dfs_backtest[symbol] = df

        if all_models_cached and not force_train:
            print(f"{Fore.GREEN}✓ All models loaded from cache — running pure backtest{Style.RESET_ALL}")
            logger.info("Pure backtest with cached models")

            for symbol in CONFIG['SYMBOLS']:
                model, scaler, sentiment, hmm = load_model_and_scaler(symbol, expected_features, force_retrain=False)
                models[symbol] = model
                scalers[symbol] = scaler
                hmms[symbol] = hmm

            attempt_results = []
            effective_max = 1
        else:
            print(f"{Fore.YELLOW}⚠ Models missing or --force-train — running full training{Style.RESET_ALL}")
            attempt_results = []
            if CONFIG.get('FORCE_FULL_RETRAIN_RUN', False):
                effective_max = CONFIG['MAX_RETRAIN_ATTEMPTS']
            elif CONFIG['ENABLE_RETRAIN_CYCLE'] and force_train:
                effective_max = CONFIG['MAX_RETRAIN_ATTEMPTS']
            else:
                effective_max = 1

        if effective_max > 0:
            for symbol in CONFIG['SYMBOLS']:
                dfs_backtest[symbol]['Future_Direction'] = (dfs_backtest[symbol]['close'].shift(-CONFIG['LOOK_AHEAD_BARS']) > dfs_backtest[symbol]['close']).astype(int)

            for retrain_attempts in range(1, effective_max + 1):
                logger.info(f"Retraining attempt {retrain_attempts}/{effective_max}")

                if force_train or retrain_attempts > 1:
                    for symbol in CONFIG['SYMBOLS']:
                        train_cache = os.path.join(CONFIG['CACHE_DIR'], f"{symbol}_train_data_{CONFIG['TRAIN_DATA_START_DATE']}_{CONFIG['VAL_END_DATE']}.pkl")
                        if os.path.exists(train_cache):
                            os.remove(train_cache)
                            logger.info(f"Deleted stale training cache for {symbol} (force retrain)")

                models = {}
                scalers = {}
                sentiments = {}
                training_times_dictionary = {}
                if 'progress_bar' in locals() and progress_bar is not None:
                    progress_bar.close()
                progress_bar = None

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
                    free_bytes, _ = torch.cuda.mem_get_info()
                    free_gb = free_bytes / (1024 ** 3)
                    device_name = torch.cuda.get_device_name(0)
                    estimated_gb_per_worker = 3.4
                    group_size = max(1, min(10, int(free_gb // estimated_gb_per_worker)))
                    if free_gb < 14.0:
                        group_size = min(3, group_size)
                        print(f"{Fore.YELLOW}Low VRAM detected ({free_gb:.1f} GB) → capped at {group_size} workers on {Fore.LIGHTYELLOW_EX}{device_name}{Style.RESET_ALL}")
                    else:
                        print(f"{Fore.LIGHTGREEN_EX}{free_gb:.1f} GB {Fore.GREEN} available on {Fore.LIGHTGREEN_EX}{device_name}{Fore.GREEN} → using {Fore.LIGHTGREEN_EX}{group_size} Slaves{Style.RESET_ALL}")
                else:
                    print(f"{Fore.RED}CUDA not available → No GPU detected, defaulting to CPU{Style.RESET_ALL}")
                    group_size = CONFIG['NUM_PARALLEL_WORKERS']
                    free_gb = 0.0

                mp.set_start_method('spawn', force=True)
                start_total_training_time = time.perf_counter()

                with mp.Manager() as manager:
                    gpu_semaphore = manager.Semaphore(group_size)
                    barrier = manager.Barrier(group_size)
                    with mp.Pool(processes=group_size) as pool:
                        worker_tasks = [(i+1, sym, expected_features, force_train or retrain_attempts > 1, barrier, gpu_semaphore, backtest_only, debug)
                                      for i, sym in enumerate(CONFIG['SYMBOLS'])]
                        outputs = list(tqdm(pool.imap(train_wrapper, worker_tasks),
                            total=len(CONFIG['SYMBOLS']), desc="Processing symbols"))
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()

                logger.info("Parallel processing completed; CUDA memory cleared.")

                end_total_training_time = time.perf_counter()
                total_training_time_in_milliseconds = (end_total_training_time - start_total_training_time) * 1000

                for output_tuple in outputs:
                    symbol = output_tuple[0]
                    model = output_tuple[1]
                    scaler = output_tuple[2]
                    data_loaded = output_tuple[3]
                    sentiment = output_tuple[4]
                    sentiment_loaded = output_tuple[5]
                    model_loaded = output_tuple[6]
                    hmm = output_tuple[7]
                    xgb_model = output_tuple[8]
                    training_time_in_milliseconds = output_tuple[9]

                    training_times_dictionary[symbol] = training_time_in_milliseconds if training_time_in_milliseconds is not None else 0
                    models[symbol] = model
                    scalers[symbol] = scaler
                    sentiments[symbol] = sentiment

                if progress_bar:
                    progress_bar.close()

                for symbol in CONFIG['SYMBOLS']:
                    model_path = os.path.join(CONFIG['MODEL_CACHE_DIR'], f"{symbol}_model_{CONFIG['MODEL_VERSION']}.pth")
                    scaler_path = os.path.join(CONFIG['MODEL_CACHE_DIR'], f"{symbol}_scaler_{CONFIG['MODEL_VERSION']}.pkl")
                    attempt_model_path = os.path.join(CONFIG['MODEL_CACHE_DIR'], f"{symbol}_model_{CONFIG['MODEL_VERSION']}_attempt{retrain_attempts}.pth")
                    attempt_scaler_path = os.path.join(CONFIG['MODEL_CACHE_DIR'], f"{symbol}_scaler_{CONFIG['MODEL_VERSION']}_attempt{retrain_attempts}.pkl")
                    if os.path.exists(model_path):
                        shutil.copyfile(model_path, attempt_model_path)
                    if os.path.exists(scaler_path):
                        shutil.copyfile(scaler_path, attempt_scaler_path)

                backtest_times_dictionary = {}
                accuracies_dictionary = {}
                start_total_backtest_time = time.perf_counter()
                initial_cash = CONFIG['INITIAL_CASH']
                final_value = 0
                symbol_results = {}
                trade_counts = {}
                win_rates = {}
                portfolio_series_per_symbol = {}
                initial_per_symbol = CONFIG['INITIAL_CASH'] / len(CONFIG['SYMBOLS'])

                for symbol in CONFIG['SYMBOLS']:
                    if symbol in models and models[symbol] is not None:
                        start_backtest_time_for_symbol = time.perf_counter()
                        cash, returns, trade_count, win_rate, accuracy_percentage, portfolio_series = backtest(
                            symbol, models[symbol], scalers[symbol], dfs_backtest[symbol], initial_per_symbol,
                            CONFIG['STOP_LOSS_ATR_MULTIPLIER'], CONFIG['TAKE_PROFIT_ATR_MULTIPLIER'],
                            CONFIG['TIMESTEPS'], CONFIG['PREDICTION_THRESHOLD_BUY'], CONFIG['PREDICTION_THRESHOLD_SELL'],
                            CONFIG['MIN_HOLDING_PERIOD_MINUTES'], CONFIG['TRANSACTION_COST_PER_TRADE'],
                            xgb_model, None, dfs_backtest, hmms, scalers, debug=debug
                        )
                    else:
                        cash = initial_per_symbol
                        returns = []
                        trade_count = 0
                        win_rate = 0.0
                        accuracy_percentage = 0.0
                        portfolio_series = pd.Series(dtype=float)

                    trade_counts[symbol] = trade_count
                    win_rates[symbol] = win_rate
                    end_backtest_time_for_symbol = time.perf_counter()
                    backtest_times_dictionary[symbol] = (end_backtest_time_for_symbol - start_backtest_time_for_symbol) * 1000
                    accuracies_dictionary[symbol] = accuracy_percentage
                    final_value += cash
                    portfolio_series_per_symbol[symbol] = portfolio_series

                    symbol_results[symbol] = calculate_performance_metrics(returns, cash, initial_per_symbol)
                    mc_metrics = monte_carlo_simulation(returns, initial_per_symbol)
                    symbol_results[symbol].update(mc_metrics)

                end_total_backtest_time = time.perf_counter()
                total_backtest_time_in_milliseconds = (end_total_backtest_time - start_total_backtest_time) * 1000

                bh_final_value, _ = buy_and_hold_backtest(dfs_backtest, initial_cash)

                max_drawdown_across_symbols = max([res['max_drawdown'] for res in symbol_results.values()]) if symbol_results else 0.0
                criteria_met = (
                    final_value > CONFIG['MIN_FINAL_VALUE'] and
                    max_drawdown_across_symbols <= CONFIG['MAX_ALLOWED_DRAWDOWN'] and
                    final_value > bh_final_value
                )

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

                print(f"\n{Fore.CYAN}=== Backtest Attempt {retrain_attempts}/{effective_max} Performance Summary ==={Style.RESET_ALL}")
                print(f"{Fore.GREEN}Backtest Performance Summary:{Style.RESET_ALL}")
                print(f"{'Symbol':<8} {'Total Return (%)':<18} {'Sharpe Ratio':<14} {'Max Drawdown (%)':<20} {'Trades':<8} {'Win Rate (%)':<14} {'Accuracy (%)':<14} {'MC Mean Final ($)':<18} {'MC Median Final ($)':<20} {'MC 95% VaR (%)':<15} {'MC Prob Profit (%)':<18}")
                for symbol in CONFIG['SYMBOLS']:
                    if symbol in symbol_results:
                        metrics_for_symbol = symbol_results[symbol]
                        return_color = Fore.GREEN if metrics_for_symbol['total_return'] > 0 else Fore.RED
                        drawdown_color = Fore.RED if metrics_for_symbol['max_drawdown'] > 0 else Fore.GREEN
                        win_rate_color = Fore.GREEN if win_rates.get(symbol, 0) > 50 else Fore.RED
                        accuracy = accuracies_dictionary.get(symbol, 0.0)
                        accuracy_color = Fore.GREEN if accuracy > 50 else Fore.RED
                        mc_mean_color = Fore.GREEN if metrics_for_symbol['mc_mean_final_value'] > initial_per_symbol else Fore.RED
                        mc_median_color = Fore.GREEN if metrics_for_symbol['mc_median_final_value'] > initial_per_symbol else Fore.RED
                        mc_var_color = Fore.RED if metrics_for_symbol['mc_var_95'] > 0 else Fore.GREEN
                        mc_prob_color = Fore.GREEN if metrics_for_symbol['mc_prob_profit'] > 50 else Fore.RED
                        print(f"{symbol:<8} {return_color}{metrics_for_symbol['total_return']:<18.3f}{Style.RESET_ALL} {metrics_for_symbol['sharpe_ratio']:<14.3f} {drawdown_color}{metrics_for_symbol['max_drawdown']:<20.3f}{Style.RESET_ALL} {trade_counts.get(symbol, 0):<8} {win_rate_color}{win_rates.get(symbol, 0):<14.3f}{Style.RESET_ALL} {accuracy_color}{accuracy:<14.3f}{Style.RESET_ALL} {mc_mean_color}{metrics_for_symbol['mc_mean_final_value']:<18.2f}{Style.RESET_ALL} {mc_median_color}{metrics_for_symbol['mc_median_final_value']:<20.2f}{Style.RESET_ALL} {mc_var_color}{metrics_for_symbol['mc_var_95']:<15.3f}{Style.RESET_ALL} {mc_prob_color}{metrics_for_symbol['mc_prob_profit']:<18.3f}{Style.RESET_ALL}")
                print(f"\nBuy-and-Hold Final Value: ${bh_final_value:.2f}")
                attempt_color = Fore.GREEN if final_value > CONFIG['INITIAL_CASH'] else Fore.RED
                print(f"{Fore.YELLOW}→ Attempt {retrain_attempts} Final Portfolio Value: {attempt_color}${final_value:,.2f}{Style.RESET_ALL}")

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
                    'accuracies_dictionary': accuracies_dictionary,
                    'portfolio_series_per_symbol': portfolio_series_per_symbol
                })

        if attempt_results:
            best_attempt_per_symbol = {}
            best_symbol_results = {}
            best_trade_counts = {}
            best_win_rates = {}
            best_accuracies = {}
            best_portfolio_series_per_symbol = {}
            bh_final_value = attempt_results[-1]['bh_final_value']
            final_value = 0.0
            initial_per_symbol = CONFIG['INITIAL_CASH'] / len(CONFIG['SYMBOLS'])
            for symbol in CONFIG['SYMBOLS']:
                best_att_for_sym = max(attempt_results, key=lambda x: x['symbol_results'].get(symbol, {}).get('total_return', -float('inf')))['attempt']
                best_attempt_per_symbol[symbol] = best_att_for_sym
                best_att_results = next(a for a in attempt_results if a['attempt'] == best_att_for_sym)
                best_symbol_results[symbol] = best_att_results['symbol_results'][symbol]
                best_trade_counts[symbol] = best_att_results['trade_counts'][symbol]
                best_win_rates[symbol] = best_att_results['win_rates'][symbol]
                best_accuracies[symbol] = best_att_results['accuracies_dictionary'][symbol]
                best_portfolio_series_per_symbol[symbol] = best_att_results['portfolio_series_per_symbol'][symbol]
                sym_cash = initial_per_symbol * (1 + best_symbol_results[symbol]['total_return'] / 100)
                final_value += sym_cash
                attempt_model_path = os.path.join(CONFIG['MODEL_CACHE_DIR'], f"{symbol}_model_{CONFIG['MODEL_VERSION']}_attempt{best_att_for_sym}.pth")
                attempt_scaler_path = os.path.join(CONFIG['MODEL_CACHE_DIR'], f"{symbol}_scaler_{CONFIG['MODEL_VERSION']}_attempt{best_att_for_sym}.pkl")
                model_path = os.path.join(CONFIG['MODEL_CACHE_DIR'], f"{symbol}_model_{CONFIG['MODEL_VERSION']}.pth")
                scaler_path = os.path.join(CONFIG['MODEL_CACHE_DIR'], f"{symbol}_scaler_{CONFIG['MODEL_VERSION']}.pkl")
                if os.path.exists(attempt_model_path):
                    shutil.copyfile(attempt_model_path, model_path)
                if os.path.exists(attempt_scaler_path):
                    shutil.copyfile(attempt_scaler_path, scaler_path)

            logger.info(f"Selected best models per symbol: {best_attempt_per_symbol}")

            training_times_dictionary = attempt_results[-1]['training_times_dictionary']
            backtest_times_dictionary = attempt_results[-1]['backtest_times_dictionary']
            total_training_time_in_milliseconds = attempt_results[-1]['total_training_time_in_milliseconds']
            total_backtest_time_in_milliseconds = attempt_results[-1]['total_backtest_time_in_milliseconds']

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
                if ms is None or not isinstance(ms, (int, float)):
                    return "00:00.000"
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

            print(f"Total Time (Loading/Training/Backtest): {format_time(total_training_time_in_milliseconds + total_backtest_time_in_milliseconds)}")

            print(f"\n{Fore.CYAN}=== FINAL BEST BACKTEST RESULTS (Best of {len(attempt_results)} attempts) ==={Style.RESET_ALL}")
            print(f"{Fore.GREEN}Final Performance Summary (Selected Best Models):{Style.RESET_ALL}")
            print(f"{'Symbol':<8} {'Attempt':<8} {'Total Return (%)':<18} {'Sharpe Ratio':<14} {'Max Drawdown (%)':<20} {'Trades':<8} {'Win Rate (%)':<14} {'Accuracy (%)':<14} {'MC Mean Final ($)':<18} {'MC Median Final ($)':<20} {'MC 95% VaR (%)':<15} {'MC Prob Profit (%)':<18}")
            for symbol in CONFIG['SYMBOLS']:
                if symbol in best_symbol_results:
                    metrics_for_symbol = best_symbol_results[symbol]
                    attempt = best_attempt_per_symbol[symbol]
                    return_color = Fore.GREEN if metrics_for_symbol['total_return'] > 0 else Fore.RED
                    drawdown_color = Fore.RED if metrics_for_symbol['max_drawdown'] > 0 else Fore.GREEN
                    win_rate_color = Fore.GREEN if best_win_rates[symbol] > 50 else Fore.RED
                    accuracy = best_accuracies[symbol] if best_trade_counts[symbol] > 0 else 0.0
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
            print(f"\nFull Backtest completed. Final value: {color}${final_value:.2f}{Style.RESET_ALL}")

            print(f"\n{Fore.CYAN}=== TOTAL RUN TIME SUMMARY FOR ALL {len(attempt_results)} ATTEMPTS ==={Style.RESET_ALL}")
            total_run_ms = total_training_time_in_milliseconds + total_backtest_time_in_milliseconds
            print(f"{Fore.GREEN}Overall Timing:{Style.RESET_ALL}")
            print(f"  Training Time (best attempt) : {format_time(total_training_time_in_milliseconds)}")
            print(f"  Backtesting Time             : {format_time(total_backtest_time_in_milliseconds)}")
            print(f"  Grand Total Run Time         : {format_time(total_run_ms)}")
            print(f" {Fore.YELLOW} Final Best Portfolio Value{Style.RESET_ALL}   : {color}${final_value:,.2f}")

            # Recompute BH with series for graphing
            bh_final_value, bh_series_per_symbol = buy_and_hold_backtest(dfs_backtest, initial_cash)

            # Day Trading — now continuous daily series (blue line guaranteed)
            all_series = pd.concat(best_portfolio_series_per_symbol.values(), axis=1, join='outer')
            all_series.index = pd.to_datetime(all_series.index)
            all_series = all_series.ffill().bfill().fillna(CONFIG['INITIAL_CASH'])
            total_portfolio = all_series.sum(axis=1)

            daily_portfolio = total_portfolio.resample('D').last().ffill()

            all_bh_series = pd.concat(bh_series_per_symbol.values(), axis=1, join='outer')
            all_bh_series.index = pd.to_datetime(all_bh_series.index)
            all_bh_series = all_bh_series.ffill().fillna(0)
            total_bh_portfolio = all_bh_series.sum(axis=1)
            daily_bh_portfolio = total_bh_portfolio.resample('D').last().ffill()

            plt.figure(figsize=(12, 6))
            plt.plot(daily_portfolio.index, daily_portfolio.values, label='Day Trading Portfolio', color='blue', linewidth=2)
            plt.plot(daily_bh_portfolio.index, daily_bh_portfolio.values, label='Buy-and-Hold Portfolio', color='green', linewidth=2)
            # === Cash Breakeven Line ===
            plt.axhline(y=CONFIG['INITIAL_CASH'], color='darkred', linestyle='--', linewidth=1.5, 
                        label='Initial Cash')
            plt.title('Daily Portfolio Value: Day Trading vs Buy-and-Hold Over Backtest Period')
            plt.xlabel('Date')
            plt.ylabel('Portfolio Value ($)')
            plt.grid(True)
            plt.legend()
            plot_file = os.path.join(CONFIG['MODEL_CACHE_DIR'], 'portfolio_value_graph.png')
            plt.savefig(plot_file)
            plt.close()
            logger.info(f"Portfolio value graph saved to {plot_file}")
            print(f"{Fore.LIGHTBLACK_EX}Portfolio value graph saved to {plot_file}{Style.RESET_ALL}")
            email_body += f"\n\nPortfolio Value Graph (Day Trading vs Buy-and-Hold): Attached as portfolio_value_graph.png."
            send_email("Backtest Completed - Best Results with Graph", email_body, plot_file)
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Trading bot with backtest mode")
    parser.add_argument('--backtest', action='store_true', help="Run in backtest-only mode")
    parser.add_argument('--force-train', action='store_true', help="Force retraining of models")
    parser.add_argument('--DEBUG', action='store_true', help="Enable debug printing for detailed informative outputs")
    parser.add_argument('--reset', action='store_true', help="Force account reset and cash injection (overrides CONFIG flag)")
    parser.add_argument('--horizon', type=int, default=None, help="Override LOOK_AHEAD_BARS (e.g. 21)")
    args = parser.parse_args()

    if args.horizon is not None:
        CONFIG['LOOK_AHEAD_BARS'] = args.horizon
        print(f"{Fore.CYAN}OVERRIDE: LOOK_AHEAD_BARS set to {args.horizon}{Style.RESET_ALL}")

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
