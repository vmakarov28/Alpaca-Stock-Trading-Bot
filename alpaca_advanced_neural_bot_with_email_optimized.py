import os
import sys
import logging
import numpy as np
import pandas as pd
import talib
from datetime import datetime, timedelta
from alpaca_trade_api.rest import REST, TimeFrame
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import Layer
from keras.saving import register_keras_serializable
from tqdm import tqdm
import time
import yfinance as yf
from concurrent.futures import ThreadPoolExecutor
import pickle
import logging.handlers
import queue
import threading
from random import uniform
from transformers import pipeline
from colorama import init, Fore
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import argparse
from contextlib import contextmanager
import traceback

# Set environment variables before any imports
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import tensorflow as tf
tf.get_logger().setLevel('FATAL')
os.environ["TF_XLA_FLAGS"] = "--tf_xla_cpu_global_jit"
tf.config.threading.set_inter_op_parallelism_threads(4)
tf.config.threading.set_intra_op_parallelism_threads(4)

# Initialize colorama
init(autoreset=True)

# Define thread-safe locks
print_lock = threading.Lock()
state_lock = threading.Lock()

# Logging filters
class SuppressTFWarnings(logging.Filter):
    def filter(self, record):
        return not any(phrase in record.getMessage() for phrase in [
            "sparse_softmax_cross_entropy is deprecated",
            "The name tf.losses",
            "oneDNN custom operations",
            "optimized to use available CPU instructions"
        ])

def suppress_child_process_logs(record):
    return not hasattr(record, 'processName') or record.processName == 'MainProcess'

# Set up asynchronous logging
log_queue = queue.Queue()
debug_queue = queue.Queue()
queue_handler = logging.handlers.QueueHandler(log_queue)
debug_queue_handler = logging.handlers.QueueHandler(debug_queue)
file_handler = logging.handlers.TimedRotatingFileHandler("trades.log", when="midnight", backupCount=7)
debug_file_handler = logging.handlers.TimedRotatingFileHandler("debug.log", when="midnight", backupCount=7)
formatter = logging.Formatter("%(asctime)s - %(message)s")
file_handler.setFormatter(formatter)
debug_file_handler.setFormatter(formatter)
logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.handlers = []
logger.addHandler(queue_handler)
logger.addFilter(SuppressTFWarnings())
logger.addFilter(suppress_child_process_logs)
debug_logger = logging.getLogger('debug')
debug_logger.setLevel(logging.DEBUG)
debug_logger.handlers = []
debug_logger.addHandler(debug_queue_handler)
debug_logger.addFilter(SuppressTFWarnings())
debug_logger.addFilter(suppress_child_process_logs)

def log_listener():
    while True:
        try:
            record = log_queue.get()
            if record is None:
                break
            file_handler.handle(record)
        except Exception as e:
            with print_lock:
                print(f"Log listener error: {e}", flush=True)

def debug_log_listener():
    while True:
        try:
            record = debug_queue.get()
            if record is None:
                break
            debug_file_handler.handle(record)
        except Exception as e:
            with print_lock:
                print(f"Debug log listener error: {e}", flush=True)

listener_thread = threading.Thread(target=log_listener, daemon=True)
debug_listener_thread = threading.Thread(target=debug_log_listener, daemon=True)
listener_thread.start()
debug_listener_thread.start()

# Alpaca API credentials
API_KEY = "PKQ7UFDXCQF7QDM95LTE"
API_SECRET = "njYD7jxxnT7lSF8cGTfBDuecSI5juARfi21IN8et"
BASE_URL = "https://paper-api.alpaca.markets"

# Trading parameters
SYMBOLS = ["TSLA", "NVDA"]
TRAIN_TIMEFRAME = "15Min"
TRADE_TIMEFRAME = "15Min"
TRAIN_DAYS = 365
TRADE_DAYS = 20  # Increased from 10 to 20 for more backtest data
LOOKBACK = 30
INITIAL_CASH = 1000.0
CASH_PER_STOCK = INITIAL_CASH / len(SYMBOLS)
MAX_POSITION_SIZE = 0.9 / len(SYMBOLS)
DATA_CACHE = "data_cache"
MODEL_CACHE = "model_cache"
STATE_FILE = "state.pkl"
CONFIDENCE_THRESHOLD = 0.4  # Lowered from 0.5 to 0.4 to allow more trades
PORTFOLIO_STOP_LOSS = 0.10
EPOCHS_PER_STOCK = 15
BATCH_SIZE = 256

# Email configuration
EMAIL_SENDER = "alpaca.ai.tradingbot@gmail.com"
EMAIL_PASSWORD = "hjdf sstp pyne rotq"
EMAIL_RECIPIENT = ["aiplane.scientist@gmail.com", "tchaikovskiy@hotmail.com", "evmakarov.md@gmail.com"]
SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 587
EMAIL_INTERVAL = 300

# Initialize Alpaca API
api = REST(API_KEY, API_SECRET, BASE_URL, api_version="v2")

# Track last portfolio email time
last_portfolio_email_time = 0

# Custom Attention Layer
@register_keras_serializable()
class Attention(Layer):
    def __init__(self, **kwargs):
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(name='attention_weight', shape=(input_shape[-1], 1),
                                 initializer='random_normal', trainable=True)
        self.b = self.add_weight(name='attention_bias', shape=(input_shape[1], 1),
                                 initializer='zeros', trainable=True)
        super(Attention, self).build(input_shape)

    def call(self, inputs):
        e = tf.keras.backend.tanh(tf.keras.backend.dot(inputs, self.W) + self.b)
        alpha = tf.keras.backend.softmax(e, axis=1)
        context = inputs * alpha
        return tf.keras.backend.sum(context, axis=1)

    def get_config(self):
        config = super(Attention, self).get_config()
        return config

def load_state():
    try:
        cash = {symbol: CASH_PER_STOCK for symbol in SYMBOLS}
        positions = {symbol: 0 for symbol in SYMBOLS}
        stop_loss_multipliers = {symbol: 1.0 for symbol in SYMBOLS}
        
        if os.path.exists(STATE_FILE):
            with state_lock:
                with open(STATE_FILE, "rb") as f:
                    saved_state = pickle.load(f)
                    cash.update(saved_state.get("cash", {}))
                    positions.update(saved_state.get("positions", {}))
                    stop_loss_multipliers.update(saved_state.get("stop_loss_multipliers", {}))
            logging.info("Loaded state from file.")
            with print_lock:
                print("Loaded state from file.", flush=True)
        
        try:
            account = api.get_account()
            alpaca_cash = float(account.cash)
            alpaca_positions = api.list_positions()
            alpaca_position_qty = {pos.symbol: int(pos.qty) for pos in alpaca_positions}
            
            for symbol in SYMBOLS:
                positions[symbol] = alpaca_position_qty.get(symbol, positions[symbol])
            
            total_cash = alpaca_cash
            for symbol in SYMBOLS:
                if positions[symbol] > 0:
                    bars = api.get_bars(symbol, TimeFrame.Minute, limit=1).df
                    if not bars.empty:
                        current_price = bars["close"].iloc[-1]
                        total_cash += positions[symbol] * current_price
                cash[symbol] = total_cash / len(SYMBOLS)
            logging.info("Synced state with Alpaca portfolio.")
            with print_lock:
                print("Synced state with Alpaca portfolio.", flush=True)
        except Exception as e:
            logging.error(f"Error syncing with Alpaca: {e}. Using file state.")
            with print_lock:
                print(f"Error syncing with Alpaca: {e}. Using file state.", flush=True)
        
        return cash, positions, stop_loss_multipliers
    except Exception as e:
        logging.error(f"Error loading state: {e}. Using default state.")
        with print_lock:
            print(f"Error loading state: {e}. Using default state.", flush=True)
        return ({symbol: CASH_PER_STOCK for symbol in SYMBOLS},
                {symbol: 0 for symbol in SYMBOLS},
                {symbol: 1.0 for symbol in SYMBOLS})

def save_state(cash, positions, stop_loss_multipliers):
    try:
        state = {
            "cash": cash,
            "positions": positions,
            "stop_loss_multipliers": stop_loss_multipliers
        }
        with state_lock:
            with open(STATE_FILE, "wb") as f:
                pickle.dump(state, f)
        logging.info("Saved state to file.")
        with print_lock:
            print("Saved state to file.", flush=True)
    except Exception as e:
        logging.error(f"Error saving state: {e}")
        with print_lock:
            print(f"Error saving state: {e}", flush=True)

def format_time_delta(seconds):
    if seconds < 0:
        return "Unknown"
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = int(seconds % 60)
    parts = []
    if hours > 0:
        parts.append(f"{hours}h")
    if minutes > 0 or hours > 0:
        parts.append(f"{minutes}m")
    parts.append(f"{seconds}s")
    return f"{Fore.RED}{' '.join(parts)}{Fore.RESET}"

def send_email(subject, body, retry_attempts=3, retry_delay=5):
    msg = MIMEMultipart()
    msg['From'] = EMAIL_SENDER
    msg['To'] = ", ".join(EMAIL_RECIPIENT)
    msg['Subject'] = subject
    msg.attach(MIMEText(body, 'plain'))

    for attempt in range(retry_attempts):
        try:
            server = smtplib.SMTP(SMTP_SERVER, SMTP_PORT, timeout=10)
            server.starttls()
            server.login(EMAIL_SENDER, EMAIL_PASSWORD)
            server.sendmail(EMAIL_SENDER, EMAIL_RECIPIENT, msg.as_string())
            server.quit()
            logging.info(f"Email sent: Subject: {subject} to {', '.join(EMAIL_RECIPIENT)}")
            with print_lock:
                print(f"Email sent: Subject: {subject} to {', '.join(EMAIL_RECIPIENT)}", flush=True)
            return True
        except Exception as e:
            logging.error(f"Email attempt {attempt+1}/{retry_attempts} failed: {e}")
            with print_lock:
                print(f"Email attempt {attempt+1}/{retry_attempts} failed: {e}", flush=True)
            if attempt < retry_attempts - 1:
                time.sleep(retry_delay * (2 ** attempt) + uniform(0, 0.5))
    logging.error(f"Failed to send email after {retry_attempts} attempts: Subject: {subject}")
    with print_lock:
        print(f"Failed to send email after {retry_attempts} attempts: Subject: {subject}", flush=True)
    return False

def check_market_open():
    for attempt in range(3):
        try:
            clock = api.get_clock()
            if clock.is_open:
                logging.info("Market is open.")
                with print_lock:
                    print("Market is open.", flush=True)
                return True, 0, None
            next_open = clock.next_open
            seconds_until_open = (next_open - datetime.now().astimezone()).total_seconds()
            if seconds_until_open < 0:
                logging.warning(f"Negative seconds_until_open: {seconds_until_open}. Setting to 300 seconds.")
                with print_lock:
                    print(f"Negative seconds_until_open: {seconds_until_open}. Setting to 300 seconds.", flush=True)
                seconds_until_open = 300
            sleep_time = max(60, seconds_until_open - 60) if seconds_until_open > 60 else 300
            time_until_open = format_time_delta(seconds_until_open)
            logging.info(f"Market is closed. Next open: {next_open}. Waiting {sleep_time:.0f} seconds.")
            return False, sleep_time, next_open
        except Exception as e:
            logging.error(f"Error checking market status on attempt {attempt+1}: {e}")
            with print_lock:
                print(f"Error checking market status on attempt {attempt+1}: {e}", flush=True)
            if attempt == 2:
                logging.warning("Failed to check market status after 3 attempts. Assuming market is closed.")
                with print_lock:
                    print("Failed to check market status after 3 attempts. Assuming market is closed.", flush=True)
                return False, 300, datetime.now().astimezone() + timedelta(seconds=300)
            time.sleep(2 ** attempt + uniform(0, 1))

def get_stock_data(symbol, timeframe, days, feed="iex"):
    cache_file = f"{DATA_CACHE}/{symbol}_{timeframe}_{days}.pkl"
    news_cache_file = f"{DATA_CACHE}/{symbol}_news_{days}.pkl"
    os.makedirs(DATA_CACHE, exist_ok=True)
    output_messages = []
    try:
        # Force fresh data fetch for TSLA to avoid cache issues
        if symbol == "TSLA" and os.path.exists(cache_file):
            os.remove(cache_file)
            logging.info(f"Deleted cached data for {symbol} to fetch fresh data.")
            output_messages.append(f"Deleted cached data for {symbol} to fetch fresh data.")
        
        if os.path.exists(cache_file) and (time.time() - os.path.getmtime(cache_file)) > 24*3600:
            logging.info(f"Cache for {symbol} is stale. Fetching fresh data.")
            output_messages.append(f"Cache for {symbol} is stale. Fetching fresh data.")
            data = pd.DataFrame()
        elif os.path.exists(cache_file):
            try:
                with open(cache_file, "rb") as f:
                    data = pickle.load(f)
                if not isinstance(data, pd.DataFrame) or data.empty:
                    raise ValueError("Invalid or empty cache data")
                logging.info(f"Loaded {len(data)} bars for {symbol} with timeframe {timeframe} from cache")
                output_messages.append(f"Loaded {len(data)} bars for {symbol} from cache.")
            except Exception as e:
                logging.warning(f"Cache load failed for {symbol}: {e}. Fetching fresh data.")
                output_messages.append(f"Cache load failed for {symbol}: {e}. Fetching fresh data.")
                data = pd.DataFrame()
        else:
            data = pd.DataFrame()
        
        if data.empty:
            end = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
            start = end - timedelta(days=days)
            for attempt in range(3):
                try:
                    time.sleep(0.5)
                    data = api.get_bars(
                        symbol,
                        timeframe,
                        start=start.strftime("%Y-%m-%d"),
                        end=end.strftime("%Y-%m-%d"),
                        feed=feed
                    ).df
                    if data.empty:
                        logging.warning(f"No data fetched for {symbol} with timeframe {timeframe} on attempt {attempt+1}")
                        output_messages.append(f"No data fetched for {symbol} with timeframe {timeframe} on attempt {attempt+1}")
                        if attempt == 2:
                            break
                        time.sleep(2 ** attempt + uniform(0, 0.5))
                        continue
                    logging.info(f"Fetched {len(data)} bars for {symbol} with timeframe {timeframe} from Alpaca")
                    output_messages.append(f"Fetched {len(data)} bars for {symbol} from Alpaca.")
                    with open(cache_file, "wb") as f:
                        pickle.dump(data, f)
                    break
                except Exception as e:
                    logging.error(f"Error fetching data for {symbol} on attempt {attempt+1}: {e}")
                    output_messages.append(f"Error fetching data for {symbol} on attempt {attempt+1}: {e}")
                    if attempt == 2:
                        break
                    time.sleep(2 ** attempt + uniform(0, 0.5))
            
            if data.empty:
                logging.info(f"Falling back to yfinance for {symbol} with timeframe {timeframe}")
                output_messages.append(f"Falling back to yfinance for {symbol}.")
                interval = "1d" if timeframe == "1Day" else "15m"
                data = yf.download(symbol, start=start, end=end, interval=interval)
                if data.empty:
                    logging.error(f"No data fetched for {symbol} with timeframe {timeframe} from yfinance")
                    output_messages.append(f"No data fetched for {symbol} from yfinance.")
                    return pd.DataFrame(), [], output_messages
                data = data.rename(columns={
                    "Open": "open",
                    "High": "high",
                    "Low": "low",
                    "Close": "close",
                    "Volume": "volume"
                })
                with open(cache_file, "wb") as f:
                    pickle.dump(data, f)
        
        # Validate data columns
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        if not all(col in data.columns for col in required_columns):
            logging.error(f"Missing required columns in data for {symbol}: {data.columns}")
            output_messages.append(f"Missing required columns in data for {symbol}: {data.columns}")
            return pd.DataFrame(), [], output_messages
        
        news_data = []
        if os.path.exists(news_cache_file):
            try:
                with open(news_cache_file, "rb") as f:
                    news_data = pickle.load(f)
                logging.info(f"Loaded news data for {symbol} from cache")
                output_messages.append(f"Loaded news data for {symbol} from cache.")
            except Exception as e:
                logging.warning(f"News cache load failed for {symbol}: {e}. Fetching fresh news.")
                output_messages.append(f"News cache load failed for {symbol}: {e}. Fetching fresh news.")
        if not news_data:
            try:
                time.sleep(0.5)
                end = datetime.now()
                start = end - timedelta(days=days)
                news = api.get_news(symbol, start=start.strftime("%Y-%m-%d"), end=end.strftime("%Y-%m-%d"))
                news_data = [{"headline": n.headline, "summary": n.summary, "timestamp": n.created_at} for n in news]
                with open(news_cache_file, "wb") as f:
                    pickle.dump(news_data, f)
                logging.info(f"Fetched {len(news_data)} news items for {symbol} from Alpaca")
                output_messages.append(f"Fetched {len(news_data)} news items for {symbol} from Alpaca.")
            except Exception as e:
                logging.error(f"Error fetching news for {symbol}: {e}")
                output_messages.append(f"Error fetching news for {symbol}: {e}")
        
        return data, news_data, output_messages
    except Exception as e:
        logging.error(f"Error in get_stock_data for {symbol}: {e}")
        output_messages.append(f"Error in get_stock_data for {symbol}: {e}")
        return pd.DataFrame(), [], output_messages

def calculate_sentiment(news_data, sentiment_analyzer):
    try:
        if not news_data or not sentiment_analyzer:
            logging.warning("No news data or sentiment analyzer available. Using neutral sentiment (0.0).")
            return 0.0, ["No news data or sentiment analyzer available. Using neutral sentiment (0.0)."]
        
        sentiments = []
        for item in news_data:
            text = (item.get("headline", "") + " " + item.get("summary", "")).strip()
            if text:
                result = sentiment_analyzer(text)[0]
                if result['label'] == 'positive':
                    score = result['score']
                elif result['label'] == 'negative':
                    score = -result['score']
                else:
                    score = 0.0
                sentiments.append(score)
        
        sentiment_score = np.mean(sentiments) if sentiments else 0.0
        logging.info(f"Calculated sentiment score: {sentiment_score:.3f}")
        return sentiment_score, [f"Calculated sentiment score: {sentiment_score:.3f}"]
    except Exception as e:
        logging.error(f"Error calculating sentiment: {e}. Using neutral sentiment (0.0).")
        return 0.0, [f"Error calculating sentiment: {e}. Using neutral sentiment (0.0)."]

def calculate_indicators(data, timeframe, sentiment_score):
    try:
        if data.empty:
            raise ValueError("Empty data provided for indicator calculation")
        debug_logger.debug(f"Calculating indicators for timeframe {timeframe}, data shape: {data.shape}")
        data = data.copy()
        data['MA20'] = talib.SMA(data['close'], timeperiod=20)
        data['RSI'] = talib.RSI(data['close'], timeperiod=14)
        macd, signal, _ = talib.MACD(data['close'], fastperiod=12, slowperiod=26, signalperiod=9)
        data['MACD'] = macd
        data['MACD_signal'] = signal
        data['OBV'] = talib.OBV(data['close'], data['volume'])
        data['VWAP'] = (data['close'] * data['volume']).cumsum() / data['volume'].cumsum()
        data['ATR'] = talib.ATR(data['high'], data['low'], data['close'], timeperiod=14)
        denominator = data['high'] - data['low']
        data['CMF'] = np.where(denominator != 0,
                              ((data['close'] - data['low']) - (data['high'] - data['close'])) / denominator * data['volume'],
                              0.0)
        data['CMF'] = data['CMF'].rolling(20).sum() / data['volume'].rolling(20).sum()
        data['Close_ATR'] = data['close'] / data['ATR']
        data['MA20_ATR'] = data['MA20'] / data['ATR']
        data['Return_1d'] = data['close'].pct_change(1)
        data['Return_5d'] = data['close'].pct_change(5)
        data['Sentiment'] = sentiment_score
        data['Trend'] = np.where(data['close'] > data['MA20'], 1, 
                               np.where(data['close'] < data['MA20'], -1, 0))
        
        debug_logger.debug(f"Indicators calculated, data shape: {data.shape}, NaN counts: {data.isna().sum().to_dict()}")
        return data
    except Exception as e:
        logging.error(f"Error calculating indicators: {e}")
        return pd.DataFrame()

def train_stop_loss(data):
    try:
        if data.empty or 'ATR' not in data or 'close' not in data:
            return 1.0, ["No data for stop-loss calculation. Using default multiplier: 1.0"]
        
        returns = data['close'].pct_change().dropna()
        atr = data['ATR'].dropna()
        volatility = returns.std()
        historical_vol = returns.rolling(20).std().dropna()
        if volatility > historical_vol.quantile(0.7):
            multiplier = 1.5
        else:
            multiplier = 1.0
        
        logging.info(f"Calculated stop-loss multiplier: {multiplier:.2f}")
        return multiplier, [f"Calculated stop-loss multiplier: {multiplier:.2f}"]
    except Exception as e:
        logging.error(f"Error training stop-loss: {e}")
        return 1.0, [f"Error training stop-loss: {e}"]

def process_chunk(chunk, lookback, batch_size, augment):
    try:
        features = chunk[['close', 'volume', 'MA20', 'RSI', 'MACD', 'MACD_signal', 'OBV', 'VWAP',
                         'ATR', 'CMF', 'Close_ATR', 'MA20_ATR', 'Return_1d', 'Return_5d', 'Sentiment', 'Trend']].copy()
        features = features.ffill().bfill().dropna()
        if len(features) <= lookback:
            debug_logger.debug(f"Chunk too small for preprocessing: {len(features)} < {lookback}")
            return None, None
        X, y = [], []
        for i in range(lookback, len(features), batch_size):
            batch_end = min(i + batch_size, len(features))
            X_batch = np.array([features.iloc[j-lookback:j].values for j in range(i, batch_end)], dtype=np.float32)
            if augment:
                noise = np.random.normal(0, 0.01, X_batch[:, :, [0, 2, 7, 10, 11]].shape)
                X_batch[:, :, [0, 2, 7, 10, 11]] += noise
            y_batch = np.array([1 if features['close'].iloc[j] > features['close'].iloc[j-1] else 0 
                              for j in range(i, batch_end)], dtype=np.int32)
            X.append(X_batch)
            y.append(y_batch)
        if not X:
            return None, None
        return np.concatenate(X), np.concatenate(y)
    except Exception as e:
        debug_logger.debug(f"Error in process_chunk: {e}")
        return None, None

def preprocess_data(data, batch_size=256, augment=False):
    try:
        if data.empty:
            raise ValueError("Empty data provided for preprocessing")
        debug_logger.debug(f"Before filling NaNs, data shape: {data.shape}")
        
        # Validate required columns
        required_columns = ['close', 'volume', 'MA20', 'RSI', 'MACD', 'MACD_signal', 'OBV', 'VWAP',
                            'ATR', 'CMF', 'Close_ATR', 'MA20_ATR', 'Return_1d', 'Return_5d', 'Sentiment', 'Trend']
        if not all(col in data.columns for col in required_columns):
            missing_cols = [col for col in required_columns if col not in data.columns]
            raise ValueError(f"Missing columns in data: {missing_cols}")
        
        # Process data sequentially
        features = data[required_columns].copy()
        features = features.ffill().bfill().dropna()
        if len(features) <= LOOKBACK:
            logging.error(f"Data too small for preprocessing: {len(features)} < {LOOKBACK}")
            return None, None, [f"Data too small for preprocessing: {len(features)} < {LOOKBACK}"]
        
        X, y = [], []
        for i in range(LOOKBACK, len(features), batch_size):
            batch_end = min(i + batch_size, len(features))
            X_batch = np.array([features.iloc[j-LOOKBACK:j].values for j in range(i, batch_end)], dtype=np.float32)
            if augment:
                noise = np.random.normal(0, 0.01, X_batch[:, :, [0, 2, 7, 10, 11]].shape)
                X_batch[:, :, [0, 2, 7, 10, 11]] += noise
            y_batch = np.array([1 if features['close'].iloc[j] > features['close'].iloc[j-1] else 0 
                              for j in range(i, batch_end)], dtype=np.int32)
            X.append(X_batch)
            y.append(y_batch)
        
        if not X or not y:
            logging.error("No valid sequences generated after preprocessing")
            return None, None, ["No valid sequences generated after preprocessing"]
        
        X = np.concatenate(X)
        y = np.concatenate(y)
        debug_logger.debug(f"Preprocessed features shape for {data.name if hasattr(data, 'name') else 'unknown'}: X shape: {X.shape}, y shape: {y.shape}")
        return X, y, []
    except Exception as e:
        logging.error(f"Error preprocessing data: {e}")
        return None, None, [f"Error preprocessing data: {e}"]

def create_dataset(X, y, batch_size=BATCH_SIZE):
    dataset = tf.data.Dataset.from_tensor_slices((X, y))
    dataset = dataset.cache().shuffle(1000).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset

def build_model(input_shape):
    try:
        tf.keras.backend.clear_session()
        inputs = tf.keras.layers.Input(shape=input_shape)
        x = tf.keras.layers.Conv1D(filters=64, kernel_size=3, activation='relu', padding='same')(inputs)
        x = tf.keras.layers.MaxPooling1D(pool_size=2)(x)
        x = tf.keras.layers.Dropout(0.3)(x)
        x = tf.keras.layers.LSTM(16, return_sequences=True)(x)  # Reduced from 32 to 16
        x = tf.keras.layers.Dropout(0.3)(x)
        x = Attention()(x)
        x = tf.keras.layers.Dense(8, activation='relu')(x)
        outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)
        model = tf.keras.Model(inputs, outputs)
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'], jit_compile=True)
        return model
    except Exception as e:
        logging.error(f"Error building model: {str(e)}")
        debug_logger.debug(f"Full error traceback: {traceback.format_exc()}")
        return None

@contextmanager
def progress_update(progress_bar, progress_lock):
    with progress_lock:
        yield
        progress_bar.update(1)

def train_model(symbol, progress_bar, progress_lock, sentiment_analyzer):
    output_messages = []
    try:
        model_file = f"{MODEL_CACHE}/{symbol}_model.h5"
        scaler_file = f"{MODEL_CACHE}/{symbol}_scaler.pkl"
        os.makedirs(MODEL_CACHE, exist_ok=True)
        
        # Force retraining for TSLA by deleting cached model/scaler
        if symbol == "TSLA":
            if os.path.exists(model_file):
                os.remove(model_file)
                logging.info(f"Deleted cached model for {symbol} to force retraining.")
                output_messages.append(f"Deleted cached model for {symbol} to force retraining.")
            if os.path.exists(scaler_file):
                os.remove(scaler_file)
                logging.info(f"Deleted cached scaler for {symbol} to force retraining.")
                output_messages.append(f"Deleted cached scaler for {symbol} to force retraining.")
        
        model, scaler = None, None
        if os.path.exists(model_file) and os.path.exists(scaler_file):
            try:
                model = tf.keras.models.load_model(model_file, custom_objects={'Attention': Attention})
                if not callable(getattr(model, 'predict', None)):
                    raise ValueError("Invalid model: missing predict method")
                with open(scaler_file, "rb") as f:
                    scaler = pickle.load(f)
                if not isinstance(scaler, MinMaxScaler):
                    raise ValueError("Invalid scaler object")
                logging.info(f"Loaded cached model and scaler for {symbol}")
                output_messages.append(f"Loaded cached model and scaler for {symbol}.")
            except Exception as e:
                logging.warning(f"Invalid cache for {symbol}: {e}. Training new model.")
                output_messages.append(f"Invalid cache for {symbol}: {e}. Training new model.")
                model, scaler = None, None
        else:
            output_messages.append(f"No cached model for {symbol}. Training new model.")
        
        raw_data, news_data, data_messages = get_stock_data(symbol, TRAIN_TIMEFRAME, TRAIN_DAYS)
        output_messages.extend(data_messages)
        if raw_data.empty:
            logging.error(f"No training data available for {symbol}")
            output_messages.append(f"No training data available for {symbol}.")
            return None, None, 1.0, output_messages
        
        sentiment_score, sentiment_messages = calculate_sentiment(news_data, sentiment_analyzer)
        output_messages.extend(sentiment_messages)
        data = calculate_indicators(raw_data, TRAIN_TIMEFRAME, sentiment_score)
        if data.empty:
            logging.error(f"Failed to calculate indicators for {symbol}")
            output_messages.append(f"Failed to calculate indicators for {symbol}.")
            return None, None, 1.0, output_messages
        
        stop_loss_multiplier, stop_loss_messages = train_stop_loss(data)
        output_messages.extend(stop_loss_messages)
        
        X, y, preprocess_messages = preprocess_data(data, batch_size=BATCH_SIZE, augment=True)
        output_messages.extend(preprocess_messages)
        if X is None or y is None:
            logging.error(f"Failed to preprocess data for {symbol}")
            output_messages.append(f"Failed to preprocess data for {symbol}.")
            return None, None, stop_loss_multiplier, output_messages
        
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
        scaler = MinMaxScaler()
        X_train = scaler.fit_transform(X_train.reshape(-1, X_train.shape[-1])).reshape(X_train.shape)
        X_val = scaler.transform(X_val.reshape(-1, X_val.shape[-1])).reshape(X_val.shape)
        model = build_model((X_train.shape[1], X_train.shape[2]))
        if model is None:
            logging.error(f"Failed to build model for {symbol}")
            output_messages.append(f"Failed to build model for {symbol}.")
            return None, None, stop_loss_multiplier, output_messages
        
        train_dataset = create_dataset(X_train, y_train, BATCH_SIZE)
        val_dataset = create_dataset(X_val, y_val, BATCH_SIZE)
        early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-5)
        epochs = EPOCHS_PER_STOCK
        start_time = time.time()
        
        for epoch in range(epochs):
            try:
                model.fit(
                    train_dataset,
                    validation_data=val_dataset,
                    epochs=1,
                    verbose=0,
                    callbacks=[early_stopping, lr_scheduler]
                )
                with progress_update(progress_bar, progress_lock):
                    pass
                elapsed_time = time.time() - start_time
                debug_logger.debug(f"Epoch {epoch+1}/{epochs} for {symbol}, elapsed time: {elapsed_time:.2f}s")
            except Exception as e:
                logging.error(f"Training failed for {symbol} at epoch {epoch+1}: {e}")
                output_messages.append(f"Training failed at epoch {epoch+1}: {e}")
                return None, None, stop_loss_multiplier, output_messages
        
        try:
            model.save(model_file)
            with open(scaler_file, "wb") as f:
                pickle.dump(scaler, f)
            logging.info(f"Saved model and scaler for {symbol}")
            output_messages.append(f"Saved model and scaler for {symbol}.")
        except Exception as e:
            logging.error(f"Failed to save model/scaler for {symbol}: {e}")
            output_messages.append(f"Failed to save model/scaler: {e}")
        
        if model is None or scaler is None:
            logging.error(f"Training failed for {symbol}: model or scaler is None")
            output_messages.append(f"Training failed for {symbol}: model or scaler is None")
            return None, None, stop_loss_multiplier, output_messages
        
        return model, scaler, stop_loss_multiplier, output_messages
    except Exception as e:
        logging.error(f"Error training model for {symbol}: {e}")
        output_messages.append(f"Error training model for {symbol}: {e}")
        send_email(
            subject=f"Training Error: {symbol}",
            body=f"Error training model for {symbol}:\n"
                 f"Message: {e}\n"
                 f"Skipping training for this symbol."
        )
        return None, None, 1.0, output_messages

def predict_signal(model, scaler, data, sentiment_score):
    try:
        if data.empty:
            raise ValueError("No data available for prediction")
        features = calculate_indicators(data, TRADE_TIMEFRAME, sentiment_score)
        if features.empty:
            return None, None, ["Failed to calculate indicators for prediction"]
        X, _, preprocess_messages = preprocess_data(features)
        if X is None:
            return None, None, preprocess_messages
        if X.shape[1:] != (LOOKBACK, len(features.columns)):
            logging.error(f"Invalid input shape for prediction: {X.shape}")
            return None, None, [f"Invalid input shape for prediction: {X.shape}"]
        X = scaler.transform(X.reshape(-1, X.shape[-1])).reshape(X.shape)
        try:
            prediction = model.predict(X[-1].reshape(1, LOOKBACK, -1), verbose=0)
        except Exception as e:
            logging.error(f"Prediction failed: {e}")
            return None, None, [f"Prediction failed: {e}"]
        confidence = prediction[0][0]
        return True, confidence, []
    except Exception as e:
        logging.error(f"Error predicting signal: {e}")
        return None, None, [f"Error predicting signal: {e}"]

def trading_strategy(sentiment_analyzer):
    global last_portfolio_email_time
    try:
        models = {}
        scalers = {}
        stop_loss_multipliers = {}
        cash, positions, stop_loss_multipliers = load_state()
        initial_portfolio_value = sum(cash.values()) + sum(
            positions[symbol] * get_stock_data(symbol, TRADE_TIMEFRAME, 1)[0]["close"].iloc[-1]
            for symbol in SYMBOLS if positions[symbol] > 0
        )
        
        os.system('cls' if os.name == 'nt' else 'clear')
        with print_lock:
            print("Starting model training for all stocks...", flush=True)
        
        total_epochs = EPOCHS_PER_STOCK * len(SYMBOLS)
        progress_lock = threading.Lock()
        symbol_outputs = {}
        with tqdm(total=total_epochs, desc="Training All Stocks", 
                  bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
                  colour="green", leave=True) as progress_bar:
            with ThreadPoolExecutor(max_workers=min(len(SYMBOLS), 2)) as executor:
                futures = {executor.submit(train_model, symbol, progress_bar, progress_lock, sentiment_analyzer): symbol for symbol in SYMBOLS}
                for future in futures:
                    symbol = futures[future]
                    try:
                        model, scaler, multiplier, messages = future.result()
                        models[symbol] = model
                        scalers[symbol] = scaler
                        stop_loss_multipliers[symbol] = multiplier
                        symbol_outputs[symbol] = messages
                        if model is None or scaler is None:
                            logging.warning(f"Skipping {symbol} due to training failure")
                            symbol_outputs[symbol].append(f"Skipping {symbol} due to training failure.")
                    except Exception as e:
                        logging.error(f"Error training {symbol}: {e}")
                        symbol_outputs[symbol] = symbol_outputs.get(symbol, []) + [f"Error training {symbol}: {e}"]
        
        progress_bar.close()
        
        with print_lock:
            for symbol in SYMBOLS:
                print(f"{Fore.LIGHTBLUE_EX}\nTraining {symbol}:{Fore.RESET}")
                for msg in symbol_outputs.get(symbol, []):
                    print(f"  {msg}")
        
        successful_models = sum(1 for symbol in SYMBOLS if models.get(symbol) is not None)
        email_body = f"Training completed for {len(SYMBOLS)} stocks.\n"
        email_body += f"Successful models: {successful_models}/{len(SYMBOLS)}\n\n"
        for symbol in SYMBOLS:
            email_body += f"{symbol}:\n"
            for msg in symbol_outputs.get(symbol, []):
                email_body += f"  {msg}\n"
        send_email(
            subject="Training Completed",
            body=email_body
        )
        
        with print_lock:
            print("\nTraining complete. Starting trading...", flush=True)
        
        while True:
            try:
                with print_lock:
                    print("\nChecking market status...", flush=True)
                is_open, sleep_time, next_open = check_market_open()
                if not is_open:
                    while sleep_time > 0:
                        current_time = datetime.now().astimezone()
                        seconds_remaining = (next_open - current_time).total_seconds()
                        if seconds_remaining <= 0:
                            break
                        time_until_open = format_time_delta(seconds_remaining)
                        with print_lock:
                            print(f"Market is closed. Opens in {time_until_open}.", 
                                  end='\r', flush=True)
                        time.sleep(1)
                        sleep_time -= 1
                    with print_lock:
                        print(" " * 80, end='\r', flush=True)
                    continue
                
                data_list = [get_stock_data(s, TRADE_TIMEFRAME, TRADE_DAYS) for s in SYMBOLS]
                
                total_cash = sum(cash.values())
                portfolio_value = total_cash
                for symbol, (data, news_data, data_messages) in zip(SYMBOLS, data_list):
                    if models[symbol] is None or data.empty:
                        logging.warning(f"Skipping {symbol} due to missing model or data")
                        with print_lock:
                            print(f"Skipping {symbol} due to missing model or data.", flush=True)
                        continue
                    current_price = round(data["close"].iloc[-1], 2)
                    if positions[symbol] > 0:
                        portfolio_value += positions[symbol] * current_price
                    
                    if portfolio_value < initial_portfolio_value * (1 - PORTFOLIO_STOP_LOSS):
                        logging.warning("Portfolio stop-loss triggered. Pausing trading.")
                        with print_lock:
                            print("Portfolio stop-loss triggered. Pausing trading.", flush=True)
                        send_email(
                            subject="Portfolio Stop-Loss Triggered",
                            body=f"Portfolio stop-loss triggered:\n"
                                 f"Portfolio Value: ${portfolio_value:.2f}\n"
                                 f"Initial Value: ${initial_portfolio_value:.2f}\n"
                                 f"Stop-Loss Threshold: {PORTFOLIO_STOP_LOSS*100:.1f}%\n"
                                 f"Trading paused."
                        )
                        return
                    
                    sentiment_score, sentiment_messages = calculate_sentiment(news_data, sentiment_analyzer)
                    signal, confidence, signal_messages = predict_signal(models[symbol], scalers[symbol], data, sentiment_score)
                    if signal is None:
                        with print_lock:
                            print(f"{symbol} - No signal generated, skipping...", flush=True)
                        continue
                    signal_type = 'Buy' if confidence > 0.5 else 'Sell' if confidence < 0.5 else 'Hold'
                    with print_lock:
                        print(f"{symbol} - Price: {current_price:.2f}, Signal: {signal_type}, "
                              f"Confidence: {confidence:.2f}, Sentiment: {sentiment_score:.2f}, "
                              f"Cash: {cash[symbol]:.2f}, Position: {positions[symbol]}", flush=True)
                    
                    atr = data['ATR'].iloc[-1] if 'ATR' in data else 1.0
                    position_size = min(cash[symbol] * MAX_POSITION_SIZE, cash[symbol]) / (atr + 1e-6)
                    qty = int(position_size // current_price)
                    if qty == 0:
                        logging.info(f"Zero quantity calculated for {symbol}: position_size={position_size:.2f}, price={current_price:.2f}")
                        with print_lock:
                            print(f"Zero quantity for {symbol}, skipping order.", flush=True)
                        continue
                    
                    stop_loss_multiplier = stop_loss_multipliers.get(symbol, 1.0)
                    stop_loss_price = round(current_price * (1 - 0.05 * stop_loss_multiplier), 2)
                    take_profit_price = round(current_price * 1.10, 2)
                    
                    open_orders = api.list_orders(status='open', symbols=[symbol])
                    has_open_buy = any(order.side == 'buy' for order in open_orders)
                    has_open_sell = any(order.side == 'sell' for order in open_orders)
                    
                    if (confidence > CONFIDENCE_THRESHOLD and cash[symbol] >= current_price and 
                        positions[symbol] == 0 and qty > 0 and not has_open_buy):
                        stop_loss = {"stop_price": stop_loss_price}
                        take_profit = {"limit_price": take_profit_price}
                        api.submit_order(
                            symbol=symbol,
                            qty=qty,
                            side="buy",
                            type="market",
                            time_in_force="gtc",
                            order_class="bracket",
                            stop_loss=stop_loss,
                            take_profit=take_profit
                        )
                        positions[symbol] += qty
                        cash[symbol] -= qty * current_price
                        logging.info(f"Buy signal: {symbol} at {current_price:.2f}, Qty: {qty}, "
                                    f"Confidence: {confidence:.2f}, Stop-Loss: {stop_loss_price:.2f}, Take-Profit: {take_profit_price:.2f}")
                        with print_lock:
                            print(f"Buy signal: {symbol} at {current_price:.2f}, Qty: {qty}, "
                                  f"Confidence: {confidence:.2f}, Stop-Loss: {stop_loss_price:.2f}, Take-Profit: {take_profit_price:.2f}", flush=True)
                        send_email(
                            subject=f"Buy Order Executed: {symbol}",
                            body=f"Buy signal for {symbol}:\n"
                                 f"Price: ${current_price:.2f}\n"
                                 f"Quantity: {qty}\n"
                                 f"Confidence: {confidence:.2f}\n"
                                 f"Stop-Loss: ${stop_loss_price:.2f}\n"
                                 f"Take-Profit: ${take_profit_price:.2f}\n"
                                 f"Cash remaining for {symbol}: ${cash[symbol]:.2f}\n"
                                 f"Sentiment: {sentiment_score:.2f}"
                        )
                    elif confidence < CONFIDENCE_THRESHOLD and positions[symbol] > 0 and not has_open_sell:
                        api.submit_order(
                            symbol=symbol,
                            qty=positions[symbol],
                            side="sell",
                            type="market",
                            time_in_force="gtc"
                        )
                        cash[symbol] += positions[symbol] * current_price
                        positions[symbol] = 0
                        logging.info(f"Sell signal: {symbol} at {current_price:.2f}, Confidence: {confidence:.2f}")
                        with print_lock:
                            print(f"Sell signal: {symbol} at {current_price:.2f}, Confidence: {confidence:.2f}", flush=True)
                        send_email(
                            subject=f"Sell Order Executed: {symbol}",
                            body=f"Sell signal for {symbol}:\n"
                                 f"Price: ${current_price:.2f}\n"
                                 f"Quantity: {positions[symbol]}\n"
                                 f"Confidence: {confidence:.2f}\n"
                                 f"Cash after sale for {symbol}: ${cash[symbol]:.2f}\n"
                                 f"Sentiment: {sentiment_score:.2f}"
                        )
                    elif abs(confidence - 0.5) <= 0.001:
                        logging.info(f"Hold signal: {symbol}, Confidence: {confidence:.2f}")
                        with print_lock:
                            print(f"Hold signal: {symbol}, Confidence: {confidence:.2f}", flush=True)
                    
                    save_state(cash, positions, stop_loss_multipliers)
                
                logging.info(f"Portfolio Value: {portfolio_value:.2f}, Cash: {total_cash:.2f}")
                with print_lock:
                    print(f"Portfolio Value: {portfolio_value:.2f}, Cash: {total_cash:.2f}", flush=True)
                current_time = time.time()
                if current_time - last_portfolio_email_time >= EMAIL_INTERVAL:
                    email_body = f"Portfolio Update:\n"
                    email_body += f"Portfolio Value: ${portfolio_value:.2f}\n"
                    email_body += f"Total Cash: ${total_cash:.2f}\n"
                    email_body += "Positions:\n"
                    for symbol in SYMBOLS:
                        email_body += f"  {symbol}: {positions[symbol]} shares, Cash: ${cash[symbol]:.2f}\n"
                    send_email(
                        subject="Portfolio Update",
                        body=email_body
                    )
                    last_portfolio_email_time = current_time
                time.sleep(900)
            
            except Exception as e:
                logging.error(f"Error in trading loop: {e}")
                with print_lock:
                    print(f"Error in trading loop: {e}", flush=True)
                send_email(
                    subject="Trading Loop Error",
                    body=f"Error in trading loop:\n"
                         f"Message: {e}\n"
                         f"Portfolio Value: ${portfolio_value:.2f}\n"
                         f"Retrying in 60 seconds."
                )
                time.sleep(60)
    
    except Exception as e:
        logging.error(f"Error in trading strategy: {e}")
        with print_lock:
            print(f"Error in trading strategy: {e}", flush=True)
        send_email(
            subject="Trading Strategy Error",
            body=f"Critical error in trading strategy:\n"
                 f"Message: {e}\n"
                 f"Bot stopped."
        )
        raise
    finally:
        save_state(cash, positions, stop_loss_multipliers)
        log_queue.put(None)
        debug_queue.put(None)

def backtest_strategy(sentiment_analyzer):
    try:
        models = {}
        scalers = {}
        stop_loss_multipliers = {}
        cash = {symbol: CASH_PER_STOCK for symbol in SYMBOLS}
        positions = {symbol: 0 for symbol in SYMBOLS}
        trades = []
        
        with print_lock:
            print("Starting model training for backtesting...", flush=True)
        
        total_epochs = EPOCHS_PER_STOCK * len(SYMBOLS)
        progress_lock = threading.Lock()
        symbol_outputs = {}
        with tqdm(total=total_epochs, desc="Training for Backtest", 
                  bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
                  colour="green", leave=True) as progress_bar:
            with ThreadPoolExecutor(max_workers=min(len(SYMBOLS), 2)) as executor:
                futures = {executor.submit(train_model, symbol, progress_bar, progress_lock, sentiment_analyzer): symbol for symbol in SYMBOLS}
                for future in futures:
                    symbol = futures[future]
                    try:
                        model, scaler, multiplier, messages = future.result()
                        models[symbol] = model
                        scalers[symbol] = scaler
                        stop_loss_multipliers[symbol] = multiplier
                        symbol_outputs[symbol] = messages
                        if model is None or scaler is None:
                            logging.warning(f"Skipping {symbol} due to training failure")
                            symbol_outputs[symbol].append(f"Skipping {symbol} due to training failure.")
                    except Exception as e:
                        logging.error(f"Error training {symbol}: {e}")
                        symbol_outputs[symbol] = symbol_outputs.get(symbol, []) + [f"Error training {symbol}: {e}"]
        
        progress_bar.close()
        
        with print_lock:
            for symbol in SYMBOLS:
                print(f"{Fore.LIGHTBLUE_EX}\nTraining {symbol}:{Fore.RESET}")
                for msg in symbol_outputs.get(symbol, []):
                    print(f"  {msg}")
        
        with print_lock:
            print("\nTraining complete. Starting backtest simulation...", flush=True)
        
        for symbol in SYMBOLS:
            data, news_data, _ = get_stock_data(symbol, TRADE_TIMEFRAME, TRADE_DAYS)
            if data.empty or models[symbol] is None:
                with print_lock:
                    print(f"Skipping {symbol} due to missing data or model.", flush=True)
                continue
            
            sentiment_score, _ = calculate_sentiment(news_data, sentiment_analyzer)
            features = calculate_indicators(data, TRADE_TIMEFRAME, sentiment_score)
            if features.empty:
                with print_lock:
                    print(f"Skipping {symbol} due to indicator calculation failure.", flush=True)
                continue
            
            X, _, preprocess_messages = preprocess_data(features)
            if X is None:
                with print_lock:
                    print(f"Skipping {symbol} due to preprocessing failure: {preprocess_messages}", flush=True)
                continue
            
            X = scalers[symbol].transform(X.reshape(-1, X.shape[-1])).reshape(X.shape)
            predictions = models[symbol].predict(X, verbose=0, batch_size=BATCH_SIZE)
            
            # Log prediction statistics
            logging.info(f"Predictions for {symbol}: mean={np.mean(predictions):.3f}, std={np.std(predictions):.3f}, min={np.min(predictions):.3f}, max={np.max(predictions):.3f}")
            with print_lock:
                print(f"Predictions for {symbol}: mean={np.mean(predictions):.3f}, std={np.std(predictions):.3f}, min={np.min(predictions):.3f}, max={np.max(predictions):.3f}", flush=True)
            
            for i, confidence in enumerate(predictions.flatten()):
                if i >= len(features) - LOOKBACK:
                    break
                current_price = round(features['close'].iloc[i + LOOKBACK], 2)
                atr = features['ATR'].iloc[i + LOOKBACK] if 'ATR' in features else 1.0
                qty = int(min(cash[symbol] * MAX_POSITION_SIZE, cash[symbol]) / (atr + 1e-6) // current_price)
                
                if qty == 0:
                    logging.info(f"Zero quantity for {symbol} at index {i}: cash={cash[symbol]:.2f}, price={current_price:.2f}")
                    continue
                
                signal_type = 'Buy' if confidence > CONFIDENCE_THRESHOLD else 'Sell' if confidence < CONFIDENCE_THRESHOLD else 'Hold'
                logging.info(f"Backtest signal for {symbol} at index {i}: price={current_price:.2f}, confidence={confidence:.3f}, signal={signal_type}")
                
                if confidence > CONFIDENCE_THRESHOLD and cash[symbol] >= current_price and qty > 0 and positions[symbol] == 0:
                    positions[symbol] += qty
                    cash[symbol] -= qty * current_price
                    trades.append({"symbol": symbol, "action": "buy", "price": current_price, "qty": qty, "time": features.index[i + LOOKBACK]})
                    with print_lock:
                        print(f"Backtest Buy: {symbol} at {current_price:.2f}, Qty: {qty}, Confidence: {confidence:.3f}", flush=True)
                elif confidence < CONFIDENCE_THRESHOLD and positions[symbol] > 0:
                    cash[symbol] += positions[symbol] * current_price
                    trades.append({"symbol": symbol, "action": "sell", "price": current_price, "qty": positions[symbol], "time": features.index[i + LOOKBACK]})
                    with print_lock:
                        print(f"Backtest Sell: {symbol} at {current_price:.2f}, Qty: {positions[symbol]}, Confidence: {confidence:.3f}", flush=True)
                    positions[symbol] = 0
                else:
                    logging.info(f"No trade for {symbol} at index {i}: confidence={confidence:.3f}, signal={signal_type}")
        
        final_value = sum(cash.values()) + sum(
            positions[symbol] * round(data["close"].iloc[-1], 2) for symbol in SYMBOLS if positions[symbol] > 0
        )
        with print_lock:
            print(f"\nBacktest Results: Initial Cash: ${INITIAL_CASH:.2f}, Final Value: ${final_value:.2f}", flush=True)
            print(f"Profit/Loss: ${(final_value - INITIAL_CASH):.2f}", flush=True)
        
        send_email(
            subject="Backtest Completed",
            body=f"Backtest Results:\n"
                 f"Initial Cash: ${INITIAL_CASH:.2f}\n"
                 f"Final Value: ${final_value:.2f}\n"
                 f"Profit/Loss: ${(final_value - INITIAL_CASH):.2f}\n"
                 f"Total Trades: {len(trades)}"
        )
    
    except Exception as e:
        with print_lock:
            print(f"Error in backtest: {e}", flush=True)
        send_email(
            subject="Backtest Error",
            body=f"Error in backtest:\n"
                 f"Message: {e}"
        )
        raise
    finally:
        log_queue.put(None)
        debug_queue.put(None)

if __name__ == "__main__":
    # Validate Alpaca API credentials
    try:
        api.get_account()
    except Exception as e:
        logging.error(f"Alpaca API authentication failed: {e}")
        print(f"Alpaca API authentication failed: {e}")
        sys.exit(1)
    
    logging.info("Bot started.")
    with print_lock:
        print("Bot started.", flush=True)
    
    # Initialize sentiment analysis pipeline
    sentiment_analyzer = None
    try:
        sentiment_analyzer = pipeline("sentiment-analysis", model="ProsusAI/finbert")
        logging.info("FinBERT initialized successfully.")
        with print_lock:
            print("FinBERT initialized successfully.", flush=True)
    except Exception as e:
        logging.error(f"Error initializing FinBERT: {e}. Sentiment analysis disabled.")
        with print_lock:
            print(f"Error initializing FinBERT: {e}. Sentiment analysis disabled.", flush=True)
        sentiment_analyzer = None
    
    parser = argparse.ArgumentParser(description="Alpaca Neural Bot")
    parser.add_argument("--mode", choices=["live", "backtest"], default="live", help="Mode: live or backtest")
    args = parser.parse_args()

    if args.mode == "live":
        with print_lock:
            print(f"Starting advanced neural network trading bot for {', '.join(SYMBOLS)}...", flush=True)
            print("Training neural network with sentiment and volatility adjustments...", flush=True)
        send_email(
            subject="Trading Bot Started",
            body=f"Advanced neural network trading bot started for {', '.join(SYMBOLS)}.\n"
                 f"Training with {TRAIN_DAYS} days of {TRAIN_TIMEFRAME} data.\n"
                 f"Initial cash: ${INITIAL_CASH:.2f}"
        )
        trading_strategy(sentiment_analyzer)
    elif args.mode == "backtest":
        with print_lock:
            print("Starting backtesting mode...", flush=True)
        backtest_strategy(sentiment_analyzer)