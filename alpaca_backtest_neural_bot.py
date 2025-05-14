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
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
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
import multiprocessing as mp
from contextlib import contextmanager

# Suppress TensorFlow warnings before import
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# Redirect stderr during TensorFlow import
if os.name == 'nt':
    stderr = sys.stderr
    sys.stderr = open('nul', 'w')
import tensorflow as tf
if os.name == 'nt':
    sys.stderr.close()
    sys.stderr = stderr

tf.get_logger().setLevel('ERROR')
os.environ["TF_XLA_FLAGS"] = "--tf_xla_cpu_global_jit"
tf.config.threading.set_inter_op_parallelism_threads(8)
tf.config.threading.set_intra_op_parallelism_threads(8)

# Initialize colorama
init(autoreset=True)

# Define thread-safe locks
print_lock = threading.Lock()

# Logging filters
class SuppressTFDeprecationWarning(logging.Filter):
    def filter(self, record):
        return "sparse_softmax_cross_entropy is deprecated" not in record.getMessage()

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
logger.addFilter(SuppressTFDeprecationWarning())
logger.addFilter(suppress_child_process_logs)
debug_logger = logging.getLogger('debug')
debug_logger.setLevel(logging.DEBUG)
debug_logger.handlers = []
debug_logger.addHandler(debug_queue_handler)
debug_logger.addFilter(SuppressTFDeprecationWarning())
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
TRADE_DAYS = 10
LOOKBACK = 30
INITIAL_CASH = 1000.0
CASH_PER_STOCK = INITIAL_CASH / len(SYMBOLS)
MAX_POSITION_SIZE = 0.9 / len(SYMBOLS)
DATA_CACHE = "data_cache"
MODEL_CACHE = "model_cache"
CONFIDENCE_THRESHOLD = 0.5
EPOCHS_PER_STOCK = 15
BATCH_SIZE = 256

# Email configuration
EMAIL_SENDER = "alpaca.ai.tradingbot@gmail.com"
EMAIL_PASSWORD = "hjdf sstp pyne rotq"
EMAIL_RECIPIENT = ["aiplane.scientist@gmail.com", "tchaikovskiy@hotmail.com", "evmakarov.md@gmail.com"]
SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 587

# Initialize Alpaca API
api = REST(API_KEY, API_SECRET, BASE_URL, api_version="v2")

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

def get_stock_data(symbol, timeframe, days, feed="iex"):
    cache_file = f"{DATA_CACHE}/{symbol}_{timeframe}_{days}.pkl"
    news_cache_file = f"{DATA_CACHE}/{symbol}_news_{days}.pkl"
    os.makedirs(DATA_CACHE, exist_ok=True)
    output_messages = []
    try:
        if os.path.exists(cache_file):
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
                    time.sleep(0.3)
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
                    output_messages.append(f"Error fetching data for {symbol} on attempt