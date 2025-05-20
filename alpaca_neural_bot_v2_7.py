# +------------------------------------------------------------------------------+
# |                            Alpaca Neural Bot v2.7                            |
# +------------------------------------------------------------------------------+
# | Author: Vladimir Makarov                                                     |
# | Project Start Date: May 9, 2025                                              |
# | License: GNU Lesser General Public License v2.1                              |
# | Version: 2.4 (Released May 18, 2025)                                         |
# |                                                                              |
# | Description:                                                                 |
# | This is an advanced AI-powered trading bot leveraging neural networks to     |
# | predict market movements and execute trades. It integrates seamlessly with   |
# | Alpaca for real-time market data and trade execution. Key features include:  |
# | - Neural Network Predictions: Trained on historical and live data for        |
# |   accurate buy/sell signals.                                                 |
# | - Sentiment Analysis: Incorporates news and social media sentiment to        |
# |   enhance decision-making.                                                   |
# | - Technical Indicators: Uses RSI, MACD, ADX, and more for market analysis.   |
# | - Dynamic Risk Management: Adjusts position sizing and stop-loss based on    |
# |   volatility and max drawdown limits.                                        |
# | - Backtesting & Live Trading: Supports simulation and real-time modes.       |
# | What makes it special: Its ability to adapt to market conditions using AI,   |
# | combined with a robust feature set, sets it apart from traditional bots.     |
# |                                                                              |
# | Configuration: Defined in the CONFIG dictionary. See code for details.       |
# |                                                                              |
# |                                                                              |
# | Dependencies:                                                                |
# | - tensorflow (Neural network framework)                                      |
# | - numpy (Numerical computations)                                             |
# | - pandas (Data manipulation)                                                 |
# | - alpaca-trade-api (Alpaca integration)                                      |
# | - transformers (Sentiment analysis)                                          | 
# | - scikit-learn (Machine learning utilities)                                  | 
# | - TA-Lib (Technical analysis)                                                |
# | - tenacity (Retry logic)                                                     |
# | - smtplib (Email notifications)                                              |
# | - argparse (Command-line parsing)                                            |
# | - tqdm (Progress bars)                                                       |
# | - colorama (Console formatting)                                              |
# | Install using: pip install <package>                                         |
# |                                                                              |
# | Double check before starting:                                                |
# | - Ensure Alpaca API keys are configured in CONFIG.                           |
# | - Requires stable internet for live trading and data fetching .              |
# | - GitHub: https://github.com/vmakarov28/Alpaca-Stock-Trading-Bot/tree/main   |
# |                                                                              |
# +------------------------------------------------------------------------------+
#Add emergancy stop feture to prevent overfitting during training


import os
import sys
import logging
import argparse
import importlib
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, LSTM, Dense, Dropout
from tensorflow.keras import backend as K
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping, Callback
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

# Suppress TensorFlow warnings
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings('ignore', category=UserWarning)

# Initialize colorama for colored console output
colorama.init()

# Configuration variables
CONFIG = {
    'SYMBOLS': ['NVDA', 'TSLA', 'AAPL', 'MSFT', 'AMZN', 'SPY'],
    'TIMEFRAME': TimeFrame(15, TimeFrameUnit.Minute),
    'TRAIN_DATA_START_DATE': '2015-01-01',
    'SIMULATION_DAYS': 365,
    'TRAIN_EPOCHS': 25,
    'BATCH_SIZE': 32,
    'INITIAL_CASH': 25000.0,
    'TIMESTEPS': 30,
    'MIN_DATA_POINTS': 100,
    'CACHE_DIR': './cache',
    'CACHE_EXPIRY_SECONDS': 24 * 60 * 60,
    'ALPACA_API_KEY': 'PKAH6YBD35AZNOVTHQ2G',
    'ALPACA_SECRET_KEY': 'AdGuIkWGOyeiegwseljcSGjUhdGw7YpBPLrepTEw',
    'EMAIL_SENDER': 'alpaca.ai.tradingbot@gmail.com',
    'EMAIL_PASSWORD': 'hjdf sstp pyne rotq',
    'EMAIL_RECEIVER': ['aiplane.scientist@gmail.com', 'tchaikovskiy@hotmail.com'],
    'SMTP_SERVER': 'smtp.gmail.com',
    'SMTP_PORT': 587,
    'LOG_FILE': 'trades.log',
    'API_RETRY_ATTEMPTS': 3,
    'API_RETRY_DELAY': 1000,
    'PREDICTION_THRESHOLD_BUY': 0.6,
    'PREDICTION_THRESHOLD_SELL': 0.4,
    'LIVE_DATA_BARS': 200,
    'MIN_HOLDING_PERIOD_MINUTES': 60,
    'MAX_DRAWDOWN_LIMIT': 0.20,
    'TRANSACTION_COST_PER_TRADE': 1.0,
    'SENTIMENT_MODEL': 'distilbert-base-uncased-finetuned-sst-2-english',
    'TRAILING_STOP_PERCENTAGE': 0.02,
    'CONFIDENCE_THRESHOLD': 0.5,
    'MAX_VOLATILITY': 3,
    'ADX_TREND_THRESHOLD': 30,
    'RISK_PERCENTAGE': 1,
    'STOP_LOSS_ATR_MULTIPLIER': 1.1,
    'TAKE_PROFIT_ATR_MULTIPLIER': 2.0,
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
sentiment_pipeline = pipeline("sentiment-analysis", model=CONFIG['SENTIMENT_MODEL'])

def check_dependencies() -> None:
    """Check for required Python modules."""
    required_modules = [
        'tensorflow', 'numpy', 'pandas', 'alpaca_trade_api', 'transformers',
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
            request = StockBarsRequest(
                symbol_or_symbols=symbol,
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
    """Compute real-time news sentiment using a pre-trained model."""
    cache_file = os.path.join(CONFIG['CACHE_DIR'], f"{symbol}_news_sentiment.pkl")
    if os.path.exists(cache_file) and (time.time() - os.path.getmtime(cache_file)) < CONFIG['CACHE_EXPIRY_SECONDS']:
        with open(cache_file, 'rb') as f:
            sentiment = pickle.load(f)
        return sentiment, True
    else:
        news_text = "Sample news text for " + symbol
        sentiment_result = sentiment_pipeline(news_text)[0]
        sentiment_score = sentiment_result['score'] if sentiment_result['label'] == 'POSITIVE' else -sentiment_result['score']
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

def preprocess_data(df: pd.DataFrame, timesteps: int, add_noise: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """Preprocess data into sequences."""
    df = df.copy()
    df.ffill(inplace=True)
    df.fillna(0, inplace=True)
    
    features = [
        'close', 'high', 'low', 'volume', 'VWAP', 'MA20', 'MA50', 'RSI',
        'MACD', 'MACD_signal', 'OBV', 'VWAP', 'ATR', 'CMF', 'Close_ATR',
        'MA20_ATR', 'Return_1d', 'Return_5d', 'Volatility', 'BB_upper',
        'BB_lower', 'Stoch_K', 'Stoch_D', 'ADX', 'Sentiment'
    ]
    X = df[features].values
    y = df['Trend'].values
    
    if add_noise:
        X += np.random.normal(0, 0.02, X.shape)
    
    X_seq = [X[i:i + timesteps] for i in range(len(X) - timesteps)]
    y_seq = [y[i + timesteps - 1] for i in range(len(X) - timesteps)]
    
    return np.array(X_seq), np.array(y_seq)

def build_model(input_shape: Tuple[int, int]) -> tf.keras.Model:
    """Build a simplified neural network model with stronger regularization."""
    K.clear_session()
    model = Sequential([
        Conv1D(32, kernel_size=3, activation='relu', input_shape=input_shape),
        MaxPooling1D(pool_size=2),
        LSTM(64, return_sequences=True),
        LSTM(64, return_sequences=False),
        Dense(128, activation='relu', kernel_regularizer=l2(0.05)),
        Dropout(0.6),
        Dense(64, activation='relu', kernel_regularizer=l2(0.05)),
        Dropout(0.6),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

class TQDMCallback(Callback):
    """Custom callback to update tqdm progress bar after each epoch."""
    def __init__(self, progress_bar):
        super().__init__()
        self.progress_bar = progress_bar

    def on_epoch_end(self, epoch, logs=None):
        self.progress_bar.update(1)

def train_model(symbol: str, X: np.ndarray, y: np.ndarray, epochs: int, batch_size: int, progress_bar) -> Tuple[tf.keras.Model, StandardScaler]:
    """Train the model with early stopping."""
    train_size = int(0.8 * len(X))
    X_train, X_val = X[:train_size], X[train_size:]
    y_train, y_val = y[:train_size], y[train_size:]
    scaler = StandardScaler()
    X_scaled_train = scaler.fit_transform(X_train.reshape(-1, X.shape[-1])).reshape(X_train.shape)
    X_scaled_val = scaler.transform(X_val.reshape(-1, X.shape[-1])).reshape(X_val.shape)
    model = build_model((X.shape[1], X.shape[2]))
    
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True
    )
    
    model.fit(
        X_scaled_train, y_train,
        validation_data=(X_scaled_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stopping, TQDMCallback(progress_bar)],
        verbose=0
    )
    
    model.save(os.path.join(CONFIG['CACHE_DIR'], f"{symbol}_model.keras"))
    with open(os.path.join(CONFIG['CACHE_DIR'], f"{symbol}_scaler.pkl"), 'wb') as f:
        pickle.dump(scaler, f)
    return model, scaler

def load_model_and_scaler(symbol: str, expected_features: int, force_train: bool) -> Tuple[Optional[tf.keras.Model], Optional[StandardScaler]]:
    """Load pre-trained model and scaler."""
    if force_train:
        return None, None
    model_path = os.path.join(CONFIG['CACHE_DIR'], f"{symbol}_model.keras")
    scaler_path = os.path.join(CONFIG['CACHE_DIR'], f"{symbol}_scaler.pkl")
    if os.path.exists(model_path) and os.path.exists(scaler_path):
        model = tf.keras.models.load_model(model_path)
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        if scaler.n_features_in_ == expected_features:
            return model, scaler
    return None, None

def backtest(
    symbol: str,
    model: tf.keras.Model,
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
    X, _ = preprocess_data(df, timesteps)
    X_scaled = scaler.transform(X.reshape(-1, X.shape[-1])).reshape(X.shape)
    predictions = model.predict(X_scaled, verbose=0).flatten()
    timestamps = df['timestamp'].iloc[timesteps:].reset_index(drop=True)
    sim_start = timestamps.max() - timedelta(days=CONFIG['SIMULATION_DAYS'])
    k_start = timestamps[timestamps >= sim_start].index[0]
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
    sim_timestamps = timestamps.values

    for i in range(k_start, len(predictions)):
        pred = predictions[i]
        price = prices[i]
        atr_val = atr[i]
        current_rsi = rsi[i]
        current_adx = adx[i]
        current_volatility = volatility[i]
        ts = pd.Timestamp(sim_timestamps[i])

        if current_volatility > max_volatility or current_adx < 20:
            continue

        if pred < confidence_threshold:
            continue

        qty = int((cash * risk_percentage) / (atr_val * stop_loss_atr_multiplier)) if cash >= price else 0

        if pred > buy_threshold and position == 0 and qty > 0 and current_rsi < 30 and current_adx > adx_trend_threshold:
            position = qty
            entry_price = price
            max_price = price
            entry_time = ts
            cash -= qty * price + transaction_cost_per_trade
            logger.info(f"{ts}: Bought {qty} shares of {symbol} at ${price:.2f}, cash: ${cash:.2f}")

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
                if price <= trailing_stop or price <= stop_loss or price >= take_profit or (pred < sell_threshold and current_rsi > 70):
                    cash += position * price - transaction_cost_per_trade
                    ret = (price - entry_price) / entry_price
                    returns.append(ret)
                    trade_count += 1
                    if ret > 0:
                        winning_trades += 1
                    logger.info(f"{ts}: Sold {position} shares of {symbol} at ${price:.2f}, return: {ret:.3f}, cash: ${cash:.2f}")
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

def main(backtest_only: bool = False, force_train: bool = False) -> None:
    """Main function to execute the trading bot with portfolio-level risk management."""
    if not backtest_only:
        logger.info("Bot started")
        stream_handler.setLevel(logging.CRITICAL)
    else:
        logger.info("Bot started in backtest mode")
    
    check_dependencies()
    validate_config(CONFIG)
    create_cache_directory()
    trading_client = TradingClient(CONFIG['ALPACA_API_KEY'], CONFIG['ALPACA_SECRET_KEY'], paper=True)
    expected_features = 25
    models = {}
    scalers = {}
    dfs = {}
    stock_info = []
    total_epochs = len(CONFIG['SYMBOLS']) * CONFIG['TRAIN_EPOCHS']
    need_training = any(load_model_and_scaler(symbol, expected_features, force_train)[0] is None for symbol in CONFIG['SYMBOLS'])
    progress_bar = tqdm(total=total_epochs, desc="Training Progress", bar_format="{l_bar}\033[32m{bar}\033[0m{r_bar}") if need_training else None

    for symbol in CONFIG['SYMBOLS']:
        df, data_loaded = load_or_fetch_data(symbol, CONFIG['TRAIN_DATA_START_DATE'], datetime.now().strftime('%Y-%m-%d'))
        validate_raw_data(df, symbol)
        sentiment, sentiment_loaded = load_news_sentiment(symbol)
        df = calculate_indicators(df, sentiment)
        validate_indicators(df, symbol)
        dfs[symbol] = df
        model, scaler = load_model_and_scaler(symbol, expected_features, force_train)
        model_loaded = model is not None and scaler is not None
        if not model_loaded:
            X, y = preprocess_data(df, CONFIG['TIMESTEPS'], add_noise=True)
            model, scaler = train_model(symbol, X, y, CONFIG['TRAIN_EPOCHS'], CONFIG['BATCH_SIZE'], progress_bar)
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
            qty_owned = int(position.qty)
        except:
            qty_owned = 0
        info.append(f"  Current amount of stocks owned: {qty_owned}")
        if not model_loaded:
            info.append(f"  Saved model and scaler for {symbol}.")
        stock_info.append(info)

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
                for symbol in CONFIG['SYMBOLS']:
                    if symbol in models:
                        df = fetch_recent_data(symbol, CONFIG['LIVE_DATA_BARS'])
                        sentiment = load_news_sentiment(symbol)[0]
                        df = calculate_indicators(df, sentiment)
                        X, _ = preprocess_data(df, CONFIG['TIMESTEPS'], add_noise=False)
                        if X.shape[0] > 0:
                            last_sequence = X[-1].reshape(1, CONFIG['TIMESTEPS'], -1)
                            X_scaled = scalers[symbol].transform(last_sequence.reshape(-1, last_sequence.shape[-1])).reshape(last_sequence.shape)
                            prediction = models[symbol].predict(X_scaled, verbose=0)[0][0]
                            price = df['close'].iloc[-1]
                            current_rsi = df['RSI'].iloc[-1]
                            current_adx = df['ADX'].iloc[-1]
                            current_volatility = df['Volatility'].iloc[-1]
                            atr_val = df['ATR'].iloc[-1]
                            try:
                                position = trading_client.get_open_position(symbol)
                                qty_owned = int(position.qty)
                                entry_time = pd.Timestamp(position.entry_time)
                                entry_price = float(position.avg_entry_price)
                                time_held = (now - entry_time).total_seconds() / 60
                            except:
                                qty_owned = 0
                                entry_time = None
                                entry_price = 0.0
                                time_held = 0

                            decision = "Hold"
                            if current_volatility <= CONFIG['MAX_VOLATILITY'] and current_adx >= 20 and prediction >= CONFIG['CONFIDENCE_THRESHOLD']:
                                if prediction > CONFIG['PREDICTION_THRESHOLD_BUY'] and qty_owned == 0 and current_rsi < 30 and current_adx > CONFIG['ADX_TREND_THRESHOLD']:
                                    decision = "Buy"
                                    cash = float(account.cash)
                                    qty = int((cash * CONFIG['RISK_PERCENTAGE']) / (atr_val * CONFIG['STOP_LOSS_ATR_MULTIPLIER']))
                                    if qty > 0:
                                        order = MarketOrderRequest(symbol=symbol, qty=qty, side=OrderSide.BUY, time_in_force=TimeInForce.GTC)
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
                                elif qty_owned > 0:
                                    max_price = max(float(position.highest_price), price) if hasattr(position, 'highest_price') else price
                                    trailing_stop = max_price * (1 - CONFIG['TRAILING_STOP_PERCENTAGE'])
                                    stop_loss = entry_price - CONFIG['STOP_LOSS_ATR_MULTIPLIER'] * atr_val
                                    take_profit = entry_price + CONFIG['TAKE_PROFIT_ATR_MULTIPLIER'] * atr_val
                                    if time_held >= CONFIG['MIN_HOLDING_PERIOD_MINUTES']:
                                        if price <= trailing_stop or price <= stop_loss or price >= take_profit or (prediction < CONFIG['PREDICTION_THRESHOLD_SELL'] and current_rsi > 70):
                                            decision = "Sell"
                                            order = MarketOrderRequest(symbol=symbol, qty=qty_owned, side=OrderSide.SELL, time_in_force=TimeInForce.GTC)
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
                # Send summary email every 15 minutes
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Trading bot with backtest mode")
    parser.add_argument('--backtest', action='store_true', help="Run in backtest-only mode")
    parser.add_argument('--force-train', action='store_true', help="Force retraining of models")
    args = parser.parse_args()
    main(backtest_only=args.backtest, force_train=args.force_train)