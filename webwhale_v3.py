`python
import os
import logging
import pandas as pd
import numpy as np
from datetime import datetime
import hmac
import hashlib
import time
import asyncio
import aiohttp
import json
from dotenv import load_dotenv
from typing import Dict, Tuple, Optional, Union, Callable,
List
from zoneinfo import ZoneInfo
from decimal import Decimal, getcontext
import aiofiles
from logging.handlers import RotatingFileHandler
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.logging import RichHandler
from rich.theme import Theme

# --- Configuration and Setup ---

getcontext().prec = 10
load_dotenv()

API_KEY = os.getenv("BYBIT_API_KEY")
API_SECRET = os.getenv("BYBIT_API_SECRET")
if not API_KEY or not API_SECRET:
    raise ValueError("BYBIT_API_KEY and BYBIT_API_SECRET
must be set in .env")

WEBSOCKET_URL = "wss://stream.bybit.com/v5/public/linear"
BASE_URL = "https://api.bybit.com"
CONFIG_FILE = "config.json"
LOG_DIRECTORY = "bot_logs"
TIMEZONE = ZoneInfo("America/Chicago")
RECONNECT_DELAY = 5
CACHE_TTL_SECONDS = 60
VALID_INTERVALS = ["1", "3", "5", "15", "30", "60", "120",
"240", "D", "W", "M"]
OUTPUT_THROTTLE_SECONDS = 60
MAX_API_RETRIES = 3

console = Console(theme=Theme({
    "logging.level.info": "cyan",
    "logging.level.warning": "yellow",
    "logging.level.error": "bold red",
    "repr.number": "bold magenta",
    "repr.string": "green",
    "table.header": "bold blue",
    "table.cell": "white",
    "signal.long": "green",
    "signal.short": "red",
    "signal.neutral": "yellow",
    "indicator.bullish": "green",
    "indicator.bearish": "red",
    "indicator.neutral": "yellow",
    "level.support": "green",
    "level.resistance": "red",
}))

os.makedirs(LOG_DIRECTORY, exist_ok=True)

# --- Utility and Helper Functions ---

class SensitiveFormatter(logging.Formatter):
    """Formatter to mask sensitive information (API keys,
secrets) in logs."""
    def format(self, record):
        msg = super().format(record)
        return msg.replace(API_KEY,
"***").replace(API_SECRET, "***")

def load_config(filepath: str) -> dict:
    """
    Loads configuration from JSON file, or creates a
default one if not found.
    Validates and corrects common configuration issues,
ensuring the bot runs smoothly.
    """
    default_config = {
        "interval": "15",
        "analysis_interval": 30,
        "retry_delay": 5,
        "momentum_period": 10,
        "momentum_ma_short": 12,
        "momentum_ma_long": 26,
        "volume_ma_period": 20,
        "atr_period": 14,
        "trend_strength_threshold": 0.4,
        "indicators": {
            "ema_alignment": {"enabled": True, "display":
True},
            "momentum": {"enabled": True, "display": True},
            "volume_confirmation": {"enabled": True,
"display": True},
            "divergence": {"enabled": True, "display":
False},
            "stoch_rsi": {"enabled": True, "display":
True},
            "rsi": {"enabled": True, "display": True},
            "macd": {"enabled": True, "display": True},
            "bollinger_bands": {"enabled": True, "display":
True},
            "bb_squeeze": {"enabled": True, "display":
False},
            "vwap_bounce": {"enabled": True, "display":
True},
            "pivot_breakout": {"enabled": True, "display":
False},
        },
        "weight_sets": {
            "low_volatility": {
                "ema_alignment": 0.4, "momentum": 0.3,
"volume_confirmation": 0.2, "divergence": 0.1,
                "stoch_rsi": 0.7, "rsi": 0.6, "macd": 0.5,
"bollinger_bands": 0.4, "bb_squeeze": 0.3,
                "vwap_bounce": 0.3, "pivot_breakout": 0.3,
            }
        },
        "rsi_period": 14,
        "bollinger_bands_period": 20,
        "bollinger_bands_std_dev": 2,
        "orderbook_limit": 50,
        "signal_config": {
            "signal_threshold": 0.3,
            "stop_loss_atr_multiplier": 2,
            "take_profit_risk_reward_ratio": 2,
            "ema_period": 20,
            "macd_fast": 12,
            "macd_slow": 26,
            "macd_signal": 9,
            "stoch_rsi_period": 14,
            "stoch_rsi_k": 3,
            "stoch_rsi_d": 3,
            "bb_squeeze_percentile": 10,
            "bb_squeeze_lookback": 20,
            "divergence_lookback": 20,
        },
        "output": {
            "save_to_json": True,
            "json_output_dir": "output",
            "alert_file": "signals.log",
            "save_to_csv": False,
            "csv_output_dir": "csv_output"
        }
    }
    if not os.path.exists(filepath):
        with open(filepath, 'w') as f:
            json.dump(default_config, f, indent=4)
        console.print(Panel(f"[bold yellow]Created new
config file at '{filepath}'.[/bold yellow]", title="[bold
cyan]Configuration Setup[/bold cyan]"))
        return default_config
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            config = json.load(f)
        _validate_config(config, default_config) # Validate
and correct config
        with open(filepath, 'w') as f:
            json.dump(config, f, indent=4) # Save corrected
config to file
        return config
    except json.JSONDecodeError as e:
        console.print(Panel(f"[bold red]JSON config error
in '{filepath}': {e}. Using defaults.[/bold red]",
title="[bold cyan]Configuration Error[/bold cyan]"))
        return default_config
    except Exception as e:
        console.print(Panel(f"[bold red]Config file error:
{e}. Using defaults.[/bold red]", title="[bold
cyan]Configuration Error[/bold cyan]"))
        return default_config

def _validate_config(config: dict, default_config: dict):
    """
    Validates the loaded configuration, correcting any
common issues and ensuring
    required parameters are present. Uses default values if
necessary and logs warnings.
    """
    required_keys = ["interval", "indicators",
"weight_sets", "signal_config", "orderbook_limit"]
    for key in required_keys:
        if key not in config:
            console.print(f"[bold red]Missing '{key}' in
config. Using default.[/bold red]")
            config[key] = default_config[key] # Fallback to
default if missing
    for ind, val in list(config["indicators"].items()):
        if isinstance(val, bool): # Correct simplified
indicator configs to detailed format
            config["indicators"][ind] = {"enabled": val,
"display": val}
    if config["interval"] not in VALID_INTERVALS:
        console.print(f"[bold yellow]Invalid interval
'{config['interval']}'. Using default '15'.[/bold yellow]")
        config["interval"] = "15" # Default to 15m if
invalid interval
    if not isinstance(config["orderbook_limit"], int) or
config["orderbook_limit"] <= 0:
        console.print(f"[bold yellow]Invalid
orderbook_limit '{config['orderbook_limit']}'. Using
default 50.[/bold yellow]")
        config["orderbook_limit"] = 50 # Default to 50 if
invalid orderbook limit

CONFIG = load_config(CONFIG_FILE)

def setup_logger(symbol: str) -> logging.Logger:
    """
    Sets up a dedicated logger for each trading symbol.
Each symbol gets:
        - A rotating file handler to save logs to
individual files in 'bot_logs/' directory.
        - A rich console handler for formatted output to
the console.
    Sensitive information is masked in logs using
SensitiveFormatter.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = os.path.join(LOG_DIRECTORY, f"{symbol}
_{timestamp}.log")
    logger = logging.getLogger(symbol) # Get logger
instance for the symbol
    logger.setLevel(logging.DEBUG) # Set default logging
level to DEBUG

    # Rotating file handler to manage log file size
    file_handler = RotatingFileHandler(log_filename,
maxBytes=10 * 1024 * 1024, backupCount=5)

file_handler.setFormatter(SensitiveFormatter("%(asctime)s -
%(levelname)s - %(message)s")) # Apply sensitive data
masking

    # Rich console handler for pretty output in console
    console_handler = RichHandler(console=console,
rich_tracebacks=True)

console_handler.setFormatter(SensitiveFormatter("%(message)
s")) # Apply sensitive data masking to console output too

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger

# --- Data Cache ---

class DataCache:
    """
    Implements a simple in-memory cache with Time-To-Live
(TTL) for storing frequently
    accessed data like prices and order books, reducing API
calls and improving response times.
    """
    def __init__(self, ttl: int = CACHE_TTL_SECONDS):
        """Initializes the DataCache with a default TTL."""
        self.cache: Dict[str, Tuple[Union[Decimal,
pd.DataFrame, dict], float]] = {} # Cache storage: key ->
(value, expiry_timestamp)
        self.ttl = ttl # Time-to-live in seconds for cache
entries

    def get(self, key: str) -> Optional[Union[Decimal,
pd.DataFrame, dict]]:
        """
        Retrieves a value from the cache if it's not
expired. If expired or not found, returns None.
        """
        cached_data = self.cache.get(key) # Try to get data
from cache
        if cached_data:
            value, timestamp = cached_data
            if time.time() - timestamp < self.ttl: # Check
if cache entry is still valid (not expired)
                return value # Return cached value if valid
            else:
                del self.cache[key] # Expire and remove
from cache if TTL exceeded
        return None # Return None if key not found or cache
expired

    def set(self, key: str, value: Union[Decimal,
pd.DataFrame, dict]):
        """Sets a value in the cache with the current
timestamp as the creation time."""
        self.cache[key] = (value, time.time()) # Store
value in cache with current timestamp

data_cache = DataCache()

# --- REST API Functions ---

async def fetch_valid_symbols(session:
aiohttp.ClientSession, logger: logging.Logger) ->
List[str]:
    """
    Fetches a list of valid trading symbols from the Bybit
API. Only 'Trading' status symbols are returned.
    """
    url = f"{BASE_URL}/v5/market/instruments-info"
    params = {"category": "linear"}
    return await _bybit_api_request(session, logger, url,
params, method="GET", endpoint_description="symbols")

async def fetch_klines(symbol: str, interval: str, limit:
int, session: aiohttp.ClientSession, logger:
logging.Logger) -> pd.DataFrame:
    """
    Fetches Kline (Candlestick) data for a given symbol and
interval from the Bybit API.
    Returns the data as a Pandas DataFrame, with error
handling and retry logic.
    """
    url = f"{BASE_URL}/v5/market/kline"
    params = {"symbol": symbol, "interval": interval,
"limit": limit, "category": "linear"}
    raw_klines = await _bybit_api_request(session, logger,
url, params, method="GET", endpoint_description="klines")
    if raw_klines:
        df = pd.DataFrame(raw_klines,
columns=["start_time", "open", "high", "low", "close",
"volume", "turnover"])
        df["start_time"] =
pd.to_datetime(df["start_time"].astype(int), unit="ms") #
Convert timestamp to datetime
        df[["open", "high", "low", "close", "volume"]] =
df[["open", "high", "low", "close",
"volume"]].apply(pd.to_numeric) # Ensure numeric types
        df = df.sort_values("start_time") # Sort by start
time to ensure chronological order
        return df
    return pd.DataFrame() # Return empty DataFrame on
failure

async def _bybit_api_request(session:
aiohttp.ClientSession, logger: logging.Logger, url: str,
params: dict, method: str = "GET", endpoint_description:
str = "API") -> Optional[Union[list, dict]]:
    """
    Handles generic requests to the Bybit API, including
authentication, request signing,
    error handling, and retries. Supports GET and POST
methods.

    Returns the JSON response data on success, or None on
failure after max retries.
    """
    timestamp = str(int(time.time() * 1000))
    param_str = "&".join([f"{k}={v}" for k, v in
sorted({**params, 'timestamp': timestamp}.items())]) #
Construct sorted parameter string
    signature = hmac.new(API_SECRET.encode(),
param_str.encode(), hashlib.sha256).hexdigest() # Generate
API signature
    headers = {"X-BAPI-API-KEY": API_KEY, "X-BAPI-
TIMESTAMP": timestamp, "X-BAPI-SIGN": signature} # Set
authentication headers

    for attempt in range(MAX_API_RETRIES):
        try:
            async with session.request(method, url,
headers=headers, params=params, timeout=10) as response:
                response.raise_for_status() # Raise
HTTPError for bad responses (4xx or 5xx)
                data = await response.json()
                if data.get("retCode") == 0:
                    return data.get("result",
{}).get("list") if endpoint_description == "klines" else
data.get("result", {}) # Adjusted for klines endpoint
structure
                else:
                    logger.error(f"{endpoint_description}
fetch error (attempt {attempt + 1}/{MAX_API_RETRIES}):
{data.get('retMsg')}, code: {data.get('retCode')}")
        except aiohttp.ClientError as e: # Catch client-
related errors (like connection refused, etc.)
            logger.warning(f"{endpoint_description} fetch
failed (attempt {attempt + 1}/{MAX_API_RETRIES}): Client
error - {e}")
        except asyncio.TimeoutError: # Catch timeout errors
            logger.warning(f"{endpoint_description} fetch
failed (attempt {attempt + 1}/{MAX_API_RETRIES}): Timeout")
        except Exception as e: # Catch any other exceptions
during API request
            logger.error(f"{endpoint_description} fetch
failed (attempt {attempt + 1}/{MAX_API_RETRIES}):
Unexpected error - {e}")

        await asyncio.sleep(RECONNECT_DELAY) # Wait before
retrying

    logger.error(f"Failed to fetch {endpoint_description}
after {MAX_API_RETRIES} attempts.")
    return None # Indicate failure after all retries

# --- WebSocket Streaming ---

async def websocket_stream(symbol: str, interval: str,
analyzer: 'TradingAnalyzer', logger: logging.Logger):
    """
    Manages WebSocket connection to Bybit, subscribes to
Kline, Ticker, and Orderbook topics,
    and continuously processes incoming messages. Handles
reconnection logic and errors.
    """
    async with aiohttp.ClientSession() as session:
        initial_df = await fetch_klines(symbol, interval,
200, session, logger) # Fetch initial data via REST
        if initial_df.empty:
            logger.error(f"Cannot proceed with {symbol}: no
initial data received.")
            return # Stop if initial data fetch fails
        analyzer.update_data(initial_df) # Initialize
analyzer with fetched data
        logger.info(f"Loaded initial {len(initial_df)}
klines for {symbol}")

        while True: # Keep trying to reconnect on
disconnect/error
            try:
                async with
session.ws_connect(WEBSOCKET_URL, heartbeat=30.0,
timeout=30) as ws: # Establish WebSocket connection
                    logger.info(f"Connected to WebSocket
for {symbol}")
                    await _subscribe_websocket(ws, symbol,
interval, logger) # Subscribe to required topics

                    async for msg in ws: # Main message
processing loop
                        if msg.type ==
aiohttp.WSMsgType.TEXT:
                            data = json.loads(msg.data)
                            if "success" in data and not
data["success"]: # Handle subscription errors
                                await
_handle_subscription_error(data, symbol, logger)
                            elif "topic" in data: # Process
valid data messages
                                await
process_websocket_message(data, symbol, interval, analyzer,
logger)
                        elif msg.type in
(aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.ERROR): #
Handle connection closures/errors
                            logger.warning(f"WebSocket
closed or errored for {symbol}: {msg}")
                            break # Reconnect on close or
error

            except aiohttp.ClientConnectorError as e: #
Catch initial connection errors
                logger.error(f"WebSocket connection error
for {symbol}: {e}")
            except Exception as e: # Catch unexpected
errors during websocket operations
                logger.error(f"Unexpected WebSocket error
for {symbol}: {e}")

            await asyncio.sleep(RECONNECT_DELAY) # Wait
before attempting to reconnect

async def _subscribe_websocket(ws:
aiohttp.WebSocketClientWebSocketResponse, symbol: str,
interval: str, logger: logging.Logger):
    """
    Sends subscription requests to the WebSocket for Kline,
Ticker, and Orderbook data.
    Handles potential errors during subscription.
    """
    subscriptions = [
        {"op": "subscribe", "args": [f"kline.{interval}.
{symbol}"]}, # Subscribe to Kline data
        {"op": "subscribe", "args": [f"tickers.{symbol}"]},
# Subscribe to Ticker data (price updates)
        {"op": "subscribe", "args": [f"orderbook.
{CONFIG['orderbook_limit']}.{symbol}"]} # Subscribe to
Orderbook data
    ]
    for sub in subscriptions:
        try:
            await ws.send_json(sub) # Send subscription
message to WebSocket
            logger.debug(f"Subscribed to: {sub['args']
[0]}")
        except Exception as e:
            logger.error(f"WebSocket subscription error for
{symbol} - {sub['args'][0]}: {e}")

async def _handle_subscription_error(data: dict, symbol:
str, logger: logging.Logger):
    """
    Handles errors reported by the WebSocket server related
to subscriptions.
    Specifically, it checks for invalid symbol errors which
are considered fatal.
    """
    ret_msg = data.get("ret_msg", "Unknown error")
    ret_code = data.get("ret_code", -1)
    if ret_code == 10001: # Error code 10001 often
indicates invalid symbol
        logger.error(f"Invalid symbol {symbol}: {ret_msg}")
# Log invalid symbol error
        raise ValueError(f"Invalid symbol: {symbol}") #
Stop processing this symbol due to invalid symbol
    else:
        logger.error(f"Subscription failed for {symbol}:
{ret_msg} (code: {ret_code})") # Log other subscription
errors

async def process_websocket_message(data: dict, symbol:
str, interval: str, analyzer: 'TradingAnalyzer', logger:
logging.Logger):
    """
    Processes incoming WebSocket messages, routing them to
specific handlers based on the message topic.
    Handles Kline, Ticker, and Orderbook data updates.
    """
    topic = data["topic"]
    try:
        if topic.startswith("kline"):
            await _process_kline_message(data, symbol,
analyzer, logger) # Process Kline data
        elif topic.startswith("tickers"):
            await _process_ticker_message(data, symbol,
analyzer, logger) # Process Ticker data
        elif topic.startswith("orderbook"):
            await _process_orderbook_message(data, symbol,
logger) # Process Orderbook data
    except Exception as e:
        logger.error(f"Error processing WebSocket message
for {symbol} topic '{topic}': {e}") # Log errors during
message processing

async def _process_kline_message(data: dict, symbol: str,
analyzer: 'TradingAnalyzer', logger: logging.Logger):
    """Processes Kline (candlestick) data received from the
WebSocket."""
    kline_data = data["data"][0] # Extract kline data from
message
    if kline_data["confirm"]: # Only process confirmed
(closed) klines
        df = pd.DataFrame([{
            "start_time":
pd.to_datetime(kline_data["start"], unit="ms"),
            "open": float(kline_data["open"]),
            "high": float(kline_data["high"]),
            "low": float(kline_data["low"]),
            "close": float(kline_data["close"]),
            "volume": float(kline_data["volume"]),
            "turnover": float(kline_data["turnover"])
        }])
        analyzer.update_data(df) # Update TradingAnalyzer
with new kline data
        logger.debug(f"Kline update for {symbol}:
{kline_data['close']}")

async def _process_ticker_message(data: dict, symbol: str,
analyzer: 'TradingAnalyzer', logger: logging.Logger):
    """Processes ticker data (price updates) received from
the WebSocket."""
    current_price = Decimal(data["data"]["lastPrice"]) #
Extract last price from ticker data
    data_cache.set(f"price_{symbol}", current_price) #
Cache current price for access by other modules if needed
    await analyzer.analyze_and_output(float(current_price),
logger) # Trigger analysis and signal output
    logger.debug(f"Price update for {symbol}:
{current_price}")

async def _process_orderbook_message(data: dict, symbol:
str, logger: logging.Logger):
    """Processes orderbook data received from the
WebSocket."""
    orderbook = {"bids": data["data"]["b"], "asks":
data["data"]["a"]} # Extract bids and asks from orderbook
data
    data_cache.set(f"orderbook_{symbol}", orderbook) #
Cache orderbook data
    logger.debug(f"Orderbook update for {symbol}")

# --- Trading Signal Functions ---

SignalFunction = Callable[[Dict[str, Union[pd.Series,
pd.DataFrame]], float, dict], int]

def base_signal(value: float, upper: float, lower: float,
inverse: bool = False) -> int:
    """
    Provides a base signal logic for indicators that
trigger signals based on upper and lower thresholds.
    Supports both direct and inverse threshold logic.
    """
    if inverse: # Inverse logic: Signal when value is
outside the thresholds (e.g., price outside Bollinger
Bands)
        return 1 if value < lower else -1 if value > upper
else 0
    else: # Direct logic: Signal when value is within/
crosses the thresholds (e.g., RSI overbought/oversold)
        return 1 if value > upper else -1 if value < lower
else 0

def ema_alignment_signal(indicators_df: Dict[str,
Union[pd.Series, pd.DataFrame]], current_price: float,
config: dict) -> int:
    """
    Generates a trading signal based on EMA alignment
relative to the current price.
    Signal logic is inverse: Long if price is below EMA,
Short if price is above EMA.
    """
    ema_value = indicators_df["ema"].iloc[-1] # Get latest
EMA value
    return base_signal(current_price, upper=ema_value,
lower=ema_value, inverse=True) # Use base_signal with
inverse logic

def momentum_signal(indicators_df: Dict[str,
Union[pd.Series, pd.DataFrame]], current_price: float,
config: dict) -> int:
    """
    Generates a trading signal based on momentum trend.
Signal direction is determined by momentum trend.
    Uptrend: Long signal, Downtrend: Short signal, Neutral:
No signal.
    """
    trend = indicators_df["mom"].iloc[-1]["trend"] # Get
latest momentum trend
    return {"Uptrend": 1, "Downtrend": -1, "Neutral": 0}
[trend] # Map trend to signal value

def volume_confirmation_signal(indicators_df: Dict[str,
Union[pd.Series, pd.DataFrame]], current_price: float,
config: dict) -> int:
    """
    Generates a trading signal based on volume confirmation
using On Balance Volume (OBV).
    Long signal if OBV is increasing, Short signal if OBV
is decreasing.
    """
    obv = indicators_df["obv"] # Get OBV series
    if len(obv) < 2:
        return 0 # Not enough OBV data to generate signal
    return 1 if obv.iloc[-1] > obv.iloc[-2] else -1 if
obv.iloc[-1] < obv.iloc[-2] else 0 # Compare current OBV
with previous

def stoch_rsi_signal(indicators_df: Dict[str,
Union[pd.Series, pd.DataFrame]], current_price: float,
config: dict) -> int:
    """
    Generates a trading signal based on Stochastic RSI
indicator.
    Overbought (>0.8): Short signal, Oversold (<0.2): Long
signal.
    """
    stoch_rsi_k = indicators_df["stoch_rsi_k"].iloc[-1] #
Get latest Stoch RSI K value
    return base_signal(stoch_rsi_k, upper=0.8, lower=0.2) #
Use base_signal for threshold-based signal

def rsi_signal(indicators_df: Dict[str, Union[pd.Series,
pd.DataFrame]], current_price: float, config: dict) -> int:
    """
    Generates a trading signal based on Relative Strength
Index (RSI).
    Overbought (>70): Short signal, Oversold (<30): Long
signal.
    """
    rsi_value = indicators_df["rsi"].iloc[-1] # Get latest
RSI value
    return base_signal(rsi_value, upper=70, lower=30) # Use
base_signal for threshold-based signal

def macd_signal(indicators_df: Dict[str, Union[pd.Series,
pd.DataFrame]], current_price: float, config: dict) -> int:
    """
    Generates a trading signal based on Moving Average
Convergence Divergence (MACD) crossover.
    MACD line crosses above Signal line: Long signal, MACD
line crosses below Signal line: Short signal.
    """
    macd_line = indicators_df["macd"]["macd"].iloc[-1] #
Get latest MACD line value
    signal_line = indicators_df["macd"]["signal"].iloc[-1]
# Get latest Signal line value
    return 1 if macd_line > signal_line else -1 if
macd_line < signal_line else 0 # Crossover logic

def bollinger_bands_signal(indicators_df: Dict[str,
Union[pd.Series, pd.DataFrame]], current_price: float,
config: dict) -> int:
    """
    Generates a trading signal based on price interaction
with Bollinger Bands.
    Price touches/crosses below lower band: Long signal,
Price touches/crosses above upper band: Short signal.
    """
    bb = indicators_df["bollinger_bands"] # Get Bollinger
Bands DataFrame
    upper_band = bb["upper_band"].iloc[-1] # Get latest
upper band value
    lower_band = bb["lower_band"].iloc[-1] # Get latest
lower band value
    return base_signal(current_price, upper=upper_band,
lower=lower_band, inverse=True) # Use base_signal with
inverse logic

def bb_squeeze_signal(indicators_df: Dict[str,
Union[pd.Series, pd.DataFrame]], current_price: float,
config: dict) -> int:
    """
    Generates a trading signal based on Bollinger Bands
Squeeze.
    A squeeze, followed by price breaking out of bands, can
indicate a potential trade.
    Signal is generated when price moves outside of
squeezed Bollinger Bands.
    """
    bb_df = indicators_df["bollinger_bands_df"] # Get
Bollinger Bands DataFrame
    lookback = config["signal_config"]
["bb_squeeze_lookback"] # Lookback period for squeeze
calculation
    if bb_df.empty or len(bb_df) < lookback + 1:
        return 0 # Not enough data for squeeze calculation
    band_width = bb_df["upper_band"] - bb_df["lower_band"]
# Calculate bandwidth of Bollinger Bands
    percentile_threshold = config["signal_config"]
["bb_squeeze_percentile"] # Percentile for squeeze
threshold
    if band_width.iloc[-1] <
np.percentile(band_width.iloc[-lookback-1:-1],
percentile_threshold): # Check if current bandwidth is
within squeeze percentile
        upper_band = bb_df["upper_band"].iloc[-1] # Get
latest upper band value
        lower_band = bb_df["lower_band"].iloc[-1] # Get
latest lower band value
        return base_signal(current_price, upper=upper_band,
lower=lower_band, inverse=True) # Signal based on breakout
from squeeze (inverse logic)
    return 0 # No squeeze signal

def vwap_bounce_signal(indicators_df: Dict[str,
Union[pd.Series, pd.DataFrame]], current_price: float,
config: dict) -> int:
    """
    Generates a trading signal based on price bouncing off
Volume Weighted Average Price (VWAP).
    Price falls below VWAP then rises back above: Long
signal, Price rises above VWAP then falls back below: Short
signal.
    """
    vwap = indicators_df["vwap"] # Get VWAP series
    if len(vwap) < 2:
        return 0 # Not enough VWAP data
    prev_price = indicators_df["close"].iloc[-2] # Get
previous close price
    vwap_value = vwap.iloc[-1] # Get latest VWAP value
    return 1 if prev_price < vwap_value and current_price >
vwap_value else -1 if prev_price > vwap_value and
current_price < vwap_value else 0 # Bounce logic

def pivot_breakout_signal(indicators_df: Dict[str,
Union[pd.Series, pd.DataFrame]], current_price: float,
config: dict, support_resistance: dict) -> int:
    """
    Generates a trading signal based on price breaking out
of Pivot Points (Resistance 1 or Support 1).
    Price breaks above Resistance 1 (R1): Long signal,
Price breaks below Support 1 (S1): Short signal.
    """
    r1, s1 = support_resistance.get("r1"),
support_resistance.get("s1") # Get R1 and S1 pivot levels
    return 1 if r1 and current_price > r1 else -1 if s1 and
current_price < s1 else 0 # Breakout logic

def divergence_signal(indicators_df: Dict[str,
Union[pd.Series, pd.DataFrame]], current_price: float,
config: dict, df: pd.DataFrame) -> int:
    """
    Generates a trading signal based on divergence between
price action and MACD Histogram.
    Bullish Divergence: Price makes lower low, MACD
Histogram makes higher low: Long signal.
    Bearish Divergence: Price makes higher high, MACD
Histogram makes lower high: Short signal.
    """
    lookback = config["signal_config"]
["divergence_lookback"] # Lookback for divergence analysis
    if len(df) < lookback:
        return 0 # Not enough data for divergence analysis
    closes = df["close"].tail(lookback).values # Get recent
close prices
    macd_hist = indicators_df["macd"]
["histogram"].tail(lookback).values # Get recent MACD
Histogram values
    min_idx, max_idx = np.argmin(closes), np.argmax(closes)
# Indices of min/max price in lookback period

    # Bullish Divergence condition
    if min_idx != len(closes) - 1 and
np.min(closes[min_idx:]) < closes[min_idx] and
np.min(macd_hist[min_idx:]) > macd_hist[min_idx]:
        return 1 # Bullish divergence signal
    # Bearish Divergence condition
    if max_idx != len(closes) - 1 and
np.max(closes[max_idx:]) > closes[max_idx] and
np.max(macd_hist[max_idx:]) < macd_hist[max_idx]:
        return -1 # Bearish divergence signal
    return 0 # No divergence signal

# --- Signal Aggregation and Output ---

async def analyze_market_data_signals(indicators_df:
Dict[str, Union[pd.Series, pd.DataFrame]],
support_resistance: dict, orderbook: Optional[dict],
config: dict, df: pd.DataFrame, current_price: float) ->
Optional[dict]:
    """
    Analyzes market data by aggregating signals from
individual indicators based on configured weights.
    Calculates a normalized score, determines signal type
(Long/Short/Neutral), confidence, and stop/take profit
levels.
    Returns a signal dictionary if a trading signal is
generated, otherwise None.
    """
    signal_functions: Dict[str, SignalFunction] = { # Map
indicator names to their signal generation functions
        "ema_alignment": ema_alignment_signal, "momentum":
momentum_signal, "volume_confirmation":
volume_confirmation_signal,
        "stoch_rsi": stoch_rsi_signal, "rsi": rsi_signal,
"macd": macd_signal, "bollinger_bands":
bollinger_bands_signal,
        "bb_squeeze": bb_squeeze_signal, "vwap_bounce":
vwap_bounce_signal, "pivot_breakout": lambda i, c, cfg:
pivot_breakout_signal(i, c, cfg, support_resistance),
        "divergence": lambda i, c, cfg:
divergence_signal(i, c, cfg, df),
    }
    weights = config["weight_sets"]["low_volatility"] # Get
weights for the 'low_volatility' weight set
    total_score, rationale_parts = 0, [] # Initialize total
score and rationale list
    active_indicators = {ind: w for ind, w in
weights.items() if config["indicators"].get(ind,
{}).get("enabled", False)} # Filter enabled indicators and
their weights

    for indicator, weight in active_indicators.items(): #
Iterate through active indicators
        score = signal_functions[indicator](indicators_df,
current_price, config) # Get signal score from indicator
function
        weighted_score = score * weight # Apply weight to
the score
        total_score += weighted_score # Accumulate total
weighted score
        if score != 0:
            rationale_parts.append(f"{indicator}:
{weighted_score:+.2f}") # Add to rationale if indicator
contributed to signal

    sum_weights = sum(active_indicators.values()) # Sum of
weights of active indicators
    if not sum_weights:
        return None # No active indicators, return None (no
signal)

    normalized_score = total_score / sum_weights #
Normalize total score by sum of weights
    signal_threshold = config["signal_config"]
["signal_threshold"] # Get signal threshold from config

    signal_type = "Long" if normalized_score >
signal_threshold else "Short" if normalized_score <
-signal_threshold else None # Determine signal type based
on normalized score
    if not signal_type:
        return None # Normalized score within neutral zone,
return None (no signal)

    confidence = "High" if abs(normalized_score) > 0.7 else
"Medium" if abs(normalized_score) > 0.3 else "Low" #
Determine confidence level based on score magnitude
    atr_value = indicators_df["atr"].iloc[-1] # Get latest
ATR value for stop/take profit calculation
    stop_loss, take_profit =
calculate_stop_take_profit(signal_type, current_price,
atr_value, config["signal_config"]) # Calculate stop loss
and take profit

    return { # Construct signal dictionary
        "signal_type": signal_type, "entry_price":
current_price, "stop_loss": stop_loss, "take_profit":
take_profit,
        "confidence": confidence, "rationale": " |
".join(rationale_parts) or "No significant contributions",
# Rationale string or default message
        "normalized_score": normalized_score, "timestamp":
datetime.now(TIMEZONE).isoformat() # ISO formatted
timestamp in configured timezone
    }

def calculate_stop_take_profit(signal_type: str,
entry_price: float, atr_value: float, signal_config: dict)
-> Tuple[float, float]:
    """
    Calculates stop loss and take profit prices based on
signal type, entry price, ATR, and configured multipliers.
    Uses ATR multiplier for stop loss and risk-reward ratio
for take profit.
    """
    sl_multiplier =
signal_config["stop_loss_atr_multiplier"] # Get stop loss
ATR multiplier from config
    tp_ratio =
signal_config["take_profit_risk_reward_ratio"] # Get take
profit risk-reward ratio from config
    if signal_type == "Long": # Long signal calculation
        stop_loss = entry_price - atr_value * sl_multiplier
# Stop loss below entry price, based on ATR
        take_profit = entry_price + (entry_price -
stop_loss) * tp_ratio # Take profit based on risk-reward
ratio
    else: # Short signal calculation
        stop_loss = entry_price + atr_value * sl_multiplier
# Stop loss above entry price, based on ATR
        take_profit = entry_price - (stop_loss -
entry_price) * tp_ratio # Take profit based on risk-reward
ratio
    return stop_loss, take_profit # Return calculated stop
loss and take profit prices

async def format_signal_output(signal: Optional[dict],
indicators: dict, indicators_df: Dict[str, Union[pd.Series,
pd.DataFrame]], current_price: float, logger:
logging.Logger, last_output_time: float) -> float:
    """
    Formats and outputs trading signals and indicator
snapshots to console and files.
    Throttles output to prevent excessive logging/console
spam based on OUTPUT_THROTTLE_SECONDS.
    """
    symbol, interval = indicators.get('Symbol', 'N/A'),
indicators.get('Interval', 'N/A') # Extract symbol and
interval for output
    current_time = time.time()
    if current_time - last_output_time <
OUTPUT_THROTTLE_SECONDS and not signal:
        return last_output_time # Throttle output if no new
signal and within throttle time

    if signal: # Output signal if generated
        _output_signal_to_console(signal, symbol, interval)
# Output signal details to console
        await _save_signal_to_files(signal, symbol, logger)
# Save signal details to JSON and CSV files
        _log_alert(signal, symbol, interval, logger) # Log
a simple alert message to alert file
    else: # Output "no signal" message if no signal
generated
        console.print(Panel(f"[bold yellow]No trading
signal for {symbol} ({interval}m) at this time.[/bold
yellow]", title="[bold cyan]Signal Status[/bold cyan]",
border_style="yellow"))

    _output_indicators_to_console(indicators_df, symbol,
interval, current_price) # Output indicator snapshot to
console
    return current_time # Update last output time

def _output_signal_to_console(signal: dict, symbol: str,
interval: str):
    """Outputs the trading signal details to the console
using a Rich table for formatted display."""
    signal_table = Table(title=f"[bold magenta]
{signal['signal_type']} Signal for {symbol} ({interval}m)[/
bold magenta]", title_justify="center")
    signal_table.add_column("Entry", style="magenta",
justify="right")
    signal_table.add_column("Stop-Loss", style="red",
justify="right")
    signal_table.add_column("Take-Profit", style="green",
justify="right")
    signal_table.add_column("Confidence", style="cyan",
justify="center")
    signal_table.add_column("Score", style="yellow",
justify="right")
    signal_table.add_column("Rationale", style="green",
justify="left")
    signal_table.add_row(
        f"[bold]{signal['entry_price']:.4f}[/bold]",
f"[bold]{signal['stop_loss']:.4f}[/bold]", f"[bold]
{signal['take_profit']:.4f}[/bold]",
        f"[bold {signal['confidence'].lower()}]
{signal['confidence']}[/bold
{signal['confidence'].lower()}]", f"[bold]
{signal['normalized_score']:.2f}[/bold]",
signal["rationale"]
    )
    console.print(Panel.fit(signal_table, title="[bold
cyan]Trading Signal[/bold cyan]", border_style="cyan")) #
Print signal table to console

async def _save_signal_to_files(signal: dict, symbol: str,
logger: logging.Logger):
    """Saves the trading signal details to JSON and CSV
files as configured."""
    output_config = CONFIG["output"]
    if output_config["save_to_json"]: # Save signal to JSON
if enabled in config
        os.makedirs(output_config["json_output_dir"],
exist_ok=True) # Ensure output directory exists
        signal_filename =
os.path.join(output_config["json_output_dir"], f"{symbol}
_{signal['timestamp'].replace(':', '-')}.json") # Filename
based on symbol and timestamp
        try:
            async with aiofiles.open(signal_filename, "w")
as f:
                await f.write(json.dumps(signal, indent=4))
# Write signal data to JSON file
            logger.info(f"Signal JSON saved to
{signal_filename}")
        except Exception as e:
            logger.error(f"Error saving signal JSON: {e}")
# Log error if JSON save fails
    if output_config["save_to_csv"]: # Save signal to CSV
if enabled in config
        os.makedirs(output_config["csv_output_dir"],
exist_ok=True) # Ensure CSV output directory exists
        csv_filepath =
os.path.join(output_config["csv_output_dir"], f"{symbol}
_signals.csv") # CSV filename based on symbol
        signal_df = pd.DataFrame([signal]) # Convert signal
dict to DataFrame for CSV output
        try:
            async with aiofiles.open(csv_filepath, "a") as
f: # Open CSV file in append mode
                await f.write(signal_df.to_csv(index=False,
header=not os.path.exists(csv_filepath))) # Append signal
to CSV, write header only if file doesn't exist
            logger.info(f"Signal CSV saved to
{csv_filepath}")
        except Exception as e:
            logger.error(f"Error saving signal CSV: {e}") #
Log error if CSV save fails

def _log_alert(signal: dict, symbol: str, interval: str,
logger: logging.Logger):
    """Logs a concise alert message to a dedicated alert
log file for quick notifications."""
    output_config = CONFIG["output"]
    try:
        with open(os.path.join(LOG_DIRECTORY,
output_config["alert_file"]), "a") as f: # Open alert file
in append mode
            f.write(f"{signal['timestamp']} - {symbol}
({interval}m): {signal['signal_type']} - Score:
{signal['normalized_score']:.2f}\n") # Write alert message
    except Exception as e:
        logger.error(f"Error writing to alert file: {e}") #
Log error if writing to alert file fails

def _output_indicators_to_console(indicators_df: Dict[str,
Union[pd.Series, pd.DataFrame]], symbol: str, interval:
str, current_price: float):
    """Outputs a snapshot of technical indicator values to
the console using a Rich table."""
    ind_table = Table(title=f"[bold blue]Technical
Indicators for {symbol} ({interval}m)[/bold blue]",
title_justify="center")
    ind_table.add_column("Indicator", style="bold blue",
justify="left")
    ind_table.add_column("Value", justify="right")
    ind_table.add_column("Status", justify="center")

    for ind_name, ind_config in
CONFIG["indicators"].items(): # Iterate through configured
indicators
        if not isinstance(ind_config, dict):
            ind_config = {"enabled": ind_config, "display":
ind_config} # Handle simplified indicator config format
        if not ind_config["display"]:
            continue # Skip if indicator display is
disabled
        _add_indicator_row_to_table(ind_table, ind_name,
indicators_df, current_price) # Add indicator row to the
table

    console.print(Panel.fit(ind_table, title="[bold
blue]Indicator Snapshot[/bold blue]", border_style="blue"))
# Print indicator table to console

def _add_indicator_row_to_table(table: Table,
indicator_name: str, indicators_df: Dict[str,
Union[pd.Series, pd.DataFrame]], current_price: float):
    """Adds a row to the indicator table for a specific
indicator, formatting value and status."""
    if indicator_name == "macd": # MACD indicator row
        macd_df = indicators_df.get("macd",
pd.DataFrame({"macd": [float('nan')], "signal":
[float('nan')]})) # Get MACD data, default to NaN DataFrame
if missing
        add_indicator_row(table, "MACD",
f"{macd_df['macd'].iloc[-1]:.4f} /
{macd_df['signal'].iloc[-1]:.4f}",
macd_status_logic(macd_df)) # Add MACD row with status
logic
    elif indicator_name == "bollinger_bands": # Bollinger
Bands indicator row
        bb_df = indicators_df.get("bollinger_bands",
pd.DataFrame({"lower_band": [float('nan')], "upper_band":
[float('nan')]})) # Get BBands data, default to NaN
DataFrame
        add_indicator_row(table, "BBands",
f"{bb_df['lower_band'].iloc[-1]:.4f} -
{bb_df['upper_band'].iloc[-1]:.4f}",
bollinger_bands_status_logic(bb_df, current_price)) # Add
BBands row with status logic
    elif indicator_name == "rsi": # RSI indicator row
        add_indicator_row(table, "RSI",
indicators_df.get("rsi",
pd.Series([float('nan')])).iloc[-1], rsi_thresholds=(30,
70)) # Add RSI row with threshold-based status
    elif indicator_name == "stoch_rsi": # Stochastic RSI
indicator row
        add_indicator_row(table, "Stoch RSI (K)",
indicators_df.get("stoch_rsi_k",
pd.Series([float('nan')])).iloc[-1],
stoch_rsi_thresholds=(0.2, 0.8)) # Add Stoch RSI row with
threshold-based status
    elif indicator_name in ["ema", "vwap"]: # EMA/VWAP
indicator rows
        add_indicator_row(table, indicator_name.upper(),
indicators_df.get(indicator_name,
pd.Series([float('nan')])).iloc[-1],
current_price=current_price) # Add EMA/VWAP row with price-
based status
    elif indicator_name == "volume_confirmation": # Volume
Confirmation (OBV) indicator row
        add_indicator_row(table, "OBV",
indicators_df.get("obv",
pd.Series([float('nan')])).iloc[-1], "neutral") # Add OBV
row with neutral status
    elif indicator_name == "momentum": # Momentum indicator
row
        mom = indicators_df.get("mom", pd.Series([{"trend":
"Neutral", "strength": 0.0}])).iloc[-1] # Get momentum
data, default to neutral if missing
        add_indicator_row(table, "Momentum",
f"{mom['strength']:.4f}", f"[{mom['trend'].lower()}]
{mom['trend']}[/{mom['trend'].lower()}]") # Add Momentum
row with trend-based status

def add_indicator_row(table: Table, indicator_name: str,
value: Union[str, float, dict], status: Union[str, tuple] =
"neutral", current_price: Optional[float] = None,
rsi_thresholds: Optional[tuple] = None,
stoch_rsi_thresholds: Optional[tuple] = None):
    """
    Adds a row to the indicator table, determining status
string based on indicator type and thresholds.
    Formats the value for display and applies appropriate
Rich text styling for status.
    """
    status_str =
_determine_indicator_status(indicator_name, value, status,
current_price, rsi_thresholds, stoch_rsi_thresholds) #
Determine status string based on logic
    display_value = f"{value:.4f}" if isinstance(value,
(int, float)) and not pd.isna(value) else str(value) #
Format value for display
    table.add_row(indicator_name, display_value,
f"[{status_str}]{status_str.capitalize()}[/{status_str}]")
# Add row to table with formatted value and status

def _determine_indicator_status(indicator_name: str, value:
Union[str, float, dict], status: Union[str, tuple] =
"neutral", current_price: Optional[float] = None,
rsi_thresholds: Optional[tuple] = None,
stoch_rsi_thresholds: Optional[tuple] = None) -> str:
    """
    Determines the status string for an indicator based on
its name, value, and optional thresholds.
    Used to apply consistent status logic across different
indicators for console output.
    """
    if isinstance(status, tuple): # Predefined status (from
status logic functions)
        return status[0].split('[')[-1].split(']')[0] #
Extract status text from Rich formatted string

    if rsi_thresholds: # RSI status logic based on
thresholds
        return "bullish" if value < rsi_thresholds[0] else
"bearish" if value > rsi_thresholds[1] else "neutral"
    if stoch_rsi_thresholds: # Stoch RSI status logic based
on thresholds
        return "bullish" if value < stoch_rsi_thresholds[0]
else "bearish" if value > stoch_rsi_thresholds[1] else
"neutral"
    if current_price is not None and isinstance(value,
(int, float)) and not pd.isna(value): # Price-based status
logic (e.g., for EMA, VWAP)
        return "bullish" if current_price > value else
"bearish" if current_price < value else "neutral"
    return status if isinstance(status, str) else "neutral"
# Default to neutral status if no specific logic applies

def macd_status_logic(macd_df: pd.DataFrame) -> Tuple[str]:
    """Determines MACD status for console output based on
MACD and Signal line positions."""
    macd_line, signal_line = macd_df["macd"].iloc[-1],
macd_df["signal"].iloc[-1] # Get latest MACD and Signal
line values
    if pd.isna(macd_line) or pd.isna(signal_line):
        return ("[yellow]Neutral[/yellow]",) # Neutral if
MACD or Signal line is NaN
    status = "bullish" if macd_line > signal_line else
"bearish" if macd_line < signal_line else "neutral" #
Determine status based on crossover
    return (f"[{status}]{status.capitalize()}[/]",) #
Return Rich formatted status string

def bollinger_bands_status_logic(bb_df: pd.DataFrame,
current_price: float) -> Tuple[str]:
    """Determines Bollinger Bands status for console output
based on price position relative to bands."""
    bb_upper, bb_lower = bb_df["upper_band"].iloc[-1],
bb_df["lower_band"].iloc[-1] # Get latest upper and lower
band values
    if pd.isna(bb_upper) or pd.isna(bb_lower) or
pd.isna(current_price):
        return ("[yellow]Neutral[/yellow]",) # Neutral if
bands or price is NaN
    status = "bullish" if current_price < bb_lower else
"bearish" if current_price > bb_upper else "neutral" #
Determine status based on price position within bands
    return (f"[{status}]{status.capitalize()}[/]",) #
Return Rich formatted status string

def vwap_status_logic(vwap_series: pd.Series,
current_price: float) -> Tuple[str]:
    """Determines VWAP status for console output based on
price position relative to VWAP."""
    vwap = vwap_series.iloc[-1] # Get latest VWAP value
    if pd.isna(vwap) or pd.isna(current_price):
        return ("[yellow]Neutral[/yellow]",) # Neutral if
VWAP or price is NaN
    status = "bullish" if current_price > vwap else
"bearish" if current_price < vwap else "neutral" #
Determine status based on price position relative to VWAP
    return (f"[{status}]{status.capitalize()}[/]",) #
Return Rich formatted status string

# --- Trading Analyzer Class ---

class TradingAnalyzer:
    """
    Analyzes trading data for a specific symbol and
interval, calculating technical indicators,
    generating trading signals, and managing output.
    """
    def __init__(self, symbol: str, interval: str, config:
dict, logger: logging.Logger):
        """Initializes the TradingAnalyzer with symbol,
interval, config, and logger."""
        self.symbol = symbol
        self.interval = interval
        self.config = config
        self.logger = logger
        self.df = pd.DataFrame() # DataFrame to store Kline
data
        self.indicator_values: Dict[str, Union[pd.Series,
pd.DataFrame]] = {} # Dictionary to store calculated
indicator values
        self.last_kline_time = None # Timestamp of the last
processed kline
        self.last_output_time = 0.0 # Timestamp of the last
output (signal or indicator snapshot)
        self.last_signal = None # Last generated trading
signal

    def update_data(self, new_df: pd.DataFrame):
        """
        Updates the historical Kline data with new incoming
data.
        Concatenates new data, removes duplicates, and
keeps the most recent 200 data points.
        Recalculates all technical indicators after
updating data.
        """
        if self.df.empty or new_df["start_time"].iloc[-1] >
self.last_kline_time: # Check if new data is actually newer
            self.df = pd.concat([self.df,
new_df]).drop_duplicates(subset="start_time").tail(200) #
Concat, deduplicate, keep last 200
            self.last_kline_time =
self.df["start_time"].iloc[-1] # Update timestamp of last
kline
            self.calculate_indicators() # Recalculate
indicators with updated data

    def calculate_indicators(self):
        """
        Calculates all configured technical indicators
using the current Kline data DataFrame.
        Stores the calculated indicator values in the
`indicator_values` dictionary.
        """
        self.indicator_values = { # Calculate and store
each indicator
            "ema":
self._calculate_ema(self.config["signal_config"]
["ema_period"]),
            "mom": self._calculate_momentum(),
            "obv": self._calculate_obv(),
            "rsi":
self._calculate_rsi(self.config["rsi_period"]),
            "stoch_rsi_k": self._calculate_stoch_rsi()
["k"],
            "macd": self._calculate_macd(),
            "bollinger_bands":
self._calculate_bollinger_bands(),
            "vwap": self._calculate_vwap(),
            "atr": self._calculate_atr(),
            "close": self.df["close"] # Store close prices
for easy access
        }
        self.indicator_values["bollinger_bands_df"] =
self._calculate_bollinger_bands() # Store BBands DataFrame
separately for BB Squeeze calculation

    def _calculate_ema(self, window: int) -> pd.Series:
        """Calculates Exponential Moving Average (EMA) for
the closing prices."""
        return self.df["close"].ewm(span=window,
adjust=False).mean()

    def _calculate_momentum(self) -> pd.Series:
        """Calculates momentum indicator with short and
long moving averages and ATR normalization."""
        if len(self.df) < self.config["momentum_ma_long"]:
# Ensure enough data for long MA calculation
            return pd.Series([{"trend": "Neutral",
"strength": 0.0}] * len(self.df), index=self.df.index) #
Return neutral momentum if insufficient data

        momentum =
self.df["close"].diff(self.config["momentum_period"]) # Raw
momentum as price difference
        short_ma =
momentum.rolling(window=self.config["momentum_ma_short"],
min_periods=1).mean() # Short moving average of momentum
        long_ma =
momentum.rolling(window=self.config["momentum_ma_long"],
min_periods=1).mean() # Long moving average of momentum
        atr = self._calculate_atr() # Calculate ATR for
normalization
        trend = np.where(short_ma > long_ma, "Uptrend",
np.where(short_ma < long_ma, "Downtrend", "Neutral")) #
Determine trend based on MA crossover
        strength = np.abs(short_ma - long_ma) /
atr.replace(0, np.nan).fillna(0) # Normalize momentum
strength by ATR

        return pd.Series([{"trend": t, "strength":
float(s)} for t, s in zip(trend, strength)],
index=self.df.index) # Return Series of momentum trend and
strength

    def _calculate_obv(self) -> pd.Series:
        """Calculates On Balance Volume (OBV) indicator."""
        direction = np.where(self.df["close"] >
self.df["close"].shift(1), 1, np.where(self.df["close"] <
self.df["close"].shift(1), -1, 0)) # Determine volume
direction
        return (direction * self.df["volume"]).cumsum() #
Calculate cumulative OBV

    def _calculate_rsi(self, window: int) -> pd.Series:
        """Calculates Relative Strength Index (RSI) for the
closing prices."""
        delta = self.df["close"].diff() # Price differences
        gain = delta.where(delta > 0,
0).rolling(window=window, min_periods=1).mean() # Average
gains
        loss = (-delta.where(delta < 0,
0)).rolling(window=window, min_periods=1).mean() # Average
losses
        rs = gain / loss.replace(0, np.nan) # Relative
strength
        rsi = 100 - (100 / (1 + rs)) # RSI formula
        return rsi.fillna(50.0) # Fill NaN values with 50
(neutral RSI)

    def _calculate_stoch_rsi(self) -> pd.DataFrame:
        """Calculates Stochastic RSI indicator, returning K
and D lines."""
        rsi =
self._calculate_rsi(self.config["signal_config"]
["stoch_rsi_period"]) # Calculate RSI first
        period = self.config["signal_config"]
["stoch_rsi_period"]
        stoch = (rsi - rsi.rolling(window=period,
min_periods=1).min()) / (rsi.rolling(window=period,
min_periods=1).max() - rsi.rolling(window=period,
min_periods=1).min() + 1e-10) # Stochastic RSI formula
        k = stoch.rolling(self.config["signal_config"]
["stoch_rsi_k"], min_periods=1).mean() # K line: moving
average of Stochastic RSI
        d = k.rolling(self.config["signal_config"]
["stoch_rsi_d"], min_periods=1).mean() # D line: moving
average of K line
        return pd.DataFrame({"stoch_rsi": stoch, "k": k,
"d": d}, index=self.df.index).fillna(0.5) # Return
DataFrame with Stoch RSI, K and D lines, fill NaN with 0.5

    def _calculate_macd(self) -> pd.DataFrame:
        """Calculates Moving Average Convergence Divergence
(MACD) indicator."""
        macd =
self.df["close"].ewm(span=self.config["signal_config"]
["macd_fast"], adjust=False).mean() -
self.df["close"].ewm(span=self.config["signal_config"]
["macd_slow"], adjust=False).mean() # MACD line: fast EMA -
slow EMA
        signal = macd.ewm(span=self.config["signal_config"]
["macd_signal"], adjust=False).mean() # Signal line: EMA of
MACD line
        histogram = macd - signal # MACD Histogram: MACD
line - Signal line
        return pd.DataFrame({"macd": macd, "signal":
signal, "histogram": histogram}, index=self.df.index) #
Return DataFrame with MACD, Signal, and Histogram

    def _calculate_bollinger_bands(self) -> pd.DataFrame:
        """Calculates Bollinger Bands indicator (upper,
middle, lower bands)."""
        sma =
self.df["close"].rolling(window=self.config["bollinger_band
s_period"], min_periods=1).mean() # Middle band: Simple
Moving Average
        std =
self.df["close"].rolling(window=self.config["bollinger_band
s_period"], min_periods=1).std().fillna(0) # Standard
deviation
        std_dev = self.config["bollinger_bands_std_dev"] #
Standard deviation multiplier from config
        upper_band = sma + (std * std_dev) # Upper
Bollinger Band
        lower_band = sma - (std * std_dev) # Lower
Bollinger Band
        return pd.DataFrame({"upper_band": upper_band,
"middle_band": sma, "lower_band": lower_band},
index=self.df.index) # Return DataFrame with upper, middle,
lower bands

    def _calculate_vwap(self) -> pd.Series:
        """Calculates Volume Weighted Average Price
(VWAP)."""
        typical_price = (self.df["high"] + self.df["low"] +
self.df["close"]) / 3 # Typical price for each period
        return (typical_price *
self.df["volume"]).cumsum() / self.df["volume"].cumsum() #
VWAP formula: cumulative(typical price * volume) /
cumulative(volume)

    def _calculate_atr(self) -> pd.Series:
        """Calculates Average True Range (ATR)
indicator."""
        tr = pd.concat([self.df["high"] - self.df["low"],
(self.df["high"] - self.df["close"].shift()).abs(),
(self.df["low"] - self.df["close"].shift()).abs()],
axis=1).max(axis=1) # True Range calculation
        return tr.rolling(window=self.config["atr_period"],
min_periods=1).mean() # ATR as moving average of True Range

    def calculate_pivot_points(self) -> dict:
        """Calculates pivot points (Pivot, R1, S1, R2, S2)
based on the latest Kline data."""
        high, low, close = self.df["high"].max(),
self.df["low"].min(), self.df["close"].iloc[-1] # Get max
high, min low, latest close from DataFrame
        pivot = (high + low + close) / 3 # Pivot Point
calculation
        return {"pivot": pivot, "r1": 2 * pivot - low,
"s1": 2 * pivot - high, "r2": pivot + (high - low), "s2":
pivot - (high - low)} # Calculate R1, S1, R2, S2

    async def analyze_and_output(self, current_price:
float, logger: logging.Logger):
        """
        Analyzes market data, generates trading signals,
and outputs the signals and indicator snapshots.
        Called on every price update from WebSocket.
        """
        if self.df.empty or len(self.df) < 2: # Ensure
sufficient data for analysis
            self.logger.warning("Insufficient data for
analysis.")
            return # Not enough data, skip analysis

        support_resistance = self.calculate_pivot_points()
# Calculate pivot points for analysis
        orderbook =
data_cache.get(f"orderbook_{self.symbol}") # Get cached
orderbook data
        signal = await
analyze_market_data_signals(self.indicator_values,
support_resistance, orderbook, self.config, self.df,
current_price) # Generate trading signal

        indicators = {"Symbol": self.symbol, "Interval":
self.interval, "Current Price": current_price} # Prepare
indicator info for output
        if signal != self.last_signal or time.time() -
self.last_output_time >= OUTPUT_THROTTLE_SECONDS: # Check
if signal changed or throttle time exceeded
            self.last_output_time = await
format_signal_output(signal, indicators,
self.indicator_values, current_price, logger,
self.last_output_time) # Format and output signal
            self.last_signal = signal # Update last signal


# --- Main Function ---

async def analyze_symbol(symbol: str, interval: str,
logger: logging.Logger):
    """
    Sets up and starts the TradingAnalyzer and WebSocket
stream for a given trading symbol and interval.
    Handles symbol-specific analysis lifecycle, including
error handling for invalid symbols.
    """
    analyzer = TradingAnalyzer(symbol, interval, CONFIG,
logger) # Initialize TradingAnalyzer for the symbol
    try:
        await websocket_stream(symbol, interval, analyzer,
logger) # Start WebSocket streaming and analysis
    except ValueError as e: # Catch ValueError raised for
invalid symbols during subscription
        logger.error(f"Stopping analysis for {symbol} due
to: {e}") # Log error and stop analysis for this symbol

async def main():
    """
    Main entry point for the WebWhale Scanner bot.
    Initializes logging, loads configuration, fetches valid
symbols, prompts user for trading pairs,
    and starts analysis tasks for each valid symbol.
    """
    console.print(Panel("[bold cyan]Initiating Real-Time
Trading Bot Sequence...[/bold cyan]", title="[bold
magenta]WebWhale Scanner v14[/bold magenta]"))
    async with aiohttp.ClientSession() as session: # Create
aiohttp client session for API requests and WebSocket
connections
        valid_symbols = await fetch_valid_symbols(session,
logging.getLogger("main")) # Fetch valid trading symbols
from Bybit API
        if not valid_symbols:
            console.print("[bold red]Failed to fetch valid
symbols. Exiting.[/bold red]")
            return # Exit if valid symbols cannot be
fetched

        symbols_input = console.input("[cyan]Enter trading
pairs (e.g., BTCUSDT, ETHUSDT) separated by commas: [/
cyan]").strip().upper() # Prompt user for trading pairs
        symbols = [s.strip() for s in
symbols_input.split(",") if s.strip()] # Split input string
into list of symbols
        valid_input_symbols = [s for s in symbols if s in
valid_symbols] # Filter out invalid symbols from user input
        invalid_symbols = set(symbols) -
set(valid_input_symbols) # Identify invalid symbols

        if invalid_symbols:
            console.print(f"[bold red]Invalid symbols
ignored: {', '.join(invalid_symbols)}. Valid symbols: {',
'.join(valid_input_symbols)}[/bold red]") # Inform user
about ignored invalid symbols
        if not valid_input_symbols:
            console.print("[bold red]No valid symbols
entered. Exiting.[/bold red]")
            return # Exit if no valid symbols are entered

        interval = CONFIG.get("interval", "15") # Get
trading interval from config, default to 15m
        if interval not in VALID_INTERVALS:
            console.print(f"[bold red]Invalid interval:
{interval}. Using '15'.[/bold red]")
            interval = "15" # Fallback to default interval
if config interval is invalid

        tasks = []
        for symbol in valid_input_symbols: # Create
analysis tasks for each valid symbol
            logger = setup_logger(symbol) # Set up logger
for each symbol
            logger.info(f"Starting real-time analysis for
{symbol} on {interval}m interval") # Log start of analysis
for symbol
            console.print(f"[cyan]Streaming {symbol} on
{interval}m interval. Press Ctrl+C to stop.[/cyan]") #
Inform user about streaming symbol
            tasks.append(analyze_symbol(symbol, interval,
logger)) # Append analysis task to task list

        try:
            await asyncio.gather(*tasks,
return_exceptions=False) # Run all analysis tasks
concurrently
        except KeyboardInterrupt:
            console.print("[bold yellow]Bot interrupted.
Shutting down.[/bold yellow]") # Handle keyboard interrupt
(Ctrl+C) for graceful shutdown
            for symbol in valid_input_symbols:
                logging.getLogger(symbol).info("Bot stopped
manually.") # Log manual stop for each symbol

if __name__ == "__main__":
    asyncio.run(main()) # Run
