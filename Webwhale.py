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
from typing import Dict, Tuple, Optional, Union, Callable, List
from zoneinfo import ZoneInfo
from decimal import Decimal, getcontext
import aiofiles
from logging.handlers import RotatingFileHandler
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.logging import RichHandler
from rich.theme import Theme
import argparse
import jsonschema  # For JSON schema validation
from functools import lru_cache  # For caching calculated indicators

# --- Enchanted Configuration and Mystical Setup ---

getcontext().prec = 10  # Precision for decimal numbers
load_dotenv()  # Load environment variables from .env file

API_KEY = os.getenv("BYBIT_API_KEY")
API_SECRET = os.getenv("BYBIT_API_SECRET")
if not API_KEY or not API_SECRET:
    raise ValueError("BYBIT_API_KEY and BYBIT_API_SECRET must be set in .env - Secrets are the soul of the machine.")

WEBSOCKET_URL = "wss://stream.bybit.com/v5/public/linear"  # Bybit WebSocket endpoint
BASE_URL = "https://api.bybit.com"  # Bybit REST API base URL
CONFIG_FILE = "config.json"  # Configuration file path
LOG_DIRECTORY = "bot_logs"  # Directory for log files
TIMEZONE = ZoneInfo("America/Chicago")  # Timezone for timestamps
RECONNECT_DELAY = 5  # Delay before reconnecting to WebSocket (seconds)
CACHE_TTL_SECONDS = 60  # Time-to-live for cached data (seconds)
VALID_INTERVALS = ["1", "3", "5", "15", "30", "60", "120", "240", "D", "W", "M"]  # Valid Kline intervals
OUTPUT_THROTTLE_SECONDS = 60  # Minimum time between console outputs (seconds)
MAX_API_RETRIES = 3  # Maximum retries for API requests
DEFAULT_WEIGHT_SET = "low_volatility" # Default weight set name
DEFAULT_INTERVAL = "15" # Default interval if not in config or invalid
DEFAULT_ORDERBOOK_LIMIT = 50 # Default orderbook limit

# Rich Console Configuration - Infusing the terminal with vibrant hues
console_theme = Theme({
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
})
console = Console(theme=console_theme)

os.makedirs(LOG_DIRECTORY, exist_ok=True)  # Ensure log directory exists

# --- Mystical Utilities and Helper Functions ---

class SensitiveFormatter(logging.Formatter):
    """Formatter to mask sensitive information (API keys, secrets) in logs - Secrets veiled in shadows."""
    def format(self, record):
        msg = super().format(record)
        return msg.replace(API_KEY, "***").replace(API_SECRET, "***")

def load_config(filepath: str, check_only: bool = False) -> dict:
    """
    Loads configuration from JSON file, creating a default if not found.
    Validates configuration against a schema and corrects common issues - Ensuring the runes are correctly inscribed.
    """
    default_config = {
        "interval": DEFAULT_INTERVAL, # Default interval set, but will be prompted
        "analysis_interval": 30,
        "retry_delay": 5,
        "momentum_period": 10,
        "momentum_ma_short": 12,
        "momentum_ma_long": 26,
        "volume_ma_period": 20,
        "atr_period": 14,
        "trend_strength_threshold": 0.4,
        "indicators": {
            "ema_alignment": {"enabled": True, "display": True, "period": 20},
            "momentum": {"enabled": True, "display": True, "period": 10, "ma_short": 12, "ma_long": 26},
            "volume_confirmation": {"enabled": True, "display": True, "period": 20},
            "divergence": {"enabled": True, "display": False, "lookback": 20},
            "stoch_rsi": {"enabled": True, "display": True, "period": 14, "k_period": 3, "d_period": 3, "overbought": 0.8, "oversold": 0.2}, # Added thresholds
            "rsi": {"enabled": True, "display": True, "period": 14, "overbought": 70, "oversold": 30}, # Added thresholds
            "macd": {"enabled": True, "display": True, "fast_period": 12, "slow_period": 26, "signal_period": 9},
            "bollinger_bands": {"enabled": True, "display": True, "period": 20, "std_dev": 2},
            "bb_squeeze": {"enabled": True, "display": False, "lookback": 20, "percentile": 10},
            "vwap_bounce": {"enabled": True, "display": True},
            "pivot_breakout": {"enabled": True, "display": False},
        },
        "weight_sets": {
            DEFAULT_WEIGHT_SET: {
                "ema_alignment": 0.4, "momentum": 0.3, "volume_confirmation": 0.2, "divergence": 0.1,
                "stoch_rsi": 0.7, "rsi": 0.6, "macd": 0.5, "bollinger_bands": 0.4, "bb_squeeze": 0.3,
                "vwap_bounce": 0.3, "pivot_breakout": 0.3,
            }
        },
        "orderbook_limit": DEFAULT_ORDERBOOK_LIMIT,
        "signal_config": {
            "signal_threshold": 0.3,
            "stop_loss_atr_multiplier": 2,
            "take_profit_risk_reward_ratio": 2,
        },
        "output": {
            "save_to_json": True,
            "json_output_dir": "output",
            "alert_file": "signals.log",
            "save_to_csv": False,
            "csv_output_dir": "csv_output"
        }
    }

    config_schema = { # Simplified schema for brevity, can be extended further
        "type": "object",
        "properties": {
            "interval": {"type": "string", "enum": VALID_INTERVALS},
            "analysis_interval": {"type": "integer"},
            "retry_delay": {"type": "integer"},
            "momentum_period": {"type": "integer"},
            "momentum_ma_short": {"type": "integer"},
            "momentum_ma_long": {"type": "integer"},
            "volume_ma_period": {"type": "integer"},
            "atr_period": {"type": "integer"},
            "trend_strength_threshold": {"type": "number"},
            "indicators": {"type": "object"}, # Further validation can be added for indicator configs
            "weight_sets": {"type": "object"},
            "orderbook_limit": {"type": "integer", "minimum": 1},
            "signal_config": {"type": "object"}, # Further validation can be added for signal_config
            "output": {"type": "object"}, # Further validation can be added for output config
        },
        "required": ["interval", "indicators", "weight_sets", "orderbook_limit", "signal_config", "output"],
    }


    if not os.path.exists(filepath):
        if not check_only:
            with open(filepath, 'w') as f:
                json.dump(default_config, f, indent=4)
            console.print(Panel(f"[bold yellow]Created new config file at '{filepath}'.[/bold yellow]", title="[bold cyan]Configuration Spellbook[/bold cyan]"))
        else:
            console.print(Panel(f"[bold yellow]Config file '{filepath}' not found. Default configuration would be created.[/bold yellow]", title="[bold cyan]Configuration Check[/bold cyan]"))
        return default_config
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            config = json.load(f)

        try:
            jsonschema.validate(instance=config, schema=config_schema)
        except jsonschema.exceptions.ValidationError as e:
            console.print(Panel(f"[bold red]Config validation error in '{filepath}': {e}. Using defaults.[/bold red]", title="[bold cyan]Configuration Error[/bold cyan]"))
            return default_config

        _validate_config(config, default_config) # Still run _validate_config for logical checks after schema validation

        if not check_only:
            with open(filepath, 'w') as f:
                json.dump(config, f, indent=4)
        return config
    except json.JSONDecodeError as e:
        console.print(Panel(f"[bold red]JSON config error in '{filepath}': {e}. Using defaults.[/bold red]", title="[bold cyan]Configuration Error[/bold cyan]"))
        return default_config
    except Exception as e:
        console.print(Panel(f"[bold red]Config file error: {e}. Using defaults.[/bold red]", title="[bold cyan]Configuration Error[/bold cyan]"))
        return default_config

def _validate_config(config: dict, default_config: dict):
    """Validates configuration parameters, ensuring essential incantations are present and logically sound."""
    if "interval" in config and config["interval"] not in VALID_INTERVALS: # Check if interval is in config before validating
        console.print(f"[bold yellow]Invalid interval '{config['interval']}'. Using default '{DEFAULT_INTERVAL}'.[/bold yellow]")
        config["interval"] = DEFAULT_INTERVAL
    if not isinstance(config["orderbook_limit"], int) or config["orderbook_limit"] <= 0:
        console.print(f"[bold yellow]Invalid orderbook_limit '{config['orderbook_limit']}'. Using default {DEFAULT_ORDERBOOK_LIMIT}.[/bold yellow]")
        config["orderbook_limit"] = DEFAULT_ORDERBOOK_LIMIT
    for indicator_name, indicator_config in config["indicators"].items():
        if not isinstance(indicator_config, dict): # For backward compatibility with old bool config
            config["indicators"][indicator_name] = {"enabled": indicator_config, "display": indicator_config}
        if "enabled" not in indicator_config or "display" not in indicator_config:
            console.print(f"[bold yellow]Indicator '{indicator_name}' config missing 'enabled' or 'display'. Using defaults.[/bold yellow]")
            config["indicators"][indicator_name] = {"enabled": indicator_config.get("enabled", False), "display": indicator_config.get("display", False), **indicator_config}


CONFIG = load_config(CONFIG_FILE)

def setup_logger(symbol: str) -> logging.Logger:
    """Sets up a logger for each symbol, channeling messages to files and console - Binding spirits of information."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = os.path.join(LOG_DIRECTORY, f"{symbol}_{timestamp}.log")
    logger = logging.getLogger(symbol)
    logger.setLevel(logging.DEBUG)

    file_handler = RotatingFileHandler(log_filename, maxBytes=10 * 1024 * 1024, backupCount=5)
    file_handler.setFormatter(SensitiveFormatter("%(asctime)s - %(levelname)s - %(message)s"))

    console_handler = RichHandler(console=console, rich_tracebacks=True)
    console_handler.setFormatter(SensitiveFormatter("%(message)s"))

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger

# --- Data Cache - Memory of the Machine ---

class DataCache:
    """Caches data with TTL, like a fleeting memory - Preserving echoes of market whispers."""
    def __init__(self, ttl: int = CACHE_TTL_SECONDS):
        self.cache: Dict[str, Tuple[Union[Decimal, pd.DataFrame, dict], float]] = {}
        self.ttl = ttl

    def get(self, key: str) -> Optional[Union[Decimal, pd.DataFrame, dict]]:
        """Retrieves value from cache if valid, else returns None - Recalling fragments from the digital aether."""
        cached_data = self.cache.get(key)
        if cached_data:
            value, timestamp = cached_data
            if time.time() - timestamp < self.ttl:
                return value
            else:
                del self.cache[key]
        return None

    def set(self, key: str, value: Union[Decimal, pd.DataFrame, dict]):
        """Sets a value in the cache with current timestamp - Inscribing knowledge into the cache."""
        self.cache[key] = (value, time.time())

data_cache = DataCache()

# --- REST API Functions - Whispers from the Oracle ---

async def fetch_valid_symbols(session: aiohttp.ClientSession, logger: logging.Logger) -> List[str]:
    """Fetches valid trading symbols from Bybit API - Seeking the lexicon of trading pairs."""
    url = f"{BASE_URL}/v5/market/instruments-info"
    params = {"category": "linear"}
    return await _bybit_api_request(session, logger, url, params, method="GET", endpoint_description="symbols")

async def fetch_klines(symbol: str, interval: str, limit: int, session: aiohttp.ClientSession, logger: logging.Logger) -> pd.DataFrame:
    """Fetches kline data from Bybit API and returns as DataFrame - Conjuring historical candles from the ether."""
    url = f"{BASE_URL}/v5/market/kline"
    params = {"symbol": symbol, "interval": interval, "limit": limit, "category": "linear"}
    raw_klines = await _bybit_api_request(session, logger, url, params, method="GET", endpoint_description="klines")
    if raw_klines:
        df = pd.DataFrame(raw_klines, columns=["start_time", "open", "high", "low", "close", "volume", "turnover"])
        df["start_time"] = pd.to_datetime(df["start_time"].astype(int), unit="ms")
        df[["open", "high", "low", "close", "volume"]] = df[["open", "high", "low", "close", "volume"]].apply(pd.to_numeric()) # Optimized apply
        df = df.sort_values("start_time", ignore_index=True) # Optimized sort and reset index
        return df
    return pd.DataFrame()

async def _bybit_api_request(session: aiohttp.ClientSession, logger: logging.Logger, url: str, params: dict, method: str = "GET", endpoint_description: str = "API") -> Optional[Union[list, dict]]:
    """Handles Bybit API requests with authentication, retries, and error handling - Invoking the API with signed requests."""
    timestamp = str(int(time.time() * 1000))
    sorted_params = "&".join([f"{k}={v}" for k, v in sorted({**params, 'timestamp': timestamp}.items())]) # Sort params once
    signature = hmac.new(API_SECRET.encode(), sorted_params.encode(), hashlib.sha256).hexdigest()
    headers = {"X-BAPI-API-KEY": API_KEY, "X-BAPI-TIMESTAMP": timestamp, "X-BAPI-SIGN": signature}

    for attempt in range(MAX_API_RETRIES):
        try:
            async with session.request(method, url, headers=headers, params=params, timeout=10) as response:
                response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
                data = await response.json()
                if data.get("retCode") == 0:
                    return data.get("result", {}).get("list") if endpoint_description == "klines" else data.get("result", {})
                else:
                    logger.error(f"{endpoint_description} fetch error (attempt {attempt + 1}/{MAX_API_RETRIES}): {data.get('retMsg')}, code: {data.get('retCode')}")
        except aiohttp.ClientError as e: # Catch specific aiohttp client errors
            logger.warning(f"{endpoint_description} fetch failed (attempt {attempt + 1}/{MAX_API_RETRIES}): Client error - {e}")
        except asyncio.TimeoutError:
            logger.warning(f"{endpoint_description} fetch failed (attempt {attempt + 1}/{MAX_API_RETRIES}): Timeout")
        except Exception as e: # Catch any other exceptions
            logger.error(f"{endpoint_description} fetch failed (attempt {attempt + 1}/{MAX_API_RETRIES}): Unexpected error - {e}")

        await asyncio.sleep(RECONNECT_DELAY)

    logger.error(f"Failed to fetch {endpoint_description} after {MAX_API_RETRIES} attempts - API invocation failed after repeated tries.")
    return None

# --- WebSocket Streaming - Real-time Divination ---

async def websocket_stream(symbol: str, interval: str, analyzer: 'TradingAnalyzer', logger: logging.Logger):
    """Manages WebSocket connection, subscriptions, and message processing - Tapping into the live market stream."""
    session = aiohttp.ClientSession() # Session created here to persist across reconnections
    initial_df = await fetch_klines(symbol, interval, 200, session, logger) # Session passed here
    if initial_df.empty:
        logger.error(f"Cannot proceed with {symbol} on {interval}m interval: no initial data received.")
        await session.close() # Close session if initial data fetch fails
        return
    analyzer.update_data(initial_df)
    logger.info(f"Loaded initial {len(initial_df)} klines for {symbol} on {interval}m interval.")

    while True:
        try:
            async with session.ws_connect(WEBSOCKET_URL, heartbeat=30.0, timeout=30) as ws: # Session used here
                logger.info(f"Connected to WebSocket for {symbol} on {interval}m interval.")
                await _subscribe_websocket(ws, symbol, interval, logger)

                async for msg in ws:
                    if msg.type == aiohttp.WSMsgType.TEXT:
                        data = json.loads(msg.data)
                        if "success" in data and not data["success"]:
                            await _handle_subscription_error(data, symbol, logger)
                        elif "topic" in data:
                            await process_websocket_message(data, symbol, interval, analyzer, logger)
                    elif msg.type in (aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.ERROR):
                        logger.warning(f"WebSocket closed or errored for {symbol} on {interval}m interval: {msg}")
                        break
        except aiohttp.ClientConnectorError as e:
            logger.error(f"WebSocket connection error for {symbol} on {interval}m interval: {e}")
        except Exception as e:
            logger.error(f"Unexpected WebSocket error for {symbol} on {interval}m interval: {e}")
        finally: # Ensure session closed even if loop breaks due to error
            if session and not session.closed: # Check if session exists and is not closed
                await session.close()
            session = aiohttp.ClientSession() # Re-initialize session for next loop iteration

        await asyncio.sleep(RECONNECT_DELAY)


async def _subscribe_websocket(ws: aiohttp.ClientWebSocketResponse, symbol: str, interval: str, logger: logging.Logger):
    """Sends subscription messages to the WebSocket - Whispering subscriptions to the stream."""
    subscriptions = [
        {"op": "subscribe", "args": [f"kline.{interval}.{symbol}"]},
        {"op": "subscribe", "args": [f"tickers.{symbol}"]},
        {"op": "subscribe", "args": [f"orderbook.{CONFIG['orderbook_limit']}.{symbol}"]}
    ]
    for sub in subscriptions:
        try:
            await ws.send_json(sub)
            logger.debug(f"Subscribed to: {sub['args'][0]} for {symbol} on {interval}m interval.")
        except Exception as e:
            logger.error(f"WebSocket subscription error for {symbol} - {sub['args'][0]}: {e}")

async def _handle_subscription_error(data: dict, symbol: str, logger: logging.Logger):
    """Handles WebSocket subscription errors - Interpreting subscription failure omens."""
    ret_msg = data.get("ret_msg", "Unknown error")
    ret_code = data.get("ret_code", -1)
    if ret_code == 10001:
        logger.error(f"Invalid symbol {symbol}: {ret_msg}")
        raise ValueError(f"Invalid symbol: {symbol}")
    else:
        logger.error(f"Subscription failed for {symbol}: {ret_msg} (code: {ret_code})")

async def process_websocket_message(data: dict, symbol: str, interval: str, analyzer: 'TradingAnalyzer', logger: logging.Logger):
    """Processes incoming WebSocket messages based on topic - Deciphering messages from the live stream."""
    topic = data["topic"]
    try:
        if topic.startswith("kline"):
            await _process_kline_message(data, symbol, analyzer, logger)
        elif topic.startswith("tickers"):
            await _process_ticker_message(data, symbol, analyzer, logger)
        elif topic.startswith("orderbook"):
            await _process_orderbook_message(data, symbol, logger)
    except Exception as e:
        logger.error(f"Error processing WebSocket message for {symbol} topic '{topic}': {e}")

async def _process_kline_message(data: dict, symbol: str, analyzer: 'TradingAnalyzer', logger: logging.Logger):
    """Processes kline data from WebSocket - Weaving candle data into the analysis tapestry."""
    kline_data = data["data"][0]
    if kline_data["confirm"]:
        df = pd.DataFrame([{
            "start_time": pd.to_datetime(kline_data["start"], unit="ms"),
            "open": float(kline_data["open"]),
            "high": float(kline_data["high"]),
            "low": float(kline_data["low"]),
            "close": float(kline_data["close"]),
            "volume": float(kline_data["volume"]),
            "turnover": float(kline_data["turnover"])
        }])
        analyzer.update_data(df)
        logger.debug(f"Kline update for {symbol}: {kline_data['close']}")

async def _process_ticker_message(data: dict, symbol: str, analyzer: 'TradingAnalyzer', logger: logging.Logger):
    """Processes ticker data from WebSocket - Receiving price updates from the market's pulse."""
    current_price = Decimal(data["data"]["lastPrice"])
    data_cache.set(f"price_{symbol}", current_price)
    await analyzer.analyze_and_output(float(current_price), logger)
    logger.debug(f"Price update for {symbol}: {current_price}")

async def _process_orderbook_message(data: dict, symbol: str, logger: logging.Logger):
    """Processes orderbook data from WebSocket - Observing the market's order depths."""
    orderbook = {"bids": data["data"]["b"], "asks": data["data"]["a"]}
    data_cache.set(f"orderbook_{symbol}", orderbook)
    logger.debug(f"Orderbook update for {symbol}")

# --- Trading Signal Functions - Incantations of Strategy ---

SignalFunction = Callable[[Dict[str, Union[pd.Series, pd.DataFrame]], float, dict, pd.DataFrame], int] # Added df to signal function signature

def base_signal(value: float, upper: float, lower: float, inverse: bool = False) -> int:
    """Base signal logic for threshold-based indicators - Foundational logic for signal generation."""
    if inverse:
        return 1 if value < lower else -1 if value > upper else 0
    else:
        return 1 if value > upper else -1 if value < lower else 0

def ema_alignment_signal(indicators_df: Dict[str, Union[pd.Series, pd.DataFrame]], current_price: float, config: dict, df: pd.DataFrame) -> int:
    """Generates signal based on EMA alignment - EMA's guidance in price direction."""
    ema_value = indicators_df["ema"].iloc[-1]
    return base_signal(current_price, upper=ema_value, lower=ema_value, inverse=True)

def momentum_signal(indicators_df: Dict[str, Union[pd.Series, pd.DataFrame]], current_price: float, config: dict, df: pd.DataFrame) -> int:
    """Generates signal based on momentum trend - Momentum's whisper on market force."""
    trend = indicators_df["mom"].iloc[-1]["trend"]
    return {"Uptrend": 1, "Downtrend": -1, "Neutral": 0}[trend]

def volume_confirmation_signal(indicators_df: Dict[str, Union[pd.Series, pd.DataFrame]], current_price: float, config: dict, df: pd.DataFrame) -> int:
    """Generates signal based on volume confirmation using OBV - Volume's affirmation of price movement."""
    obv = indicators_df["obv"]
    if len(obv) < 2:
        return 0
    return 1 if obv.iloc[-1] > obv.iloc[-2] else -1 if obv.iloc[-1] < obv.iloc[-2] else 0

def stoch_rsi_signal(indicators_df: Dict[str, Union[pd.Series, pd.DataFrame]], current_price: float, config: dict, df: pd.DataFrame) -> int:
    """Generates signal based on Stochastic RSI - Stochastic RSI's insight into overbought/oversold conditions."""
    stoch_rsi_k = indicators_df["stoch_rsi_k"].iloc[-1]
    overbought = config["indicators"]["stoch_rsi"].get("overbought", 0.8) # Get thresholds from config
    oversold = config["indicators"]["stoch_rsi"].get("oversold", 0.2)
    return base_signal(stoch_rsi_k, upper=overbought, lower=oversold)

def rsi_signal(indicators_df: Dict[str, Union[pd.Series, pd.DataFrame]], current_price: float, config: dict, df: pd.DataFrame) -> int:
    """Generates signal based on RSI - RSI's measure of market strength."""
    rsi_value = indicators_df["rsi"].iloc[-1]
    overbought = config["indicators"]["rsi"].get("overbought", 70) # Get thresholds from config
    oversold = config["indicators"]["rsi"].get("oversold", 30)
    return base_signal(rsi_value, upper=overbought, lower=oversold)

def macd_signal(indicators_df: Dict[str, Union[pd.Series, pd.DataFrame]], current_price: float, config: dict, df: pd.DataFrame) -> int:
    """Generates signal based on MACD crossover - MACD's convergence and divergence signals."""
    macd_line = indicators_df["macd"]["macd"].iloc[-1]
    signal_line = indicators_df["macd"]["signal"].iloc[-1]
    return 1 if macd_line > signal_line else -1 if macd_line < signal_line else 0

def bollinger_bands_signal(indicators_df: Dict[str, Union[pd.Series, pd.DataFrame]], current_price: float, config: dict, df: pd.DataFrame) -> int:
    """Generates signal based on Bollinger Bands - Bollinger Bands' embrace of price volatility."""
    bb = indicators_df["bollinger_bands"]
    upper_band = bb["upper_band"].iloc[-1]
    lower_band = bb["lower_band"].iloc[-1]
    return base_signal(current_price, upper=upper_band, lower=lower_band, inverse=True)

def bb_squeeze_signal(indicators_df: Dict[str, Union[pd.Series, pd.DataFrame]], current_price: float, config: dict, df: pd.DataFrame) -> int:
    """Generates signal based on Bollinger Bands Squeeze - BB Squeeze's prelude to explosive moves."""
    bb_df = indicators_df["bollinger_bands_df"]
    lookback = config["indicators"]["bb_squeeze"]["lookback"] # Get lookback from indicator config
    percentile_threshold = config["indicators"]["bb_squeeze"]["percentile"] # Get percentile from indicator config
    if bb_df.empty or len(bb_df) < lookback + 1:
        return 0
    band_width = bb_df["upper_band"] - bb_df["lower_band"]
    if band_width.iloc[-1] < np.percentile(band_width.iloc[-lookback-1:-1], percentile_threshold):
        upper_band = bb_df["upper_band"].iloc[-1]
        lower_band = bb_df["lower_band"].iloc[-1]
        return base_signal(current_price, upper=upper_band, lower=lower_band, inverse=True)
    return 0

def vwap_bounce_signal(indicators_df: Dict[str, Union[pd.Series, pd.DataFrame]], current_price: float, config: dict, df: pd.DataFrame) -> int:
    """Generates signal based on VWAP bounce - VWAP's gravitational pull on price."""
    vwap = indicators_df["vwap"]
    if len(vwap) < 2:
        return 0
    prev_price = indicators_df["close"].iloc[-2]
    vwap_value = vwap.iloc[-1]
    return 1 if prev_price < vwap_value and current_price > vwap_value else -1 if prev_price > vwap_value and current_price < vwap_value else 0

def pivot_breakout_signal(indicators_df: Dict[str, Union[pd.Series, pd.DataFrame]], current_price: float, config: dict, df: pd.DataFrame, support_resistance: dict) -> int:
    """Generates signal based on Pivot Point breakouts - Pivot Points' lines of destiny."""
    r1, s1 = support_resistance.get("r1"), support_resistance.get("s1")
    return 1 if r1 and current_price > r1 else -1 if s1 and current_price < s1 else 0

def divergence_signal(indicators_df: Dict[str, Union[pd.Series, pd.DataFrame]], current_price: float, config: dict, df: pd.DataFrame) -> int:
    """Generates signal based on divergence between price and MACD histogram - Divergence's hidden messages in price and momentum."""
    lookback = config["indicators"]["divergence"]["lookback"] # Get lookback from indicator config
    if len(df) < lookback:
        return 0
    closes = df["close"].tail(lookback).values
    macd_hist = indicators_df["macd"]["histogram"].tail(lookback).values
    min_idx, max_idx = np.argmin(closes), np.argmax(closes)

    if min_idx != len(closes) - 1 and np.min(closes[min_idx:]) < closes[min_idx] and np.min(macd_hist[min_idx:]) > macd_hist[min_idx]:
        return 1
    if max_idx != len(closes) - 1 and np.max(closes[max_idx:]) > closes[max_idx] and np.max(macd_hist[max_idx:]) < macd_hist[max_idx]:
        return -1
    return 0

# --- Signal Aggregation and Output - Orchestration of Signals and Output ---

async def analyze_market_data_signals(indicators_df: Dict[str, Union[pd.Series, pd.DataFrame]], support_resistance: dict, orderbook: Optional[dict], config: dict, df: pd.DataFrame, current_price: float) -> Optional[dict]:
    """Analyzes market data using configured indicators and generates trading signals - Synthesizing insights into actionable signals."""
    signal_functions: Dict[str, SignalFunction] = {
        "ema_alignment": ema_alignment_signal, "momentum": momentum_signal, "volume_confirmation": volume_confirmation_signal,
        "stoch_rsi": stoch_rsi_signal, "rsi": rsi_signal, "macd": macd_signal, "bollinger_bands": bollinger_bands_signal,
        "bb_squeeze": bb_squeeze_signal, "vwap_bounce": vwap_bounce_signal, "pivot_breakout": pivot_breakout_signal,
        "divergence": divergence_signal,
    }
    weight_set_name = CONFIG.get("weight_set", DEFAULT_WEIGHT_SET) # Get selected weight set name, default to "low_volatility"
    weights = config["weight_sets"].get(weight_set_name, config["weight_sets"].get(DEFAULT_WEIGHT_SET, {})) # Fallback to default weight set if selected not found
    total_score, rationale_parts = 0, []
    active_indicators = {ind: w for ind, w in weights.items() if config["indicators"].get(ind, {}).get("enabled", False)}

    for indicator, weight in active_indicators.items():
        score = signal_functions[indicator](indicators_df, current_price, config, df)
        weighted_score = score * weight
        total_score += weighted_score
        if score != 0:
            rationale_parts.append(f"{indicator}: {weighted_score:+.2f}")

    sum_weights = sum(active_indicators.values())
    if not sum_weights:
        return None

    normalized_score = total_score / sum_weights
    signal_threshold = config["signal_config"]["signal_threshold"]

    signal_type = "Long" if normalized_score > signal_threshold else "Short" if normalized_score < -signal_threshold else None
    if not signal_type:
        return None


    confidence = "High" if abs(normalized_score) > 0.7 else "Medium" if abs(normalized_score) > 0.3 else "Low"
    atr_value = indicators_df["atr"].iloc[-1]
    stop_loss, take_profit = calculate_stop_take_profit(signal_type, current_price, atr_value, config["signal_config"])

    return {
        "signal_type": signal_type, "entry_price": current_price, "stop_loss": stop_loss, "take_profit": take_profit,
        "confidence": confidence, "rationale": " | ".join(rationale_parts) or "No significant contributions",
        "normalized_score": normalized_score, "timestamp": datetime.now(TIMEZONE).isoformat()
    }

def calculate_stop_take_profit(signal_type: str, entry_price: float, atr_value: float, signal_config: dict) -> Tuple[float, float]:
    """Calculates stop loss and take profit levels based on ATR and risk-reward ratio - Defining protective boundaries and profit targets."""
    sl_multiplier = signal_config["stop_loss_atr_multiplier"]
    tp_ratio = signal_config["take_profit_risk_reward_ratio"]
    if signal_type == "Long":
        stop_loss = entry_price - atr_value * sl_multiplier
        take_profit = entry_price + (entry_price - stop_loss) * tp_ratio
    else:
        stop_loss = entry_price + atr_value * sl_multiplier
        take_profit = entry_price - (stop_loss - entry_price) * tp_ratio
    return stop_loss, take_profit

async def format_signal_output(signal: Optional[dict], indicators: dict, indicators_df: Dict[str, Union[pd.Series, pd.DataFrame]], current_price: float, logger: logging.Logger, last_output_time: float) -> float:
    """Formats and outputs trading signals to console and files - Presenting insights with clarity and grace."""
    symbol, interval = indicators.get('Symbol', 'N/A'), indicators.get('Interval', 'N/A')
    current_time = time.time()
    if current_time - last_output_time < OUTPUT_THROTTLE_SECONDS and not signal:
        return last_output_time

    if signal:
        _output_signal_to_console(signal, symbol, interval)
        await _save_signal_to_files(signal, symbol, logger)
        _log_alert(signal, symbol, interval, logger)
    else:
        console.print(Panel(f"[bold yellow]No trading signal for {symbol} ({interval}m) at this time.[/bold yellow]", title="[bold cyan]Signal Status[/bold cyan]", border_style="yellow"))

    _output_indicators_to_console(indicators_df, symbol, interval, current_price)
    return current_time

def _output_signal_to_console(signal: dict, symbol: str, interval: str):
    """Outputs trading signal to console using Rich table - Displaying the signal's essence in a structured form."""
    signal_table = Table(title=f"[bold magenta]{signal['signal_type']} Signal for {symbol} ({interval}m)[/bold magenta]", title_justify="center")
    signal_table.add_column("Entry", style="magenta", justify="right")
    signal_table.add_column("Stop-Loss", style="red", justify="right")
    signal_table.add_column("Take-Profit", style="green", justify="right")
    signal_table.add_column("Confidence", style="cyan", justify="center")
    signal_table.add_column("Score", style="yellow", justify="right")
    signal_table.add_column("Rationale", style="green", justify="left")
    signal_table.add_row(
        f"[bold]{signal['entry_price']:.4f}[/bold]", f"[bold]{signal['stop_loss']:.4f}[/bold]", f"[bold]{signal['take_profit']:.4f}[/bold]",
        f"[bold {signal['confidence'].lower()}]{signal['confidence']}[/bold {signal['confidence'].lower()}]", f"[bold]{signal['normalized_score']:.2f}[/bold]", signal["rationale"]
    )
    console.print(Panel.fit(signal_table, title="[bold cyan]Trading Signal[/bold cyan]", border_style="cyan"))

async def _save_signal_to_files(signal: dict, symbol: str, logger: logging.Logger):
    """Saves signal data to JSON and CSV files if configured - Preserving the signal's trace in digital scrolls."""
    output_config = CONFIG["output"]
    if output_config["save_to_json"]:
        os.makedirs(output_config["json_output_dir"], exist_ok=True)
        signal_filename = os.path.join(output_config["json_output_dir"], f"{symbol}_{signal['timestamp'].replace(':', '-')}.json")
        try:
            async with aiofiles.open(signal_filename, "w") as f:
                await f.write(json.dumps(signal, indent=4))
            logger.info(f"Signal JSON saved to {signal_filename}")
        except Exception as e:
            logger.error(f"Error saving signal JSON: {e}")
    if output_config["save_to_csv"]:
        os.makedirs(output_config["csv_output_dir"], exist_ok=True)
        csv_filepath = os.path.join(output_config["csv_output_dir"], f"{symbol}_signals.csv")
        signal_df = pd.DataFrame([signal])
        try:
            async with aiofiles.open(csv_filepath, "a") as f:
                await f.write(signal_df.to_csv(index=False, header=not os.path.exists(csv_filepath)))
            logger.info(f"Signal CSV saved to {csv_filepath}")
        except Exception as e:
            logger.error(f"Error saving signal CSV: {e}")

def _log_alert(signal: dict, symbol: str, interval: str, logger: logging.Logger):
    """Logs a simple alert message to a dedicated alert file - Sending forth a concise alert to the log."""
    output_config = CONFIG["output"]
    try:
        with open(os.path.join(LOG_DIRECTORY, output_config["alert_file"]), "a") as f:
            f.write(f"{signal['timestamp']} - {symbol} ({interval}m): {signal['signal_type']} - Score: {signal['normalized_score']:.2f}\n")
    except Exception as e:
        logger.error(f"Error writing to alert file: {e}")

def _output_indicators_to_console(indicators_df: Dict[str, Union[pd.Series, pd.DataFrame]], symbol: str, interval: str, current_price: float):
    """Outputs indicator values to console using Rich table - Revealing the indicator landscape in a structured view."""
    ind_table = Table(title=f"[bold blue]Technical Indicators for {symbol} ({interval}m)[/bold blue]", title_justify="center")
    ind_table.add_column("Indicator", style="bold blue", justify="left")
    ind_table.add_column("Value", justify="right")
    ind_table.add_column("Status", justify="center")

    for ind_name, ind_config in CONFIG["indicators"].items():
        if not ind_config.get("display", False): # Directly check display from config
            continue
        _add_indicator_row_to_table(ind_table, ind_name, indicators_df, current_price, ind_config) # Pass indicator config

    add_indicator_row(ind_table, "ATR", indicators_df.get("atr", pd.Series([float('nan')])).iloc[-1], "neutral")
    console.print(Panel.fit(ind_table, title="[bold blue]Indicator Snapshot[/bold blue]", border_style="blue"))

def _add_indicator_row_to_table(table: Table, indicator_name: str, indicators_df: Dict[str, Union[pd.Series, pd.DataFrame]], current_price: float, indicator_config: dict):
    """Adds a row for a specific indicator to the Rich table - Inscribing indicator details into the table."""
    if indicator_name == "macd":
        macd_df = indicators_df.get("macd", pd.DataFrame({"macd": [float('nan')], "signal": [float('nan')]}))
        add_indicator_row(table, "MACD", f"{macd_df['macd'].iloc[-1]:.4f} / {macd_df['signal'].iloc[-1]:.4f}", macd_status_logic(macd_df))
    elif indicator_name == "bollinger_bands":
        bb_df = indicators_df.get("bollinger_bands", pd.DataFrame({"lower_band": [float('nan')], "upper_band": [float('nan')]}))
        add_indicator_row(table, "BBands", f"{bb_df['lower_band'].iloc[-1]:.4f} - {bb_df['upper_band'].iloc[-1]:.4f}", bollinger_bands_status_logic(bb_df, current_price))
    elif indicator_name == "rsi":
        rsi_thresholds = (indicator_config.get("oversold", 30), indicator_config.get("overbought", 70)) # Get thresholds from config
        add_indicator_row(table, "RSI", indicators_df.get("rsi", pd.Series([float('nan')])).iloc[-1], rsi_thresholds=rsi_thresholds)
    elif indicator_name == "stoch_rsi":
        stoch_rsi_thresholds = (indicator_config.get("oversold", 0.2), indicator_config.get("overbought", 0.8)) # Get thresholds from config
        add_indicator_row(table, "Stoch RSI (K)", indicators_df.get("stoch_rsi_k", pd.Series([float('nan')])).iloc[-1], stoch_rsi_thresholds=stoch_rsi_thresholds)
    elif indicator_name in ["ema", "vwap"]:
        add_indicator_row(table, indicator_name.upper(), indicators_df.get(indicator_name, pd.Series([float('nan')])).iloc[-1], current_price=current_price)
    elif indicator_name == "volume_confirmation":
        add_indicator_row(table, "OBV", indicators_df.get("obv", pd.Series([float('nan')])).iloc[-1], "neutral")
    elif indicator_name == "momentum":
        mom = indicators_df.get("mom", pd.Series([{"trend": "Neutral", "strength": 0.0}])).iloc[-1]
        add_indicator_row(table, "Momentum", f"{mom['strength']:.4f}", f"[{mom['trend'].lower()}]{mom['trend']}[/{mom['trend'].lower()}]")

def add_indicator_row(table: Table, indicator_name: str, value: Union[str, float, dict], status: Union[str, tuple] = "neutral", current_price: Optional[float] = None, rsi_thresholds: Optional[tuple] = None, stoch_rsi_thresholds: Optional[tuple] = None):
    """Adds a row to the indicator table with formatted status - Crafting a row of indicator wisdom."""
    status_str = _determine_indicator_status(indicator_name, value, status, current_price, rsi_thresholds, stoch_rsi_thresholds)
    display_value = f"{value:.4f}" if isinstance(value, (int, float)) and not pd.isna(value) else str(value)
    table.add_row(indicator_name, display_value, f"[{status_str}]{status_str.capitalize()}[/{status_str}]")

def _determine_indicator_status(indicator_name: str, value: Union[str, float, dict], status: Union[str, tuple] = "neutral", current_price: Optional[float] = None, rsi_thresholds: Optional[tuple] = None, stoch_rsi_thresholds: Optional[tuple] = None) -> str:
    """Determines the status string for an indicator based on its value and thresholds - Judging the indicator's disposition."""
    if isinstance(status, tuple):
        return status[0].split('[')[-1].split(']')[0]

    if rsi_thresholds:
        return "bullish" if value < rsi_thresholds[0] else "bearish" if value > rsi_thresholds[1] else "neutral"
    if stoch_rsi_thresholds:
        return "bullish" if value < stoch_rsi_thresholds[0] else "bearish" if value > stoch_rsi_thresholds[1] else "neutral"
    if current_price is not None and isinstance(value, (int, float)) and not pd.isna(value):
        return "bullish" if current_price > value else "bearish" if current_price < value else "neutral"
    return status if isinstance(status, str) else "neutral"

def macd_status_logic(macd_df: pd.DataFrame) -> Tuple[str]:
    """Determines MACD status for console output - Interpreting MACD's signals."""
    macd_line, signal_line = macd_df["macd"].iloc[-1], macd_df["signal"].iloc[-1]
    if pd.isna(macd_line) or pd.isna(signal_line):
        return ("[yellow]Neutral[/yellow]",)
    status = "bullish" if macd_line > signal_line else "bearish" if macd_line < signal_line else "neutral"
    return (f"[{status}]{status.capitalize()}[/]",)

def bollinger_bands_status_logic(bb_df: pd.DataFrame, current_price: float) -> Tuple[str]:
    """Determines Bollinger Bands status for console output - Reading price position within Bollinger's embrace."""
    bb_upper, bb_lower = bb_df["upper_band"].iloc[-1], bb_df["lower_band"].iloc[-1]
    if pd.isna(bb_upper) or pd.isna(bb_lower) or pd.isna(current_price):
        return ("[yellow]Neutral[/yellow]",)
    status = "bullish" if current_price < bb_lower else "bearish" if current_price > bb_upper else "neutral"
    return (f"[{status}]{status.capitalize()}[/]",)

def vwap_status_logic(vwap_series: pd.Series, current_price: float) -> Tuple[str]:
    """Determines VWAP status for console output - Gauging price alignment with VWAP's anchor."""
    vwap = vwap_series.iloc[-1]
    if pd.isna(vwap) or pd.isna(current_price):
        return ("[yellow]Neutral[/yellow]",)
    status = "bullish" if current_price > vwap else "bearish" if current_price < vwap else "neutral"
    return (f"[{status}]{status.capitalize()}[/]",)

# --- Trading Analyzer Class - The Heart of Analysis ---

class TradingAnalyzer:
    """Analyzes trading data, calculates indicators, and generates signals - The engine of market insight."""
    indicator_values: Dict[str, Union[pd.Series, pd.DataFrame]] # Type hint for indicator_values

    def __init__(self, symbol: str, interval: str, config: dict, logger: logging.Logger):
        self.symbol = symbol
        self.interval = interval
        self.config = config
        self.logger = logger
        self.df = pd.DataFrame()
        self.indicator_values = {} # Initialize with empty dict
        self.last_kline_time = None
        self.last_output_time = 0.0
        self.last_signal = None

    def update_data(self, new_df: pd.DataFrame):
        """Updates historical data with new klines and recalculates indicators - Refreshing the analytical canvas."""
        if self.df.empty or not self.last_kline_time or new_df["start_time"].iloc[-1] > self.last_kline_time: # Handle initial load and ensure new data is indeed newer
            combined_df = pd.concat([self.df, new_df]).drop_duplicates(subset="start_time", keep='last') # Keep='last' to prioritize new data in case of duplicates
            self.df = combined_df.tail(200).reset_index(drop=True) # Reset index after concat and deduplication
            self.last_kline_time = self.df["start_time"].iloc[-1] if not self.df.empty else None # Update last_kline_time only if DataFrame is not empty
            self.calculate_indicators()

    @lru_cache(maxsize=32) # Cache indicator calculations for performance
    def calculate_indicators(self):
        """Calculates all technical indicators based on current DataFrame - Performing the calculations of market metrics."""
        config_indicators = self.config["indicators"] # Access indicator config once for efficiency
        self.indicator_values = {
            "ema": self._calculate_ema(config_indicators["ema_alignment"]),
            "mom": self._calculate_momentum(config_indicators["momentum"]),
            "obv": self._calculate_obv(),
            "rsi": self._calculate_rsi(config_indicators["rsi"]),
            "stoch_rsi_k": self._calculate_stoch_rsi(config_indicators["stoch_rsi"])["k"],
            "macd": self._calculate_macd(config_indicators["macd"]),
            "bollinger_bands": self._calculate_bollinger_bands(config_indicators["bollinger_bands"]),
            "vwap": self._calculate_vwap(),
            "atr": self._calculate_atr(config_indicators["atr"]),
            "close": self.df["close"]
        }
        self.indicator_values["bollinger_bands_df"] = self._calculate_bollinger_bands(config_indicators["bollinger_bands"])
        return self.indicator_values # Return for potential caching benefits even if not directly used now

    def _calculate_ema(self, indicator_config: dict) -> pd.Series:
        """Calculates Exponential Moving Average - EMA's smoothing gaze upon price."""
        window = indicator_config.get("period", 20) # Default to 20 if not in indicator config
        return self.df["close"].ewm(span=window, adjust=False).mean()

    def _calculate_momentum(self, indicator_config: dict) -> pd.Series:
        """Calculates momentum with moving averages and ATR normalization - Momentum's quantified force."""
        period = indicator_config.get("period", 10) # Default to 10 if not in indicator config
        ma_short = indicator_config.get("ma_short", 12) # Default to 12 if not in indicator config
        ma_long = indicator_config.get("ma_long", 26) # Default to 26 if not in indicator config

        if len(self.df) < ma_long:
            return pd.Series([{"trend": "Neutral", "strength": 0.0}] * len(self.df), index=self.df.index)

        momentum = self.df["close"].diff(period)
        short_ma = momentum.rolling(window=ma_short, min_periods=1).mean()
        long_ma = momentum.rolling(window=ma_long, min_periods=1).mean()
        atr = self._calculate_atr(self.config["indicators"]["atr"]) # Use atr indicator config
        trend = np.where(short_ma > long_ma, "Uptrend", np.where(short_ma < long_ma, "Downtrend", "Neutral"))
        strength = np.abs(short_ma - long_ma) / (atr + 1e-9) # Adding small epsilon to avoid division by zero

        return pd.Series([{"trend": t, "strength": float(s)} for t, s in zip(trend, strength)], index=self.df.index)

    def _calculate_obv(self) -> pd.Series:
        """Calculates On Balance Volume - OBV's cumulative measure of volume flow."""
        price_change = self.df["close"].diff()
        direction = np.sign(price_change).fillna(0) # Optimized direction calculation with sign and fillna
        return (direction * self.df["volume"]).cumsum()

    def _calculate_rsi(self, indicator_config: dict) -> pd.Series:
        """Calculates Relative Strength Index - RSI's scale of market vigor."""
        window = indicator_config.get("period", 14) # Default to 14 if not in indicator config
        delta = self.df["close"].diff()
        gain = np.maximum(delta, 0) # Optimized gain/loss calculation using numpy maximum/minimum
        loss = -np.minimum(delta, 0)
        avg_gain = gain.rolling(window=window, min_periods=1).mean()
        avg_loss = loss.rolling(window=window, min_periods=1).mean()
        rs = avg_gain / (avg_loss + 1e-9) # Adding small epsilon to avoid division by zero
        rsi = 100 - (100 / (1 + rs))
        return rsi.fillna(50.0)

    def _calculate_stoch_rsi(self, indicator_config: dict) -> pd.DataFrame:
        """Calculates Stochastic RSI - Stoch RSI's refined view of momentum extremes."""
        period = indicator_config.get("period", 14) # Default to 14 if not in indicator config
        k_period = indicator_config.get("k_period", 3) # Default to 3 if not in indicator config
        d_period = indicator_config.get("d_period", 3) # Default to 3 if not in indicator config

        rsi = self._calculate_rsi(indicator_config) # Pass indicator config to RSI
        rsi_min = rsi.rolling(window=period, min_periods=1).min() # Calculate min and max in one go
        rsi_max = rsi.rolling(window=period, min_periods=1).max()
        stoch_rsi = (rsi - rsi_min) / (rsi_max - rsi_min + 1e-9) # Adding small epsilon to avoid division by zero
        k = stoch_rsi.rolling(k_period, min_periods=1).mean()
        d = k.rolling(d_period, min_periods=1).mean()
        return pd.DataFrame({"stoch_rsi": stoch_rsi, "k": k, "d": d}, index=self.df.index).fillna(0.5)

    def _calculate_macd(self, indicator_config: dict) -> pd.DataFrame:
        """Calculates Moving Average Convergence Divergence - MACD's rhythm of trend shifts."""
        fast_period = indicator_config.get("fast_period", 12) # Default to 12 if not in indicator config
        slow_period = indicator_config.get("slow_period", 26) # Default to 26 if not in indicator config
        signal_period = indicator_config.get("signal_period", 9) # Default to 9 if not in indicator config

        ema_fast = self.df["close"].ewm(span=fast_period, adjust=False).mean() # Calculate EMAs directly
        ema_slow = self.df["close"].ewm(span=slow_period, adjust=False).mean()
        macd = ema_fast - ema_slow
        signal = macd.ewm(span=signal_period, adjust=False).mean()
        histogram = macd - signal
        return pd.DataFrame({"macd": macd, "signal": signal, "histogram": histogram}, index=self.df.index)

    def _calculate_bollinger_bands(self, indicator_config: dict) -> pd.DataFrame:
        """Calculates Bollinger Bands - Bollinger Bands' envelope of volatility."""
        period = indicator_config.get("period", 20) # Default to 20 if not in indicator config
        std_dev = indicator_config.get("std_dev", 2) # Default to 2 if not in indicator config

        sma = self.df["close"].rolling(window=period, min_periods=1).mean()
        std = self.df["close"].rolling(window=period, min_periods=1).std(ddof=0).fillna(0) # Use ddof=0 for population std
        upper_band = sma + (std * std_dev)
        lower_band = sma - (std * std_dev)
        return pd.DataFrame({"upper_band": upper_band, "middle_band": sma, "lower_band": lower_band}, index=self.df.index)

    def _calculate_vwap(self) -> pd.Series:
        """Calculates Volume Weighted Average Price - VWAP's volume-aligned price measure."""
        typical_price = (self.df["high"] + self.df["low"] + self.df["close"]) / 3
        cumulative_tp_volume = (typical_price * self.df["volume"]).cumsum() # Calculate cumulative products and volume once
        cumulative_volume = self.df["volume"].cumsum()
        return cumulative_tp_volume / (cumulative_volume + 1e-9) # Adding small epsilon to avoid division by zero

    def _calculate_atr(self, indicator_config: dict) -> pd.Series:
        """Calculates Average True Range - ATR's gauge of market turbulence."""
        period = indicator_config.get("period", 14) # Default to 14 if not in indicator config

        high_low = self.df["high"] - self.df["low"]
        high_close_prev = abs(self.df["high"] - self.df["close"].shift(1)) # Calculate TR components efficiently
        low_close_prev = abs(self.df["low"] - self.df["close"].shift(1))
        tr = pd.concat([high_low, high_close_prev, low_close_prev], axis=1).max(axis=1) # Combine TR components
        return tr.rolling(window=period, min_periods=1).mean()

    @lru_cache(maxsize=2) # Cache pivot point calculations if called frequently in short time
    def calculate_pivot_points(self) -> dict:
        """Calculates pivot points, Resistance 1, Support 1 - Pivot Points, strategic levels of price action."""
        high, low, close = self.df["high"].max(), self.df["low"].min(), self.df["close"].iloc[-1]
        pivot = (high + low + close) / 3
        return {"pivot": pivot, "r1": 2 * pivot - low, "s1": 2 * pivot - high, "r2": pivot + (high - low), "s2": pivot - (high - low)}

    async def analyze_and_output(self, current_price: float, logger: logging.Logger):
        """Analyzes market data, generates signals, and outputs results - The culmination of analysis into output."""
        if self.df.empty or len(self.df) < 2:
            self.logger.warning("Insufficient data for analysis.")
            return

        indicator_values = self.calculate_indicators() # Get cached indicator values
        support_resistance = self.calculate_pivot_points() # Get cached pivot points
        orderbook = data_cache.get(f"orderbook_{self.symbol}")
        signal = await analyze_market_data_signals(indicator_values, support_resistance, orderbook, self.config, self.df, current_price)

        indicators = {"Symbol": self.symbol, "Interval": self.interval, "Current Price": current_price}
        if signal != self.last_signal or time.time() - self.last_output_time >= OUTPUT_THROTTLE_SECONDS:
            self.last_output_time = await format_signal_output(signal, indicators, indicator_values, current_price, logger, self.last_output_time) # Pass indicator_values
            self.last_signal = signal

# --- Main Function - The Grand Invocation ---

async def analyze_symbol(symbol: str, interval: str, logger: logging.Logger):
    """Sets up and runs the TradingAnalyzer and WebSocket stream for a given symbol - Focusing the analytical gaze on a single pair."""
    analyzer = TradingAnalyzer(symbol, interval, CONFIG, logger)
    try:
        await websocket_stream(symbol, interval, analyzer, logger)
    except ValueError as e:
        logger.error(f"Stopping analysis for {symbol} on {interval}m interval due to: {e}")

async def main(check_config_only: bool = False, weight_set: Optional[str] = None, cli_symbol: Optional[str] = None, cli_interval: Optional[str] = None):
    """Main function to start the trading bot - Initiating the WebWhale Scanner's sequence."""
    console.print(Panel("[bold cyan]Initiating Real-Time Trading Bot Sequence...[/bold cyan]", title="[bold magenta]WebWhale Scanner v14[/bold magenta]"))

    if check_config_only:
        load_config(CONFIG_FILE, check_only=True)
        console.print("[bold green]Configuration check completed.[/bold green]")
        return

    global CONFIG
    CONFIG = load_config(CONFIG_FILE)
    if weight_set: # Override weight set from command line if provided
        CONFIG["weight_set"] = weight_set
        console.print(f"[cyan]Using weight set: [bold]{weight_set}[/bold][/cyan]")
    elif "weight_set" in CONFIG:
        console.print(f"[cyan]Using weight set from config: [bold]{CONFIG['weight_set']}[/bold][/cyan]")
    else:
        console.print(f"[cyan]Using default weight set: [bold]{DEFAULT_WEIGHT_SET}[/bold][/cyan]")


    async with aiohttp.ClientSession() as session:
        valid_symbols = await fetch_valid_symbols(session, logging.getLogger("main"))
        if not valid_symbols:
            console.print("[bold red]Failed to fetch valid symbols. Exiting.[/bold red]")
            return

        if cli_symbol and cli_interval: # Symbol and interval provided via command line
            symbols = [cli_symbol.strip().upper()]
            interval = cli_interval.strip()

            if symbols[0] not in valid_symbols:
                console.print(f"[bold red]Invalid symbol provided: {symbols[0]}. Valid symbols: {', '.join(valid_symbols)}[/bold red]")
                return
            if interval not in VALID_INTERVALS:
                console.print(f"[bold red]Invalid interval provided: {interval}. Valid intervals: {', '.join(VALID_INTERVALS)}[/bold red]")
                return

        else: # Prompt for symbol and interval
            symbols_input = console.input("[cyan]Enter trading pairs (e.g., BTCUSDT, ETHUSDT) separated by commas, or a single symbol: [/cyan]").strip().upper()
            symbols = [s.strip() for s in symbols_input.split(",") if s.strip()]

            interval_input = console.input(f"[cyan]Enter interval ({', '.join(VALID_INTERVALS)}): [/cyan]").strip()
            interval = interval_input if interval_input in VALID_INTERVALS else DEFAULT_INTERVAL
            if interval != interval_input:
                console.print(f"[bold yellow]Invalid interval entered: '{interval_input}'. Using default '{interval}'.[/bold yellow]")

            valid_input_symbols = [s for s in symbols if s in valid_symbols]
            invalid_symbols = set(symbols) - set(valid_input_symbols)

            if invalid_symbols:
                console.print(f"[bold red]Invalid symbols ignored: {', '.join(invalid_symbols)}. Valid symbols: {', '.join(valid_input_symbols)}[/bold red]")
            if not valid_input_symbols:
                console.print("[bold red]No valid symbols entered. Exiting.[/bold red]")
                return
            symbols = valid_input_symbols # Use only valid symbols


        tasks = []
        for symbol in symbols:
            logger = setup_logger(symbol)
            logger.info(f"Starting real-time analysis for {symbol} on {interval}m interval.")
            console.print(f"[cyan]Streaming {symbol} on {interval}m interval. Press Ctrl+C to stop.[/cyan]")
            tasks.append(analyze_symbol(symbol, interval, logger))

        try:
            await asyncio.gather(*tasks, return_exceptions=False)
        except KeyboardInterrupt:
            console.print("[bold yellow]Bot interrupted. Shutting down.[/bold yellow]")
            for symbol in symbols:
                logging.getLogger(symbol).info("Bot stopped manually.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="WebWhale Scanner Bot - The Pyrmethus Edition")
    parser.add_argument('-c', '--check-config', action='store_true', help='Check configuration file and exit')
    parser.add_argument('-w', '--weight-set', type=str, help='Specify weight set to use from config.json')
    parser.add_argument('-s', '--symbol', type=str, help='Trading symbol to analyze (e.g., BTCUSDT)')
    parser.add_argument('-i', '--interval', type=str, help='Kline interval (e.g., 15, 1H, D)')
    args = parser.parse_args()

    asyncio.run(main(check_config_only=args.check_config, weight_set=args.weight_set, cli_symbol=args.symbol, cli_interval=args.interval))
```
