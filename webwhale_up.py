#!/usr/bin/env python3

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
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
from zoneinfo import ZoneInfo
from decimal import Decimal, getcontext
import aiofiles
from logging.handlers import RotatingFileHandler
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.logging import RichHandler
from rich.theme import Theme
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from dataclasses import dataclass
from contextlib import asynccontextmanager
from collections import defaultdict
import argparse

# --- Configuration and Constants ---

getcontext().prec = 10
load_dotenv()

API_KEY = os.getenv("BYBIT_API_KEY")
API_SECRET = os.getenv("BYBIT_API_SECRET")
if not API_KEY or not API_SECRET:
    raise ValueError("BYBIT_API_KEY and BYBIT_API_SECRET must be set in .env")

WEBSOCKET_URL = "wss://stream.bybit.com/v5/public/linear"
BASE_URL = "https://api.bybit.com"
CONFIG_FILE = "config.json"
LOG_DIRECTORY = "bot_logs"
TIMEZONE = ZoneInfo("America/Chicago")
RECONNECT_DELAY = 5
CACHE_TTL_SECONDS = 60
VALID_INTERVALS = ["1", "3", "5", "15", "30", "60", "120", "240", "D", "W", "M"]
OUTPUT_THROTTLE_SECONDS = 60
MAX_API_RETRIES = 3
MAX_BATCH_SIZE = 100
DEFAULT_CACHE_SIZE = 1000
HEARTBEAT_INTERVAL = 30.0
WEBSOCKET_TIMEOUT = 30

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

# --- Data Classes and Validators ---

@dataclass
class IndicatorConfig:
    """Configuration for technical indicators"""
    enabled: bool = True
    display: bool = True
    period: int = 14
    ma_short: int = 12
    ma_long: int = 26
    k_period: int = 3
    d_period: int = 3
    overbought: float = 0.8
    oversold: float = 0.2
    std_dev: float = 2.0

def validate_indicator_config(config: dict) -> bool:
    """Validates indicator configuration parameters"""
    required_fields = ["enabled", "display"]
    for indicator, settings in config["indicators"].items():
        if not all(field in settings for field in required_fields):
            return False
    return True

def validate_trading_signal(signal: dict) -> bool:
    """Validates trading signal structure and values"""
    required_fields = ["signal_type", "entry_price", "stop_loss", "take_profit", "confidence", "normalized_score"]
    return all(field in signal and signal[field] is not None for field in required_fields)

# --- Utility Functions ---

class SensitiveFormatter(logging.Formatter):
    """Formatter to mask sensitive information in logs"""
    def format(self, record):
        msg = super().format(record)
        return msg.replace(API_KEY, "***").replace(API_SECRET, "***")

def load_config(filepath: str, check_only: bool = False) -> dict:
    """Loads and validates configuration with defaults"""
    default_config = {
        "interval": "15",
        "analysis_interval": 30,
        "retry_delay": 5,
        "momentum_period": 10,
        "volume_ma_period": 20,
        "atr_period": 14,
        "trend_strength_threshold": 0.4,
        "indicators": {
            "ema_alignment": {"enabled": True, "display": True, "period": 20},
            "momentum": {"enabled": True, "display": True, "period": 10, "ma_short": 12, "ma_long": 26},
            "volume_confirmation": {"enabled": True, "display": True},
            "divergence": {"enabled": True, "display": False},
            "stoch_rsi": {"enabled": True, "display": True, "period": 14, "k_period": 3, "d_period": 3},
            "rsi": {"enabled": True, "display": True, "period": 14},
            "macd": {"enabled": True, "display": True, "period": 12},
            "bollinger_bands": {"enabled": True, "display": True, "period": 20, "std_dev": 2.0},
            "bb_squeeze": {"enabled": True, "display": False},
            "vwap_bounce": {"enabled": True, "display": True},
            "pivot_breakout": {"enabled": True, "display": False},
        },
        "weight_sets": {
            "low_volatility": {
                "ema_alignment": 0.4, "momentum": 0.3, "volume_confirmation": 0.2, "divergence": 0.1,
                "stoch_rsi": 0.7, "rsi": 0.6, "macd": 0.5, "bollinger_bands": 0.4, "bb_squeeze": 0.3,
                "vwap_bounce": 0.3, "pivot_breakout": 0.3,
            }
        },
        "signal_config": {
            "signal_threshold": 0.3,
            "stop_loss_atr_multiplier": 2,
            "take_profit_risk_reward_ratio": 2,
            "macd_fast": 12,
            "macd_slow": 26,
            "macd_signal": 9,
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
        if not check_only:
            with open(filepath, 'w') as f:
                json.dump(default_config, f, indent=4)
            console.print(Panel(f"[bold yellow]Created new config file at '{filepath}'.[/bold yellow]", title="[bold cyan]Configuration Setup[/bold cyan]"))
        return default_config
    
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            config = json.load(f)
        if not validate_indicator_config(config):
            console.print("[bold red]Invalid indicator configuration. Using defaults.[/bold red]")
            return default_config
        return {**default_config, **config}  # Merge with defaults
    except Exception as e:
        console.print(Panel(f"[bold red]Config error: {e}. Using defaults.[/bold red]", title="[bold cyan]Configuration Error[/bold cyan]"))
        return default_config

def setup_logger(symbol: str) -> logging.Logger:
    """Sets up a dedicated logger for each trading symbol"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = os.path.join(LOG_DIRECTORY, f"{symbol}_{timestamp}.log")
    logger = logging.getLogger(symbol)
    logger.setLevel(logging.DEBUG)
    
    file_handler = RotatingFileHandler(log_filename, maxBytes=10 * 1024 * 1024, backupCount=5)
    file_handler.setFormatter(SensitiveFormatter("%(asctime)s - %(levelname)s - %(message)s"))
    
    console_handler = RichHandler(console=console, rich_tracebacks=True)
    console_handler.setFormatter(SensitiveFormatter("%(message)s"))
    
    logger.handlers = []  # Clear existing handlers
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger

# --- Optimized Data Cache ---

class OptimizedDataCache:
    """Thread-safe cache with size management"""
    def __init__(self, ttl: int = CACHE_TTL_SECONDS, max_size: int = DEFAULT_CACHE_SIZE):
        self.cache: Dict[str, Tuple[Any, float]] = {}
        self.ttl = ttl
        self.max_size = max_size
        self._lock = asyncio.Lock()

    async def get(self, key: str) -> Optional[Any]:
        async with self._lock:
            if key in self.cache:
                value, timestamp = self.cache[key]
                if time.time() - timestamp < self.ttl:
                    return value
                del self.cache[key]
            return None

    async def set(self, key: str, value: Any) -> None:
        async with self._lock:
            if len(self.cache) >= self.max_size:
                oldest_key = min(self.cache.items(), key=lambda x: x[1][1])[0]
                del self.cache[oldest_key]
            self.cache[key] = (value, time.time())

# --- REST API Functions ---

async def fetch_valid_symbols(session: aiohttp.ClientSession, logger: logging.Logger) -> List[str]:
    """Fetches valid trading symbols from Bybit"""
    url = f"{BASE_URL}/v5/market/instruments-info"
    params = {"category": "linear"}
    result = await _bybit_api_request(session, logger, url, params, "GET", "symbols")
    return [item["symbol"] for item in result] if result else []

async def fetch_klines(symbol: str, interval: str, limit: int, session: aiohttp.ClientSession, logger: logging.Logger) -> pd.DataFrame:
    """Fetches Kline data with optimized processing"""
    url = f"{BASE_URL}/v5/market/kline"
    params = {"symbol": symbol, "interval": interval, "limit": limit, "category": "linear"}
    raw_klines = await _bybit_api_request(session, logger, url, params, "GET", "klines")
    if raw_klines:
        df = pd.DataFrame(raw_klines, columns=["start_time", "open", "high", "low", "close", "volume", "turnover"])
        df["start_time"] = pd.to_datetime(df["start_time"].astype(np.int64), unit="ms")
        df[["open", "high", "low", "close", "volume"]] = df[["open", "high", "low", "close", "volume"]].astype(np.float64)
        return df.sort_values("start_time")
    return pd.DataFrame()

async def _bybit_api_request(session: aiohttp.ClientSession, logger: logging.Logger, url: str, params: dict, method: str = "GET", endpoint_description: str = "API") -> Optional[Union[list, dict]]:
    """Optimized API request with retry logic"""
    timestamp = str(int(time.time() * 1000))
    param_str = "&".join([f"{k}={v}" for k, v in sorted({**params, 'timestamp': timestamp}.items())])
    signature = hmac.new(API_SECRET.encode(), param_str.encode(), hashlib.sha256).hexdigest()
    headers = {"X-BAPI-API-KEY": API_KEY, "X-BAPI-TIMESTAMP": timestamp, "X-BAPI-SIGN": signature}

    for attempt in range(MAX_API_RETRIES):
        try:
            async with session.request(method, url, headers=headers, params=params, timeout=aiohttp.ClientTimeout(total=10)) as response:
                response.raise_for_status()
                data = await response.json()
                if data.get("retCode") == 0:
                    return data.get("result", {}).get("list", data.get("result", {}))
                logger.error(f"{endpoint_description} error: {data.get('retMsg')}, code: {data.get('retCode')}")
        except Exception as e:
            logger.warning(f"{endpoint_description} failed (attempt {attempt + 1}/{MAX_API_RETRIES}): {e}")
            if attempt < MAX_API_RETRIES - 1:
                await asyncio.sleep(RECONNECT_DELAY)
    logger.error(f"Failed to fetch {endpoint_description} after {MAX_API_RETRIES} attempts")
    return None

# --- WebSocket Streaming ---

@asynccontextmanager
async def managed_websocket(session: aiohttp.ClientSession, url: str, **kwargs) -> aiohttp.ClientWebSocketResponse:
    """WebSocket connection manager"""
    ws = await session.ws_connect(url, **kwargs)
    try:
        yield ws
    finally:
        await ws.close()

async def optimized_websocket_stream(symbol: str, interval: str, analyzer: 'OptimizedTradingAnalyzer', logger: logging.Logger):
    """Optimized WebSocket streaming with symbol and timeframe promotion"""
    async with aiohttp.ClientSession() as session:
        initial_df = await fetch_klines(symbol, interval, 200, session, logger)
        if initial_df.empty:
            logger.error(f"Cannot proceed with {symbol} on {interval}m: no initial data")
            return
        analyzer.update_data(initial_df)
        logger.info(f"Loaded {len(initial_df)} initial klines for {symbol} on {interval}m")

        while True:
            try:
                async with managed_websocket(session, WEBSOCKET_URL, heartbeat=HEARTBEAT_INTERVAL, timeout=WEBSOCKET_TIMEOUT) as ws:
                    logger.info(f"Connected to WebSocket for {symbol} on {interval}m")
                    await _subscribe_websocket(ws, symbol, interval, logger)
                    async for msg in ws:
                        if msg.type == aiohttp.WSMsgType.TEXT:
                            data = json.loads(msg.data)
                            if "success" in data and not data["success"]:
                                await _handle_subscription_error(data, symbol, logger)
                            elif "topic" in data:
                                await process_websocket_message(data, symbol, interval, analyzer, logger)
                        elif msg.type in (aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.ERROR):
                            logger.warning(f"WebSocket closed for {symbol} on {interval}m: {msg}")
                            break
            except Exception as e:
                logger.error(f"WebSocket error for {symbol} on {interval}m: {e}")
                await asyncio.sleep(RECONNECT_DELAY)

async def _subscribe_websocket(ws: aiohttp.WebSocketClientWebSocketResponse, symbol: str, interval: str, logger: logging.Logger):
    """Subscribes to WebSocket topics with symbol and timeframe"""
    subscriptions = [
        {"op": "subscribe", "args": [f"kline.{interval}.{symbol}"]},
        {"op": "subscribe", "args": [f"tickers.{symbol}"]},
        {"op": "subscribe", "args": [f"orderbook.{CONFIG['orderbook_limit']}.{symbol}"]},
    ]
    for sub in subscriptions:
        await ws.send_json(sub)
        logger.debug(f"Subscribed to: {sub['args'][0]}")

async def _handle_subscription_error(data: dict, symbol: str, logger: logging.Logger):
    """Handles WebSocket subscription errors"""
    ret_msg = data.get("ret_msg", "Unknown error")
    ret_code = data.get("ret_code", -1)
    if ret_code == 10001:
        logger.error(f"Invalid symbol {symbol}: {ret_msg}")
        raise ValueError(f"Invalid symbol: {symbol}")
    logger.error(f"Subscription failed for {symbol}: {ret_msg} (code: {ret_code})")

async def process_websocket_message(data: dict, symbol: str, interval: str, analyzer: 'OptimizedTradingAnalyzer', logger: logging.Logger):
    """Processes WebSocket messages with symbol and timeframe context"""
    topic = data["topic"]
    if topic.startswith("kline"):
        await _process_kline_message(data, symbol, analyzer, logger)
    elif topic.startswith("tickers"):
        await _process_ticker_message(data, symbol, analyzer, logger)
    elif topic.startswith("orderbook"):
        await _process_orderbook_message(data, symbol, logger)

async def _process_kline_message(data: dict, symbol: str, analyzer: 'OptimizedTradingAnalyzer', logger: logging.Logger):
    """Processes Kline data"""
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

async def _process_ticker_message(data: dict, symbol: str, analyzer: 'OptimizedTradingAnalyzer', logger: logging.Logger):
    """Processes ticker data"""
    current_price = Decimal(data["data"]["lastPrice"])
    data_cache.set(f"price_{symbol}", current_price)
    await analyzer.analyze_and_output(float(current_price), logger)
    logger.debug(f"Price update for {symbol}: {current_price}")

async def _process_orderbook_message(data: dict, symbol: str, logger: logging.Logger):
    """Processes orderbook data"""
    orderbook = {"bids": data["data"]["b"], "asks": data["data"]["a"]}
    data_cache.set(f"orderbook_{symbol}", orderbook)
    logger.debug(f"Orderbook update for {symbol}")

# --- Trading Signal Functions ---

SignalFunction = Callable[[Dict[str, Union[pd.Series, pd.DataFrame]], float, dict], int]

def base_signal(value: float, upper: float, lower: float, inverse: bool = False) -> int:
    """Base signal logic for threshold-based indicators"""
    if inverse:
        return 1 if value < lower else -1 if value > upper else 0
    return 1 if value > upper else -1 if value < lower else 0

def ema_alignment_signal(indicators_df: Dict[str, Union[pd.Series, pd.DataFrame]], current_price: float, config: dict) -> int:
    return base_signal(current_price, indicators_df["ema"].iloc[-1], indicators_df["ema"].iloc[-1], inverse=True)

def momentum_signal(indicators_df: Dict[str, Union[pd.Series, pd.DataFrame]], current_price: float, config: dict) -> int:
    trend = indicators_df["mom"].iloc[-1]["trend"]
    return {"Uptrend": 1, "Downtrend": -1, "Neutral": 0}[trend]

def volume_confirmation_signal(indicators_df: Dict[str, Union[pd.Series, pd.DataFrame]], current_price: float, config: dict) -> int:
    obv = indicators_df["obv"]
    return 1 if len(obv) >= 2 and obv.iloc[-1] > obv.iloc[-2] else -1 if len(obv) >= 2 and obv.iloc[-1] < obv.iloc[-2] else 0

def stoch_rsi_signal(indicators_df: Dict[str, Union[pd.Series, pd.DataFrame]], current_price: float, config: dict) -> int:
    return base_signal(indicators_df["stoch_rsi_k"].iloc[-1], config["indicators"]["stoch_rsi"]["overbought"], config["indicators"]["stoch_rsi"]["oversold"])

def rsi_signal(indicators_df: Dict[str, Union[pd.Series, pd.DataFrame]], current_price: float, config: dict) -> int:
    return base_signal(indicators_df["rsi"].iloc[-1], 70, 30)

def macd_signal(indicators_df: Dict[str, Union[pd.Series, pd.DataFrame]], current_price: float, config: dict) -> int:
    macd_line = indicators_df["macd"]["macd"].iloc[-1]
    signal_line = indicators_df["macd"]["signal"].iloc[-1]
    return 1 if macd_line > signal_line else -1 if macd_line < signal_line else 0

def bollinger_bands_signal(indicators_df: Dict[str, Union[pd.Series, pd.DataFrame]], current_price: float, config: dict) -> int:
    bb = indicators_df["bollinger_bands"]
    return base_signal(current_price, bb["upper_band"].iloc[-1], bb["lower_band"].iloc[-1], inverse=True)

def bb_squeeze_signal(indicators_df: Dict[str, Union[pd.Series, pd.DataFrame]], current_price: float, config: dict) -> int:
    bb_df = indicators_df["bollinger_bands_df"]
    lookback = config["signal_config"]["bb_squeeze_lookback"]
    if bb_df.empty or len(bb_df) < lookback + 1:
        return 0
    band_width = bb_df["upper_band"] - bb_df["lower_band"]
    if band_width.iloc[-1] < np.percentile(band_width.iloc[-lookback-1:-1], config["signal_config"]["bb_squeeze_percentile"]):
        return base_signal(current_price, bb_df["upper_band"].iloc[-1], bb_df["lower_band"].iloc[-1], inverse=True)
    return 0

def vwap_bounce_signal(indicators_df: Dict[str, Union[pd.Series, pd.DataFrame]], current_price: float, config: dict) -> int:
    vwap = indicators_df["vwap"]
    if len(vwap) < 2:
        return 0
    prev_price = indicators_df["close"].iloc[-2]
    vwap_value = vwap.iloc[-1]
    return 1 if prev_price < vwap_value and current_price > vwap_value else -1 if prev_price > vwap_value and current_price < vwap_value else 0

def pivot_breakout_signal(indicators_df: Dict[str, Union[pd.Series, pd.DataFrame]], current_price: float, config: dict, support_resistance: dict) -> int:
    r1, s1 = support_resistance.get("r1"), support_resistance.get("s1")
    return 1 if r1 and current_price > r1 else -1 if s1 and current_price < s1 else 0

def divergence_signal(indicators_df: Dict[str, Union[pd.Series, pd.DataFrame]], current_price: float, config: dict, df: pd.DataFrame) -> int:
    lookback = config["signal_config"]["divergence_lookback"]
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

# --- Signal Aggregation and Output ---

async def analyze_market_data_signals(indicators_df: Dict[str, Union[pd.Series, pd.DataFrame]], support_resistance: dict, orderbook: Optional[dict], config: dict, df: pd.DataFrame, current_price: float) -> Optional[dict]:
    """Analyzes market data and generates signals"""
    signal_functions: Dict[str, SignalFunction] = {
        "ema_alignment": ema_alignment_signal, "momentum": momentum_signal, "volume_confirmation": volume_confirmation_signal,
        "stoch_rsi": stoch_rsi_signal, "rsi": rsi_signal, "macd": macd_signal, "bollinger_bands": bollinger_bands_signal,
        "bb_squeeze": bb_squeeze_signal, "vwap_bounce": vwap_bounce_signal, "pivot_breakout": lambda i, c, cfg: pivot_breakout_signal(i, c, cfg, support_resistance),
        "divergence": lambda i, c, cfg: divergence_signal(i, c, cfg, df),
    }
    weights = config["weight_sets"]["low_volatility"]
    total_score, rationale_parts = 0, []
    active_indicators = {ind: w for ind, w in weights.items() if config["indicators"].get(ind, {}).get("enabled", False)}

    for indicator, weight in active_indicators.items():
        score = signal_functions[indicator](indicators_df, current_price, config)
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

    signal = {
        "signal_type": signal_type, "entry_price": current_price, "stop_loss": stop_loss, "take_profit": take_profit,
        "confidence": confidence, "rationale": " | ".join(rationale_parts) or "No significant contributions",
        "normalized_score": normalized_score, "timestamp": datetime.now(TIMEZONE).isoformat()
    }
    
    return signal if validate_trading_signal(signal) else None

def calculate_stop_take_profit(signal_type: str, entry_price: float, atr_value: float, signal_config: dict) -> Tuple[float, float]:
    """Calculates stop loss and take profit"""
    sl_multiplier = signal_config["stop_loss_atr_multiplier"]
    tp_ratio = signal_config["take_profit_risk_reward_ratio"]
    if signal_type == "Long":
        stop_loss = entry_price - atr_value * sl_multiplier
        take_profit = entry_price + (entry_price - stop_loss) * tp_ratio
    else:
        stop_loss = entry_price + atr_value * sl_multiplier
        take_profit = entry_price - (stop_loss - entry_price) * tp_ratio
    return stop_loss, take_profit

def calculate_position_size(account_balance: float, risk_percentage: float, entry_price: float, stop_loss: float) -> float:
    """Calculates position size based on risk"""
    risk_amount = account_balance * (risk_percentage / 100)
    price_difference = abs(entry_price - stop_loss)
    return risk_amount / price_difference if price_difference else 0

def generate_trade_report(signal: dict, position_size: float, account_balance: float) -> str:
    """Generates detailed trade report"""
    risk_amount = abs(position_size * (signal["entry_price"] - signal["stop_loss"]))
    potential_profit = abs(position_size * (signal["take_profit"] - signal["entry_price"]))
    return f"""
    Trade Report:
    Signal Type: {signal["signal_type"]}
    Entry Price: {signal["entry_price"]:.8f}
    Stop Loss: {signal["stop_loss"]:.8f}
    Take Profit: {signal["take_profit"]:.8f}
    Position Size: {position_size:.8f}
    Risk Amount: {risk_amount:.2f}
    Potential Profit: {potential_profit:.2f}
    Risk/Reward Ratio: {potential_profit/risk_amount:.2f}
    """

async def format_signal_output(signal: Optional[dict], indicators: dict, indicators_df: Dict[str, Union[pd.Series, pd.DataFrame]], current_price: float, logger: logging.Logger, last_output_time: float, account_balance: float = 10000.0, risk_percentage: float = 1.0) -> float:
    """Formats and outputs signals with position sizing"""
    symbol, interval = indicators.get('Symbol', 'N/A'), indicators.get('Interval', 'N/A')
    current_time = time.time()
    if current_time - last_output_time < OUTPUT_THROTTLE_SECONDS and not signal:
        return last_output_time

    if signal:
        position_size = calculate_position_size(account_balance, risk_percentage, signal["entry_price"], signal["stop_loss"])
        report = generate_trade_report(signal, position_size, account_balance)
        _output_signal_to_console(signal, symbol, interval, report)
        await _save_signal_to_files(signal, symbol, logger)
        _log_alert(signal, symbol, interval, logger)
    else:
        console.print(Panel(f"[bold yellow]No trading signal for {symbol} ({interval}m) at {current_price:.4f}[/bold yellow]", title="[bold cyan]Signal Status[/bold cyan]"))

    _output_indicators_to_console(indicators_df, symbol, interval, current_price)
    return current_time

def _output_signal_to_console(signal: dict, symbol: str, interval: str, report: str):
    """Outputs signal to console"""
    signal_table = Table(title=f"[bold magenta]{signal['signal_type']} Signal for {symbol} ({interval}m)[/bold magenta]")
    signal_table.add_column("Parameter", style="cyan")
    signal_table.add_column("Value", style="white")
    signal_table.add_row("Entry", f"{signal['entry_price']:.4f}")
    signal_table.add_row("Stop-Loss", f"{signal['stop_loss']:.4f}")
    signal_table.add_row("Take-Profit", f"{signal['take_profit']:.4f}")
    signal_table.add_row("Confidence", f"{signal['confidence']}")
    signal_table.add_row("Score", f"{signal['normalized_score']:.2f}")
    signal_table.add_row("Rationale", signal["rationale"])
    console.print(Panel(signal_table, title="[bold cyan]Trading Signal[/bold cyan]"))
    console.print(Panel(report, title="[bold green]Trade Report[/bold green]"))

async def _save_signal_to_files(signal: dict, symbol: str, logger: logging.Logger):
    """Saves signals to files"""
    output_config = CONFIG["output"]
    if output_config["save_to_json"]:
        os.makedirs(output_config["json_output_dir"], exist_ok=True)
        signal_filename = os.path.join(output_config["json_output_dir"], f"{symbol}_{signal['timestamp'].replace(':', '-')}.json")
        async with aiofiles.open(signal_filename, "w") as f:
            await f.write(json.dumps(signal, indent=4))
        logger.info(f"Signal saved to {signal_filename}")

def _log_alert(signal: dict, symbol: str, interval: str, logger: logging.Logger):
    """Logs alerts"""
    with open(os.path.join(LOG_DIRECTORY, CONFIG["output"]["alert_file"]), "a") as f:
        f.write(f"{signal['timestamp']} - {symbol} ({interval}m): {signal['signal_type']} - Score: {signal['normalized_score']:.2f}\n")

def _output_indicators_to_console(indicators_df: Dict[str, Union[pd.Series, pd.DataFrame]], symbol: str, interval: str, current_price: float):
    """Outputs indicator snapshot"""
    ind_table = Table(title=f"[bold blue]Indicators for {symbol} ({interval}m)[/bold blue]")
    ind_table.add_column("Indicator", style="bold blue")
    ind_table.add_column("Value", justify="right")
    ind_table.add_column("Status", justify="center")

    for ind_name, ind_config in CONFIG["indicators"].items():
        if ind_config.get("display", False):
            _add_indicator_row_to_table(ind_table, ind_name, indicators_df, current_price)

    console.print(Panel(ind_table, title="[bold blue]Indicator Snapshot[/bold blue]"))

def _add_indicator_row_to_table(table: Table, indicator_name: str, indicators_df: Dict[str, Union[pd.Series, pd.DataFrame]], current_price: float):
    """Adds indicator row to table"""
    if indicator_name == "macd":
        macd_df = indicators_df.get("macd", pd.DataFrame({"macd": [np.nan], "signal": [np.nan]}))
        status = "bullish" if macd_df["macd"].iloc[-1] > macd_df["signal"].iloc[-1] else "bearish" if macd_df["macd"].iloc[-1] < macd_df["signal"].iloc[-1] else "neutral"
        table.add_row("MACD", f"{macd_df['macd'].iloc[-1]:.4f} / {macd_df['signal'].iloc[-1]:.4f}", f"[{status}]{status.capitalize()}[/]")
    elif indicator_name == "bollinger_bands":
        bb_df = indicators_df.get("bollinger_bands", pd.DataFrame({"lower_band": [np.nan], "upper_band": [np.nan]}))
        status = "bullish" if current_price < bb_df["lower_band"].iloc[-1] else "bearish" if current_price > bb_df["upper_band"].iloc[-1] else "neutral"
        table.add_row("BBands", f"{bb_df['lower_band'].iloc[-1]:.4f} - {bb_df['upper_band'].iloc[-1]:.4f}", f"[{status}]{status.capitalize()}[/]")
    elif indicator_name in ["ema", "rsi", "stoch_rsi_k", "vwap"]:
        value = indicators_df.get(indicator_name, pd.Series([np.nan])).iloc[-1]
        status = "bullish" if current_price > value else "bearish" if current_price < value else "neutral"
        table.add_row(indicator_name.upper(), f"{value:.4f}", f"[{status}]{status.capitalize()}[/]")

# --- Performance Monitoring ---

class PerformanceMonitor:
    """Monitors performance metrics"""
    def __init__(self):
        self.start_time = time.time()
        self.operation_times = defaultdict(list)

    async def record_operation(self, operation_name: str, start_time: float):
        duration = time.time() - start_time
        self.operation_times[operation_name].append(duration)

    def get_statistics(self) -> dict:
        return {
            name: {
                "avg": np.mean(times),
                "min": np.min(times),
                "max": np.max(times),
                "count": len(times)
            }
            for name, times in self.operation_times.items()
        }

# --- Trading Analyzer ---

class OptimizedTradingAnalyzer:
    """Optimized trading analyzer with symbol and timeframe promotion"""
    def __init__(self, symbol: str, interval: str, config: dict, logger: logging.Logger):
        self.symbol = symbol
        self.interval = interval
        self.config = config
        self.logger = logger
        self.df = pd.DataFrame()
        self.indicator_values: Dict[str, Union[pd.Series, pd.DataFrame]] = {}
        self.last_kline_time = None
        self.last_output_time = 0.0
        self.last_signal = None
        self.cache = OptimizedDataCache()
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.performance = PerformanceMonitor()

    async def update_data(self, new_df: pd.DataFrame):
        """Updates data with performance tracking"""
        start_time = time.time()
        if self.df.empty or not self.last_kline_time or new_df["start_time"].iloc[-1] > self.last_kline_time:
            self.df = pd.concat([self.df, new_df]).drop_duplicates(subset="start_time").tail(200).reset_index(drop=True)
            self.last_kline_time = self.df["start_time"].iloc[-1] if not self.df.empty else None
            await self._parallel_calculate_indicators()
        await self.performance.record_operation(f"update_data_{self.symbol}_{self.interval}", start_time)

    async def _parallel_calculate_indicators(self):
        """Calculates indicators in parallel"""
        tasks = [
            self.executor.submit(self._calculate_ema),
            self.executor.submit(self._calculate_momentum),
            self.executor.submit(self._calculate_obv),
            self.executor.submit(self._calculate_rsi),
            self.executor.submit(self._calculate_stoch_rsi),
            self.executor.submit(self._calculate_macd),
            self.executor.submit(self._calculate_bollinger_bands),
            self.executor.submit(self._calculate_vwap),
            self.executor.submit(self._calculate_atr)
        ]
        results = await asyncio.gather(*[asyncio.to_thread(f.result) for f in tasks])
        self.indicator_values.update({
            "ema": results[0], "mom": results[1], "obv": results[2], "rsi": results[3],
            "stoch_rsi_k": results[4]["k"], "macd": results[5], "bollinger_bands": results[6],
            "vwap": results[7], "atr": results[8], "close": self.df["close"],
            "bollinger_bands_df": results[6]
        })

    def _calculate_ema(self) -> pd.Series:
        return pd.Series(np.array(self.df["close"].ewm(span=self.config["indicators"]["ema_alignment"]["period"], adjust=False).mean()), index=self.df.index)

    def _calculate_momentum(self) -> pd.Series:
        config = self.config["indicators"]["momentum"]
        if len(self.df) < config["ma_long"]:
            return pd.Series([{"trend": "Neutral", "strength": 0.0}] * len(self.df))
        momentum = np.diff(self.df["close"].values, config["period"])
        momentum = np.pad(momentum, (config["period"], 0), mode='constant')
        short_ma = pd.Series(momentum).rolling(window=config["ma_short"], min_periods=1).mean().values
        long_ma = pd.Series(momentum).rolling(window=config["ma_long"], min_periods=1).mean().values
        trend = np.where(short_ma > long_ma, "Uptrend", np.where(short_ma < long_ma, "Downtrend", "Neutral"))
        atr = self._calculate_atr().values
        strength = np.abs(short_ma - long_ma) / (atr + 1e-9)
        return pd.Series([{"trend": t, "strength": float(s)} for t, s in zip(trend, strength)], index=self.df.index)

    def _calculate_obv(self) -> pd.Series:
        direction = np.where(self.df["close"] > self.df["close"].shift(1), 1, np.where(self.df["close"] < self.df["close"].shift(1), -1, 0))
        return pd.Series((direction * self.df["volume"]).cumsum(), index=self.df.index)

    def _calculate_rsi(self) -> pd.Series:
        delta = self.df["close"].diff()
        gain = delta.where(delta > 0, 0).rolling(window=self.config["indicators"]["rsi"]["period"], min_periods=1).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=self.config["indicators"]["rsi"]["period"], min_periods=1).mean()
        rs = gain / loss.replace(0, np.nan)
        return pd.Series(100 - (100 / (1 + rs)), index=self.df.index).fillna(50.0)

    def _calculate_stoch_rsi(self) -> pd.DataFrame:
        rsi = self._calculate_rsi()
        period = self.config["indicators"]["stoch_rsi"]["period"]
        stoch = (rsi - rsi.rolling(window=period, min_periods=1).min()) / (rsi.rolling(window=period, min_periods=1).max() - rsi.rolling(window=period, min_periods=1).min() + 1e-10)
        k = stoch.rolling(self.config["indicators"]["stoch_rsi"]["k_period"], min_periods=1).mean()
        d = k.rolling(self.config["indicators"]["stoch_rsi"]["d_period"], min_periods=1).mean()
        return pd.DataFrame({"stoch_rsi": stoch, "k": k, "d": d}, index=self.df.index).fillna(0.5)

    def _calculate_macd(self) -> pd.DataFrame:
        macd = self.df["close"].ewm(span=self.config["signal_config"]["macd_fast"], adjust=False).mean() - self.df["close"].ewm(span=self.config["signal_config"]["macd_slow"], adjust=False).mean()
        signal = macd.ewm(span=self.config["signal_config"]["macd_signal"], adjust=False).mean()
        histogram = macd - signal
        return pd.DataFrame({"macd": macd, "signal": signal, "histogram": histogram}, index=self.df.index)

    def _calculate_bollinger_bands(self) -> pd.DataFrame:
        sma = self.df["close"].rolling(window=self.config["indicators"]["bollinger_bands"]["period"], min_periods=1).mean()
        std = self.df["close"].rolling(window=self.config["indicators"]["bollinger_bands"]["period"], min_periods=1).std().fillna(0)
        upper_band = sma + (std * self.config["indicators"]["bollinger_bands"]["std_dev"])
        lower_band = sma - (std * self.config["indicators"]["bollinger_bands"]["std_dev"])
        return pd.DataFrame({"upper_band": upper_band, "middle_band": sma, "lower_band": lower_band}, index=self.df.index)

    def _calculate_vwap(self) -> pd.Series:
        typical_price = (self.df["high"] + self.df["low"] + self.df["close"]) / 3
        return pd.Series((typical_price * self.df["volume"]).cumsum() / self.df["volume"].cumsum(), index=self.df.index)

    def _calculate_atr(self) -> pd.Series:
        tr = pd.concat([self.df["high"] - self.df["low"], (self.df["high"] - self.df["close"].shift()).abs(), (self.df["low"] - self.df["close"].shift()).abs()], axis=1).max(axis=1)
        return pd.Series(tr.rolling(window=self.config["atr_period"], min_periods=1).mean(), index=self.df.index)

    def calculate_pivot_points(self) -> dict:
        high, low, close = self.df["high"].max(), self.df["low"].min(), self.df["close"].iloc[-1]
        pivot = (high + low + close) / 3
        return {"pivot": pivot, "r1": 2 * pivot - low, "s1": 2 * pivot - high, "r2": pivot + (high - low), "s2": pivot - (high - low)}

    async def analyze_and_output(self, current_price: float, logger: logging.Logger):
        """Analyzes and outputs signals with performance tracking"""
        start_time = time.time()
        if self.df.empty or len(self.df) < 2:
            logger.warning(f"Insufficient data for {self.symbol} on {self.interval}m")
            return

        support_resistance = self.calculate_pivot_points()
        orderbook = await self.cache.get(f"orderbook_{self.symbol}")
        signal = await analyze_market_data_signals(self.indicator_values, support_resistance, orderbook, self.config, self.df, current_price)

        indicators = {"Symbol": self.symbol, "Interval": self.interval, "Current Price": current_price}
        if signal != self.last_signal or time.time() - self.last_output_time >= OUTPUT_THROTTLE_SECONDS:
            self.last_output_time = await format_signal_output(signal, indicators, self.indicator_values, current_price, logger, self.last_output_time)
            self.last_signal = signal
        await self.performance.record_operation(f"analyze_{self.symbol}_{self.interval}", start_time)

# --- Main Functions ---

async def analyze_symbol(symbol: str, interval: str, logger: logging.Logger):
    """Analyzes a single symbol with timeframe"""
    analyzer = OptimizedTradingAnalyzer(symbol, interval, CONFIG, logger)
    try:
        await optimized_websocket_stream(symbol, interval, analyzer, logger)
    except ValueError as e:
        logger.error(f"Stopping analysis for {symbol} on {interval}m: {e}")

async def main(check_config_only: bool = False, cli_symbol: Optional[str] = None, cli_interval: Optional[str] = None, risk_percentage: float = 1.0):
    """Main entry point with enhanced symbol and timeframe handling"""
    console.print(Panel(f"[bold cyan]WebWhale Scanner v15 - Starting {datetime.now(TIMEZONE).strftime('%Y-%m-%d %H:%M:%S')}[/bold cyan]", title="[bold magenta]Trading Bot[/bold magenta]"))
    
    global CONFIG, data_cache
    CONFIG = load_config(CONFIG_FILE, check_config_only)
    data_cache = OptimizedDataCache()
    
    if check_config_only:
        console.print("[bold green]Configuration check completed[/bold green]")
        return

    async with aiohttp.ClientSession() as session:
        valid_symbols = await fetch_valid_symbols(session, logging.getLogger("main"))
        if not valid_symbols:
            console.print("[bold red]Failed to fetch valid symbols[/bold red]")
            return

        symbols = [cli_symbol] if cli_symbol else console.input("[cyan]Enter trading pairs (e.g., BTCUSDT, ETHUSDT): [/cyan]").strip().upper().split(",")
        symbols = [s.strip() for s in symbols if s.strip()]
        valid_symbols_set = set(valid_symbols)
        valid_input_symbols = [s for s in symbols if s in valid_symbols_set]
        
        if not valid_input_symbols:
            console.print("[bold red]No valid symbols entered[/bold red]")
            return

        interval = cli_interval if cli_interval in VALID_INTERVALS else CONFIG.get("interval", "15")
        if interval not in VALID_INTERVALS:
            console.print(f"[bold yellow]Invalid interval {interval}, using 15m[/bold yellow]")
            interval = "15"

        tasks = []
        for symbol in valid_input_symbols:
            logger = setup_logger(symbol)
            logger.info(f"Starting analysis for {symbol} on {interval}m")
            console.print(f"[cyan]Streaming {symbol} on {interval}m[/cyan]")
            tasks.append(analyze_symbol(symbol, interval, logger))

        try:
            await asyncio.gather(*tasks, return_exceptions=True)
        except KeyboardInterrupt:
            console.print("[bold yellow]Shutting down gracefully[/bold yellow]")
        finally:
            for handler in logging.root.handlers[:]:
                handler.close()
                logging.root.removeHandler(handler)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="WebWhale Scanner Trading Bot")
    parser.add_argument('-c', '--check-config', action='store_true', help='Check configuration and exit')
    parser.add_argument('-s', '--symbol', type=str, help='Trading symbol (e.g., BTCUSDT)')
    parser.add_argument('-i', '--interval', type=str, help='Trading interval (e.g., 15)')
    parser.add_argument('-r', '--risk', type=float, default=1.0, help='Risk percentage per trade')
    args = parser.parse_args()

    asyncio.run(main(
        check_config_only=args.check_config,
        cli_symbol=args.symbol,
        cli_interval=args.interval,
        risk_percentage=args.risk
    ))
