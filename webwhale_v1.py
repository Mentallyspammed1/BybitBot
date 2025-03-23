import os
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
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

# --- Configuration and Setup ---

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
OUTPUT_THROTTLE_SECONDS = 60  # Limit signal output frequency
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
    def format(self, record):
        msg = super().format(record)
        return msg.replace(API_KEY, "***").replace(API_SECRET, "***")

def load_config(filepath: str) -> dict:
    default_config = {
        "interval": "15",
        "analysis_interval": 30,
        "momentum_period": 10,
        "momentum_ma_short": 12,
        "momentum_ma_long": 26,
        "volume_ma_period": 20,
        "atr_period": 14,
        "trend_strength_threshold": 0.4,
        "indicators": {
            "ema_alignment": {"enabled": True, "display": True},
            "momentum": {"enabled": True, "display": True},
            "volume_confirmation": {"enabled": True, "display": True},
            "divergence": {"enabled": True, "display": False},
            "stoch_rsi": {"enabled": True, "display": True},
            "rsi": {"enabled": True, "display": True},
            "macd": {"enabled": True, "display": True},
            "bollinger_bands": {"enabled": True, "display": True},
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
        console.print(Panel(f"[bold yellow]Created new config file at '{filepath}'.[/bold yellow]", title="[bold cyan]Configuration Setup[/bold cyan]"))
        return default_config
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            config = json.load(f)
        # Validate and upgrade config
        required_keys = ["interval", "indicators", "weight_sets", "signal_config", "orderbook_limit"]
        for key in required_keys:
            if key not in config:
                console.print(f"[bold red]Missing '{key}' in config. Using default.[/bold red]")
                config[key] = default_config[key]
        for ind, val in list(config["indicators"].items()):
            if isinstance(val, bool):
                config["indicators"][ind] = {"enabled": val, "display": val}
        if config["interval"] not in VALID_INTERVALS:
            config["interval"] = "15"
        if not isinstance(config["orderbook_limit"], int) or config["orderbook_limit"] <= 0:
            config["orderbook_limit"] = 50
        with open(filepath, 'w') as f:
            json.dump(config, f, indent=4)
        return config
    except Exception as e:
        console.print(Panel(f"[bold red]Config error: {e}. Using defaults.[/bold red]", title="[bold cyan]Configuration Error[/bold cyan]"))
        return default_config

CONFIG = load_config(CONFIG_FILE)

def setup_logger(symbol: str) -> logging.Logger:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = os.path.join(LOG_DIRECTORY, f"{symbol}_{timestamp}.log")
    logger = logging.getLogger(symbol)
    logger.setLevel(logging.DEBUG)  # Increased verbosity
    file_handler = RotatingFileHandler(log_filename, maxBytes=10 * 1024 * 1024, backupCount=5)
    file_handler.setFormatter(SensitiveFormatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(file_handler)
    console_handler = RichHandler(console=console, rich_tracebacks=True)
    console_handler.setFormatter(SensitiveFormatter("%(message)s"))
    logger.addHandler(console_handler)
    return logger

# --- Data Cache ---

class DataCache:
    def __init__(self, ttl: int = CACHE_TTL_SECONDS):
        self.cache: Dict[str, Tuple[Union[Decimal, pd.DataFrame, dict], float]] = {}
        self.ttl = ttl

    def get(self, key: str) -> Optional[Union[Decimal, pd.DataFrame, dict]]:
        if key in self.cache:
            value, timestamp = self.cache[key]
            if time.time() - timestamp < self.ttl:
                return value
            del self.cache[key]
        return None

    def set(self, key: str, value: Union[Decimal, pd.DataFrame, dict]):
        self.cache[key] = (value, time.time())

data_cache = DataCache()

# --- REST API Functions ---

async def fetch_valid_symbols(session: aiohttp.ClientSession, logger: logging.Logger) -> List[str]:
    url = f"{BASE_URL}/v5/market/instruments-info"
    params = {"category": "linear"}
    timestamp = str(int(time.time() * 1000))
    param_str = "&".join([f"{k}={v}" for k, v in sorted({**params, 'timestamp': timestamp}.items())])
    signature = hmac.new(API_SECRET.encode(), param_str.encode(), hashlib.sha256).hexdigest()
    headers = {"X-BAPI-API-KEY": API_KEY, "X-BAPI-TIMESTAMP": timestamp, "X-BAPI-SIGN": signature}
    for attempt in range(MAX_API_RETRIES):
        try:
            async with session.get(url, headers=headers, params=params) as response:
                response.raise_for_status()
                data = await response.json()
                if data.get("retCode") == 0:
                    return [item["symbol"] for item in data["result"]["list"] if item["status"] == "Trading"]
                logger.error(f"Symbols fetch error: {data.get('retMsg')}")
        except Exception as e:
            logger.warning(f"Symbols fetch failed (attempt {attempt + 1}/{MAX_API_RETRIES}): {e}")
            await asyncio.sleep(RECONNECT_DELAY)
    logger.error("Failed to fetch valid symbols")
    return []

async def fetch_klines(symbol: str, interval: str, limit: int, session: aiohttp.ClientSession, logger: logging.Logger) -> pd.DataFrame:
    url = f"{BASE_URL}/v5/market/kline"
    params = {"symbol": symbol, "interval": interval, "limit": limit, "category": "linear"}
    timestamp = str(int(time.time() * 1000))
    param_str = "&".join([f"{k}={v}" for k, v in sorted({**params, 'timestamp': timestamp}.items())])
    signature = hmac.new(API_SECRET.encode(), param_str.encode(), hashlib.sha256).hexdigest()
    headers = {"X-BAPI-API-KEY": API_KEY, "X-BAPI-TIMESTAMP": timestamp, "X-BAPI-SIGN": signature}
    for attempt in range(MAX_API_RETRIES):
        try:
            async with session.get(url, headers=headers, params=params) as response:
                response.raise_for_status()
                data = await response.json()
                if data.get("retCode") == 0 and data["result"]["list"]:
                    df = pd.DataFrame(data["result"]["list"], columns=["start_time", "open", "high", "low", "close", "volume", "turnover"])
                    df["start_time"] = pd.to_datetime(df["start_time"].astype(int), unit="ms")
                    df[["open", "high", "low", "close", "volume"]] = df[["open", "high", "low", "close", "volume"]].apply(pd.to_numeric)
                    df = df.sort_values("start_time")
                    return df
                logger.error(f"Klines fetch error: {data.get('retMsg')}")
        except Exception as e:
            logger.warning(f"Klines fetch failed (attempt {attempt + 1}/{MAX_API_RETRIES}): {e}")
            await asyncio.sleep(RECONNECT_DELAY)
    logger.error(f"Failed to fetch initial klines for {symbol}")
    return pd.DataFrame()

# --- WebSocket Streaming ---

async def websocket_stream(symbol: str, interval: str, analyzer: 'TradingAnalyzer', logger: logging.Logger):
    async with aiohttp.ClientSession() as session:
        # Fetch initial data
        initial_df = await fetch_klines(symbol, interval, 200, session, logger)
        if initial_df.empty:
            logger.error(f"Cannot proceed with {symbol}: no initial data")
            return
        analyzer.update_data(initial_df)
        logger.info(f"Loaded initial {len(initial_df)} klines for {symbol}")

        while True:
            try:
                async with session.ws_connect(WEBSOCKET_URL, heartbeat=30.0) as ws:
                    logger.info(f"Connected to WebSocket for {symbol}")
                    subscriptions = [
                        {"op": "subscribe", "args": [f"kline.{interval}.{symbol}"]},
                        {"op": "subscribe", "args": [f"tickers.{symbol}"]},
                        {"op": "subscribe", "args": [f"orderbook.{CONFIG['orderbook_limit']}.{symbol}"]}
                    ]
                    for sub in subscriptions:
                        await ws.send_json(sub)

                    async for msg in ws:
                        if msg.type == aiohttp.WSMsgType.TEXT:
                            data = json.loads(msg.data)
                            if "success" in data:
                                if data["success"]:
                                    logger.debug(f"Subscription confirmed: {data.get('op')}")
                                else:
                                    ret_msg = data.get("ret_msg", "Unknown error")
                                    ret_code = data.get("ret_code", -1)
                                    if ret_code == 10001:  # Invalid symbol
                                        logger.error(f"Invalid symbol {symbol}: {ret_msg}")
                                        return
                                    logger.error(f"Subscription failed: {ret_msg} (code: {ret_code})")
                            elif "topic" in data:
                                await process_websocket_message(data, symbol, interval, analyzer, logger)
                        elif msg.type in (aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.ERROR):
                            logger.warning(f"WebSocket closed or errored: {msg}")
                            break
            except Exception as e:
                logger.error(f"WebSocket error for {symbol}: {e}")
                await asyncio.sleep(RECONNECT_DELAY)

async def process_websocket_message(data: dict, symbol: str, interval: str, analyzer: 'TradingAnalyzer', logger: logging.Logger):
    topic = data["topic"]
    if topic.startswith("kline"):
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
    elif topic.startswith("tickers"):
        current_price = Decimal(data["data"]["lastPrice"])
        data_cache.set(f"price_{symbol}", current_price)
        await analyzer.analyze_and_output(float(current_price), logger)
        logger.debug(f"Price update for {symbol}: {current_price}")
    elif topic.startswith("orderbook"):
        orderbook = {"bids": data["data"]["b"], "asks": data["data"]["a"]}
        data_cache.set(f"orderbook_{symbol}", orderbook)
        logger.debug(f"Orderbook update for {symbol}")

# --- Trading Signal Functions ---

SignalFunction = Callable[[Dict[str, Union[pd.Series, pd.DataFrame]], float, dict], int]

def base_signal(value: float, upper: Optional[float] = None, lower: Optional[float] = None, inverse: bool = False) -> int:
    if upper is not None and value > upper:
        return -1 if not inverse else 1
    if lower is not None and value < lower:
        return 1 if not inverse else -1
    return 0

def ema_alignment_signal(indicators_df: Dict[str, Union[pd.Series, pd.DataFrame]], current_price: float, config: dict) -> int:
    return base_signal(current_price, upper=indicators_df["ema"].iloc[-1], lower=indicators_df["ema"].iloc[-1], inverse=True)

def momentum_signal(indicators_df: Dict[str, Union[pd.Series, pd.DataFrame]], current_price: float, config: dict) -> int:
    trend = indicators_df["mom"].iloc[-1]["trend"]
    return {"Uptrend": 1, "Downtrend": -1, "Neutral": 0}[trend]

def volume_confirmation_signal(indicators_df: Dict[str, Union[pd.Series, pd.DataFrame]], current_price: float, config: dict) -> int:
    obv = indicators_df["obv"]
    return 1 if len(obv) >= 2 and obv.iloc[-1] > obv.iloc[-2] else -1 if len(obv) >= 2 and obv.iloc[-1] < obv.iloc[-2] else 0

def stoch_rsi_signal(indicators_df: Dict[str, Union[pd.Series, pd.DataFrame]], current_price: float, config: dict) -> int:
    return base_signal(indicators_df["stoch_rsi_k"].iloc[-1], upper=0.8, lower=0.2)

def rsi_signal(indicators_df: Dict[str, Union[pd.Series, pd.DataFrame]], current_price: float, config: dict) -> int:
    return base_signal(indicators_df["rsi"].iloc[-1], upper=70, lower=30)

def macd_signal(indicators_df: Dict[str, Union[pd.Series, pd.DataFrame]], current_price: float, config: dict) -> int:
    macd_line, signal_line = indicators_df["macd"]["macd"].iloc[-1], indicators_df["macd"]["signal"].iloc[-1]
    return 1 if macd_line > signal_line else -1 if macd_line < signal_line else 0

def bollinger_bands_signal(indicators_df: Dict[str, Union[pd.Series, pd.DataFrame]], current_price: float, config: dict) -> int:
    bb = indicators_df["bollinger_bands"]
    return base_signal(current_price, upper=bb["upper_band"].iloc[-1], lower=bb["lower_band"].iloc[-1], inverse=True)

def bb_squeeze_signal(indicators_df: Dict[str, Union[pd.Series, pd.DataFrame]], current_price: float, config: dict) -> int:
    bb_df = indicators_df["bollinger_bands"]
    lookback = config["signal_config"]["bb_squeeze_lookback"]
    if len(bb_df) < lookback + 1:
        return 0
    band_width = bb_df["upper_band"] - bb_df["lower_band"]
    if band_width.iloc[-1] < np.percentile(band_width.iloc[-lookback-1:-1], config["signal_config"]["bb_squeeze_percentile"]):
        return base_signal(current_price, upper=bb_df["upper_band"].iloc[-1], lower=bb_df["lower_band"].iloc[-1], inverse=True)
    return 0

def vwap_bounce_signal(indicators_df: Dict[str, Union[pd.Series, pd.DataFrame]], current_price: float, config: dict) -> int:
    vwap = indicators_df["vwap"]
    if len(vwap) < 2:
        return 0
    prev_price = indicators_df["close"].iloc[-2]
    return 1 if prev_price < vwap.iloc[-1] and current_price > vwap.iloc[-1] else -1 if prev_price > vwap.iloc[-1] and current_price < vwap.iloc[-1] else 0

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
    signal_functions: Dict[str, SignalFunction] = {
        "ema_alignment": ema_alignment_signal, "momentum": momentum_signal, "volume_confirmation": volume_confirmation_signal,
        "stoch_rsi": stoch_rsi_signal, "rsi": rsi_signal, "macd": macd_signal, "bollinger_bands": bollinger_bands_signal,
        "bb_squeeze": bb_squeeze_signal, "vwap_bounce": vwap_bounce_signal, "pivot_breakout": lambda i, c, cfg: pivot_breakout_signal(i, c, cfg, support_resistance),
        "divergence": lambda i, c, cfg: divergence_signal(i, c, cfg, df),
    }
    weights = config["weight_sets"]["low_volatility"]
    total_score, rationale_parts = 0, []
    active_indicators = {}
    for ind, weight in weights.items():
        indicator_config = config["indicators"].get(ind, {"enabled": False, "display": False})
        if isinstance(indicator_config, bool):
            indicator_config = {"enabled": indicator_config, "display": indicator_config}
        if indicator_config["enabled"]:
            active_indicators[ind] = weight

    for ind, weight in active_indicators.items():
        score = signal_functions[ind](indicators_df, current_price, config)
        weighted_score = score * weight
        total_score += weighted_score
        if score != 0:
            rationale_parts.append(f"{ind}: {weighted_score:+.2f}")

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
    symbol, interval = indicators.get('Symbol', 'N/A'), indicators.get('Interval', 'N/A')
    current_time = time.time()
    if current_time - last_output_time < OUTPUT_THROTTLE_SECONDS and not signal:
        return last_output_time  # Skip if throttled and no signal

    if signal:
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
        if CONFIG["output"]["save_to_json"]:
            os.makedirs(CONFIG["output"]["json_output_dir"], exist_ok=True)
            async with aiofiles.open(os.path.join(CONFIG["output"]["json_output_dir"], f"{symbol}_{signal['timestamp'].replace(':', '-')}.json"), "w") as f:
                await f.write(json.dumps(signal, indent=4))
        if CONFIG["output"]["save_to_csv"]:
            os.makedirs(CONFIG["output"]["csv_output_dir"], exist_ok=True)
            csv_path = os.path.join(CONFIG["output"]["csv_output_dir"], f"{symbol}_signals.csv")
            signal_df = pd.DataFrame([signal])
            async with aiofiles.open(csv_path, "a") as f:
                await f.write(signal_df.to_csv(index=False, header=not os.path.exists(csv_path)))
        async with aiofiles.open(os.path.join(LOG_DIRECTORY, CONFIG["output"]["alert_file"]), "a") as f:
            await f.write(f"{signal['timestamp']} - {symbol} ({interval}m): {signal['signal_type']} - Score: {signal['normalized_score']:.2f}\n")
    else:
        console.print(Panel(f"[bold yellow]No trading signal for {symbol} ({interval}m) at this time.[/bold yellow]", title="[bold cyan]Signal Status[/bold cyan]", border_style="yellow"))

    ind_table = Table(title=f"[bold blue]Technical Indicators for {symbol} ({interval}m)[/bold blue]", title_justify="center")
    ind_table.add_column("Indicator", style="bold blue", justify="left")
    ind_table.add_column("Value", justify="right")
    ind_table.add_column("Status", justify="center")

    for ind_name, ind_config in CONFIG["indicators"].items():
        if not isinstance(ind_config, dict):
            ind_config = {"enabled": ind_config, "display": ind_config}
        if not ind_config["display"]:
            continue
        if ind_name == "macd":
            macd_df = indicators_df.get("macd", pd.DataFrame({"macd": [float('nan')], "signal": [float('nan')]}))
            add_indicator_row(ind_table, "MACD", f"{macd_df['macd'].iloc[-1]:.4f} / {macd_df['signal'].iloc[-1]:.4f}", macd_status_logic(macd_df))
        elif ind_name == "bollinger_bands":
            bb_df = indicators_df.get("bollinger_bands", pd.DataFrame({"lower_band": [float('nan')], "upper_band": [float('nan')]}))
            add_indicator_row(ind_table, "BBands", f"{bb_df['lower_band'].iloc[-1]:.4f} - {bb_df['upper_band'].iloc[-1]:.4f}", bollinger_bands_status_logic(bb_df, current_price))
        elif ind_name == "rsi":
            add_indicator_row(ind_table, "RSI", indicators_df.get("rsi", pd.Series([float('nan')])).iloc[-1], rsi_thresholds=(30, 70))
        elif ind_name == "stoch_rsi":
            add_indicator_row(ind_table, "Stoch RSI (K)", indicators_df.get("stoch_rsi_k", pd.Series([float('nan')])).iloc[-1], stoch_rsi_thresholds=(0.2, 0.8))
        elif ind_name in ["ema", "vwap"]:
            add_indicator_row(ind_table, ind_name.upper(), indicators_df.get(ind_name, pd.Series([float('nan')])).iloc[-1], current_price=current_price)
        elif ind_name == "volume_confirmation":
            add_indicator_row(ind_table, "OBV", indicators_df.get("obv", pd.Series([float('nan')])).iloc[-1], "neutral")
        elif ind_name == "momentum":
            mom = indicators_df.get("mom", pd.Series([{"trend": "Neutral", "strength": 0.0}])).iloc[-1]
            add_indicator_row(ind_table, "Momentum", f"{mom['strength']:.4f}", f"[{mom['trend'].lower()}]{mom['trend']}[/{mom['trend'].lower()}]")
    add_indicator_row(ind_table, "ATR", indicators_df.get("atr", pd.Series([float('nan')])).iloc[-1], "neutral")
    console.print(Panel.fit(ind_table, title="[bold blue]Indicator Snapshot[/bold blue]", border_style="blue"))
    return current_time

def add_indicator_row(table: Table, indicator_name: str, value: Union[str, float, dict], status: Union[str, tuple] = "neutral", current_price: Optional[float] = None, rsi_thresholds: Optional[tuple] = None, stoch_rsi_thresholds: Optional[tuple] = None):
    if isinstance(status, tuple):
        table.add_row(indicator_name, str(value), status[0])
    elif rsi_thresholds:
        status_str = "bullish" if value < rsi_thresholds[0] else "bearish" if value > rsi_thresholds[1] else "neutral"
        table.add_row(indicator_name, f"{value:.2f}", f"[{status_str}]{status_str.capitalize()}[/{status_str}]")
    elif stoch_rsi_thresholds:
        status_str = "bullish" if value < stoch_rsi_thresholds[0] else "bearish" if value > stoch_rsi_thresholds[1] else "neutral"
        table.add_row(indicator_name, f"{value:.4f}", f"[{status_str}]{status_str.capitalize()}[/{status_str}]")
    else:
        if current_price is not None and isinstance(value, (int, float)) and not pd.isna(value):
            status_str = "bullish" if current_price > value else "bearish" if current_price < value else "neutral"
        else:
            status_str = status if isinstance(status, str) else "neutral"
        table.add_row(indicator_name, f"{value:.4f}" if isinstance(value, (int, float)) and not pd.isna(value) else str(value), f"[{status_str}]{status_str.capitalize()}[/{status_str}]")

def macd_status_logic(macd_df: pd.DataFrame) -> Tuple[str]:
    macd_line, signal_line = macd_df["macd"].iloc[-1], macd_df["signal"].iloc[-1]
    return ("[yellow]Neutral[/yellow]",) if pd.isna(macd_line) or pd.isna(signal_line) else (f"[{'bullish' if macd_line > signal_line else 'bearish' if macd_line < signal_line else 'neutral'}]{'Bullish' if macd_line > signal_line else 'Bearish' if macd_line < signal_line else 'Neutral'}[/]",)

def bollinger_bands_status_logic(bb_df: pd.DataFrame, current_price: float) -> Tuple[str]:
    bb_upper, bb_lower = bb_df["upper_band"].iloc[-1], bb_df["lower_band"].iloc[-1]
    return ("[yellow]Neutral[/yellow]",) if pd.isna(bb_upper) or pd.isna(bb_lower) or pd.isna(current_price) else (f"[{'bullish' if current_price < bb_lower else 'bearish' if current_price > bb_upper else 'neutral'}]{'Bullish' if current_price < bb_lower else 'Bearish' if current_price > bb_upper else 'Neutral'}[/]",)

# --- Trading Analyzer Class ---

class TradingAnalyzer:
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

    def update_data(self, new_df: pd.DataFrame):
        if self.df.empty or new_df["start_time"].iloc[-1] > self.last_kline_time:
            self.df = pd.concat([self.df, new_df]).drop_duplicates(subset="start_time").tail(200)
            self.last_kline_time = self.df["start_time"].iloc[-1]
            self.update_indicators()

    def update_indicators(self):
        if not self.indicator_values:
            self.calculate_indicators()
        else:
            new_close = self.df["close"].iloc[-1]
            # Incremental updates
            self.indicator_values["close"] = self.df["close"]
            self.indicator_values["ema"] = self.update_ema(new_close)
            self.indicator_values["mom"] = self.update_momentum()
            self.indicator_values["obv"] = self.update_obv()
            self.indicator_values["rsi"] = self.update_rsi(new_close)
            self.indicator_values["stoch_rsi_k"] = self.update_stoch_rsi()
            self.indicator_values["macd"] = self.update_macd(new_close)
            self.indicator_values["bollinger_bands"] = self.update_bollinger_bands(new_close)
            self.indicator_values["vwap"] = self.update_vwap()
            self.indicator_values["atr"] = self.update_atr()

    def update_ema(self, new_close: float) -> pd.Series:
        alpha = 2 / (self.config["signal_config"]["ema_period"] + 1)
        last_ema = self.indicator_values["ema"].iloc[-2] if len(self.indicator_values["ema"]) > 1 else new_close
        new_ema = alpha * new_close + (1 - alpha) * last_ema
        return pd.concat([self.indicator_values["ema"][:-1], pd.Series([new_ema], index=[self.df.index[-1]])])

    def update_momentum(self) -> pd.Series:
        if len(self.df) < self.config["momentum_ma_long"]:
            return pd.Series([{"trend": "Neutral", "strength": 0.0}] * len(self.df), index=self.df.index)
        momentum = self.df["close"].diff(self.config["momentum_period"])
        short_ma = momentum.rolling(window=self.config["momentum_ma_short"], min_periods=1).mean()
        long_ma = momentum.rolling(window=self.config["momentum_ma_long"], min_periods=1).mean()
        atr = self.indicator_values["atr"]
        trend = np.where(short_ma > long_ma, "Uptrend", np.where(short_ma < long_ma, "Downtrend", "Neutral"))
        strength = np.abs(short_ma - long_ma) / atr.replace(0, np.nan).fillna(0)
        return pd.Series([{"trend": t, "strength": float(s)} for t, s in zip(trend, strength)], index=self.df.index)

    def update_obv(self) -> pd.Series:
        direction = np.where(self.df["close"] > self.df["close"].shift(1), 1, np.where(self.df["close"] < self.df["close"].shift(1), -1, 0))
        return (direction * self.df["volume"]).cumsum()

    def update_rsi(self, new_close: float) -> pd.Series:
        delta = self.df["close"].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(window=self.config["rsi_period"], min_periods=1).mean()
        avg_loss = loss.rolling(window=self.config["rsi_period"], min_periods=1).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        return (100 - (100 / (1 + rs))).fillna(50.0)

    def update_stoch_rsi(self) -> pd.Series:
        rsi = self.indicator_values["rsi"]
        period = self.config["signal_config"]["stoch_rsi_period"]
        stoch = (rsi - rsi.rolling(window=period, min_periods=1).min()) / (rsi.rolling(window=period, min_periods=1).max() - rsi.rolling(window=period, min_periods=1).min() + 1e-10)
        return stoch.rolling(self.config["signal_config"]["stoch_rsi_k"], min_periods=1).mean()

    def update_macd(self, new_close: float) -> pd.DataFrame:
        alpha_fast = 2 / (self.config["signal_config"]["macd_fast"] + 1)
        alpha_slow = 2 / (self.config["signal_config"]["macd_slow"] + 1)
        alpha_signal = 2 / (self.config["signal_config"]["macd_signal"] + 1)
        last_fast = self.indicator_values["macd"]["macd"].iloc[-2] if len(self.indicator_values["macd"]) > 1 else new_close
        last_slow = self.indicator_values["macd"]["signal"].iloc[-2] if len(self.indicator_values["macd"]) > 1 else new_close
        new_fast = alpha_fast * new_close + (1 - alpha_fast) * last_fast
        new_slow = alpha_slow * new_close + (1 - alpha_slow) * last_slow
        macd = new_fast - new_slow
        last_signal = self.indicator_values["macd"]["signal"].iloc[-2] if len(self.indicator_values["macd"]) > 1 else macd
        new_signal = alpha_signal * macd + (1 - alpha_signal) * last_signal
        return pd.concat([self.indicator_values["macd"][:-1], pd.DataFrame({"macd": [macd], "signal": [new_signal], "histogram": [macd - new_signal]}, index=[self.df.index[-1]])])

    def update_bollinger_bands(self, new_close: float) -> pd.DataFrame:
        window = self.config["bollinger_bands_period"]
        if len(self.df) < window:
            sma = self.df["close"].mean()
            std = self.df["close"].std() or 0
        else:
            sma = self.df["close"].iloc[-window:].mean()
            std = self.df["close"].iloc[-window:].std()
        upper = sma + std * self.config["bollinger_bands_std_dev"]
        lower = sma - std * self.config["bollinger_bands_std_dev"]
        return pd.concat([self.indicator_values["bollinger_bands"][:-1], pd.DataFrame({"upper_band": [upper], "middle_band": [sma], "lower_band": [lower]}, index=[self.df.index[-1]])])

    def update_vwap(self) -> pd.Series:
        typical_price = (self.df["high"] + self.df["low"] + self.df["close"]) / 3
        return (typical_price * self.df["volume"]).cumsum() / self.df["volume"].cumsum()

    def update_atr(self) -> pd.Series:
        tr = pd.concat([self.df["high"] - self.df["low"], (self.df["high"] - self.df["close"].shift()).abs(), (self.df["low"] - self.df["close"].shift()).abs()], axis=1).max(axis=1)
        return tr.rolling(window=self.config["atr_period"], min_periods=1).mean()

    def calculate_indicators(self):
        self.indicator_values = {
            "ema": self.df["close"].ewm(span=self.config["signal_config"]["ema_period"], adjust=False).mean(),
            "mom": self.update_momentum(),
            "obv": self.update_obv(),
            "rsi": self.update_rsi(self.df["close"].iloc[-1]),
            "stoch_rsi_k": self.update_stoch_rsi(),
            "macd": self.df["close"].ewm(span=self.config["signal_config"]["macd_fast"], adjust=False).mean() - self.df["close"].ewm(span=self.config["signal_config"]["macd_slow"], adjust=False).mean(),
            "bollinger_bands": self.update_bollinger_bands(self.df["close"].iloc[-1]),
            "vwap": self.update_vwap(),
            "atr": self.update_atr(),
            "close": self.df["close"]
        }
        self.indicator_values["macd"] = pd.DataFrame({
            "macd": self.indicator_values["macd"],
            "signal": self.indicator_values["macd"].ewm(span=self.config["signal_config"]["macd_signal"], adjust=False).mean(),
            "histogram": self.indicator_values["macd"] - self.indicator_values["macd"].ewm(span=self.config["signal_config"]["macd_signal"], adjust=False).mean()
        }, index=self.df.index)

    def calculate_pivot_points(self) -> dict:
        high, low, close = self.df["high"].max(), self.df["low"].min(), self.df["close"].iloc[-1]
        pivot = (high + low + close) / 3
        return {"pivot": pivot, "r1": 2 * pivot - low, "s1": 2 * pivot - high, "r2": pivot + (high - low), "s2": pivot - (high - low)}

    async def analyze_and_output(self, current_price: float, logger: logging.Logger):
        if self.df.empty or len(self.df) < 2:
            self.logger.warning("Insufficient data for analysis.")
            return
        support_resistance = self.calculate_pivot_points()
        orderbook = data_cache.get(f"orderbook_{self.symbol}")
        signal = await analyze_market_data_signals(self.indicator_values, support_resistance, orderbook, self.config, self.df, current_price)
        indicators = {"Symbol": self.symbol, "Interval": self.interval}
        if signal != self.last_signal or time.time() - self.last_output_time >= OUTPUT_THROTTLE_SECONDS:
            self.last_output_time = await format_signal_output(signal, indicators, self.indicator_values, current_price, logger, self.last_output_time)
            self.last_signal = signal

# --- Main Functions ---

async def analyze_symbol(symbol: str, interval: str, logger: logging.Logger):
    analyzer = TradingAnalyzer(symbol, interval, CONFIG, logger)
    await websocket_stream(symbol, interval, analyzer, logger)

async def main():
    console.print(Panel("[bold cyan]Initiating Real-Time Trading Bot Sequence...[/bold cyan]", title="[bold magenta]WebWhale Scanner v2[/bold magenta]"))
    async with aiohttp.ClientSession() as session:
        valid_symbols = await fetch_valid_symbols(session, logging.getLogger("main"))
        if not valid_symbols:
            console.print("[bold red]Failed to fetch valid symbols. Exiting.[/bold red]")
            return

        symbols_input = console.input("[cyan]Enter trading pairs (e.g., TRUMPUSDT, ETHUSDT) separated by commas: [/cyan]").strip().upper()
        symbols = [s.strip() for s in symbols_input.split(",") if s.strip()]
        valid_input_symbols = [s for s in symbols if s in valid_symbols]
        invalid_symbols = set(symbols) - set(valid_input_symbols)
        if invalid_symbols:
            console.print(f"[bold red]Invalid symbols ignored: {', '.join(invalid_symbols)}. Valid symbols: {', '.join(valid_input_symbols)}[/bold red]")
        if not valid_input_symbols:
            console.print("[bold red]No valid symbols entered. Exiting.[/bold red]")
            return

        interval = CONFIG.get("interval", "15")
        if interval not in VALID_INTERVALS:
            console.print(f"[bold red]Invalid interval: {interval}. Using '15'.[/bold red]")
            interval = "15"

        tasks = []
        for symbol in valid_input_symbols:
            logger = setup_logger(symbol)
            logger.info(f"Starting real-time analysis for {symbol} on {interval}m interval")
            console.print(f"[cyan]Streaming {symbol} on {interval}m interval. Press Ctrl+C to stop.[/cyan]")
            tasks.append(analyze_symbol(symbol, interval, logger))

        try:
            await asyncio.gather(*tasks)
        except KeyboardInterrupt:
            console.print("[bold yellow]Bot interrupted. Shutting down.[/bold yellow]")
            for symbol in valid_input_symbols:
                logging.getLogger(symbol).info("Bot stopped manually.")

if __name__ == "__main__":
    asyncio.run(main())
