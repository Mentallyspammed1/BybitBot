#!/usr/bin/env python3

import os
import logging
import pandas as pd
import numpy as np
from datetime import datetime, date
import time
import asyncio
import ccxt.async_support as ccxt
from dotenv import load_dotenv
from typing import Any, Dict, List, Optional, Tuple, Union, Callable, Coroutine
from zoneinfo import ZoneInfo
from decimal import Decimal, getcontext
from logging.handlers import RotatingFileHandler
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.logging import RichHandler
from rich.theme import Theme
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from dataclasses import dataclass
from collections import defaultdict, deque
import argparse
import json
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import psutil
import aiofiles
import zstandard as zstd
import io
import random
import glob
import subprocess
import aiohttp

# --- Configuration and Constants ---

getcontext().prec = 10
load_dotenv()

API_KEY = os.getenv("BYBIT_API_KEY")
API_SECRET = os.getenv("BYBIT_API_SECRET")
if not API_KEY or not API_SECRET:
    raise ValueError("BYBIT_API_KEY and BYBIT_API_SECRET must be set in .env")

CONFIG_FILE = "config.json"
LOG_DIRECTORY = "bot_logs"
TIMEZONE = ZoneInfo("America/Chicago")
CACHE_TTL_SECONDS = 30  # Reduced from 60 - Mobile Optimization
DEFAULT_CACHE_SIZE = 200  # Reduced from 500 - Mobile Optimization
POLLING_INTERVAL = 15  # Adjusted base interval - Mobile Optimization
MAX_POSITION_SIZE = 2500.0  # Further reduced for mobile - Mobile Optimization
MAX_DAILY_LOSS = 250.0  # Reduced risk - Mobile Optimization
OUTPUT_THROTTLE_SECONDS = 120  # Increased to save battery - Mobile Optimization
MAX_API_RETRIES = 5

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
    "performance.summary": "bold cyan",
}))

os.makedirs(LOG_DIRECTORY, exist_ok=True)

# --- Termux Enhancement Classes ---

class TermuxResourceMonitor:
    """Monitors Termux resource usage with enhanced memory management."""
    def __init__(self):
        self.memory_threshold = 70  # Lower threshold for earlier warnings - Optimized
        self.storage_threshold = 75
        self.cpu_threshold = 65
        self.check_interval = 180  # 3 minutes - Optimized
        self.last_check = 0
        self.last_gc = 0
        self.gc_interval = 300  # 5 minutes - Optimized

    async def check_resources(self) -> dict:
        """Checks memory, swap, storage, and CPU usage. Initiates garbage collection if needed."""
        current_time = time.time()
        if current_time - self.last_check < self.check_interval:
            return getattr(self, 'last_status', {'warning': False, 'critical': False})

        try:
            # Invoke garbage collection to reclaim memory - A mystical cleansing
            if current_time - self.last_gc > self.gc_interval:
                import gc
                gc.collect()
                self.last_gc = current_time

            memory = psutil.virtual_memory()
            swap = psutil.swap_memory()
            disk = psutil.disk_usage('/')
            cpu = psutil.cpu_percent(interval=0.1)

            status = {
                'memory_used': memory.percent,
                'memory_available': memory.available / (1024 * 1024),  # MB - Clarity for mobile
                'swap_used': swap.percent,
                'storage_used': disk.percent,
                'storage_free': disk.free / (1024 * 1024 * 1024),  # GB - Clarity for mobile
                'cpu_used': cpu,
                'warning': False,
                'critical': False
            }

            # Smart thresholds - More sensitive for mobile
            if memory.percent > self.memory_threshold or swap.percent > 80:
                status['warning'] = True
            if memory.available < 100 * 1024 * 1024:  # Less than 100MB free - Critical threshold
                status['critical'] = True
            if disk.free < 500 * 1024 * 1024:  # Less than 500MB free - Critical threshold
                status['critical'] = True

            self.last_status = status
            self.last_check = current_time
            return status

        except Exception as e:
            return {'error': str(e), 'warning': True, 'critical': False}

    def format_status(self, status: dict) -> str:
        if 'error' in status:
            return f"[bold red]Resource monitoring error: {status['error']}[/bold red]"
        return f"Mem: {status['memory_used']:.1f}% | Swap: {status['swap_used']:.1f}% | Disk: {status['storage_used']:.1f}% | CPU: {status['cpu_used']:.1f}%"

class TermuxDataManager:
    def __init__(self, base_path: str = "/data/data/com.termux/files/home/trading_data"):
        self.base_path = base_path
        self.max_file_size = 5 * 1024 * 1024  # 5MB
        self.compression_level = 3

    async def save_data(self, data: dict, filename: str) -> bool:
        try:
            os.makedirs(self.base_path, exist_ok=True)
            filepath = os.path.join(self.base_path, f"{filename}.zst")
            json_str = json.dumps(data)
            compressed = zstd.compress(json_str.encode(), level=self.compression_level)
            if len(compressed) > self.max_file_size:
                return False
            async with aiofiles.open(filepath, 'wb') as f:
                await f.write(compressed)
            return True
        except Exception as e:
            logging.error(f"Data save error: {e}")
            return False

    async def load_data(self, filename: str) -> Optional[dict]:
        try:
            filepath = os.path.join(self.base_path, f"{filename}.zst")
            if not os.path.exists(filepath):
                return None
            async with aiofiles.open(filepath, 'rb') as f:
                compressed = await f.read()
            return json.loads(zstd.decompress(compressed).decode())
        except Exception as e:
            logging.error(f"Data load error: {e}")
            return None

class BatteryAwareTrading:
    def __init__(self):
        self.min_battery = 30
        self.critical_battery = 15
        self.check_interval = 600  # 10 minutes

    async def get_battery_status(self) -> Optional[dict]:
        try:
            output = subprocess.check_output(['termux-battery-status']).decode()
            status = json.loads(output)
            return {
                'percent': status['percentage'],
                'low_battery': status['percentage'] <= self.min_battery,
                'power_plugged': status['status'] in ['CHARGING', 'FULL']
            }
        except Exception:
            return None

    async def adjust_trading_params(self, analyzer: 'OptimizedTradingAnalyzer', battery_status: Optional[dict]) -> None:
        if not battery_status or time.time() - getattr(self, '_last_check', 0) < self.check_interval:
            return
        self._last_check = time.time()
        if battery_status['low_battery'] and not battery_status['power_plugged']:
            analyzer.polling_interval = min(60, analyzer.polling_interval * 2)
            analyzer.config["signal_config"]["base_risk"] *= 0.5
            if battery_status['percent'] <= self.critical_battery:
                analyzer.polling_interval = 120
                analyzer.config["signal_config"]["base_risk"] *= 0.25
            console.print(f"[yellow]Battery {battery_status['percent']}% - Adjusted polling to {analyzer.polling_interval}s[/yellow]")

class NetworkManager:
    """Manages network connections with adaptive polling and retry logic."""
    def __init__(self):
        self.retry_delay = 1.5  # Reduced initial delay - Optimized
        self.max_retries = 4 # Reduced max retries - Optimized
        self.connection_timeout = 2.5 # Reduced timeout - Optimized
        self.last_latency = 0
        self._session = None
        self._backoff_factor = 1.5
        self.latency_history = deque(maxlen=10) # Track latency history for adaptive polling - Optimized

    async def get_session(self) -> aiohttp.ClientSession:
        """Returns or creates an aiohttp client session."""
        if self._session is None or self._session.closed:
            timeout = aiohttp.ClientTimeout(total=self.connection_timeout)
            self._session = aiohttp.ClientSession(timeout=timeout)
        return self._session

    async def check_connection(self) -> bool:
        """Checks network connection and measures latency."""
        try:
            session = await self.get_session()
            start = time.time()
            async with session.get('https://api.bybit.com/v2/public/time') as resp:
                if resp.status == 200:
                    latency = time.time() - start
                    self.last_latency = latency
                    self.latency_history.append(latency) # Record latency - Optimized
                    return True
            return False
        except Exception as e:
            logging.error(f"Connection check failed: {e}")
            return False

    async def wait_for_connection(self) -> None:
        """Waits for a stable network connection before proceeding."""
        retries = 0
        console.print("[cyan]Awaiting network connection...[/cyan]") # Visual feedback
        while retries < self.max_retries:
            if await self.check_connection():
                console.print("[green]Network connected[/green]") # Visual feedback
                return
            retries += 1
            console.print(f"[yellow]Network check failed, retry {retries}/{self.max_retries}...[/yellow]") # Visual feedback
            await asyncio.sleep(self.retry_delay * (self._backoff_factor ** retries)) # Exponential backoff

        console.print("[bold red]Failed to establish network connection after multiple retries.[/bold red]") # Critical failure
        raise ConnectionError("No network after retries")

    def get_optimal_polling_interval(self, base_interval: float) -> float:
        """Adjusts polling interval based on network latency."""
        if not self.latency_history:
            return base_interval

        avg_latency = sum(self.latency_history) / len(self.latency_history)
        if avg_latency > 1.0:  # High latency
            return min(base_interval * 2, 60)  # Cap at 60 seconds - Conservative adjustment
        elif avg_latency < 0.2:  # Low latency
            return max(base_interval * 0.75, 5)  # Don't go below 5 seconds - Responsive but not too aggressive
        return base_interval

class TermuxNotificationManager:
    def __init__(self):
        self.notification_cmd = 'termux-notification'
        self.vibration_cmd = 'termux-vibrate'
        self.sound_cmd = 'termux-toast'  # Using toast for audio feedback - Mobile friendly

    async def send_notification(self, title: str, message: str, priority: str = 'high', sound: bool = True) -> None:
        try:
            subprocess.run([self.notification_cmd, '--title', title, '--content', message, '--priority', priority])
            if sound:
                subprocess.run([self.sound_cmd, '-s', f"{title}: {message[:20]}..."]) # Shortened message for toast
            subprocess.run([self.vibration_cmd, '-d', '500'])
        except Exception as e:
            logging.error(f"Notification error: {e}")

class MobileUI:
    def __init__(self, console: Console):
        self.console = console
        self.max_table_width = 50
        self.compact_mode = True

    def format_signal_table(self, signal: dict) -> Table:
        table = Table(title=f"{signal['signal_type']}", width=self.max_table_width, style="cyan") # Added style
        table.add_column("Param", width=10, style="magenta") # Added style
        table.add_column("Value", style="green") # Added style
        table.add_row("Entry", f"{signal['entry_price']:.4f}")
        table.add_row("SL", f"{signal['stop_loss']:.4f}")
        table.add_row("TP", f"{signal['take_profit']:.4f}")
        return table

    def display_status(self, status: dict) -> None:
        self.console.print(f"{status['symbol']}: [bold cyan]{status['price']:.4f}[/bold cyan] ([{self._get_change_color(status['change'])}]{status['change']:+.2f}%[/])")

    def _get_change_color(self, change: float) -> str:
        if change > 0.5:
            return "green"
        elif change < -0.5:
            return "red"
        else:
            return "yellow"

class DataCompressor:
    def __init__(self):
        self.compression_level = 3

    async def compress_dataframe(self, df: pd.DataFrame) -> bytes:
        if df.empty:
            return b''
        buffer = io.BytesIO()
        df.to_parquet(buffer, compression='zstd', compression_level=self.compression_level, engine='fastparquet')
        return buffer.getvalue()

    async def decompress_dataframe(self, data: bytes) -> pd.DataFrame:
        return pd.read_parquet(io.BytesIO(data))

class MemoryEfficientStorage:
    def __init__(self, max_items: int = 500):
        self.max_items = max_items
        self.data = {}
        self.timestamps = deque(maxlen=max_items)
        self._lock = asyncio.Lock()

    async def add_item(self, key: str, value: Any) -> None:
        async with self._lock:
            if key in self.data:
                self.timestamps.remove(key)
            elif len(self.timestamps) >= self.max_items:
                oldest_key = self.timestamps.popleft()
                del self.data[oldest_key]
            self.data[key] = value
            self.timestamps.append(key)

    async def get_item(self, key: str) -> Optional[Any]:
        async with self._lock:
            if key in self.data:
                self.timestamps.remove(key)
                self.timestamps.append(key)
                return self.data[key]
            return None

class PowerEfficientTasks:
    def __init__(self):
        self.tasks = []
        self.is_running = False
        self.power_save_mode = False
        self._lock = asyncio.Lock()

    async def add_task(self, coro: Coroutine, interval: float, power_intensive: bool = False) -> None:
        async with self._lock:
            self.tasks.append({'coro': coro, 'interval': interval, 'last_run': 0, 'power_intensive': power_intensive})

    async def run(self) -> None:
        self.is_running = True
        while self.is_running:
            current_time = time.time()
            async with self._lock:
                for task in self.tasks:
                    if current_time - task['last_run'] >= task['interval'] and (not self.power_save_mode or not task['power_intensive']):
                        await task['coro']()
                        task['last_run'] = current_time
            await asyncio.sleep(5 if self.power_save_mode else 1)

    def set_power_save_mode(self, enabled: bool) -> None:
        self.power_save_mode = enabled

class CrashRecoverySystem:
    def __init__(self, save_path: str = "recovery_data"):
        self.save_path = save_path
        self.checkpoint_interval = 600  # 10 minutes
        self.max_checkpoints = 2

    async def save_checkpoint(self, state: dict) -> None:
        os.makedirs(self.save_path, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = os.path.join(self.save_path, f"checkpoint_{timestamp}.zst")
        async with aiofiles.open(filepath, 'wb') as f:
            await f.write(zstd.compress(json.dumps(state).encode()))
        checkpoints = sorted(glob.glob(os.path.join(self.save_path, "checkpoint_*.zst")))
        while len(checkpoints) > self.max_checkpoints:
            os.remove(checkpoints.pop(0))

    async def load_latest_checkpoint(self) -> Optional[dict]:
        checkpoints = sorted(glob.glob(os.path.join(self.save_path, "checkpoint_*.zst")))
        if not checkpoints:
            return None
        async with aiofiles.open(checkpoints[-1], 'rb') as f:
            return json.loads(zstd.decompress(await f.read()).decode())

class DataOptimizer:
    """Optimizes DataFrames for memory efficiency on mobile devices."""
    def __init__(self):
        self.chunk_size = 100  # Further reduced - Mobile Optimization
        self.max_rows = 1000  # Reduced history - Mobile Optimization
        self.precision = 4
        self._dtypes = {
            'open': 'float32',
            'high': 'float32',
            'low': 'float32',
            'close': 'float32',
            'volume': 'float32',
            'start_time': 'datetime64[ns]' # Ensure datetime is efficient - Optimized
        }

    async def process_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Processes and optimizes a DataFrame for memory usage."""
        if df.empty:
            return df

        # Optimize memory usage by converting to efficient dtypes - Mobile memory magic
        for col, dtype in self._dtypes.items():
            if col in df.columns:
                df[col] = df[col].astype(dtype)

        # Remove unnecessary columns - Streamlining data for mobile
        essential_columns = set(self._dtypes.keys())
        df = df[[col for col in df.columns if col in essential_columns]]

        # Trim to max rows - Limiting history for mobile performance
        if len(df) > self.max_rows:
            df = df.tail(self.max_rows)

        return df.copy()

    def estimate_memory_usage(self, df: pd.DataFrame) -> float:
        """Estimates DataFrame memory usage in MB."""
        return df.memory_usage(deep=True).sum() / (1024 * 1024)

class PerformanceMonitor:
    """Monitors and logs performance metrics."""
    def __init__(self):
        self.start_time = time.time()
        self.operation_times = defaultdict(deque, maxlen=100)
        self.memory_samples = deque(maxlen=60)  # 1-hour of minute samples - Mobile friendly sampling
        self._last_memory_sample = 0
        self.sample_interval = 60  # 1 minute

    async def record_operation(self, operation: str, duration: float):
        """Records the duration of an operation."""
        self.operation_times[operation].append(duration)

    async def sample_memory(self):
        """Samples and records memory usage periodically."""
        current_time = time.time()
        if current_time - self._last_memory_sample >= self.sample_interval:
            memory = psutil.Process().memory_info()
            self.memory_samples.append({
                'timestamp': current_time,
                'rss': memory.rss / (1024 * 1024),  # MB
                'vms': memory.vms / (1024 * 1024)   # MB
            })
            self._last_memory_sample = current_time

    def get_statistics(self) -> dict:
        """Calculates and returns performance statistics."""
        stats = {
            'uptime': time.time() - self.start_time,
            'operations': {},
            'memory': {
                'current': psutil.Process().memory_info().rss / (1024 * 1024),
                'average': sum(s['rss'] for s in self.memory_samples) / len(self.memory_samples) if self.memory_samples else 0
            }
        }

        for op, times in self.operation_times.items():
            if times:
                stats['operations'][op] = {
                    'avg': sum(times) / len(times),
                    'min': min(times),
                    'max': max(times),
                    'count': len(times)
                }

        return stats

class OptimizedLogger:
    def __init__(self, max_buffer_size: int = 50):
        self.max_buffer_size = max_buffer_size
        self.buffer = deque(maxlen=max_buffer_size)
        self._lock = asyncio.Lock()
        self.log_file = os.path.join(LOG_DIRECTORY, "trading.log")

    async def log(self, message: str, level: str = "info"):
        entry = f"{datetime.now(TIMEZONE).isoformat()} - {level.upper()} - {message}\n"
        async with self._lock:
            self.buffer.append(entry)
            if len(self.buffer) >= self.max_buffer_size:
                await self.flush()

    async def flush(self):
        if not self.buffer:
            return
        async with self._lock:
            entries = list(self.buffer)
            self.buffer.clear()
        async with aiofiles.open(self.log_file, 'a') as f:
            await f.writelines(entries)

# --- Core Classes (Optimized Versions) ---

class OptimizedDataCache:
    def __init__(self, ttl: int = CACHE_TTL_SECONDS, max_size: int = DEFAULT_CACHE_SIZE):
        self.cache = {}
        self.timestamps = deque(maxlen=max_size)
        self.ttl = ttl
        self._lock = asyncio.Lock()

    async def get(self, key: str) -> Optional[Any]:
        async with self._lock:
            if key in self.cache and time.time() - self.cache[key][1] < self.ttl:
                return self.cache[key][0]
            return None

    async def set(self, key: str, value: Any) -> None:
        async with self._lock:
            if len(self.timestamps) >= self.timestamps.maxlen:
                oldest_key = self.timestamps.popleft()
                del self.cache[oldest_key]
            self.cache[key] = (value, time.time())
            self.timestamps.append(key)

async def initialize_exchange() -> ccxt.bybit:
    exchange = ccxt.bybit({
        'apiKey': API_KEY,
        'secret': API_SECRET,
        'enableRateLimit': True,
        'options': {'defaultType': 'linear'}
    })
    await exchange.load_markets()
    return exchange

@retry(stop=stop_after_attempt(MAX_API_RETRIES), wait=wait_exponential(multiplier=1, min=2, max=10), retry=retry_if_exception_type((ccxt.NetworkError, ccxt.ExchangeError)))
async def fetch_klines(exchange: ccxt.bybit, symbol: str, interval: str, limit: int = 200) -> pd.DataFrame:
    try:
        ohlcv = await exchange.fetch_ohlcv(f"{symbol}/USDT", timeframe=interval, limit=limit)
        df = pd.DataFrame(ohlcv, columns=["start_time", "open", "high", "low", "close", "volume"])
        df["start_time"] = pd.to_datetime(df["start_time"], unit="ms")
        return df
    except (ccxt.NetworkError, ccxt.ExchangeError) as e:
        await logger.log(f"API error for {symbol}: {e}", "error") # Log API errors
        raise  # Re-raise for retry

class VolatilityAnalyzer:
    def __init__(self, window: int = 20):
        self.window = window
        self.volatility_value = 0.0

    def analyze_volatility(self, df: pd.DataFrame) -> dict:
        if len(df) < self.window:
            return {'is_volatile': False, 'volatility_ratio': 0, 'current_volatility': 0}
        returns = np.log(df['close'] / df['close'].shift(1))
        rolling_std = returns.rolling(window=self.window).std()
        self.volatility_value = rolling_std.iloc[-1] if not pd.isna(rolling_std.iloc[-1]) else 0
        avg_vol = rolling_std.mean() or 0
        return {
            'is_volatile': self.volatility_value > (avg_vol * 2),
            'volatility_ratio': self.volatility_value / avg_vol if avg_vol != 0 else 0,
            'current_volatility': self.volatility_value
        }

class OptimizedTradingAnalyzer:
    """Analyzes market data and generates trading signals with resource awareness."""
    def __init__(self, symbol: str, interval: str, config: dict, exchange: ccxt.bybit, logger: OptimizedLogger):
        self.symbol = symbol
        self.interval = interval
        self.config = config
        self.exchange = exchange
        self.logger = logger
        self.df = pd.DataFrame()
        self.indicator_values = {}
        self.last_kline_time = None
        self.last_output_time = 0.0
        self.polling_interval = POLLING_INTERVAL # Base polling interval
        self.cache = OptimizedDataCache()
        self.executor = ThreadPoolExecutor(max_workers=2)  # Reduced workers for mobile - Optimized
        self.vol_analyzer = VolatilityAnalyzer()
        self.data_optimizer = DataOptimizer()
        self.memory_storage = MemoryEfficientStorage()
        self._shutdown = False

    async def cleanup(self):
        """Cleanup resources when analyzer is stopped."""
        self._shutdown = True
        self.executor.shutdown(wait=True)
        self.df = pd.DataFrame()  # Release DataFrame memory
        self.indicator_values.clear()  # Clear indicator data
        await self.cache.clear()  # Clear cache

    async def update_data(self, new_df: pd.DataFrame):
        """Updates historical data and recalculates indicators."""
        if self._shutdown:
            return

        try:
            optimized_df = await self.data_optimizer.process_dataframe(new_df) # Optimize new data - Memory efficiency
            if self.df.empty or optimized_df["start_time"].iloc[-1] > self.last_kline_time:
                self.df = pd.concat([self.df, optimized_df]).drop_duplicates(subset="start_time").tail(200) # Limit DataFrame size - Memory efficiency
                self.last_kline_time = self.df["start_time"].iloc[-1]
                await self._calculate_indicators()
        except Exception as e:
            await self.logger.log(f"Error updating data: {e}", "error")

    async def _calculate_indicators(self):
        """Calculates technical indicators asynchronously."""
        tasks = [
            self.executor.submit(self._calculate_ema),
            self.executor.submit(self._calculate_rsi),
            self.executor.submit(self._calculate_atr)
        ]
        results = await asyncio.gather(*[asyncio.to_thread(f.result) for f in tasks])
        self.indicator_values.update({"ema": results[0], "rsi": results[1], "atr": results[2], "close": self.df["close"]})
        await self.memory_storage.add_item(f"{self.symbol}_{self.interval}_indicators", self.indicator_values) # Cache indicators

    def _calculate_ema(self) -> pd.Series:
        """Calculates Exponential Moving Average."""
        period = self.config["indicators"]["ema_alignment"]["period"]
        alpha = 2 / (period + 1)
        ema = [self.df["close"].iloc[0]]
        for price in self.df["close"].iloc[1:]:
            ema.append(ema[-1] + alpha * (price - ema[-1]))
        return pd.Series(ema, index=self.df.index)

    def _calculate_rsi(self) -> pd.Series:
        """Calculates Relative Strength Index."""
        period = self.config["indicators"]["rsi"]["period"]
        delta = self.df["close"].diff()
        gain = delta.where(delta > 0, 0).rolling(window=period, min_periods=1).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period, min_periods=1).mean()
        rs = gain / loss.replace(0, np.nan)
        return pd.Series(100 - (100 / (1 + rs)), index=self.df.index).fillna(50.0)

    def _calculate_atr(self) -> pd.Series:
        """Calculates Average True Range."""
        atr_period = self.config.get("atr_period", 14) # Default ATR period
        tr = pd.concat([self.df["high"] - self.df["low"], (self.df["high"] - self.df["close"].shift()).abs(), (self.df["low"] - self.df["close"].shift()).abs()], axis=1).max(axis=1)
        return pd.Series(tr.rolling(window=atr_period, min_periods=1).mean(), index=self.df.index)

    async def analyze_and_output(self, current_price: float):
        """Analyzes market conditions and outputs trading signals if generated."""
        if len(self.df) < 2:
            return
        signal = await self._generate_signal(current_price)
        if signal and time.time() - self.last_output_time >= OUTPUT_THROTTLE_SECONDS:
            table = mobile_ui.format_signal_table(signal)
            console.print(Panel(table, title="[bold blue]Trade Signal[/bold blue]")) # Styled panel title
            await notification_manager.send_notification(f"{self.symbol} {signal['signal_type']}", f"Price: {current_price:.4f}")
            self.last_output_time = time.time()

    async def _generate_signal(self, current_price: float) -> Optional[dict]:
        """Generates a trading signal based on indicators and price action."""
        ema = self.indicator_values["ema"].iloc[-1]
        rsi = self.indicator_values["rsi"].iloc[-1]
        atr = self.indicator_values["atr"].iloc[-1]
        score = (1 if current_price > ema else -1) + (1 if rsi < 30 else -1 if rsi > 70 else 0)
        signal_type = "Long" if score > 0 else "Short" if score < 0 else None
        if not signal_type:
            return None
        stop_loss = current_price - atr * 2 if signal_type == "Long" else current_price + atr * 2
        take_profit = current_price + atr * 4 if signal_type == "Long" else current_price - atr * 4
        return {
            "signal_type": signal_type,
            "entry_price": current_price,
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "confidence": "Medium",
            "normalized_score": score / 2.0,
            "Symbol": self.symbol
        }

# --- Main Functions ---

async def analyze_symbol(symbol: str, interval: str, exchange: ccxt.bybit, logger: OptimizedLogger):
    """Analyzes a specific trading symbol and interval."""
    analyzer = OptimizedTradingAnalyzer(symbol, interval, CONFIG, exchange, logger)
    initial_df = await fetch_klines(exchange, symbol, interval)
    await analyzer.update_data(initial_df)
    await logger.log(f"Started analysis for {symbol} on {interval}") # Log start

    subprocess.run(['termux-wake-lock'])  # Keep device awake during trading - Mobile consideration
    try:
        while True:
            resource_status = await resource_monitor.check_resources() # Check resource usage regularly
            if resource_status.get('critical', False):
                await logger.log("Critical resource usage detected, pausing analysis", "warning") # Log critical resource
                power_tasks.set_power_save_mode(True) # Enter power save mode
                await asyncio.sleep(300) # Longer sleep in critical state
                continue # Skip trading operations in critical state

            battery_status = await battery_monitor.get_battery_status() # Check battery status
            if battery_status:
                await battery_monitor.adjust_trading_params(analyzer, battery_status) # Adjust params based on battery

            df = await fetch_klines(exchange, symbol, interval, limit=1) # Fetch latest kline
            if not df.empty:
                current_price = df["close"].iloc[-1]
                await analyzer.update_data(df) # Update data with new kline
                await analyzer.analyze_and_output(current_price) # Analyze and output signals
                await logger.flush() # Flush logs periodically
                await perf_monitor.record_operation("analyze_symbol", time.time() - perf_monitor.start_time) # Record operation time
                await perf_monitor.sample_memory() # Sample memory usage
            await asyncio.sleep(network_manager.get_optimal_polling_interval(analyzer.polling_interval)) # Adaptive polling
    finally:
        subprocess.run(['termux-wake-unlock']) # Release wake lock on exit
        await analyzer.cleanup() # Cleanup analyzer resources
        await logger.log(f"Stopped analysis for {symbol} on {interval}", "info") # Log stop

CONFIG = {}
resource_monitor = TermuxResourceMonitor()
data_manager = TermuxDataManager()
battery_monitor = BatteryAwareTrading()
network_manager = NetworkManager()
notification_manager = TermuxNotificationManager()
mobile_ui = MobileUI(console)
power_tasks = PowerEfficientTasks()
logger = OptimizedLogger()
perf_monitor = PerformanceMonitor() # Initialize performance monitor

async def load_config(config_file: str, check_only: bool = False) -> dict:
    """Loads configuration from JSON file, with validation and optional check-only mode."""
    try:
        async with aiofiles.open(config_file, 'r') as f:
            config = json.loads(await f.read())
            if check_only:
                console.print("[green]Configuration file is valid JSON.[/green]")
                return {} # Return empty dict in check-only mode
            # Basic validation - Expand as needed
            if not isinstance(config.get("symbols"), list) or not config.get("signal_config"):
                raise ValueError("Invalid configuration format.")
            return config
    except FileNotFoundError:
        console.print(f"[bold red]Configuration file '{config_file}' not found.[/bold red]")
        raise
    except json.JSONDecodeError:
        console.print(f"[bold red]Error decoding JSON in '{config_file}'.[/bold red]")
        raise
    except ValueError as e:
        console.print(f"[bold red]Configuration error: {e}[/bold red]")
        raise

async def main(check_config_only: bool = False, cli_symbol: Optional[str] = None, cli_interval: Optional[str] = None):
    """Main function to initialize and run the trading bot."""
    try:
        # Initialize performance monitoring - Start monitoring early
        perf_monitor = PerformanceMonitor()

        # Load configuration - Load config first
        global CONFIG
        CONFIG = await load_config(CONFIG_FILE, check_config_only)
        if check_config_only:
            return

        # Initialize system components - Order matters
        await network_manager.wait_for_connection() # Ensure network is up
        exchange = await initialize_exchange() # Initialize exchange after network
        logger = OptimizedLogger() # Initialize logger

        # Check initial resources - Check resources before starting
        resource_status = await resource_monitor.check_resources()
        if resource_status.get('critical', True):
            console.print("[bold red]Insufficient resources to start trading.[/bold red]")
            return

        try:
            # Get symbols - User input for symbols
            symbols = [cli_symbol] if cli_symbol else console.input(
                "[cyan]Enter trading pairs (e.g., BTC,ETH): [/cyan]"
            ).strip().upper().split(",")

            # Validate interval - Ensure interval is valid
            interval = cli_interval if cli_interval in VALID_INTERVALS else "15m"

            # Start analysis tasks - Run analyzers concurrently
            tasks = []
            for symbol in symbols:
                analyzer_task = analyze_symbol(symbol, interval, exchange, logger)
                tasks.append(analyzer_task)

            # Enable wake lock - Keep screen on during trading
            subprocess.run(['termux-wake-lock'])
            console.print("[green]Trading bot started, monitoring symbols...[/green]") # Startup message

            # Run all analysis tasks concurrently - Unleash the magic
            await asyncio.gather(*tasks)

        except KeyboardInterrupt:
            console.print("[yellow]Graceful shutdown initiated...[/yellow]") # Graceful shutdown message
        finally:
            # Cleanup resources - Ensure resources are released
            subprocess.run(['termux-wake-unlock']) # Release wake lock
            await exchange.close() # Close exchange connection
            await logger.flush() # Flush remaining logs

            # Display performance summary - Show performance metrics at the end
            stats = perf_monitor.get_statistics()
            console.print(Panel(
                f"[bold]Uptime[/bold]: {stats['uptime']:.2f}s\n"
                f"[bold]Memory Usage[/bold]: [bold cyan]{stats['memory']['current']:.1f}MB[/bold cyan] (Avg: {stats['memory']['average']:.1f}MB)", # Styled memory info
                title="[bold cyan]Performance Summary[/bold cyan]", # Styled panel title
                style="performance.summary" # Apply theme style
            ))
            console.print("[green]Trading bot stopped.[/green]") # Stop confirmation

    except Exception as e:
        console.print(f"[bold red]Critical error: {str(e)}[/bold red]") # Critical error display
        raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="WebWhale Scanner for Termux")
    parser.add_argument('-c', '--check-config', action='store_true', help='Check configuration file validity')
    parser.add_argument('-s', '--symbol', type=str, help='Specify a single trading symbol to analyze')
    parser.add_argument('-i', '--interval', type=str, help='Specify the interval (e.g., 1m, 15m, 1h)')
    args = parser.parse_args()

    try:
        asyncio.run(main(args.check_config, args.symbol, args.interval))
    except KeyboardInterrupt:
        console.print("\n[yellow]Trading bot stopped by user[/yellow]")
    except Exception as e:
        console.print(f"\n[bold red]Fatal error: {str(e)}[/bold red]")
    finally:
        console.print("[green]Trading session ended[/green]")
