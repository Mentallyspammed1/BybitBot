# -*- coding: utf-8 -*-
"""
Neonta Trading Bot - Technical Analysis Tool for Bybit Exchange.

This script fetches market data from Bybit, performs technical analysis using
various indicators, outputs trading insights, and generates potential scalping
signals with entry, stop-loss, and confidence levels.

Disclaimer: This is for educational and informational purposes only.
Trading involves risk. Past performance is not indicative of future results.
Use at your own risk.
"""

import hashlib
import hmac
import json
import logging
import os
import time
from datetime import datetime, timezone
from decimal import ROUND_DOWN, ROUND_UP, Decimal, getcontext
from enum import Enum
from logging.handlers import RotatingFileHandler
from typing import Dict, Final, List, Optional, Tuple, Union
from zoneinfo import ZoneInfo

import ccxt
import numpy as np
import pandas as pd
import requests
from colorama import Fore, Style, init
from dotenv import load_dotenv
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# --- Initial Setup ---
# Set precision for Decimal calculations (adjust if needed)
getcontext().prec = 18  # Increased precision for crypto prices
# Initialize colorama
init(autoreset=True)
# Load environment variables from .env file
load_dotenv()

# --- Constants ---
# API Keys and Secrets - Ensure these are set in your .env file
API_KEY: Final[Optional[str]] = os.getenv("BYBIT_API_KEY")
API_SECRET: Final[Optional[str]] = os.getenv("BYBIT_API_SECRET")
if not API_KEY or not API_SECRET:
    raise ValueError("BYBIT_API_KEY and BYBIT_API_SECRET must be set in .env")

# Base URL for Bybit API
BASE_URL: Final[str] = os.getenv("BYBIT_BASE_URL", "https://api.bybit.com")

# Configuration File and Logging Directories
CONFIG_FILE: Final[str] = "config.json"
LOG_DIRECTORY: Final[str] = "bot_logs"
# Define local timezone for display purposes
try:
    LOCAL_TIMEZONE: Final[ZoneInfo] = ZoneInfo("America/Chicago")  # Or your local timezone
except Exception:
    LOCAL_TIMEZONE: Final[ZoneInfo] = ZoneInfo("Etc/UTC")
    print(
        f"{Fore.YELLOW}Warning: Could not load America/Chicago timezone."
        f" Using UTC for display.{Style.RESET_ALL}"
    )


# API Request Settings
MAX_API_RETRIES: Final[int] = 3
RETRY_DELAY_SECONDS: Final[int] = 5
# Codes triggering retries for requests library
RETRY_ERROR_CODES: Final[List[int]] = [429, 500, 502, 503, 504]

# Valid time intervals for Kline data (using Enum for clarity)
class BybitInterval(Enum):
    MIN_1 = "1"
    MIN_3 = "3"
    MIN_5 = "5"
    MIN_15 = "15"
    MIN_30 = "30"
    HOUR_1 = "60"
    HOUR_2 = "120"
    HOUR_4 = "240"
    DAY_1 = "D"
    WEEK_1 = "W"
    MONTH_1 = "M"

    @classmethod
    def values(cls) -> List[str]:
        return [item.value for item in cls]


VALID_INTERVALS: Final[List[str]] = BybitInterval.values()

# Color Constants for console output
NEON_GREEN: Final = Fore.LIGHTGREEN_EX
NEON_BLUE: Final = Fore.CYAN
NEON_PURPLE: Final = Fore.MAGENTA
NEON_YELLOW: Final = Fore.YELLOW
NEON_RED: Final = Fore.LIGHTRED_EX
RESET: Final = Style.RESET_ALL

# Ensure log directory exists
os.makedirs(LOG_DIRECTORY, exist_ok=True)

# Default configuration structure
DEFAULT_CONFIG: Final[Dict] = {
    "interval": "15",
    "analysis_interval_seconds": 30,
    "api_retry_delay_seconds": 5,
    "momentum_period": 10,
    "momentum_ma_short": 12,
    "momentum_ma_long": 26,
    "volume_ma_period": 20,
    "atr_period": 14,
    "trend_strength_threshold": 0.4,
    "sideways_atr_multiplier": 1.5,
    "rsi_period": 14,
    "stoch_rsi_period": 14,
    "stoch_rsi_k": 3,
    "stoch_rsi_d": 3,
    "stoch_rsi_stoch_window": 12,
    "rsi_oversold": 30,
    "rsi_overbought": 70,
    "stoch_rsi_oversold": 20,
    "stoch_rsi_overbought": 80,
    "bollinger_bands_period": 20,
    "bollinger_bands_std_dev": 2.0,
    "orderbook_limit": 50,
    "orderbook_cluster_threshold": 1000.0,
    "ccxt_recv_window_ms": 10000,
    "scalping": {
        "enabled": True,
        "sl_atr_multiplier": 1.5,
        "min_confidence_level": 50,
        "rsi_scalp_oversold": 35,
        "rsi_scalp_overbought": 65,
        "stoch_rsi_scalp_oversold": 25,
        "stoch_rsi_scalp_overbought": 75,
        "max_signals": 2,
    },
}

# --- Logging Setup ---

class SensitiveFormatter(logging.Formatter):
    """Custom logging formatter to mask API keys and secrets in logs."""

    def format(self, record: logging.LogRecord) -> str:
        """Mask sensitive info before formatting."""
        msg = super().format(record)
        api_key_str = str(API_KEY) if API_KEY else ""
        api_secret_str = str(API_SECRET) if API_SECRET else ""
        # Mask even if they appear partially
        if api_key_str:
            msg = msg.replace(api_key_str, "***API_KEY***")
        if api_secret_str:
            msg = msg.replace(api_secret_str, "***API_SECRET***")
        return msg


def setup_logger(symbol: str) -> logging.Logger:
    """Set up a logger with file and stream handlers for a specific symbol."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = os.path.join(LOG_DIRECTORY, f"{symbol}_{timestamp}.log")
    logger = logging.getLogger(symbol)
    logger.setLevel(logging.DEBUG)  # Set logger level to capture DEBUG

    # Prevent adding multiple handlers if logger already exists
    if logger.hasHandlers():
        logger.handlers.clear()

    # File handler (logs DEBUG and above)
    file_handler = RotatingFileHandler(
        log_filename, maxBytes=10 * 1024 * 1024, backupCount=5, encoding="utf-8"
    )
    file_handler.setLevel(logging.DEBUG) # Log detailed info to file
    file_formatter = SensitiveFormatter(
        "%(asctime)s - %(levelname)s - [%(name)s:%(funcName)s:%(lineno)d] - %(message)s"
    )
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    # Stream handler (logs INFO and above to console)
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO) # Console level set to INFO
    stream_formatter = SensitiveFormatter(
        NEON_BLUE + "%(asctime)s" + RESET + " [%(levelname)s] %(message)s"
    )
    stream_handler.setFormatter(stream_formatter)
    logger.addHandler(stream_handler)

    logger.propagate = False
    return logger


# --- Configuration Loading ---

def load_config(filepath: str) -> Dict:
    """Load config from JSON, create/merge with defaults if needed."""
    config = DEFAULT_CONFIG.copy()

    if os.path.exists(filepath):
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                loaded_config = json.load(f)

                # Deep merge (simple version for one level nesting)
                for key, value in loaded_config.items():
                    if isinstance(value, dict) and isinstance(config.get(key), dict):
                        config[key].update(value)
                    else:
                        # Warn if loaded type mismatches default type
                        default_val = config.get(key)
                        if default_val is not None and not isinstance(value, type(default_val)):
                             print(
                                f"{NEON_YELLOW}Warning: Config type mismatch for '{key}'. "
                                f"Expected {type(default_val).__name__}, got {type(value).__name__}. "
                                f"Using loaded value: {value}{RESET}"
                            )
                        config[key] = value
                print(f"{NEON_GREEN}Loaded configuration from {filepath}{RESET}")

        except (FileNotFoundError, json.JSONDecodeError, TypeError) as e:
            print(
                f"{NEON_YELLOW}Could not load/parse config '{filepath}': {e}. "
                f"Using defaults/creating file.{RESET}"
            )
            if not os.path.exists(filepath) or isinstance(e, json.JSONDecodeError):
                try:
                    with open(filepath, "w", encoding="utf-8") as f:
                        json.dump(config, f, indent=2)
                    print(
                        f"{NEON_YELLOW}Created/Overwrote config file with defaults: {filepath}{RESET}"
                    )
                except IOError as io_e:
                    print(
                        f"{NEON_RED}Could not create config file '{filepath}': {io_e}{RESET}"
                    )
    else:
        try:
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(config, f, indent=2)
            print(f"{NEON_YELLOW}Created new config file with defaults: {filepath}{RESET}")
        except IOError as io_e:
            print(f"{NEON_RED}Could not create config file '{filepath}': {io_e}{RESET}")

    return config


# Load config globally once
CONFIG: Final[Dict] = load_config(CONFIG_FILE)


# --- API Request Handling ---

def create_session() -> requests.Session:
    """Create a requests session with retry logic."""
    session = requests.Session()
    retries = Retry(
        total=MAX_API_RETRIES,
        backoff_factor=0.5,
        status_forcelist=RETRY_ERROR_CODES,
        allowed_methods=["GET", "POST"],
    )
    adapter = HTTPAdapter(max_retries=retries)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session


def bybit_request(
    session: requests.Session,
    method: str,
    endpoint: str,
    params: Optional[Dict] = None,
    logger: Optional[logging.Logger] = None,
) -> Optional[Dict]:
    """
    Send a signed request to Bybit API v5 using a provided session.

    Args:
        session: The requests.Session object to use.
        method: HTTP method ('GET' or 'POST').
        endpoint: API endpoint path (e.g., '/v5/market/kline').
        params: Dictionary of parameters for the request.
        logger: Logger instance for logging errors/info.

    Returns:
        Parsed JSON response dictionary if successful, None otherwise.
    """
    params = params or {}
    if not API_KEY or not API_SECRET: # Re-check just in case
         if logger: logger.critical("API Key or Secret not found during request.")
         return None

    try:
        timestamp_ms = str(int(datetime.now(timezone.utc).timestamp() * 1000))
        recv_window = str(CONFIG.get("ccxt_recv_window_ms", 10000))

        # Signature generation logic for V5
        if method == "GET":
            query_string = "&".join(
                f"{key}={value}" for key, value in sorted(params.items())
            )
            payload_str = timestamp_ms + API_KEY + recv_window + query_string
        elif method == "POST":
            # V5 typically uses JSON body for POST, sign the body
            body_string = json.dumps(params, separators=(",", ":")) if params else ""
            payload_str = timestamp_ms + API_KEY + recv_window + body_string
        else:
            if logger:
                logger.error(f"Unsupported HTTP method: {method}")
            return None

        signature = hmac.new(
            API_SECRET.encode("utf-8"), payload_str.encode("utf-8"), hashlib.sha256
        ).hexdigest()

        headers = {
            "X-BAPI-API-KEY": API_KEY,
            "X-BAPI-TIMESTAMP": timestamp_ms,
            "X-BAPI-SIGN": signature,
            "X-BAPI-RECV-WINDOW": recv_window,
            "X-BAPI-SIGN-TYPE": "2",  # Required for HMAC SHA256
            "Content-Type": "application/json",
        }
        url = f"{BASE_URL}{endpoint}"

        request_kwargs = {"method": method, "url": url, "headers": headers, "timeout": 15} # Increased timeout slightly

        if method == "GET":
            request_kwargs["params"] = params
        elif method == "POST":
            request_kwargs["json"] = params # Send params as JSON body

        response = session.request(**request_kwargs)
        response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)
        json_response = response.json()

        # Check Bybit's specific return code
        ret_code = json_response.get("retCode")
        ret_msg = json_response.get("retMsg", "")

        if ret_code == 0:
            return json_response
        else:
            # Log timestamp mismatch errors specifically
            if ret_code == 10002 and "timestamp" in ret_msg.lower():
                error_msg = (
                    f"Bybit API Timestamp Error (10002): {ret_msg}. "
                    "CHECK SYSTEM CLOCK! Your time might be too far off server time."
                )
                if logger:
                    logger.error(
                        f"{NEON_RED}{error_msg}{RESET} (Using recv_window={recv_window}ms)"
                    )
            else:
                error_msg = f"Bybit API Error: {ret_code} - {ret_msg}"
                if logger:
                    logger.error(
                        f"{NEON_RED}{error_msg}{RESET} (Endpoint: {endpoint}, Params: {params})"
                    )
            return None

    except requests.exceptions.RequestException as e:
        if logger:
            logger.error(f"{NEON_RED}API Request Failed: {e}{RESET} (Endpoint: {endpoint})")
        return None
    except Exception as e:
        if logger:
            logger.exception(
                f"{NEON_RED}Unexpected error during Bybit request: {e}{RESET} (Endpoint: {endpoint})"
            )
        return None


# --- Data Fetching Functions ---

def fetch_current_price(
    session: requests.Session, symbol: str, logger: logging.Logger
) -> Optional[Decimal]:
    """Fetch the current last traded price for a given symbol."""
    endpoint = "/v5/market/tickers"
    params = {"category": "linear", "symbol": symbol}
    response = bybit_request(session, "GET", endpoint, params, logger)

    if not response or "result" not in response or "list" not in response["result"]:
        logger.error(f"{NEON_RED}Invalid response structure fetching ticker for {symbol}{RESET}")
        return None

    tickers = response["result"]["list"]
    if not tickers:
        logger.warning(f"{NEON_YELLOW}Ticker list empty for {symbol}{RESET}")
        return None

    found_ticker = next((t for t in tickers if t.get("symbol") == symbol), None)

    if not found_ticker:
        logger.error(f"{NEON_RED}Symbol {symbol} not found in ticker response list.{RESET}")
        return None

    last_price_str = found_ticker.get("lastPrice")
    if not last_price_str:
        logger.error(f"{NEON_RED}No 'lastPrice' in ticker data for {symbol}{RESET}")
        return None

    try:
        price = Decimal(last_price_str)
        if not price.is_finite():
            logger.error(
                f"{NEON_RED}Invalid decimal value for 'lastPrice': {last_price_str}{RESET}"
            )
            return None
        return price
    except Exception as e:
        logger.error(
            f"{NEON_RED}Error parsing 'lastPrice' ('{last_price_str}') for {symbol}: {e}{RESET}"
        )
        return None


def fetch_klines(
    session: requests.Session,
    symbol: str,
    interval: str,
    limit: int = 200,
    logger: logging.Logger = None,
) -> pd.DataFrame:
    """Fetch Kline (candlestick) data."""
    endpoint = "/v5/market/kline"
    params = {
        "symbol": symbol,
        "interval": interval,
        "limit": limit,
        "category": "linear",
    }
    response = bybit_request(session, "GET", endpoint, params, logger)

    if not response or "result" not in response or "list" not in response["result"]:
        if logger:
            logger.error(f"Invalid response structure fetching klines for {symbol}")
        return pd.DataFrame()

    data = response["result"]["list"]
    if not data:
        if logger:
            logger.warning(f"Kline data list empty for {symbol} interval {interval}.")
        return pd.DataFrame()

    columns = ["start_time", "open", "high", "low", "close", "volume", "turnover"]
    df = pd.DataFrame(data, columns=columns)

    try:
        df["start_time"] = pd.to_numeric(df["start_time"])
        df["start_time"] = pd.to_datetime(df["start_time"], unit="ms", utc=True)

        numeric_cols = ["open", "high", "low", "close", "volume", "turnover"]
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        essential_cols = ["open", "high", "low", "close", "volume"]
        initial_len = len(df)
        df.dropna(subset=essential_cols, inplace=True)
        if len(df) < initial_len and logger:
            logger.warning(
                f"Dropped {initial_len - len(df)} rows with NaN values in essential kline columns."
            )

        if df.empty:
            if logger:
                logger.warning("Kline DataFrame empty after NaN drop.")
            return pd.DataFrame()

        # Convert to appropriate types (float for price/vol, keep time as datetime)
        for col in ["open", "high", "low", "close", "volume", "turnover"]:
             df[col] = df[col].astype(float)

        df = df.sort_values(by="start_time").reset_index(drop=True)
        df.set_index("start_time", inplace=True) # Set time as index

        return df

    except KeyError as e:
        if logger:
            logger.error(f"Missing expected column in kline data: {e}")
        return pd.DataFrame()
    except Exception as e:
        if logger:
            logger.exception(f"Error processing kline data: {e}")
        return pd.DataFrame()


def fetch_orderbook(
    exchange: ccxt.Exchange, symbol: str, limit: int, logger: logging.Logger
) -> Optional[dict]:
    """Fetch order book data using the provided ccxt exchange instance with retry."""
    retry_count = 0
    max_retries = MAX_API_RETRIES # Use global constant
    retry_delay = CONFIG.get("api_retry_delay_seconds", RETRY_DELAY_SECONDS)
    recv_window = CONFIG.get("ccxt_recv_window_ms", 10000)

    while retry_count <= max_retries:
        try:
            # logger.debug(f"Fetching orderbook for {symbol} (Attempt {retry_count+1})...")
            orderbook_data = exchange.fetch_order_book(symbol, limit=limit)

            # Basic validation
            if (
                orderbook_data
                and "bids" in orderbook_data
                and "asks" in orderbook_data
                and isinstance(orderbook_data["bids"], list)
                and isinstance(orderbook_data["asks"], list)
            ):
                # logger.debug(f"Orderbook fetch successful for {symbol}.")
                return orderbook_data
            else:
                logger.warning(
                    f"{NEON_YELLOW}Fetched orderbook data invalid/incomplete. "
                    f"Attempt {retry_count + 1}/{max_retries + 1}.{RESET}"
                )

        except ccxt.AuthenticationError as e:
            logger.error(
                f"{NEON_RED}ccxt Authentication Error fetching orderbook: {e}. "
                f"Check API keys/permissions.{RESET}"
            )
            return None  # No point retrying auth errors

        except (ccxt.NetworkError, ccxt.ExchangeNotAvailable, ccxt.RequestTimeout) as e:
            logger.warning(
                f"{NEON_YELLOW}ccxt Network/Timeout Error: {e}. "
                f"Attempt {retry_count + 1}/{max_retries + 1}. Retrying in {retry_delay}s...{RESET}"
            )

        except ccxt.ExchangeError as e:
            if "10002" in str(e) and "timestamp" in str(e).lower():
                logger.error(
                    f"{NEON_RED}ccxt Timestamp Error (10002): {e}. "
                    f"PLEASE CHECK SYSTEM CLOCK SYNC! recvWindow={recv_window}ms.{RESET}"
                )
            else:
                logger.warning(
                    f"{NEON_YELLOW}ccxt Exchange Error: {e}. "
                    f"Attempt {retry_count + 1}/{max_retries + 1}. Retrying in {retry_delay}s...{RESET}"
                )

        except Exception as e:
            logger.exception(
                f"{NEON_RED}Unexpected error fetching orderbook with ccxt: {e}. "
                f"Attempt {retry_count + 1}/{max_retries + 1}. Retrying...{RESET}"
            )

        retry_count += 1
        if retry_count <= max_retries:
            time.sleep(retry_delay)

    logger.error(
        f"{NEON_RED}Max retries ({max_retries}) reached for orderbook fetch. Giving up.{RESET}"
    )
    return None


# --- Trading Analyzer Class ---

class TradingAnalyzer:
    """
    Analyzes trading data, calculates indicators, generates insights & signals.
    """
    # Potentially use slots if memory becomes an issue with many instances
    # __slots__ = ['df', 'logger', 'config', 'symbol', 'interval', 'levels',
    #              'fib_levels', 'scalping_signals', 'indicator_values']

    def __init__(
        self,
        df: pd.DataFrame,
        logger: logging.Logger,
        config: Dict,
        symbol: str,
        interval: str,
    ):
        """
        Initializes TradingAnalyzer.

        Args:
            df: Pandas DataFrame with Kline data (must have DateTimeIndex).
            logger: Logger instance.
            config: Configuration dictionary.
            symbol: Trading symbol (e.g., 'BTCUSDT').
            interval: Kline interval (e.g., '15').

        Raises:
            ValueError: If the input DataFrame is empty or lacks a DateTimeIndex.
        """
        if df.empty:
            logger.error("Input DataFrame is empty. Cannot initialize TradingAnalyzer.")
            raise ValueError("Input DataFrame is empty.")
        if not isinstance(df.index, pd.DatetimeIndex):
            logger.error("Input DataFrame must have a DatetimeIndex.")
            raise ValueError("Input DataFrame must have a DatetimeIndex.")

        self.df: pd.DataFrame = df.copy()
        self.logger: logging.Logger = logger
        self.config: Dict = config
        self.symbol: str = symbol
        self.interval: str = interval
        self.levels: Dict[str, Union[Dict[str, Decimal], Decimal]] = {}
        self.fib_levels: Dict[str, Decimal] = {}
        self.scalping_signals: List[Dict] = []
        self.indicator_values: Dict[str, List[Union[float, np.nan]]] = {} # Store last value

        # Pre-calculate indicators safely
        try:
            self._calculate_all_indicators()
        except Exception as e:
            self.logger.exception(
                f"Error during initial indicator calculation in __init__: {e}"
            )
            # Allow initialization, but analysis might fail later if cols missing

    def _safe_get_config(self, *keys: str, default=None):
        """
        Safely retrieve nested config values with type checking/casting.

        Args:
            *keys: Sequence of keys to access nested dictionary.
            default: The default value to return if key path not found,
                     value is None, or type mismatch occurs.

        Returns:
            The retrieved value, casted value, or the default.
        """
        value = self.config
        try:
            for key in keys:
                value = value[key]

            if value is None:
                # self.logger.debug(f"Config path {' > '.join(keys)} is None. Using default: {default}")
                return default

            if default is not None and not isinstance(value, type(default)):
                try:
                    casted_value = type(default)(value)
                    self.logger.debug(
                        f"Config value for {' > '.join(keys)} ({value}, type {type(value).__name__}) "
                        f"casted to type {type(default).__name__}."
                    )
                    return casted_value
                except (ValueError, TypeError):
                    self.logger.warning(
                        f"Config type mismatch for {' > '.join(keys)}. Expected {type(default).__name__}, "
                        f"got {type(value).__name__}. Cannot cast '{value}'. Using default: {default}"
                    )
                    return default
            return value
        except (KeyError, TypeError):
            # self.logger.debug(f"Config path not found/invalid: {' > '.join(keys)}. Using default: {default}")
            return default

    def _calculate_all_indicators(self) -> None:
        """Calculate all required technical indicators."""
        required_cols = {"open", "high", "low", "close", "volume"}
        if not required_cols.issubset(self.df.columns):
            missing = required_cols - set(self.df.columns)
            self.logger.error(f"Indicator calc skipped: Missing required columns: {missing}")
            return

        self.logger.debug("Calculating indicators...")
        start_time = time.perf_counter()

        # Define calculation tasks with error handling wrappers
        indicator_tasks = {
            "atr": lambda: self.calculate_atr(window=self._safe_get_config('atr_period', default=14)),
            "rsi": lambda: self.calculate_rsi(window=self._safe_get_config('rsi_period', default=14)),
            "rsi_20": lambda: self.calculate_rsi(window=20),
            "rsi_100": lambda: self.calculate_rsi(window=100),
            "stoch_rsi": lambda: self.calculate_stoch_rsi(
                rsi_window=self._safe_get_config('stoch_rsi_period', default=14),
                stoch_window=self._safe_get_config('stoch_rsi_stoch_window', default=12),
                k_window=self._safe_get_config('stoch_rsi_k', default=3),
                d_window=self._safe_get_config('stoch_rsi_d', default=3)
            ),
            "macd": lambda: self.calculate_macd(),
            "bollinger_bands": lambda: self.calculate_bollinger_bands(
                window=self._safe_get_config('bollinger_bands_period', default=20),
                num_std_dev=self._safe_get_config('bollinger_bands_std_dev', default=2.0)
            ),
            "momentum_ma": lambda: self.calculate_momentum_ma(), # Modifies df in place
            "mfi": lambda: self.calculate_mfi(window=14),
            "cci": lambda: self.calculate_cci(window=20),
            "wr": lambda: self.calculate_williams_r(window=14),
            "adx": lambda: self.calculate_adx(window=14),
            "obv": lambda: self.calculate_obv(),
            "adi": lambda: self.calculate_adi(),
            "sma_10": lambda: self.calculate_sma(window=10),
            "psar": lambda: self.calculate_psar(),
            "fve": lambda: self.calculate_fve(),
        }

        for name, task in indicator_tasks.items():
            try:
                result = task()
                if isinstance(result, pd.Series):
                    self.df[name] = result
                elif isinstance(result, pd.DataFrame):
                    for col in result.columns:
                        self.df[col] = result[col]
                # Handle tasks that modify df in place (like momentum_ma) - no assignment needed
                elif result is None and name == "momentum_ma":
                     pass # Expected for in-place modification
                elif result is not None:
                     self.logger.warning(f"Unexpected result type from {name} calculation: {type(result)}")

            except Exception as e:
                self.logger.error(f"Error calculating {name}: {e}", exc_info=True)
                # Optionally add placeholder columns with NaN if calculation fails
                # placeholder_cols = {'stoch_rsi': ['stoch_rsi', 'stoch_k', 'stoch_d'], ...}
                # if name in placeholder_cols:
                #     for col_name in placeholder_cols[name]: self.df[col_name] = np.nan
                # elif name != 'momentum_ma': # Avoid overwriting if momentum fails partially
                #     self.df[name] = np.nan


        end_time = time.perf_counter()
        self.logger.debug(
            f"Finished calculating indicators in {end_time - start_time:.4f} seconds."
        )

    # --- Indicator Calculation Methods ---
    # (Ensure all return pd.Series/pd.DataFrame aligned with self.df.index, or modify df in place)

    def calculate_atr(self, window: int = 14) -> pd.Series:
        """Calculates Average True Range (ATR)."""
        if not all(col in self.df.columns for col in ["high", "low", "close"]):
            self.logger.warning("ATR calculation skipped: Missing required columns.")
            return pd.Series(np.nan, index=self.df.index, dtype="float64")

        high_low = self.df["high"] - self.df["low"]
        high_close = (self.df["high"] - self.df["close"].shift()).abs()
        low_close = (self.df["low"] - self.df["close"].shift()).abs()
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(
            axis=1, skipna=False
        )
        # Wilder's smoothing (equivalent to EMA with alpha = 1/window)
        atr = tr.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
        return atr # Keep NaNs at the start

    def calculate_rsi(self, window: int = 14) -> pd.Series:
        """Calculates Relative Strength Index (RSI)."""
        if "close" not in self.df.columns:
            self.logger.warning("RSI calculation skipped: Missing 'close' column.")
            return pd.Series(np.nan, index=self.df.index, dtype="float64")

        delta = self.df["close"].diff()
        gain = delta.where(delta > 0, 0.0)
        loss = -delta.where(delta < 0, 0.0)

        avg_gain = gain.ewm(com=window - 1, adjust=False).mean()
        avg_loss = loss.ewm(com=window - 1, adjust=False).mean()

        # Handle division by zero
        rs = avg_gain / avg_loss.replace(0, np.nan)
        rsi = 100.0 - (100.0 / (1.0 + rs))

        # If avg_loss is 0, RSI is 100; if avg_gain is 0 (and avg_loss > 0), RSI is 0.
        rsi = np.where(avg_loss == 0, 100.0, rsi)
        rsi = np.where(avg_gain == 0, 0.0, rsi) # Handle case where gain is 0 but loss isn't
        rsi = pd.Series(rsi, index=self.df.index)
        rsi.iloc[0:window] = np.nan # RSI is undefined for the first 'window' periods

        return rsi


    def calculate_stoch_rsi(
        self,
        rsi_window: int = 14,
        stoch_window: int = 12,
        k_window: int = 3,
        d_window: int = 3,
    ) -> pd.DataFrame:
        """Calculates Stochastic RSI (%K and %D)."""
        rsi_col_name = "rsi"
        if rsi_col_name not in self.df or self.df[rsi_col_name].isnull().all():
            self.logger.warning(
                f"Stoch RSI calc skipped: '{rsi_col_name}' column missing/invalid."
            )
            return pd.DataFrame(
                index=self.df.index, columns=["stoch_rsi", "stoch_k", "stoch_d"], dtype="float64"
            )

        rsi = self.df[rsi_col_name]
        min_rsi = rsi.rolling(window=stoch_window, min_periods=1).min()
        max_rsi = rsi.rolling(window=stoch_window, min_periods=1).max()
        range_rsi = (max_rsi - min_rsi).replace(0, np.nan) # Avoid division by zero

        stoch_rsi_raw = (rsi - min_rsi) / range_rsi
        # Ensure values are between 0 and 1, handling potential float precision issues
        stoch_rsi_raw = stoch_rsi_raw.clip(0, 1)

        k_line = stoch_rsi_raw.rolling(window=k_window, min_periods=1).mean() * 100.0
        d_line = k_line.rolling(window=d_window, min_periods=1).mean()
        stoch_rsi = stoch_rsi_raw * 100.0

        return pd.DataFrame(
            {"stoch_rsi": stoch_rsi, "stoch_k": k_line, "stoch_d": d_line},
            index=self.df.index,
        )

    def calculate_macd(self) -> pd.DataFrame:
        """Calculates MACD, Signal Line, and Histogram."""
        if "close" not in self.df.columns:
            self.logger.warning("MACD calc skipped: Missing 'close'.")
            return pd.DataFrame(
                index=self.df.index, columns=["macd", "signal", "histogram"], dtype="float64"
            )

        close_prices = self.df["close"]
        exp12 = close_prices.ewm(span=12, adjust=False).mean()
        exp26 = close_prices.ewm(span=26, adjust=False).mean()
        macd = exp12 - exp26
        signal = macd.ewm(span=9, adjust=False).mean()
        histogram = macd - signal

        return pd.DataFrame(
            {"macd": macd, "signal": signal, "histogram": histogram}, index=self.df.index
        )

    def calculate_bollinger_bands(
        self, window: int = 20, num_std_dev: float = 2.0
    ) -> pd.DataFrame:
        """Calculates Bollinger Bands."""
        if "close" not in self.df.columns:
            self.logger.warning("BBands calc skipped: Missing 'close'.")
            return pd.DataFrame(
                index=self.df.index,
                columns=["upper_band", "middle_band", "lower_band"],
                dtype="float64",
            )

        close_price = self.df["close"]
        middle_band = close_price.rolling(window=window, min_periods=1).mean()
        rolling_std = close_price.rolling(window=window, min_periods=1).std()

        upper_band = middle_band + (rolling_std * num_std_dev)
        lower_band = middle_band - (rolling_std * num_std_dev)

        return pd.DataFrame(
            {"upper_band": upper_band, "middle_band": middle_band, "lower_band": lower_band},
            index=self.df.index,
        )

    def calculate_momentum_ma(self) -> None:
        """Calculate Momentum and its MAs. Modifies DataFrame in place."""
        if not all(c in self.df.columns for c in ["close", "volume"]):
            self.logger.warning("Momentum MA calc skipped: Missing 'close' or 'volume'.")
            # Add NaN columns if they don't exist to prevent errors later
            for col in ["momentum", "momentum_ma_short", "momentum_ma_long", "volume_ma"]:
                 if col not in self.df: self.df[col] = np.nan
            return

        mom_period = self._safe_get_config("momentum_period", default=10)
        mom_ma_short = self._safe_get_config("momentum_ma_short", default=12)
        mom_ma_long = self._safe_get_config("momentum_ma_long", default=26)
        vol_ma_period = self._safe_get_config("volume_ma_period", default=20)

        self.df["momentum"] = self.df["close"].diff(mom_period)
        self.df["momentum_ma_short"] = self.df["momentum"].rolling(
            window=mom_ma_short, min_periods=1
        ).mean()
        self.df["momentum_ma_long"] = self.df["momentum"].rolling(
            window=mom_ma_long, min_periods=1
        ).mean()
        self.df["volume_ma"] = self.df["volume"].rolling(
            window=vol_ma_period, min_periods=1
        ).mean()

    def calculate_cci(self, window: int = 20, constant: float = 0.015) -> pd.Series:
        """Calculates Commodity Channel Index (CCI)."""
        if not all(col in self.df.columns for col in ["high", "low", "close"]):
            self.logger.warning("CCI calc skipped: Missing required columns.")
            return pd.Series(np.nan, index=self.df.index, dtype="float64")

        tp = (self.df["high"] + self.df["low"] + self.df["close"]) / 3
        sma_tp = tp.rolling(window=window, min_periods=1).mean()
        # Calculate Mean Absolute Deviation (MAD)
        mad_tp = tp.rolling(window=window, min_periods=1).apply(
            lambda x: np.mean(np.abs(x - np.mean(x))), raw=True
        )

        cci = (tp - sma_tp) / (constant * mad_tp.replace(0, np.nan)) # Avoid division by zero
        return cci

    def calculate_williams_r(self, window: int = 14) -> pd.Series:
        """Calculates Williams %R."""
        if not all(col in self.df.columns for col in ["high", "low", "close"]):
            self.logger.warning("Williams %R calc skipped: Missing required columns.")
            return pd.Series(np.nan, index=self.df.index, dtype="float64")

        high_h = self.df["high"].rolling(window=window, min_periods=1).max()
        low_l = self.df["low"].rolling(window=window, min_periods=1).min()
        range_hl = (high_h - low_l).replace(0, np.nan) # Avoid division by zero

        wr = -100 * (high_h - self.df["close"]) / range_hl
        return wr

    def calculate_mfi(self, window: int = 14) -> pd.Series:
        """Calculates Money Flow Index (MFI)."""
        if not all(col in self.df.columns for col in ["high", "low", "close", "volume"]):
            self.logger.warning("MFI calc skipped: Missing required columns.")
            return pd.Series(np.nan, index=self.df.index, dtype="float64")

        tp = (self.df["high"] + self.df["low"] + self.df["close"]) / 3
        rmf = tp * self.df["volume"] # Raw Money Flow

        # Calculate positive and negative money flow
        delta_tp = tp.diff()
        positive_mf = rmf.where(delta_tp > 0, 0)
        negative_mf = rmf.where(delta_tp < 0, 0)

        # Sum over the window period
        pos_mf_sum = positive_mf.rolling(window=window, min_periods=1).sum()
        neg_mf_sum = negative_mf.rolling(window=window, min_periods=1).sum()

        # Calculate Money Flow Ratio and MFI
        mfr = pos_mf_sum / neg_mf_sum.replace(0, np.nan) # Avoid division by zero
        mfi = 100.0 - (100.0 / (1.0 + mfr))
        mfi = np.where(neg_mf_sum == 0, 100.0, mfi) # Handle case where neg_mf_sum is 0

        mfi.iloc[0:window] = np.nan # MFI undefined for first 'window' periods
        return mfi

    def calculate_adx(self, window: int = 14) -> pd.Series:
        """Calculates Average Directional Index (ADX)."""
        if not all(col in self.df.columns for col in ["high", "low", "close"]):
            self.logger.warning("ADX calc skipped: Missing required columns.")
            return pd.Series(np.nan, index=self.df.index, dtype="float64")

        high = self.df["high"]
        low = self.df["low"]
        close = self.df["close"]

        # Calculate True Range (TR) - use the existing ATR function with window=1
        tr = self.calculate_atr(window=1)

        # Calculate Directional Movement (+DM, -DM)
        move_up = high.diff().fillna(0)
        move_down = low.diff().fillna(0) # Note: diff is current - previous

        plus_dm = np.where((move_up > -move_down) & (move_up > 0), move_up, 0.0)
        minus_dm = np.where((-move_down > move_up) & (-move_down > 0), -move_down, 0.0)
        # Smoothed versions using Wilder's smoothing (EMA with alpha=1/window)
        atr_smooth = tr.ewm(alpha=1 / window, adjust=False).mean()
        plus_di = 100 * (
            pd.Series(plus_dm, index=self.df.index)
            .ewm(alpha=1 / window, adjust=False)
            .mean()
            / atr_smooth.replace(0, np.nan) # Avoid division by zero
        )
        minus_di = 100 * (
             pd.Series(minus_dm, index=self.df.index)
            .ewm(alpha=1 / window, adjust=False)
            .mean()
            / atr_smooth.replace(0, np.nan) # Avoid division by zero
        )

        # Calculate Directional Index (DX)
        di_diff = abs(plus_di - minus_di)
        di_sum = (plus_di + minus_di).replace(0, np.nan) # Avoid division by zero
        dx = 100 * (di_diff / di_sum)

        # Calculate ADX
        adx = dx.ewm(alpha=1 / window, adjust=False).mean()
        adx.iloc[0 : window * 2] = np.nan # ADX needs more time to stabilize

        return adx

    def calculate_obv(self) -> pd.Series:
        """Calculates On-Balance Volume (OBV)."""
        if not all(col in self.df.columns for col in ["close", "volume"]):
            self.logger.warning("OBV calc skipped: Missing required columns.")
            return pd.Series(np.nan, index=self.df.index, dtype="float64")

        signed_volume = self.df["volume"] * np.sign(self.df["close"].diff()).fillna(0)
        return signed_volume.cumsum()

    def calculate_adi(self) -> pd.Series:
        """Calculates Accumulation/Distribution Indicator (ADI)."""
        if not all(col in self.df.columns for col in ["high", "low", "close", "volume"]):
            self.logger.warning("ADI calc skipped: Missing required columns.")
            return pd.Series(np.nan, index=self.df.index, dtype="float64")

        high = self.df["high"]
        low = self.df["low"]
        close = self.df["close"]
        volume = self.df["volume"]

        # Calculate Money Flow Multiplier (MFM)
        h_minus_l = (high - low).replace(0, np.nan) # Avoid division by zero
        mfm = ((close - low) - (high - close)) / h_minus_l
        mfm = mfm.fillna(0)  # Fill NaNs resulting from H=L with 0

        # Calculate Money Flow Volume (MFV)
        mfv = mfm * volume

        # Calculate ADI
        adi = mfv.cumsum()
        return adi

    def calculate_sma(self, window: int) -> pd.Series:
        """Calculates Simple Moving Average (SMA)."""
        if "close" not in self.df.columns:
            self.logger.warning(f"SMA {window} calc skipped: Missing 'close'.")
            return pd.Series(np.nan, index=self.df.index, dtype="float64")

        sma = self.df["close"].rolling(window=window, min_periods=1).mean()
        return sma

    def calculate_psar(
        self, acceleration: float = 0.02, max_acceleration: float = 0.2
    ) -> pd.Series:
        """Calculates Parabolic SAR (PSAR)."""
        if not all(col in self.df.columns for col in ["high", "low"]):
            self.logger.warning("PSAR calc skipped: Missing required columns.")
            return pd.Series(np.nan, index=self.df.index, dtype="float64")

        high = self.df["high"]
        low = self.df["low"]
        length = len(self.df)
        psar = np.full(length, np.nan)
        if length < 2:
             return pd.Series(psar, index=self.df.index)

        # Initial values (handle potential NaNs at the start)
        start_idx = 0
        while start_idx < length and (pd.isna(low.iloc[start_idx]) or pd.isna(high.iloc[start_idx])):
            start_idx += 1

        if start_idx >= length - 1: # Not enough valid data
            return pd.Series(psar, index=self.df.index)

        # Determine initial trend based on first two valid points
        bull = high.iloc[start_idx + 1] > high.iloc[start_idx]
        psar[start_idx] = low.iloc[start_idx] if bull else high.iloc[start_idx]
        ep = high.iloc[start_idx] if bull else low.iloc[start_idx]
        af = acceleration

        # Iterate through the rest of the data
        for i in range(start_idx + 1, length):
            prev_psar = psar[i - 1]
            current_high = high.iloc[i]
            current_low = low.iloc[i]

            # Skip if data is missing
            if pd.isna(prev_psar) or pd.isna(current_high) or pd.isna(current_low):
                psar[i] = prev_psar # Carry forward last valid PSAR
                continue

            # Calculate current PSAR based on previous trend
            current_psar = prev_psar + af * (ep - prev_psar)

            if bull:
                # Potential reversal to Bearish
                if current_low < current_psar:
                    bull = False
                    psar[i] = ep # Switch point becomes the EP of the previous trend
                    ep = current_low
                    af = acceleration
                # Continue Bullish trend
                else:
                    psar[i] = min(current_psar, low.iloc[i-1], low.iloc[i-2] if i > start_idx + 1 else current_psar) # Don't let PSAR go above recent lows
                    if current_high > ep:
                        ep = current_high
                        af = min(af + acceleration, max_acceleration)
            else: # Bearish
                # Potential reversal to Bullish
                if current_high > current_psar:
                    bull = True
                    psar[i] = ep # Switch point
                    ep = current_high
                    af = acceleration
                # Continue Bearish trend
                else:
                    psar[i] = max(current_psar, high.iloc[i-1], high.iloc[i-2] if i > start_idx + 1 else current_psar) # Don't let PSAR go below recent highs
                    if current_low < ep:
                        ep = current_low
                        af = min(af + acceleration, max_acceleration)

        return pd.Series(psar, index=self.df.index)


    def calculate_fve(self, window: int = 13) -> pd.Series:
        """Calculates Force Index Volume Estimator (similar to Force Index)."""
        if not all(col in self.df.columns for col in ["close", "volume"]):
            self.logger.warning("FVE calc skipped: Missing required columns.")
            return pd.Series(np.nan, index=self.df.index, dtype="float64")

        force = self.df["close"].diff() * self.df["volume"]
        fve = force.ewm(span=window, adjust=False).mean() # Standard Force Index smoothing
        return fve

    # --- Level Calculation Methods ---

    def calculate_fibonacci_retracement(
        self, high: float, low: float, current_price: Decimal
    ) -> None:
        """Calculate Fibonacci Retracement levels and store them."""
        self.fib_levels = {}
        # Reset only Fib-related levels in the main dict
        if "Support" in self.levels:
            self.levels["Support"] = {
                k: v
                for k, v in self.levels["Support"].items()
                if not k.startswith("Fib")
            }
        if "Resistance" in self.levels:
            self.levels["Resistance"] = {
                k: v
                for k, v in self.levels["Resistance"].items()
                if not k.startswith("Fib")
            }

        try:
            if pd.isna(high) or pd.isna(low) or high <= low:
                return # Need valid high > low

            diff = Decimal(str(high)) - Decimal(str(low))
            if diff <= 0: return

            fib_ratios = {
                "Fib 23.6%": Decimal("0.236"),
                "Fib 38.2%": Decimal("0.382"),
                "Fib 50.0%": Decimal("0.5"),
                "Fib 61.8%": Decimal("0.618"),
                "Fib 78.6%": Decimal("0.786"),
            }
            high_dec = Decimal(str(high))

            for label, ratio in fib_ratios.items():
                level_price = high_dec - diff * ratio
                self.fib_levels[label] = level_price
                if pd.notna(current_price):
                    if level_price < current_price:
                        self.levels.setdefault("Support", {})[label] = level_price
                    elif level_price > current_price:
                        self.levels.setdefault("Resistance", {})[label] = level_price

        except Exception as e:
            self.logger.error(f"Fibonacci calculation error: {e}", exc_info=True)

    def calculate_pivot_points(self, high: float, low: float, close: float) -> None:
        """Calculate standard pivot points and store them."""
        try:
            if pd.isna(high) or pd.isna(low) or pd.isna(close) or high < low:
                 return # Need valid data

            # Use Decimal for precision
            high_d, low_d, close_d = map(Decimal, map(str, [high, low, close]))

            pivot = (high_d + low_d + close_d) / 3
            range_hl = high_d - low_d
            if range_hl <= 0: return

            # Calculate levels using Decimal
            r1 = (2 * pivot) - low_d
            s1 = (2 * pivot) - high_d
            r2 = pivot + range_hl
            s2 = pivot - range_hl
            r3 = high_d + 2 * (pivot - low_d)
            s3 = low_d - 2 * (high_d - pivot)

            # Update only pivot keys, preserving others
            pivot_levels = {
                "pivot": pivot, "r1": r1, "s1": s1, "r2": r2, "s2": s2, "r3": r3, "s3": s3
            }
            # Ensure sub-dictionaries exist before updating
            self.levels.setdefault("Resistance", {}).update({"R1": r1, "R2": r2, "R3": r3})
            self.levels.setdefault("Support", {}).update({"S1": s1, "S2": s2, "S3": s3})
            self.levels["Pivot"] = pivot # Store pivot separately or within S/R? Storing separately.

        except Exception as e:
            self.logger.error(f"Pivot point calculation error: {e}", exc_info=True)

    def find_nearest_levels(
        self, current_price: Decimal, num_levels: int = 3
    ) -> Tuple[List[Tuple[str, Decimal]], List[Tuple[str, Decimal]]]:
        """
        Find the nearest support and resistance levels from calculated levels.

        Args:
            current_price: The current market price as a Decimal.
            num_levels: The maximum number of nearest S/R levels to return.

        Returns:
            A tuple containing two lists: (nearest_supports, nearest_resistances).
            Each list contains tuples of (level_name, level_price_Decimal).
        """
        support_levels = []
        resistance_levels = []
        try:
            if pd.isna(current_price):
                return [], []

            def process_level(level_label: str, level_value: Union[Decimal, float, int]):
                try:
                    # Ensure level value is Decimal for accurate comparison
                    if isinstance(level_value, (float, int)):
                        level_value_dec = Decimal(str(level_value))
                    elif isinstance(level_value, Decimal):
                        level_value_dec = level_value
                    else: return # Skip invalid types

                    if pd.isna(level_value_dec): return # Skip NaN level values
                except Exception:
                    return # Skip if conversion fails

                distance = abs(level_value_dec - current_price)
                if level_value_dec < current_price:
                    support_levels.append((level_label, level_value_dec, distance))
                elif level_value_dec > current_price:
                    resistance_levels.append((level_label, level_value_dec, distance))

            # Iterate through nested and direct levels
            for level_type, level_group in self.levels.items():
                if isinstance(level_group, dict):
                    for sub_label, sub_value in level_group.items():
                        process_level(f"{level_type} {sub_label}", sub_value)
                elif isinstance(level_group, (Decimal, float, int)):
                     process_level(level_type, level_group) # Handle top-level items like Pivot

            # Sort by distance (3rd element)
            support_levels.sort(key=lambda x: x[2])
            resistance_levels.sort(key=lambda x: x[2])

            # Return only name and Decimal value
            return (
                [(n, v) for n, v, _ in support_levels[:num_levels]],
                [(n, v) for n, v, _ in resistance_levels[:num_levels]],
            )
        except Exception as e:
            self.logger.error(f"Error finding nearest levels: {e}", exc_info=True)
            return [], []

    # --- Orderbook Analysis ---
    def analyze_orderbook_levels(
        self, orderbook: Optional[dict], current_price: Decimal
    ) -> str:
        """Analyze order book for significant clusters near key S/R levels."""
        if not orderbook or "bids" not in orderbook or "asks" not in orderbook:
            return f"{NEON_YELLOW}  Orderbook data unavailable or invalid.{RESET}"

        try:
            # Validate bids/asks format more robustly
            bids_raw = orderbook.get('bids', [])
            asks_raw = orderbook.get('asks', [])
            if not isinstance(bids_raw, list) or not isinstance(asks_raw, list):
                return f"{NEON_YELLOW}  Orderbook bids/asks format incorrect (not lists).{RESET}"

            # Filter out invalid entries [[price, size], ...]
            valid_bids = [b for b in bids_raw if isinstance(b, (list, tuple)) and len(b) == 2 and all(isinstance(x, (int, float)) for x in b)]
            valid_asks = [a for a in asks_raw if isinstance(a, (list, tuple)) and len(a) == 2 and all(isinstance(x, (int, float)) for x in a)]

            if not valid_bids and not valid_asks:
                return f"{NEON_YELLOW}  Orderbook bids/asks empty or invalid after filtering.{RESET}"

            bids = pd.DataFrame(valid_bids, columns=['price', 'size'], dtype=float)
            asks = pd.DataFrame(valid_asks, columns=['price', 'size'], dtype=float)
            bids.dropna(inplace=True)
            asks.dropna(inplace=True)

            if bids.empty and asks.empty:
                return f"{NEON_YELLOW}  Orderbook empty after validation.{RESET}"

            analysis_output = ""
            threshold = self._safe_get_config('orderbook_cluster_threshold', default=1000.0)
            price_range_factor = Decimal("0.001") # Check within 0.1% (adjust as needed)
            current_price_dec = current_price # Already Decimal

            def check_cluster(level_name: str, level_price_dec: Decimal, bids_df: pd.DataFrame, asks_df: pd.DataFrame):
                nonlocal analysis_output
                try:
                    if pd.isna(level_price_dec): return # Skip invalid level

                    level_price_float = float(level_price_dec) # For DataFrame filtering
                    price_delta = float(level_price_dec * price_range_factor)
                    lower_bound = level_price_float - price_delta
                    upper_bound = level_price_float + price_delta

                    # Filter using float bounds
                    bid_volume = bids_df.loc[
                        (bids_df['price'] >= lower_bound) & (bids_df['price'] <= upper_bound), 'size'
                    ].sum()
                    ask_volume = asks_df.loc[
                        (asks_df['price'] >= lower_bound) & (asks_df['price'] <= upper_bound), 'size'
                    ].sum()

                    # Report significant clusters
                    price_fmt = self._get_price_format(level_price_dec) # Dynamic format
                    if bid_volume > threshold:
                        analysis_output += (
                            f"  {NEON_GREEN}OB Cluster (Bid):{RESET} {bid_volume:,.0f} "
                            f"near {level_name} @ ${level_price_dec:{price_fmt}}\n"
                        )
                    if ask_volume > threshold:
                        analysis_output += (
                            f"  {NEON_RED}OB Cluster (Ask):{RESET} {ask_volume:,.0f} "
                            f"near {level_name} @ ${level_price_dec:{price_fmt}}\n"
                        )
                except Exception as e:
                    self.logger.debug(
                        f"Skipping OB check for level {level_name} due to error: {e}"
                    )

            # Check calculated S/R levels (which should store Decimals)
            all_levels_to_check = {}
            for level_type in ["Support", "Resistance", "Pivot"]: # Include Pivot explicitly
                 if level_type in self.levels:
                     level_data = self.levels[level_type]
                     if isinstance(level_data, dict):
                          for name, price in level_data.items():
                               if isinstance(price, Decimal):
                                    all_levels_to_check[f"{level_type} {name}"] = price
                     elif isinstance(level_data, Decimal): # Handle Pivot stored directly
                           all_levels_to_check[level_type] = level_data


            for name, price_dec in all_levels_to_check.items():
                 check_cluster(name, price_dec, bids, asks)

            return analysis_output.strip() if analysis_output else "  No significant OB clusters near levels."

        except Exception as e:
            self.logger.error(f"Error analyzing orderbook levels: {e}", exc_info=True)
            return f"{NEON_RED}  Error during orderbook analysis.{RESET}"

    # --- Trend Determination ---
    def determine_trend_momentum(self) -> Dict[str, Union[str, float]]:
        """Determine trend direction and strength using Momentum MAs and ATR."""
        mom_short_col = "momentum_ma_short"
        mom_long_col = "momentum_ma_long"
        atr_col = "atr"

        # Check if necessary columns exist and have valid data in the last row
        required_cols = [mom_short_col, mom_long_col, atr_col]
        if not all(col in self.df.columns for col in required_cols):
            self.logger.warning(f"Trend determination skipped: Missing columns: {required_cols - set(self.df.columns)}")
            return {"trend": "Insufficient Data", "strength": 0.0}

        latest_vals = self.df[required_cols].iloc[-1]
        if latest_vals.isnull().any():
             self.logger.warning(f"Trend determination skipped: NaN values in latest row for {latest_vals[latest_vals.isnull()].index.tolist()}")
             return {"trend": "Insufficient Data", "strength": 0.0}

        mom_short = latest_vals[mom_short_col]
        mom_long = latest_vals[mom_long_col]
        atr = latest_vals[atr_col]

        if atr <= 0:
            trend_strength = 0.0 # Cannot determine strength without positive ATR
            self.logger.debug(f"ATR is non-positive ({atr}), trend strength set to 0.")
        else:
            # Normalize the difference by ATR
            trend_strength = abs(mom_short - mom_long) / atr

        # Determine trend direction
        if mom_short > mom_long:
            trend = "Uptrend"
        elif mom_short < mom_long:
            trend = "Downtrend"
        else:
            trend = "Neutral"

        return {"trend": trend, "strength": float(trend_strength)} # Return strength as float

    # --- Scalping Signal Generation ---

    def _check_long_signal(
        self, latest: pd.Series, prev: pd.Series, current_price_dec: Decimal
    ) -> Tuple[int, List[str]]:
        """Checks conditions for a potential LONG scalping signal."""
        confidence = 0
        factors = []
        rsi_scalp_os = self._safe_get_config("scalping", "rsi_scalp_oversold", default=35)
        stoch_rsi_scalp_os = self._safe_get_config("scalping", "stoch_rsi_scalp_oversold", default=25)
        latest_vol_ma = latest.get('volume_ma', 0) # Default to 0 if missing

        # RSI Condition
        if latest['rsi'] < rsi_scalp_os:
            confidence += 25
            factors.append(f"RSI<{rsi_scalp_os:.0f}")

        # Stochastic RSI Condition (with cross check)
        if latest['stoch_k'] < stoch_rsi_scalp_os and latest['stoch_d'] < stoch_rsi_scalp_os:
            if pd.notna(prev['stoch_k']) and pd.notna(prev['stoch_d']):
                if latest['stoch_k'] > latest['stoch_d'] and prev['stoch_k'] <= prev['stoch_d']:
                    confidence += 35
                    factors.append(f"StochK<{stoch_rsi_scalp_os:.0f}+Cross↑")
                else:
                    confidence += 15
                    factors.append(f"StochK<{stoch_rsi_scalp_os:.0f}")
            else:
                confidence += 15
                factors.append(f"StochK<{stoch_rsi_scalp_os:.0f}")

        # Bollinger Band Condition (use Decimal for comparison)
        lower_band_dec = Decimal(str(latest['lower_band']))
        if current_price_dec <= lower_band_dec * Decimal("1.001"):
            confidence += 15
            factors.append("Near LowerBB")

        # Support Level Condition
        _, nearest_resistances = self.find_nearest_levels(current_price_dec, num_levels=1)
        nearest_support_price = nearest_resistances[0][1] if nearest_resistances else None # find_nearest_levels returns S/R relative to price
        if nearest_support_price and current_price_dec <= nearest_support_price * Decimal("1.002"):
             confidence += 15
             factors.append(f"Near Supp({nearest_resistances[0][0]})") # Use the correct name


        # MACD Condition
        if pd.notna(prev['histogram']) and latest['histogram'] > 0 and prev['histogram'] <= 0:
            confidence += 20
            factors.append("MACD Hist↑")
        elif latest['macd'] > latest['signal']:
            confidence += 10
            factors.append("MACD>Sig")

        # Volume Condition
        if latest_vol_ma > 0 and latest['volume'] > latest_vol_ma * 1.1:
            confidence += 10
            factors.append("Vol↑")

        return min(confidence, 100), factors


    def _check_short_signal(
         self, latest: pd.Series, prev: pd.Series, current_price_dec: Decimal
    ) -> Tuple[int, List[str]]:
        """Checks conditions for a potential SHORT scalping signal."""
        confidence = 0
        factors = []
        rsi_scalp_ob = self._safe_get_config("scalping", "rsi_scalp_overbought", default=65)
        stoch_rsi_scalp_ob = self._safe_get_config("scalping", "stoch_rsi_scalp_overbought", default=75)
        latest_vol_ma = latest.get('volume_ma', 0) # Default to 0 if missing

        # RSI Condition
        if latest['rsi'] > rsi_scalp_ob:
            confidence += 25
            factors.append(f"RSI>{rsi_scalp_ob:.0f}")

        # Stochastic RSI Condition (with cross check)
        if latest['stoch_k'] > stoch_rsi_scalp_ob and latest['stoch_d'] > stoch_rsi_scalp_ob:
            if pd.notna(prev['stoch_k']) and pd.notna(prev['stoch_d']):
                if latest['stoch_k'] < latest['stoch_d'] and prev['stoch_k'] >= prev['stoch_d']:
                    confidence += 35
                    factors.append(f"StochK>{stoch_rsi_scalp_ob:.0f}+Cross↓")
                else:
                    confidence += 15
                    factors.append(f"StochK>{stoch_rsi_scalp_ob:.0f}")
            else:
                confidence += 15
                factors.append(f"StochK>{stoch_rsi_scalp_ob:.0f}")

        # Bollinger Band Condition (use Decimal for comparison)
        upper_band_dec = Decimal(str(latest['upper_band']))
        if current_price_dec >= upper_band_dec * Decimal("0.999"):
            confidence += 15
            factors.append("Near UpperBB")

        # Resistance Level Condition
        nearest_supports, _ = self.find_nearest_levels(current_price_dec, num_levels=1)
        nearest_resistance_price = nearest_supports[0][1] if nearest_supports else None # find_nearest_levels returns S/R relative to price
        if nearest_resistance_price and current_price_dec >= nearest_resistance_price * Decimal("0.998"):
             confidence += 15
             factors.append(f"Near Res({nearest_supports[0][0]})") # Use the correct name


        # MACD Condition
        if pd.notna(prev['histogram']) and latest['histogram'] < 0 and prev['histogram'] >= 0:
            confidence += 20
            factors.append("MACD Hist↓")
        elif latest['macd'] < latest['signal']:
            confidence += 10
            factors.append("MACD<Sig")

        # Volume Condition
        if latest_vol_ma > 0 and latest['volume'] > latest_vol_ma * 1.1:
            confidence += 10
            factors.append("Vol↑")

        return min(confidence, 100), factors


    def generate_scalping_signals(self, current_price: Decimal) -> None:
        """Generate potential short and long scalping signals."""
        self.scalping_signals = []
        if not self._safe_get_config("scalping", "enabled", default=False):
            return

        if len(self.df) < 2:
            self.logger.warning("Scalping signals need at least 2 candles.")
            return

        latest = self.df.iloc[-1]
        prev = self.df.iloc[-2]

        # Check for required indicators (ensure they exist and are not NaN)
        required_indicators = [
            "atr", "rsi", "stoch_k", "stoch_d", "macd", "signal", "histogram",
            "upper_band", "lower_band", "close", "volume", "volume_ma",
        ]
        missing_inds = [
            ind for ind in required_indicators if ind not in latest.index or pd.isna(latest[ind])
        ]
        if missing_inds:
            self.logger.warning(
                f"Scalping signal generation skipped: Missing/NaN indicator data: {', '.join(missing_inds)}"
            )
            return

        atr = latest["atr"]
        if atr <= 0 or pd.isna(atr):
            self.logger.warning(f"Scalping signal generation skipped: Invalid ATR ({atr}).")
            return

        # Use Decimal for calculations involving price and ATR
        current_price_dec = current_price
        atr_dec = Decimal(str(atr))
        sl_multiplier = Decimal(str(self._safe_get_config("scalping", "sl_atr_multiplier", default=1.5)))
        atr_sl_offset = atr_dec * sl_multiplier
        min_confidence = self._safe_get_config("scalping", "min_confidence_level", default=50)


        # --- Long Signal ---
        long_confidence, long_factors = self._check_long_signal(latest, prev, current_price_dec)
        if long_confidence >= min_confidence:
            entry = current_price_dec
            stop_loss = entry - atr_sl_offset
            stop_loss = max(Decimal("0.00000001"), stop_loss) # Ensure SL is positive
            self.scalping_signals.append({
                "type": "LONG",
                "entry": entry,
                "stop_loss": stop_loss,
                "confidence": long_confidence,
                "factors": ", ".join(long_factors),
            })

        # --- Short Signal ---
        short_confidence, short_factors = self._check_short_signal(latest, prev, current_price_dec)
        if short_confidence >= min_confidence:
            entry = current_price_dec
            stop_loss = entry + atr_sl_offset
            self.scalping_signals.append({
                "type": "SHORT",
                "entry": entry,
                "stop_loss": stop_loss,
                "confidence": short_confidence,
                "factors": ", ".join(short_factors),
            })

        # Sort and limit signals
        max_signals = self._safe_get_config("scalping", "max_signals", default=2)
        self.scalping_signals = sorted(
            self.scalping_signals, key=lambda x: x["confidence"], reverse=True
        )[:max_signals]

    # --- Main Analysis and Output ---

    def _get_price_format(self, price: Optional[Decimal]) -> str:
         """Determines appropriate f-string format based on price magnitude."""
         if price is None or pd.isna(price) or not isinstance(price, Decimal):
             return ".2f" # Default fallback
         price_str = str(price.normalize())
         if '.' in price_str:
             decimals = len(price_str.split('.')[-1])
             # Limit decimals reasonably (e.g., max 8 for most cryptos)
             decimals = max(1, min(8, decimals))
             return f".{decimals}f"
         else:
             return ".2f" # Default if integer price

    def _update_indicator_values(self) -> None:
         """Stores the latest value of each calculated indicator."""
         self.indicator_values = {}
         if self.df.empty:
             self.logger.error("DataFrame is empty, cannot update indicator values.")
             return

         latest_row = self.df.iloc[-1]
         exclude_cols = { # Set for faster lookup
             "open", "high", "low", "close", "volume", "turnover", "momentum",
             "tr", "move_up", "move_down", "plus_dm", "minus_dm" # Exclude intermediate/raw cols
         }
         for col in latest_row.index:
             if col not in exclude_cols:
                 value = latest_row[col]
                 # Store as list containing the single value (or NaN)
                 self.indicator_values[col] = [value] if pd.notna(value) else [np.nan]


    def analyze(self, current_price: Decimal, timestamp_str: str, session: requests.Session, exchange: ccxt.Exchange):
        """
        Perform full analysis: levels, indicators, orderbook, signals, output.

        Args:
            current_price: Current market price as Decimal.
            timestamp_str: UTC timestamp string for the analysis cycle start.
            session: requests.Session for Bybit API calls.
            exchange: ccxt exchange instance for orderbook.
        """
        min_candles_config = max( # Determine minimum candles needed based on longest lookback periods in config
            self._safe_get_config("rsi_period", default=14),
            self._safe_get_config("bollinger_bands_period", default=20),
            self._safe_get_config("momentum_ma_long", default=26),
            self._safe_get_config("atr_period", default=14),
            self._safe_get_config("stoch_rsi_period", default=14) + self._safe_get_config("stoch_rsi_stoch_window", default=12), # StochRSI needs RSI period + Stoch window
             # Add other relevant periods (e.g., ADX, MFI windows)
        )
        min_candles_needed = max(50, min_candles_config + 5) # Absolute minimum + buffer

        if len(self.df) < min_candles_needed:
            self.logger.warning(
                f"Insufficient data ({len(self.df)} candles, need ~{min_candles_needed}) for full analysis. Skipping."
            )
            return

        # --- Recalculate Indicators (ensures latest data is used) ---
        # self.logger.debug("Re-calculating all indicators for analysis cycle...")
        # self._calculate_all_indicators() # Already called in __init__, could be optimized to only update last candle if needed

        # --- Calculate Levels ---
        nearest_supports, nearest_resistances = [], []
        try:
            lookback_period = self._safe_get_config("bollinger_bands_period", default=20) # Use a common lookback like BB period
            lookback_period = min(lookback_period, len(self.df))
            recent_df = self.df.iloc[-lookback_period:]

            high = recent_df["high"].max() if not recent_df.empty else np.nan
            low = recent_df["low"].min() if not recent_df.empty else np.nan
            close = self.df["close"].iloc[-1] if not self.df.empty else np.nan

            if not all(pd.notna(v) for v in [high, low, close, current_price]):
                self.logger.warning("Cannot calculate levels: Invalid high/low/close/price.")
            else:
                self.calculate_fibonacci_retracement(high, low, current_price)
                self.calculate_pivot_points(high, low, close)
                nearest_supports, nearest_resistances = self.find_nearest_levels(
                    current_price
                )
        except Exception as e:
            self.logger.error(f"Error calculating levels: {e}", exc_info=True)


        # --- Fetch and Analyze Orderbook ---
        ob_limit = self._safe_get_config("orderbook_limit", default=50)
        orderbook_data = fetch_orderbook(exchange, self.symbol, ob_limit, self.logger)
        orderbook_analysis_str = self.analyze_orderbook_levels(
            orderbook_data, current_price
        )

        # --- Determine Trend ---
        trend_data = self.determine_trend_momentum()

        # --- Generate Signals ---
        self.generate_scalping_signals(current_price)

        # --- Update Latest Indicator Values for Report ---
        self._update_indicator_values()

        # --- Format and Print Output ---
        output = self._format_analysis_output(
            current_price,
            timestamp_str,
            trend_data,
            nearest_supports,
            nearest_resistances,
            orderbook_analysis_str,
        )
        print(output)

        # Log clean text version
        clean_output = output
        for color_code in [NEON_GREEN, NEON_RED, NEON_YELLOW, NEON_BLUE, NEON_PURPLE, RESET]:
            clean_output = clean_output.replace(color_code, "")
        self.logger.info(f"Analysis Report:\n{clean_output}") # Add context to log


    def _format_header(self, current_price: Decimal, timestamp_str: str) -> str:
        """Formats the header section of the report."""
        display_timestamp = datetime.now(LOCAL_TIMEZONE).strftime("%Y-%m-%d %H:%M:%S %Z")
        price_format = self._get_price_format(current_price)
        return f"""
==================================================
{NEON_BLUE}Analysis Report:{RESET} {self.symbol} ({self.interval})
{NEON_BLUE}Timestamp:{RESET} {display_timestamp} | {NEON_BLUE}Current Price:{RESET} {current_price:{price_format}}
--------------------------------------------------"""

    def _format_trend(self, trend_data: Dict) -> str:
        """Formats the trend section."""
        trend = trend_data.get("trend", "Unknown")
        strength = trend_data.get("strength", 0.0)
        trend_color = NEON_GREEN if trend == "Uptrend" else NEON_RED if trend == "Downtrend" else NEON_YELLOW
        return f"""{NEON_PURPLE}Trend & Momentum:{RESET}
  Trend: {trend_color}{trend}{RESET} (Strength: {strength:.2f})"""

    def _format_indicators(self, price_format: str) -> str:
        """Formats the key indicators section."""
        output = f"{NEON_PURPLE}Key Indicators (Last Value):{RESET}\n"
        key_inds_map = { # Map display name to DataFrame column name
            "RSI": "rsi", "Stoch K": "stoch_k", "Stoch D": "stoch_d",
            "ATR": "atr", "MACD": "macd", "MACD Hist": "histogram",
            "ADX": "adx", "MFI": "mfi", "CCI": "cci", "WR": "wr",
            "Upper BB": "upper_band", "Lower BB": "lower_band", "PSAR": "psar",
        }
        indicator_lines = []
        for name, col in key_inds_map.items():
            val_list = self.indicator_values.get(col)
            if val_list and pd.notna(val_list[-1]):
                val = val_list[-1]
                if 'atr' in col.lower(): fmt = ".5f"
                elif 'band' in col.lower() or 'psar' in col.lower(): fmt = price_format # Use price format for price-based indicators
                else: fmt = ".2f" # Default format
                indicator_lines.append(f"  {name:<10}: {val:{fmt}}")
            else:
                indicator_lines.append(f"  {name:<10}: N/A")

        # Arrange in two columns
        half = (len(indicator_lines) + 1) // 2
        for i in range(half):
            left = indicator_lines[i]
            right = indicator_lines[i + half] if (i + half) < len(indicator_lines) else ""
            output += f"{left:<25}{right}\n"
        return output

    def _format_rsi_cross(self) -> str:
        """Formats the RSI 20/100 cross information."""
        output = ""
        rsi20_list = self.indicator_values.get('rsi_20')
        rsi100_list = self.indicator_values.get('rsi_100')

        # Check if data exists and we have at least 2 rows for comparison
        if rsi20_list and rsi100_list and len(self.df) >= 2:
            rsi20_last = rsi20_list[-1]
            rsi100_last = rsi100_list[-1]
            rsi20_prev = self.df['rsi_20'].iloc[-2]
            rsi100_prev = self.df['rsi_100'].iloc[-2]

            if all(pd.notna(v) for v in [rsi20_last, rsi100_last, rsi20_prev, rsi100_prev]):
                if rsi20_last > rsi100_last and rsi20_prev <= rsi100_prev:
                    output += f"  {NEON_GREEN}RSI 20/100 Cross:{RESET} ▲ Bullish\n"
                elif rsi20_last < rsi100_last and rsi20_prev >= rsi100_prev:
                    output += f"  {NEON_RED}RSI 20/100 Cross:{RESET} ▼ Bearish\n"
                # else: output += "  RSI 20/100 Cross: Neutral\n" # Optional: Show neutral state
        return output


    def _format_levels(
        self, supports: List, resistances: List, price_format: str
    ) -> str:
        """Formats the support and resistance levels section."""
        output = f"{NEON_PURPLE}Support & Resistance Levels:{RESET}\n"
        if supports:
            for s_name, s_val in supports:
                output += f"  S: {s_name:<20} @ ${s_val:{price_format}}\n"
        else:
            output += "  No nearby support identified.\n"
        if resistances:
            for r_name, r_val in resistances:
                output += f"  R: {r_name:<20} @ ${r_val:{price_format}}\n"
        else:
            output += "  No nearby resistance identified.\n"
        return output

    def _format_scalping_signals(self, price_format: str) -> str:
        """Formats the potential scalping signals section."""
        output = f"{NEON_PURPLE}Potential Scalping Signals:{RESET}\n"
        if self.scalping_signals:
            for signal in self.scalping_signals:
                sig_col = NEON_GREEN if signal["type"] == "LONG" else NEON_RED
                # Determine decimals from price_format string
                try:
                    decimals = int(price_format.split('.')[-1][:-1])
                    quantizer = Decimal(f"1e-{decimals}")
                except ValueError:
                     quantizer = Decimal("1e-2") # Fallback

                # Quantize entry and SL to match display precision
                entry_val = signal["entry"].quantize(
                    quantizer, rounding=ROUND_DOWN if signal["type"] == "LONG" else ROUND_UP
                )
                sl_val = signal["stop_loss"].quantize(
                    quantizer, rounding=ROUND_DOWN if signal["type"] == "LONG" else ROUND_UP
                )

                output += (
                    f"  {sig_col}{signal['type']} Signal ({signal['confidence']}%):{RESET}"
                    f" Entry ~${entry_val:{price_format}}, SL ${sl_val:{price_format}}\n"
                    f"    Factors: {signal['factors']}\n"
                )
        else:
            enabled = self._safe_get_config("scalping", "enabled", default=False)
            min_conf = self._safe_get_config("scalping", "min_confidence_level", default=50)
            output += (
                f"  No signals meeting min confidence ({min_conf}%)"
                + ("." if enabled else " (Signals Disabled).") + "\n"
            )
        return output

    def _format_analysis_output(
        self,
        current_price: Decimal,
        timestamp_str: str,
        trend_data: Dict,
        supports: List,
        resistances: List,
        ob_analysis: str,
    ) -> str:
        """Formats the final analysis report string by combining parts."""
        price_format = self._get_price_format(current_price)

        output = self._format_header(current_price, timestamp_str)
        output += "\n" + self._format_trend(trend_data)
        output += "\n" + self._format_indicators(price_format)
        output += self._format_rsi_cross() # Add RSI cross info
        output += self._format_levels(supports, resistances, price_format)
        output += f"{NEON_PURPLE}Orderbook Analysis:{RESET}\n{ob_analysis}\n" # Already formatted
        output += self._format_scalping_signals(price_format)
        output += "=================================================="
        return output


# --- Main Execution Logic ---

def main():
    """Main function to run the trading analysis bot."""
    symbol = ""
    while True:
        s = input(f"{NEON_BLUE}Enter trading symbol (e.g., BTCUSDT): {RESET}").upper().strip()
        # Basic validation: not empty and no obviously invalid chars
        if s and not any(c in s for c in [" ", "/", "\\", ":", "*", "?", '"', "<", ">", "|"]):
            symbol = s
            break
        print(f"{NEON_RED}Invalid symbol format. Please enter a valid symbol.{RESET}")

    interval = ""
    while True:
        interval_input = input(
            f"{NEON_BLUE}Enter timeframe ({', '.join(VALID_INTERVALS)}) "
            f"[Default: {CONFIG.get('interval', '15')}]: {RESET}"
        ).strip()
        if not interval_input:
            interval = CONFIG.get("interval", "15") # Use default from config
            print(f"{NEON_YELLOW}Using default interval: {interval}{RESET}")
            break
        if interval_input in VALID_INTERVALS:
            interval = interval_input
            break
        print(f"{NEON_RED}Invalid interval. Valid options: {', '.join(VALID_INTERVALS)}{RESET}")

    # Initialize logger *after* getting symbol
    logger = setup_logger(symbol)
    analysis_interval_seconds = CONFIG.get("analysis_interval_seconds", 30)
    api_retry_delay = CONFIG.get("api_retry_delay_seconds", RETRY_DELAY_SECONDS)

    logger.info("--- Neonta Analysis Bot Started ---")
    logger.info(f"Symbol: {symbol}, Interval: {interval}, Update Freq: {analysis_interval_seconds}s")
    logger.info(f"Scalping Signals Enabled: {CONFIG.get('scalping', {}).get('enabled', False)}")
    logger.info(f"Using config file: {CONFIG_FILE}")
    logger.info(f"Log directory: {LOG_DIRECTORY}")
    logger.info(f"Display timezone: {LOCAL_TIMEZONE.key}")

    # --- Create persistent sessions/instances ---
    requests_session = create_session()
    ccxt_exchange = None
    try:
        ccxt_exchange = ccxt.bybit({
            'apiKey': API_KEY,
            'secret': API_SECRET,
            'options': {
                'recvWindow': CONFIG.get('ccxt_recv_window_ms', 10000),
                'defaultType': 'linear', # Explicitly set for linear contracts
            },
            'enableRateLimit': True, # Let ccxt handle rate limits
        })
        # Optional: Load markets to ensure connectivity early
        ccxt_exchange.load_markets()
        logger.info("ccxt exchange initialized successfully.")
    except ccxt.AuthenticationError as auth_err:
         logger.critical(f"{NEON_RED}ccxt Authentication Failed: {auth_err}. Check API Key/Secret.{RESET}")
         return # Exit if auth fails
    except Exception as e:
        logger.critical(f"{NEON_RED}Failed to initialize ccxt Bybit exchange: {e}{RESET}")
        return # Exit if ccxt fails to initialize

    analyzer: Optional[TradingAnalyzer] = None

    while True:
        loop_start_time = time.time()
        logger.debug(f"Starting analysis cycle for {symbol}...")
        try:
            # 1. Fetch Price
            current_price = fetch_current_price(requests_session, symbol, logger)
            if current_price is None:
                logger.warning(f"Failed to fetch current price. Retrying in {api_retry_delay}s...")
                time.sleep(api_retry_delay)
                continue

            # 2. Fetch Klines
            # Determine limit dynamically based on longest indicator needs
            limit_needed = max(
                 200, # Base limit
                 CONFIG.get('momentum_ma_long', 26) + 50,
                 CONFIG.get('bollinger_bands_period', 20) + 50,
                 CONFIG.get('rsi_period', 14) + CONFIG.get('stoch_rsi_stoch_window', 12) + 50 # Stoch RSI
                 # Add other long periods + buffer
            )
            df = fetch_klines(
                requests_session, symbol, interval, limit=limit_needed, logger=logger
            )
            if df.empty or len(df) < 50: # Need a reasonable minimum
                logger.warning(
                    f"Failed/Insufficient kline data ({len(df)} candles). Retrying in {api_retry_delay}s..."
                )
                time.sleep(api_retry_delay)
                continue

            # 3. Analyze Data
            try:
                logger.debug("Initializing/Updating TradingAnalyzer...")
                # Re-initialize analyzer with fresh data each loop
                analyzer = TradingAnalyzer(df, logger, CONFIG, symbol, interval)

                timestamp_utc_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S %Z")
                logger.debug(f"Starting analysis at {timestamp_utc_str}...")

                analyzer.analyze(current_price, timestamp_utc_str, requests_session, ccxt_exchange)

                logger.debug("Analysis complete.")

            except ValueError as ve: # Catch init errors (e.g., empty df)
                logger.error(f"Failed to initialize/run analyzer: {ve}. Retrying...")
                time.sleep(api_retry_delay)
                continue
            except Exception as analysis_e:
                logger.exception(f"Unexpected error during analysis phase: {analysis_e}")
                # Optional delay if analysis errors are frequent
                # time.sleep(api_retry_delay / 2)


        # Handle specific network/API errors that might bypass lower-level retries
        except (requests.exceptions.RequestException, ccxt.NetworkError, ccxt.ExchangeError) as net_err:
            logger.warning(f"Network/API Error in main loop: {net_err}. Retrying in {api_retry_delay}s...")
            time.sleep(api_retry_delay)
        # Handle user interruption
        except KeyboardInterrupt:
            logger.info(f"{NEON_YELLOW}Analysis stopped by user.{RESET}")
            print(f"\n{NEON_YELLOW}Exiting Neonta Bot.{RESET}")
            break
        # Catch any other unexpected errors in the main loop
        except Exception as e:
            logger.exception(
                f"{NEON_RED}Critical unexpected error in main loop: {e}. Retrying in {api_retry_delay}s...{RESET}"
            )
            time.sleep(api_retry_delay)

        # --- Loop Timing Control ---
        loop_end_time = time.time()
        elapsed_time = loop_end_time - loop_start_time
        sleep_time = max(0, analysis_interval_seconds - elapsed_time)
        logger.debug(f"Analysis cycle took {elapsed_time:.2f}s.")
        if sleep_time > 0:
            logger.debug(f"Sleeping for {sleep_time:.2f}s.")
            time.sleep(sleep_time)
        else:
            logger.warning(
                f"Analysis loop execution time ({elapsed_time:.2f}s) exceeded interval "
                f"({analysis_interval_seconds}s). Running next cycle immediately."
            )


if __name__ == "__main__":
    try:
        main()
    except ValueError as val_err: # Catch startup config errors (e.g., missing API keys)
        print(f"{NEON_RED}Startup Configuration Error: {val_err}{RESET}")
        # Attempt to log startup errors if possible
        try:
            startup_logger = setup_logger("STARTUP_ERROR")
            startup_logger.critical(f"Startup Configuration Error: {val_err}")
        except Exception:
            pass # Ignore if logger setup fails during critical error
    except Exception as e:
        print(f"{NEON_RED}Fatal error during startup: {e}{RESET}")
        try:
            startup_logger = setup_logger("STARTUP_ERROR")
            startup_logger.critical("Fatal error during startup", exc_info=True)
        except Exception:
            pass
