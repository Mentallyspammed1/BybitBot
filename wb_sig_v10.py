from colorama import init, Fore, Style
from datetime import datetime
import os
import logging
import requests
import pandas as pd
import numpy as np
import hmac
import hashlib
import time
import json
from logging.handlers import RotatingFileHandler
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from dotenv import load_dotenv
from typing import Dict, Tuple, List, Union, Optional, Any
from zoneinfo import ZoneInfo
from decimal import Decimal, getcontext
import ccxt

# Initialize colorama and set precision
getcontext().prec = 10
init(autoreset=True)
load_dotenv()

# Neon Color Scheme
NEON_GREEN = Fore.LIGHTGREEN_EX
NEON_BLUE = Fore.CYAN
NEON_PURPLE = Fore.MAGENTA
NEON_YELLOW = Fore.YELLOW
NEON_RED = Fore.LIGHTRED_EX
NEON_CYAN = Fore.CYAN
RESET = Style.RESET_ALL

# Constants
API_KEY = os.getenv("BYBIT_API_KEY")
API_SECRET = os.getenv("BYBIT_API_SECRET")
if not API_KEY or not API_SECRET:
    raise ValueError("BYBIT_API_KEY and BYBIT_API_SECRET must be set in .env")

BASE_URL = os.getenv("BYBIT_BASE_URL", "https://api.bybit.com")
CONFIG_FILE = "config.json"
LOG_DIRECTORY = "bot_logs"
TIMEZONE = ZoneInfo("America/Chicago")
MAX_API_RETRIES = 3
RETRY_DELAY_SECONDS = 5
VALID_INTERVALS = ["1", "3", "5", "15", "30", "60", "120", "240", "D", "W", "M"]
RETRY_ERROR_CODES = [429, 500, 502, 503, 504]
DEFAULT_ATR_PERIOD = 14
DEFAULT_CCI_WINDOW = 20
DEFAULT_WILLIAMS_R_WINDOW = 14
DEFAULT_MFI_WINDOW = 14
DEFAULT_STOCH_RSI_WINDOW = 14
DEFAULT_STOCH_WINDOW = 12
DEFAULT_K_WINDOW = 3
DEFAULT_D_WINDOW = 3
DEFAULT_RSI_WINDOW = 14
DEFAULT_BOLLINGER_BANDS_PERIOD = 20
DEFAULT_BOLLINGER_BANDS_STD_DEV = 2
DEFAULT_SMA_10_WINDOW = 10
DEFAULT_EMA_SHORT_PERIOD = 9
DEFAULT_EMA_LONG_PERIOD = 21
FIB_LEVELS = [0.0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0]
LOOP_DELAY_SECONDS = 10

os.makedirs(LOG_DIRECTORY, exist_ok=True)


class SensitiveFormatter(logging.Formatter):
    """Formatter to redact sensitive information from logs."""

    def format(self, record: logging.LogRecord) -> str:
        msg = super().format(record)
        return msg.replace(API_KEY, "***").replace(API_SECRET, "***")


def load_config(filepath: str) -> Dict[str, Any]:
    """Load configuration from JSON file, creating default if not found."""
    default_config = {
        "interval": "1m",
        "analysis_interval": 5,
        "retry_delay": 5,
        "momentum_period": 7,
        "volume_ma_period": 15,
        "atr_period": DEFAULT_ATR_PERIOD,
        "ema_short_period": DEFAULT_EMA_SHORT_PERIOD,
        "ema_long_period": DEFAULT_EMA_LONG_PERIOD,
        "rsi_period": DEFAULT_RSI_WINDOW,
        "bollinger_bands_period": DEFAULT_BOLLINGER_BANDS_PERIOD,
        "bollinger_bands_std_dev": DEFAULT_BOLLINGER_BANDS_STD_DEV,
        "orderbook_limit": 50,
        "signal_score_threshold": 1.5,
        "stoch_rsi_oversold_threshold": 25,
        "stoch_rsi_overbought_threshold": 75,
        "stop_loss_multiple": 1.8,
        "take_profit_multiple": 0.7,
        "volume_confirmation_multiplier": 2.0,
        "scalping_signal_threshold": 3,
        "fibonacci_window": 50,  # Added for Fibonacci range
        "indicators": {
            "ema_alignment": True,
            "momentum": True,
            "volume_confirmation": True,
            "stoch_rsi": True,
            "rsi": True,
            "bollinger_bands": True,
            "vwap": True,
            "cci": True,
            "wr": True,
            "psar": True,
            "sma_10": True,
            "mfi": True,
        },
        "weight_sets": {
            "scalping": {
                "ema_alignment": 0.2,
                "momentum": 0.3,
                "volume_confirmation": 0.2,
                "stoch_rsi": 0.6,
                "rsi": 0.2,
                "bollinger_bands": 0.3,
                "vwap": 0.4,
                "cci": 0.3,
                "wr": 0.3,
                "psar": 0.2,
                "sma_10": 0.1,
            }
        },
    }

    if not os.path.exists(filepath):
        with open(filepath, "w") as f:
            json.dump(default_config, f, indent=4)
        print(f"{NEON_YELLOW}Created new config file with defaults{RESET}")
        return default_config

    try:
        with open(filepath, "r", encoding="utf-8") as f:
            config = json.load(f)
            _ensure_config_keys(config, default_config)
            return config
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"{NEON_YELLOW}Could not load config: {e}. Using defaults.{RESET}")
        return default_config


def _ensure_config_keys(config: Dict[str, Any], default_config: Dict[str, Any]) -> None:
    """Ensure all keys from default_config are in config, adding defaults if missing."""
    for key, value in default_config.items():
        if key not in config:
            config[key] = value
            print(f"{NEON_YELLOW}Added missing key '{key}' to config{RESET}")
        elif isinstance(value, dict) and isinstance(config[key], dict):
            _ensure_config_keys(config[key], value)


CONFIG = load_config(CONFIG_FILE)


def create_session() -> requests.Session:
    """Create a requests session with retry logic."""
    session = requests.Session()
    retries = Retry(
        total=MAX_API_RETRIES,
        backoff_factor=0.5,
        status_forcelist=RETRY_ERROR_CODES,
        allowed_methods=["GET", "POST"],
    )
    session.mount("https://", HTTPAdapter(max_retries=retries))
    return session


def setup_logger(symbol: str) -> logging.Logger:
    """Set up a logger for the given symbol."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = os.path.join(LOG_DIRECTORY, f"{symbol}_{timestamp}.log")
    logger = logging.getLogger(symbol)
    logger.setLevel(logging.INFO)

    file_handler = RotatingFileHandler(log_filename, maxBytes=10 * 1024 * 1024, backupCount=5)
    file_handler.setFormatter(SensitiveFormatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(SensitiveFormatter(f"{NEON_BLUE}%(asctime)s{RESET} - %(levelname)s - %(message)s"))
    logger.addHandler(stream_handler)
    return logger


def bybit_request(
    method: str, endpoint: str, params: Optional[dict] = None, logger: Optional[logging.Logger] = None
) -> Optional[dict]:
    """Send a request to Bybit API with retry logic."""
    session = create_session()
    params = params or {}
    timestamp = str(int(datetime.now(TIMEZONE).timestamp() * 1000))
    signature_params = params.copy()
    signature_params["timestamp"] = timestamp
    param_str = "&".join(f"{k}={v}" for k, v in sorted(signature_params.items()))
    signature = hmac.new(API_SECRET.encode(), param_str.encode(), hashlib.sha256).hexdigest()

    headers = {
        "X-BAPI-API-KEY": API_KEY,
        "X-BAPI-TIMESTAMP": timestamp,
        "X-BAPI-SIGN": signature,
        "Content-Type": "application/json",
    }
    url = f"{BASE_URL}{endpoint}"
    request_kwargs = {"method": method, "url": url, "headers": headers, "timeout": 10}
    if method == "GET":
        request_kwargs["params"] = params
    elif method == "POST":
        request_kwargs["json"] = params

    try:
        response = session.request(**request_kwargs)
        response.raise_for_status()
        json_response = response.json()
        if json_response.get("retCode") == 0:
            return json_response
        if logger:
            logger.error(f"{NEON_RED}API error: {json_response.get('retMsg')}{RESET}")
        return None
    except requests.exceptions.RequestException as e:
        if logger:
            logger.error(f"{NEON_RED}API request failed: {e}{RESET}")
        return None


def fetch_current_price(symbol: str, logger: logging.Logger) -> Optional[Decimal]:
    """Fetch the current price of a trading symbol."""
    endpoint = "/v5/market/tickers"
    params = {"category": "linear", "symbol": symbol}
    response = bybit_request("GET", endpoint, params, logger)
    if not response or "result" not in response or not response["result"].get("list"):
        logger.error(f"{NEON_RED}Failed to fetch ticker data{RESET}")
        return None
    for ticker in response["result"]["list"]:
        if ticker.get("symbol") == symbol:
            try:
                return Decimal(ticker.get("lastPrice", "0"))
            except Exception as e:
                logger.error(f"{NEON_RED}Error parsing price: {e}{RESET}")
                return None
    logger.error(f"{NEON_RED}Symbol {symbol} not found{RESET}")
    return None


def fetch_klines(
    symbol: str, interval: str, limit: int = 250, logger: logging.Logger = None
) -> pd.DataFrame:
    """Fetch kline data with caching for efficiency."""
    endpoint = "/v5/market/kline"
    params = {"symbol": symbol, "interval": interval, "limit": limit, "category": "linear"}
    response = bybit_request("GET", endpoint, params, logger)
    if not response or "result" not in response or not response["result"].get("list"):
        if logger:
            logger.error(f"{NEON_RED}Failed to fetch klines{RESET}")
        return pd.DataFrame()

    try:
        data = response["result"]["list"]
        columns = ["start_time", "open", "high", "low", "close", "volume"]
        if data and len(data[0]) > 6 and data[0][6]:
            columns.append("turnover")
        df = pd.DataFrame(data, columns=columns)
        df["start_time"] = pd.to_datetime(pd.to_numeric(df["start_time"]), unit="ms")
        for col in columns[1:]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        return df.astype({c: float for c in columns if c != "start_time"})
    except Exception as e:
        if logger:
            logger.error(f"{NEON_RED}Error processing klines: {e}{RESET}")
        return pd.DataFrame()


def fetch_orderbook(symbol: str, limit: int, logger: logging.Logger) -> Optional[dict]:
    """Fetch orderbook data using ccxt with retries."""
    exchange = ccxt.bybit()
    for attempt in range(MAX_API_RETRIES + 1):
        try:
            orderbook = exchange.fetch_order_book(symbol, limit=limit)
            if orderbook:
                return orderbook
            logger.error(f"{NEON_RED}Empty orderbook response{RESET}")
        except (ccxt.ExchangeError, ccxt.NetworkError) as e:
            logger.warning(f"{NEON_YELLOW}Fetch error: {e}. Retrying...{RESET}")
            time.sleep(RETRY_DELAY_SECONDS)
        except Exception as e:
            logger.error(f"{NEON_RED}Unexpected error: {e}{RESET}")
            time.sleep(RETRY_DELAY_SECONDS)
    logger.error(f"{NEON_RED}Max retries reached for orderbook{RESET}")
    return None


class TradingAnalyzer:
    """Analyze trading data and generate scalping signals."""

    def __init__(
        self, df: pd.DataFrame, logger: logging.Logger, config: Dict[str, Any], symbol: str, interval: str
    ):
        self.df = df
        self.logger = logger
        self.config = config
        self.symbol = symbol
        self.interval = interval
        self.indicator_values: Dict[str, float] = {}
        self.scalping_signals: Dict[str, int] = {"BUY": 0, "SELL": 0, "HOLD": 0}
        self.weights = config["weight_sets"]["scalping"]
        self.fib_levels_data: Dict[str, Decimal] = {}

    def calculate_atr(self, period: int = DEFAULT_ATR_PERIOD) -> pd.Series:
        """Calculate Average True Range (ATR)."""
        try:
            high_low = self.df["high"] - self.df["low"]
            high_close = np.abs(self.df["high"] - self.df["close"].shift())
            low_close = np.abs(self.df["low"] - self.df["close"].shift())
            true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            atr = true_range.rolling(period).mean()
            self.indicator_values["ATR"] = float(atr.iloc[-1]) if not atr.empty else np.nan
            return atr
        except KeyError as e:
            self.logger.error(f"{NEON_RED}ATR error: Missing column {e}{RESET}")
            return pd.Series(dtype="float64")

    def calculate_fibonacci_levels(self, window: int = 50) -> Dict[str, Decimal]:
        """Calculate Fibonacci retracement levels over a specified window."""
        df_slice = self.df.tail(window)
        high = Decimal(str(df_slice["high"].max()))
        low = Decimal(str(df_slice["low"].min()))
        diff = high - low

        levels = {}
        for level in FIB_LEVELS:
            level_name = f"Fib_{level*100:.1f}%"
            levels[level_name] = high - (diff * Decimal(str(level)))
        self.fib_levels_data = levels
        return levels

    def get_nearest_fibonacci_levels(self, current_price: Decimal, num_levels: int = 5) -> List[Tuple[str, Decimal]]:
        """Find nearest Fibonacci levels to the current price."""
        if not self.fib_levels_data:
            self.calculate_fibonacci_levels(self.config.get("fibonacci_window", 50))
        level_distances = [(name, level, abs(current_price - level)) for name, level in self.fib_levels_data.items()]
        level_distances.sort(key=lambda x: x[2])
        return [(name, level) for name, level, _ in level_distances[:num_levels]]

    def calculate_sma(self, window: int) -> pd.Series:
        """Calculate Simple Moving Average (SMA)."""
        try:
            sma = self.df["close"].rolling(window=window).mean()
            self.indicator_values[f"SMA{window}"] = float(sma.iloc[-1]) if not sma.empty else np.nan
            return sma
        except KeyError as e:
            self.logger.error(f"{NEON_RED}SMA error: {e}{RESET}")
            return pd.Series(dtype="float64")

    def calculate_ema(self, window: int) -> pd.Series:
        """Calculate Exponential Moving Average (EMA)."""
        try:
            ema = self.df["close"].ewm(span=window, adjust=False).mean()
            self.indicator_values[f"EMA{window}"] = float(ema.iloc[-1]) if not ema.empty else np.nan
            return ema
        except KeyError as e:
            self.logger.error(f"{NEON_RED}EMA error: {e}{RESET}")
            return pd.Series(dtype="float64")

    def calculate_ema_alignment(self) -> float:
        """Calculate EMA alignment score."""
        ema_short = self.calculate_ema(self.config.get("ema_short_period", DEFAULT_EMA_SHORT_PERIOD))
        ema_long = self.calculate_ema(self.config.get("ema_long_period", DEFAULT_EMA_LONG_PERIOD))
        if ema_short.empty or ema_long.empty:
            return 0.0
        latest_short = ema_short.iloc[-1]
        latest_long = ema_long.iloc[-1]
        current_price = self.df["close"].iloc[-1]
        return 1.0 if latest_short > latest_long and current_price > latest_short else -1.0 if latest_short < latest_long and current_price < latest_short else 0.0

    def calculate_momentum(self) -> pd.Series:
        """Calculate Momentum."""
        try:
            period = self.config["momentum_period"]
            momentum = ((self.df["close"] - self.df["close"].shift(period)) / self.df["close"].shift(period)) * 100
            self.indicator_values["Momentum"] = float(momentum.iloc[-1]) if not momentum.empty else np.nan
            return momentum
        except (KeyError, ZeroDivisionError) as e:
            self.logger.error(f"{NEON_RED}Momentum error: {e}{RESET}")
            return pd.Series(dtype="float64")

    def calculate_cci(self, window: int = DEFAULT_CCI_WINDOW) -> pd.Series:
        """Calculate Commodity Channel Index (CCI)."""
        try:
            typical_price = (self.df["high"] + self.df["low"] + self.df["close"]) / 3
            sma_tp = typical_price.rolling(window=window).mean()
            mean_dev = typical_price.rolling(window=window).apply(lambda x: np.abs(x - x.mean()).mean(), raw=True)
            cci = (typical_price - sma_tp) / (0.015 * mean_dev)
            self.indicator_values["CCI"] = float(cci.iloc[-1]) if not cci.empty else np.nan
            return cci
        except (KeyError, ZeroDivisionError) as e:
            self.logger.error(f"{NEON_RED}CCI error: {e}{RESET}")
            return pd.Series(dtype="float64")

    def calculate_williams_r(self, window: int = DEFAULT_WILLIAMS_R_WINDOW) -> pd.Series:
        """Calculate Williams %R."""
        try:
            highest_high = self.df["high"].rolling(window=window).max()
            lowest_low = self.df["low"].rolling(window=window).min()
            wr = (highest_high - self.df["close"]) / (highest_high - lowest_low) * -100
            self.indicator_values["Williams_R"] = float(wr.iloc[-1]) if not wr.empty else np.nan
            return wr
        except KeyError as e:
            self.logger.error(f"{NEON_RED}Williams %R error: {e}{RESET}")
            return pd.Series(dtype="float64")

    def calculate_mfi(self, window: int = DEFAULT_MFI_WINDOW) -> pd.Series:
        """Calculate Money Flow Index (MFI)."""
        try:
            typical_price = (self.df["high"] + self.df["low"] + self.df["close"]) / 3
            raw_money_flow = typical_price * self.df["volume"]
            positive_flow = pd.Series([mf if tp > tp_prev else 0 for tp, tp_prev, mf in zip(typical_price[1:], typical_price[:-1], raw_money_flow[1:])])
            negative_flow = pd.Series([mf if tp < tp_prev else 0 for tp, tp_prev, mf in zip(typical_price[1:], typical_price[:-1], raw_money_flow[1:])])
            positive_mf = positive_flow.rolling(window=window).sum()
            negative_mf = negative_flow.rolling(window=window).sum()
            money_ratio = positive_mf / negative_mf
            mfi = 100 - (100 / (1 + money_ratio))
            self.indicator_values["MFI"] = float(mfi.iloc[-1]) if not mfi.empty else np.nan
            return mfi
        except (KeyError, ZeroDivisionError) as e:
            self.logger.error(f"{NEON_RED}MFI error: {e}{RESET}")
            return pd.Series(dtype="float64")

    def calculate_vwap(self) -> pd.Series:
        """Calculate Volume Weighted Average Price (VWAP)."""
        try:
            typical_price = (self.df["high"] + self.df["low"] + self.df["close"]) / 3
            vwap = (self.df["volume"] * typical_price).cumsum() / self.df["volume"].cumsum()
            self.indicator_values["VWAP"] = float(vwap.iloc[-1]) if not vwap.empty else np.nan
            return vwap
        except KeyError as e:
            self.logger.error(f"{NEON_RED}VWAP error: {e}{RESET}")
            return pd.Series(dtype="float64")

    def calculate_psar(self, acceleration: float = 0.01, max_acceleration: float = 0.2) -> pd.Series:
        """Calculate Parabolic SAR (PSAR)."""
        psar = pd.Series(index=self.df.index, dtype="float64")
        psar.iloc[0] = self.df["low"].iloc[0]
        trend, ep, af = 1, self.df["high"].iloc[0], acceleration

        for i in range(1, len(self.df)):
            psar.iloc[i] = psar.iloc[i - 1] + af * (ep - psar.iloc[i - 1])
            if trend == 1:
                if self.df["low"].iloc[i] < psar.iloc[i]:
                    trend, psar.iloc[i], ep, af = -1, ep, self.df["low"].iloc[i], acceleration
                elif self.df["high"].iloc[i] > ep:
                    ep, af = self.df["high"].iloc[i], min(af + acceleration, max_acceleration)
            else:
                if self.df["high"].iloc[i] > psar.iloc[i]:
                    trend, psar.iloc[i], ep, af = 1, ep, self.df["high"].iloc[i], acceleration
                elif self.df["low"].iloc[i] < ep:
                    ep, af = self.df["low"].iloc[i], min(af + acceleration, max_acceleration)
        self.indicator_values["PSAR"] = float(psar.iloc[-1]) if not psar.empty else np.nan
        return psar

    def calculate_sma_10(self) -> pd.Series:
        """Calculate SMA with 10-period window."""
        return self.calculate_sma(DEFAULT_SMA_10_WINDOW)

    def calculate_stoch_rsi(self) -> pd.DataFrame:
        """Calculate Stochastic RSI."""
        try:
            rsi = self.calculate_rsi()
            stoch_rsi = (rsi - rsi.rolling(DEFAULT_STOCH_WINDOW).min()) / (rsi.rolling(DEFAULT_STOCH_WINDOW).max() - rsi.rolling(DEFAULT_STOCH_WINDOW).min())
            k_line = stoch_rsi.rolling(window=DEFAULT_K_WINDOW).mean()
            d_line = k_line.rolling(window=DEFAULT_D_WINDOW).mean()
            self.indicator_values["StochRSI_K"] = float(k_line.iloc[-1]) if not k_line.empty else np.nan
            self.indicator_values["StochRSI_D"] = float(d_line.iloc[-1]) if not d_line.empty else np.nan
            return pd.DataFrame({"stoch_rsi": stoch_rsi, "k": k_line, "d": d_line})
        except (ZeroDivisionError, KeyError) as e:
            self.logger.error(f"{NEON_RED}Stoch RSI error: {e}{RESET}")
            return pd.DataFrame()

    def calculate_rsi(self, window: int = DEFAULT_RSI_WINDOW) -> pd.Series:
        """Calculate Relative Strength Index (RSI)."""
        try:
            delta = self.df["close"].diff()
            gain = delta.where(delta > 0, 0).rolling(window=window).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
            rs = gain / loss
            rsi = np.where(loss == 0, 100, 100 - (100 / (1 + rs)))
            self.indicator_values["RSI"] = float(rsi[-1]) if len(rsi) > 0 else np.nan
            return pd.Series(rsi, index=self.df.index)
        except (KeyError, ZeroDivisionError) as e:
            self.logger.error(f"{NEON_RED}RSI error: {e}{RESET}")
            return pd.Series(dtype="float64")

    def calculate_bollinger_bands(self) -> pd.DataFrame:
        """Calculate Bollinger Bands."""
        try:
            period = self.config.get("bollinger_bands_period", DEFAULT_BOLLINGER_BANDS_PERIOD)
            std_dev = self.config.get("bollinger_bands_std_dev", DEFAULT_BOLLINGER_BANDS_STD_DEV)
            rolling_mean = self.df["close"].rolling(window=period).mean()
            rolling_std = self.df["close"].rolling(window=period).std()
            bb_upper = rolling_mean + (rolling_std * std_dev)
            bb_lower = rolling_mean - (rolling_std * std_dev)
            self.indicator_values["BB_Upper"] = float(bb_upper.iloc[-1]) if not bb_upper.empty else np.nan
            self.indicator_values["BB_Middle"] = float(rolling_mean.iloc[-1]) if not rolling_mean.empty else np.nan
            self.indicator_values["BB_Lower"] = float(bb_lower.iloc[-1]) if not bb_lower.empty else np.nan
            return pd.DataFrame({"bb_upper": bb_upper, "bb_mid": rolling_mean, "bb_lower": bb_lower})
        except KeyError as e:
            self.logger.error(f"{NEON_RED}Bollinger Bands error: {e}{RESET}")
            return pd.DataFrame()

    def generate_trading_signal(self, current_price: Decimal, orderbook_data: Optional[dict]) -> str:
        """Generate trading signal based on scalping strategy."""
        self.scalping_signals = {"BUY": 0, "SELL": 0, "HOLD": 0}
        signal_score = 0

        if self.config["indicators"]["ema_alignment"]:
            signal_score += self.calculate_ema_alignment() * self.weights["ema_alignment"]
        if self.config["indicators"]["momentum"]:
            self.calculate_momentum()
            signal_score += self._check_momentum()
        if self.config["indicators"]["volume_confirmation"]:
            signal_score += self._check_volume_confirmation()
        if self.config["indicators"]["stoch_rsi"]:
            self.calculate_stoch_rsi()
            signal_score += self._check_stoch_rsi()
        if self.config["indicators"]["rsi"]:
            self.calculate_rsi()
            signal_score += self._check_rsi()
        if self.config["indicators"]["cci"]:
            self.calculate_cci()
            signal_score += self._check_cci()
        if self.config["indicators"]["wr"]:
            self.calculate_williams_r()
            signal_score += self._check_williams_r()
        if self.config["indicators"]["psar"]:
            self.calculate_psar()
            signal_score += self._check_psar()
        if self.config["indicators"]["sma_10"]:
            self.calculate_sma_10()
            signal_score += self._check_sma_10()
        if self.config["indicators"]["vwap"]:
            self.calculate_vwap()
            signal_score += self._check_vwap()
        if self.config["indicators"]["mfi"]:
            self.calculate_mfi()
            signal_score += self._check_mfi()
        if self.config["indicators"]["bollinger_bands"]:
            signal_score += self._check_bollinger_bands()

        if orderbook_data:
            signal_score += self._check_orderbook(orderbook_data, current_price)

        threshold = self.config["scalping_signal_threshold"]
        if signal_score >= threshold:
            self.scalping_signals["BUY"] = 1
            return "BUY"
        elif signal_score <= -threshold:
            self.scalping_signals["SELL"] = 1
            return "SELL"
        self.scalping_signals["HOLD"] = 1
        return "HOLD"

    def _check_momentum(self) -> float:
        """Check momentum for signal scoring."""
        momentum = self.indicator_values.get("Momentum", np.nan)
        return self.weights["momentum"] if momentum > 0 else -self.weights["momentum"] if momentum < 0 else 0.0

    def _check_volume_confirmation(self) -> float:
        """Check volume confirmation for signal scoring."""
        volume_ma = self.df["volume"].rolling(window=self.config["volume_ma_period"]).mean().iloc[-1]
        current_volume = self.df["volume"].iloc[-1]
        multiplier = self.config["volume_confirmation_multiplier"]
        return self.weights["volume_confirmation"] if current_volume > volume_ma * multiplier else -self.weights["volume_confirmation"] if current_volume < volume_ma / multiplier else 0.0

    def _check_stoch_rsi(self) -> float:
        """Check Stochastic RSI for signal scoring."""
        k, d = self.indicator_values.get("StochRSI_K", np.nan), self.indicator_values.get("StochRSI_D", np.nan)
        if pd.isna(k) or pd.isna(d):
            return 0.0
        oversold, overbought = self.config["stoch_rsi_oversold_threshold"], self.config["stoch_rsi_overbought_threshold"]
        boost = 0.7  # Simplified confidence boost
        return self.weights["stoch_rsi"] + boost if k < oversold and d < oversold else -self.weights["stoch_rsi"] - boost if k > overbought and d > overbought else 0.0

    def _check_rsi(self) -> float:
        """Check RSI for signal scoring."""
        rsi = self.indicator_values.get("RSI", np.nan)
        return self.weights["rsi"] + 0.3 if rsi < 30 else -self.weights["rsi"] - 0.3 if rsi > 70 else 0.0

    def _check_cci(self) -> float:
        """Check CCI for signal scoring."""
        cci = self.indicator_values.get("CCI", np.nan)
        return self.weights["cci"] if cci < -100 else -self.weights["cci"] if cci > 100 else 0.0

    def _check_williams_r(self) -> float:
        """Check Williams %R for signal scoring."""
        wr = self.indicator_values.get("Williams_R", np.nan)
        return self.weights["wr"] if wr < -80 else -self.weights["wr"] if wr > -20 else 0.0

    def _check_psar(self) -> float:
        """Check PSAR for signal scoring."""
        psar = self.indicator_values.get("PSAR", np.nan)
        last_close = self.df["close"].iloc[-1]
        return self.weights["psar"] if last_close > psar else -self.weights["psar"] if last_close < psar else 0.0

    def _check_sma_10(self) -> float:
        """Check SMA_10 for signal scoring."""
        sma_10 = self.indicator_values.get("SMA10", np.nan)
        last_close = self.df["close"].iloc[-1]
        return self.weights["sma_10"] if last_close > sma_10 else -self.weights["sma_10"] if last_close < sma_10 else 0.0

    def _check_vwap(self) -> float:
        """Check VWAP for signal scoring."""
        vwap = self.indicator_values.get("VWAP", np.nan)
        last_close = self.df["close"].iloc[-1]
        return self.weights["vwap"] if last_close > vwap else -self.weights["vwap"] if last_close < vwap else 0.0

    def _check_mfi(self) -> float:
        """Check MFI for signal scoring."""
        mfi = self.indicator_values.get("MFI", np.nan)
        return self.weights["mfi"] + 0.3 if mfi < 20 else -self.weights["mfi"] - 0.3 if mfi > 80 else 0.0

    def _check_bollinger_bands(self) -> float:
        """Check Bollinger Bands for signal scoring."""
        bbands = self.calculate_bollinger_bands()
        if bbands.empty:
            return 0.0
        last_close = self.df["close"].iloc[-1]
        return self.weights["bollinger_bands"] if last_close < bbands["bb_lower"].iloc[-1] else -self.weights["bollinger_bands"] if last_close > bbands["bb_upper"].iloc[-1] else 0.0

    def _check_orderbook(self, orderbook_data: dict, current_price: Decimal) -> float:
        """Analyze order book for signal scoring."""
        bid_volume = sum(float(bid[1]) for bid in orderbook_data["bids"][:5])
        ask_volume = sum(float(ask[1]) for ask in orderbook_data["asks"][:5])
        return 0.1 if bid_volume > ask_volume * 1.2 else -0.1 if ask_volume > bid_volume * 1.2 else 0.0

    def calculate_entry_tp_sl(self, current_price: Decimal, signal: str) -> Tuple[float, float, float]:
        """Calculate entry, take profit, and stop loss levels."""
        atr = self.indicator_values.get("ATR", 0.0)
        entry = float(current_price)
        tp_multiple = self.config["take_profit_multiple"]
        sl_multiple = self.config["stop_loss_multiple"]
        take_profit = entry + (atr * tp_multiple) if signal == "BUY" else entry - (atr * tp_multiple)
        stop_loss = entry - (atr * sl_multiple) if signal == "BUY" else entry + (atr * sl_multiple)
        return entry, take_profit, stop_loss

    def calculate_confidence(self) -> float:
        """Calculate confidence score (placeholder)."""
        return 75.0  # Simplified for now, could be based on signal strength


def analyze_symbol(symbol: str, config: Dict[str, Any]) -> None:
    """Analyze trading data for a symbol and output signals."""
    logger = setup_logger(symbol)
    logger.info(f"Analyzing {symbol} with interval: {config['interval']}")

    klines = fetch_klines(symbol, config["interval"], logger=logger)
    if klines.empty:
        logger.error(f"{NEON_RED}Failed to fetch klines{RESET}")
        return

    orderbook_data = fetch_orderbook(symbol, config["orderbook_limit"], logger)
    if not orderbook_data:
        logger.warning(f"{NEON_YELLOW}Orderbook fetch failed{RESET}")

    analyzer = TradingAnalyzer(klines.copy(), logger, config, symbol, config["interval"])
    analyzer.calculate_atr()

    current_price = fetch_current_price(symbol, logger) or Decimal(str(klines["close"].iloc[-1]))
    signal = analyzer.generate_trading_signal(current_price, orderbook_data)
    entry, tp, sl = analyzer.calculate_entry_tp_sl(current_price, signal)
    confidence = analyzer.calculate_confidence()
    fib_levels = analyzer.get_nearest_fibonacci_levels(current_price)

    output = (
        f"\n{NEON_BLUE}--- Scalping Analysis for {symbol} ({config['interval']}) ---{RESET}\n"
        f"Current Price: {NEON_GREEN}{current_price}{RESET} - {datetime.now(TIMEZONE).strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"Signal: {NEON_GREEN}{signal}{RESET}\n"
        f"Entry: {NEON_YELLOW}{entry:.4f}{RESET}\n"
        f"Take Profit: {NEON_GREEN}{tp:.4f}{RESET}\n"
        f"Stop Loss: {NEON_RED}{sl:.4f}{RESET}\n"
        f"Confidence: {NEON_CYAN}{confidence:.2f}%{RESET}\n"
        "Indicators:\n" + "\n".join(f"  {k}: {NEON_YELLOW}{v:.4f}{RESET}" for k, v in analyzer.indicator_values.items()) +
        "\nFibonacci Levels:\n" + "\n".join(f"  {name}: {NEON_PURPLE}{level:.4f}{RESET}" for name, level in fib_levels)
    )
    print(output)
    logger.info(output)


def main() -> None:
    """Run scalping analysis in a loop."""
    symbol = input(f"{NEON_YELLOW}Enter symbol (e.g., BTCUSDT): {RESET}").upper()
    interval = input(f"{NEON_YELLOW}Enter interval (e.g., 1m, 5m): {RESET}").lower()

    if interval not in VALID_INTERVALS:
        print(f"{NEON_RED}Invalid interval: {VALID_INTERVALS}{RESET}")
        return

    CONFIG["interval"] = interval
    print(f"{NEON_CYAN}--- Neonta Scalping Bot v1.1 ---{RESET}")

    try:
        while True:
            analyze_symbol(symbol, CONFIG)
            time.sleep(LOOP_DELAY_SECONDS)
    except KeyboardInterrupt:
        print(f"{NEON_CYAN}--- Stopped by User ---{RESET}")


if __name__ == "__main__":
    main()
