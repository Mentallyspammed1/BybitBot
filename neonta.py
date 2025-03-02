import os
import logging
import requests
import pandas as pd
import numpy as np
from datetime import datetime
import hmac
import hashlib
import time
from dotenv import load_dotenv
from typing import Dict, Tuple, List, Union, Optional
from colorama import init, Fore, Style
from zoneinfo import ZoneInfo
from decimal import Decimal, getcontext
import json
from logging.handlers import RotatingFileHandler
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import ccxt

getcontext().prec = 10
init(autoreset=True)
load_dotenv()

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
NEON_GREEN = Fore.LIGHTGREEN_EX
NEON_BLUE = Fore.CYAN
NEON_PURPLE = Fore.MAGENTA
NEON_YELLOW = Fore.YELLOW
NEON_RED = Fore.LIGHTRED_EX
RESET = Style.RESET_ALL

os.makedirs(LOG_DIRECTORY, exist_ok=True)

class SensitiveFormatter(logging.Formatter):
    def format(self, record):
        msg = super().format(record)
        return msg.replace(API_KEY, "***").replace(API_SECRET, "***")

def load_config(filepath: str) -> dict:
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
        "sideways_atr_multiplier": 1.5,
        "indicators": {
            "ema_alignment": True,
            "momentum": True,
            "volume_confirmation": True,
            "divergence": True,
            "stoch_rsi": True,
            "rsi": True,
            "macd": False,
            "bollinger_bands": True,
        },
        "weight_sets": {
            "low_volatility": {
                "ema_alignment": 0.4,
                "momentum": 0.3,
                "volume_confirmation": 0.2,
                "divergence": 0.1,
                "stoch_rsi": 0.7,
                "rsi": 0.0,
                "macd": 0.0,
                "bollinger_bands": 0.0,
            }
        },
        "rsi_period": 14,
        "bollinger_bands_period": 20,
        "bollinger_bands_std_dev": 2,
        "orderbook_limit": 100,
        "orderbook_cluster_threshold": 1000,
    }

    if not os.path.exists(filepath):
        with open(filepath, 'w') as f:
            json.dump(default_config, f, indent=2)
        print(f"{NEON_YELLOW}Created new config file with defaults{RESET}")
        return default_config

    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        print(f"{NEON_YELLOW}Could not load or parse config. Loading defaults.{RESET}")
        return default_config

CONFIG = load_config(CONFIG_FILE)

def create_session() -> requests.Session:
    session = requests.Session()
    retries = Retry(
        total=MAX_API_RETRIES,
        backoff_factor=0.5,
        status_forcelist=RETRY_ERROR_CODES,
        allowed_methods=["GET", "POST"]
    )
    session.mount('https://', HTTPAdapter(max_retries=retries))
    return session

def setup_logger(symbol: str) -> logging.Logger:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = os.path.join(LOG_DIRECTORY, f"{symbol}_{timestamp}.log")
    logger = logging.getLogger(symbol)
    logger.setLevel(logging.INFO)

    file_handler = RotatingFileHandler(
        log_filename,
        maxBytes=10*1024*1024,
        backupCount=5
    )
    file_handler.setFormatter(SensitiveFormatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(SensitiveFormatter(NEON_BLUE + "%(asctime)s" + RESET + " - %(levelname)s - %(message)s"))
    logger.addHandler(stream_handler)

    return logger

def bybit_request(method: str, endpoint: str, params: Optional[dict] = None, logger: Optional[logging.Logger] = None) -> Optional[dict]:
    session = create_session()
    try:
        params = params or {}
        timestamp = str(int(datetime.now(TIMEZONE).timestamp() * 1000))
        signature_params = params.copy()
        signature_params['timestamp'] = timestamp
        param_str = "&".join(f"{key}={value}" for key, value in sorted(signature_params.items()))
        signature = hmac.new(API_SECRET.encode(), param_str.encode(), hashlib.sha256).hexdigest()
        headers = {
            "X-BAPI-API-KEY": API_KEY,
            "X-BAPI-TIMESTAMP": timestamp,
            "X-BAPI-SIGN": signature,
            "Content-Type": "application/json"
        }
        url = f"{BASE_URL}{endpoint}"
        request_kwargs = {
            'method': method,
            'url': url,
            'headers': headers,
            'timeout': 10
        }
        if method == "GET":
            request_kwargs['params'] = params
        elif method == "POST":
            request_kwargs['json'] = params

        response = session.request(**request_kwargs)
        response.raise_for_status()
        json_response = response.json()
        if json_response and json_response.get("retCode") == 0:
            return json_response
        else:
            if logger:
                logger.error(f"{NEON_RED}Bybit API error: {json_response.get('retCode')} - {json_response.get('retMsg')}{RESET}")
            return None

    except requests.exceptions.RequestException as e:
        if logger:
            logger.error(f"{NEON_RED}API request failed: {e}{RESET}")
        return None

def fetch_current_price(symbol: str, logger: logging.Logger) -> Union[Decimal, None]:
    endpoint = "/v5/market/tickers"
    params = {
        "category": "linear",
        "symbol": symbol
    }
    response = bybit_request("GET", endpoint, params, logger)
    if not response or response.get("retCode") != 0 or not response.get("result"):
        logger.error(f"{NEON_RED}Failed to fetch ticker data: {response}{RESET}")
        return None
    tickers = response["result"].get("list", [])
    for ticker in tickers:
        if ticker.get("symbol") == symbol:
            last_price_str = ticker.get("lastPrice")
            if not last_price_str:
                logger.error(f"{NEON_RED}No lastPrice in ticker data{RESET}")
                return None
            try:
                return Decimal(last_price_str)
            except Exception as e:
                logger.error(f"{NEON_RED}Error parsing last price: {e}{RESET}")
                return None
    logger.error(f"{NEON_RED}Symbol {symbol} not found in ticker data{RESET}")
    return None

def fetch_klines(symbol: str, interval: str, limit: int = 200, logger: logging.Logger = None) -> pd.DataFrame:
    try:
        endpoint = "/v5/market/kline"
        params = {"symbol": symbol, "interval": interval, "limit": limit, "category": "linear"}
        response = bybit_request("GET", endpoint, params, logger)
        if (
            response
            and response.get("retCode") == 0
            and response.get("result")
            and response["result"].get("list")
        ):
            data = response["result"]["list"]
            columns = ["start_time", "open", "high", "low", "close", "volume"]
            if data and len(data[0]) > 6 and data[0][6]:
                columns.append("turnover")
            df = pd.DataFrame(data, columns=columns)
            df["start_time"] = pd.to_numeric(df["start_time"])
            df["start_time"] = pd.to_datetime(df["start_time"], unit="ms")
            for col in ["open", "high", "low", "close", "volume", "turnover"]:
                df[col] = pd.to_numeric(df[col], errors="coerce")
            for col in ["open", "high", "low", "close", "volume", "turnover"]:
                if col not in df.columns:
                    df[col] = np.nan
            if not {"close", "high", "low", "volume"}.issubset(df.columns):
                if logger:
                    logger.error(f"{NEON_RED}Kline data missing required columns after processing.{RESET}")
                return pd.DataFrame()

            return df.astype({c: float for c in columns if c != "start_time"})

        if logger:
            logger.error(f"{NEON_RED}Failed to fetch klines: {response}{RESET}")
        return pd.DataFrame()
    except (requests.exceptions.RequestException, KeyError, ValueError, TypeError) as e:
        if logger:
            logger.exception(f"{NEON_RED}Error fetching klines: {e}{RESET}")
        return pd.DataFrame()

def fetch_orderbook(symbol: str, limit: int, logger: logging.Logger) -> Optional[dict]:
    retry_count = 0
    exchange = ccxt.bybit()
    while retry_count <= MAX_API_RETRIES:
        try:
            orderbook_data = exchange.fetch_order_book(symbol, limit=limit)
            if orderbook_data:
                return orderbook_data
            else:
                logger.error(f"{NEON_RED}Failed to fetch orderbook data from ccxt (empty response). Retrying in {RETRY_DELAY_SECONDS} seconds...{RESET}")
                time.sleep(RETRY_DELAY_SECONDS)
                retry_count += 1
        except ccxt.ExchangeError as e:
            if "orderbook_limit" in str(e).lower():
                logger.warning(f"{NEON_YELLOW}ccxt ExchangeError: orderbook_limit issue. Retrying in {RETRY_DELAY_SECONDS} seconds...{RESET}")
            else:
                logger.error(f"{NEON_RED}ccxt ExchangeError fetching orderbook: {e}. Retrying in {RETRY_DELAY_SECONDS} seconds...{RESET}")
            time.sleep(RETRY_DELAY_SECONDS)
            retry_count += 1
        except ccxt.NetworkError as e:
            logger.error(f"{NEON_RED}ccxt NetworkError fetching orderbook: {e}. Retrying in {RETRY_DELAY_SECONDS} seconds...{RESET}")
            time.sleep(RETRY_DELAY_SECONDS)
            retry_count += 1
        except Exception as e:
            logger.exception(f"{NEON_RED}Unexpected error fetching orderbook with ccxt: {e}. Retrying in {RETRY_DELAY_SECONDS} seconds...{RESET}")
            time.sleep(RETRY_DELAY_SECONDS)
            retry_count += 1

    logger.error(f"{NEON_RED}Max retries reached for orderbook fetch using ccxt. Aborting.{RESET}")
    return None


class TradingAnalyzer:
    def __init__(self, df: pd.DataFrame, logger: logging.Logger, config: dict, symbol: str, interval: str):
        self.df = df
        self.logger = logger
        self.levels = {}
        self.fib_levels = {}
        self.config = config
        self.signal = None
        self.weight_sets = config["weight_sets"]
        self.user_defined_weights = self.weight_sets["low_volatility"]
        self.symbol = symbol
        self.interval = interval

    def calculate_sma(self, window: int) -> pd.Series:
        try:
            return self.df["close"].rolling(window=window).mean()
        except KeyError as e:
            self.logger.error(f"{NEON_RED}Missing 'close' column for SMA calculation: {e}{RESET}")
            return pd.Series(dtype="float64")

    def calculate_momentum(self, period: int = 10) -> pd.Series:
        try:
            return ((self.df["close"] - self.df["close"].shift(period)) / self.df["close"].shift(period)) * 100
        except (KeyError, ZeroDivisionError) as e:
            self.logger.error(f"{NEON_RED}Momentum calculation error: {e}{RESET}")
            return pd.Series(dtype="float64")

    def calculate_cci(self, window: int = 20, constant: float = 0.015) -> pd.Series:
        try:
            typical_price = (self.df["high"] + self.df["low"] + self.df["close"]) / 3
            sma_typical_price = typical_price.rolling(window=window).mean()
            mean_deviation = typical_price.rolling(window=window).apply(lambda x: np.abs(x - x.mean()).mean(), raw=True)
            cci = (typical_price - sma_typical_price) / (constant * mean_deviation)
            return cci
        except (KeyError, ZeroDivisionError) as e:
            self.logger.error(f"{NEON_RED}CCI calculation error: {e}{RESET}")
            return pd.Series(dtype="float64")
        except Exception as e:
            self.logger.exception(f"{NEON_RED}Unexpected error during CCI calculation: {e}{RESET}")
            return pd.Series(dtype="float64")

    def calculate_williams_r(self, window: int = 14) -> pd.Series:
        try:
            highest_high = self.df["high"].rolling(window=window).max()
            lowest_low = self.df["low"].rolling(window=window).min()
            wr = (highest_high - self.df["close"]) / (highest_high - lowest_low) * -100
            return wr
        except KeyError as e:
            self.logger.error(f"{NEON_RED}Williams %R calculation error: {e}{RESET}")
            return pd.Series(dtype="float64")

    def calculate_mfi(self, window: int = 14) -> pd.Series:
        try:
            typical_price = (self.df["high"] + self.df["low"] + self.df["close"]) / 3
            raw_money_flow = typical_price * self.df["volume"]
            positive_flow = []
            negative_flow = []
            for i in range(1, len(typical_price)):
                if typical_price[i] > typical_price[i - 1]:
                    positive_flow.append(raw_money_flow[i - 1])
                    negative_flow.append(0)
                elif typical_price[i] < typical_price[i - 1]:
                    negative_flow.append(raw_money_flow[i - 1])
                    positive_flow.append(0)
                else:
                    positive_flow.append(0)
                    negative_flow.append(0)
            positive_mf = pd.Series(positive_flow).rolling(window=window).sum()
            negative_mf = pd.Series(negative_flow).rolling(window=window).sum()
            money_ratio = positive_mf / negative_mf
            mfi = 100 - (100 / (1 + money_ratio))
            return mfi
        except (KeyError, ZeroDivisionError) as e:
            self.logger.error(f"{NEON_RED}MFI calculation error: {e}{RESET}")
            return pd.Series(dtype="float64")

    def calculate_fibonacci_retracement(self, high: float, low: float, current_price: float) -> Dict[str, float]:
        try:
            diff = high - low
            if diff == 0:
                return {}
            fib_levels = {
                "Fib 23.6%": high - diff * 0.236,
                "Fib 38.2%": high - diff * 0.382,
                "Fib 50.0%": high - diff * 0.5,
                "Fib 61.8%": high - diff * 0.618,
                "Fib 78.6%": high - diff * 0.786,
                "Fib 88.6%": high - diff * 0.886,
                "Fib 94.1%": high - diff * 0.941,
            }
            self.levels = {"Support": {}, "Resistance": {}}
            for label, value in fib_levels.items():
                if value < current_price:
                    self.levels["Support"][label] = value
                elif value > current_price:
                    self.levels["Resistance"][label] = value
            self.fib_levels = fib_levels
            return self.fib_levels
        except ZeroDivisionError:
            self.logger.error(f"{NEON_RED}Fibonacci calculation error: Division by zero.{RESET}")
            return {}
        except Exception as e:
            self.logger.exception(f"{NEON_RED}Fibonacci calculation error: {e}{RESET}")
            return {}

    def calculate_pivot_points(self, high: float, low: float, close: float):
        try:
            pivot = (high + low + close) / 3
            r1 = (2 * pivot) - low
            s1 = (2 * pivot) - high
            r2 = pivot + (high - low)
            s2 = pivot - (high - low)
            r3 = high + 2 * (pivot - low)
            s3 = low - 2 * (high - pivot)
            self.levels.update(
                {
                    "pivot": pivot,
                    "r1": r1,
                    "s1": s1,
                    "r2": r2,
                    "s2": s2,
                    "r3": r3,
                    "s3": s3,
                }
            )
        except Exception as e:
            self.logger.exception(f"{NEON_RED}Pivot point calculation error: {e}{RESET}")
            self.levels = {}

    def find_nearest_levels(self, current_price: float, num_levels: int = 5) -> Tuple[List[Tuple[str, float]], List[Tuple[str, float]]]:
        try:
            support_levels = []
            resistance_levels = []

            def process_level(label, value):
                if value < current_price:
                    support_levels.append((label, value))
                elif value > current_price:
                    resistance_levels.append((label, value))

            for label, value in self.levels.items():
                if isinstance(value, dict):
                    for sub_label, sub_value in value.items():
                        if isinstance(sub_value, (float, Decimal)):
                            process_level(f"{label} ({sub_label})", sub_value)
                elif isinstance(value, (float, Decimal)):
                    process_level(label, value)

            support_levels.sort(key=lambda x: abs(x[1] - current_price))
            resistance_levels.sort(key=lambda x: abs(x[1] - current_price))

            nearest_supports = support_levels[:num_levels]
            nearest_resistances = resistance_levels[:num_levels]
            return nearest_supports, nearest_resistances
        except (KeyError, TypeError) as e:
            self.logger.error(f"{NEON_RED}Error finding nearest levels: {e}{RESET}")
            return [], []
        except Exception as e:
            self.logger.exception(f"{NEON_RED}Unexpected error finding nearest levels: {e}{RESET}")
            return [], []

    def calculate_atr(self, window: int = 20) -> pd.Series:
        try:
            high_low = self.df["high"] - self.df["low"]
            high_close = (self.df["high"] - self.df["close"].shift()).abs()
            low_close = (self.df["low"] - self.df["close"].shift()).abs()
            tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            atr = tr.rolling(window=window).mean()
            return atr
        except KeyError as e:
            self.logger.error(f"{NEON_RED}ATR calculation error: {e}{RESET}")
            return pd.Series(dtype="float64")

    def calculate_rsi(self, window: int = 14) -> pd.Series:
        try:
            delta = self.df["close"].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
            rs = gain / loss
            rsi = np.where(loss == 0, 100, 100 - (100 / (1 + rs)))
            return pd.Series(rsi, index=self.df.index)
        except ZeroDivisionError:
            self.logger.error(f"{NEON_RED}RSI calculation error: Division by zero (handled). Returning NaN.{RESET}")
            return pd.Series(np.nan, index=self.df.index)
        except KeyError as e:
            self.logger.error(f"{NEON_RED}RSI calculation error: Missing column - {e}{RESET}")
            return pd.Series(dtype="float64")
        except Exception as e:
            self.logger.exception(f"{NEON_RED}Unexpected error during RSI calculation: {e}{RESET}")
            return pd.Series(dtype="float64")

    def calculate_stoch_rsi(self, rsi_window: int = 14, stoch_window: int = 12,
                            k_window: int = 3, d_window: int = 3) -> pd.DataFrame:
        try:
            rsi = self.calculate_rsi(window=rsi_window)
            stoch_rsi = (rsi - rsi.rolling(stoch_window).min()) / (
                  rsi.rolling(stoch_window).max() - rsi.rolling(stoch_window).min())
            k_line = stoch_rsi.rolling(window=k_window).mean()
            d_line = k_line.rolling(window=d_window).mean()
            return pd.DataFrame({"stoch_rsi": stoch_rsi, "k": k_line, "d": d_line})
        except (ZeroDivisionError, KeyError) as e:
            self.logger.error(f"{NEON_RED}Stochastic RSI calculation error: {e}{RESET}")
            return pd.DataFrame()

    def calculate_momentum_ma(self) -> None:
        try:
            self.df["momentum"] = self.df["close"].diff(self.config["momentum_period"])
            self.df["momentum_ma_short"] = self.df["momentum"].rolling(window=self.config["momentum_ma_short"]).mean()
            self.df["momentum_ma_long"] = self.df["momentum"].rolling(window=self.config["momentum_ma_long"]).mean()
            self.df["volume_ma"] = self.df["volume"].rolling(window=self.config["volume_ma_period"]).mean()
        except KeyError as e:
            self.logger.error(f"{NEON_RED}Momentum/MA calculation error: Missing column {e}{RESET}")

    def calculate_macd(self) -> pd.DataFrame:
        try:
            close_prices = self.df["close"]
            ma_short = close_prices.ewm(span=12, adjust=False).mean()
            ma_long = close_prices.ewm(span=26, adjust=False).mean()
            macd = ma_short - ma_long
            signal = macd.ewm(span=9, adjust=False).mean()
            histogram = macd - signal
            return pd.DataFrame({"macd": macd, "signal": signal, "histogram": histogram})
        except KeyError:
            self.logger.error(f"{NEON_RED}Missing 'close' column for MACD calculation.{RESET}")
            return pd.DataFrame()

    def detect_macd_divergence(self) -> str | None:
        if self.df.empty or len(self.df) < 30:
            return None

        macd_df = self.calculate_macd()
        if macd_df.empty:
            return None

        prices = self.df["close"]
        macd_histogram = macd_df["histogram"]

        if (prices.iloc[-2] > prices.iloc[-1] and
            macd_histogram.iloc[-2] < macd_histogram.iloc[-1]):
            return "bullish"
        elif (prices.iloc[-2] < prices.iloc[-1] and
              macd_histogram.iloc[-2] > macd_histogram.iloc[-1]):
            return "bearish"

        return None

    def calculate_ema(self, window: int) -> pd.Series:
        try:
            return self.df["close"].ewm(span=window, adjust=False).mean()
        except KeyError as e:
            self.logger.error(f"{NEON_RED}Missing 'close' column for EMA calculation: {e}{RESET}")
            return pd.Series(dtype="float64")

    def determine_trend_momentum(self) -> dict:
        if self.df.empty or len(self.df) < 26:
            return {"trend": "Insufficient Data", "strength": 0}

        atr = self.calculate_atr()
        if atr.iloc[-1] == 0:
            self.logger.warning(f"{NEON_YELLOW}ATR is zero, cannot calculate trend strength.{RESET}")
            return {"trend": "Neutral", "strength": 0}

        self.calculate_momentum_ma()

        if self.df["momentum_ma_short"].iloc[-1] > self.df["momentum_ma_long"].iloc[-1]:
            trend = "Uptrend"
        elif self.df["momentum_ma_short"].iloc[-1] < self.df["momentum_ma_long"].iloc[-1]:
            trend = "Downtrend"
        else:
            trend = "Neutral"

        trend_strength = abs(self.df["momentum_ma_short"].iloc[-1] - self.df["momentum_ma_long"].iloc[-1]) / atr.iloc[-1]
        return {"trend": trend, "strength": trend_strength}

    def calculate_adx(self, window: int = 14) -> float:
        try:
            df = self.df.copy()
            df["TR"] = pd.concat([
                df["high"] - df["low"],
                abs(df["high"] - df["close"].shift()),
                abs(df["low"] - df["close"].shift())
            ], axis=1).max(axis=1)

            df["+DM"] = np.where(
                (df["high"] - df["high"].shift()) > (df["low"].shift() - df["low"]),
                np.maximum(df["high"] - df["high"].shift(), 0),
                0
            )
            df["-DM"] = np.where(
                (df["low"].shift() - df["low"]) > (df["high"] - df["high"].shift()),
                np.maximum(df["low"].shift() - df["low"], 0),
                0
            )

            df["TR"] = df["TR"].rolling(window).sum()
            df["+DM"] = df["+DM"].rolling(window).sum()
            df["-DM"] = df["-DM"].rolling(window).sum()

            df["+DI"] = 100 * (df["+DM"] / df["TR"])
            df["-DI"] = 100 * (df["-DM"] / df["TR"])
            df["DX"] = 100 * (abs(df["+DI"] - df["-DI"]) / (df["+DI"] + df["-DI"]))

            adx = df["DX"].rolling(window).mean().iloc[-1]
            return adx

        except (KeyError, ZeroDivisionError) as e:
            self.logger.error(f"{NEON_RED}ADX calculation error: {e}{RESET}")
            return 0.0
        except Exception as e:
            self.logger.exception(f"{NEON_RED}Unexpected ADX calculation error: {e}{RESET}")
            return 0.0

    def calculate_obv(self) -> pd.Series:
        try:
            obv = np.where(self.df["close"] > self.df["close"].shift(1),
                                    self.df["volume"],
                           np.where(self.df["close"] < self.df["close"].shift(1),
                                    -self.df["volume"],
                                    0))
            return pd.Series(np.cumsum(obv), index=self.df.index)
        except KeyError as e:
            self.logger.error(f"{NEON_RED}OBV calculation error: Missing column {e}{RESET}")
            return pd.Series(dtype="float64")

    def calculate_adi(self) -> pd.Series:
        try:
            money_flow_multiplier = ((self.df["close"] - self.df["low"]) - (self.df["high"] - self.df["close"])) / (self.df["high"] - self.df["low"])
            money_flow_volume = money_flow_multiplier * self.df["volume"]
            return money_flow_volume.cumsum()
        except (KeyError, ZeroDivisionError) as e:
            self.logger.error(f"{NEON_RED}ADI calculation error: {e}{RESET}")
            return pd.Series(dtype="float64")

    def calculate_psar(self, acceleration=0.02, max_acceleration=0.2) -> pd.Series:
        psar = pd.Series(index=self.df.index, dtype="float64")
        psar.iloc[0] = self.df["low"].iloc[0]

        trend = 1
        ep = self.df["high"].iloc[0]
        af = acceleration

        for i in range(1, len(self.df)):
            if trend == 1:
                psar.iloc[i] = psar.iloc[i-1] + af * (ep - psar.iloc[i-1])
                if self.df["low"].iloc[i] < psar.iloc[i]:
                    trend = -1
                    psar.iloc[i] = ep
                    ep = self.df["low"].iloc[i]
                    af = acceleration
                else:
                    if self.df["high"].iloc[i] > ep:
                        ep = self.df["high"].iloc[i]
                        af = min(af + acceleration, max_acceleration)

            elif trend == -1:
                psar.iloc[i] = psar.iloc[i-1] + af * (ep - psar.iloc[i-1])
                if self.df["high"].iloc[i] > psar.iloc[i]:
                    trend = 1
                    psar.iloc[i] = ep
                    ep = self.df["high"].iloc[i]
                    af = acceleration
                else:
                    if self.df["low"].iloc[i] < ep:
                        ep = self.df["low"].iloc[i]
                        af = min(af + acceleration, max_acceleration)
        return psar

    def calculate_fve(self) -> pd.Series:
        try:
            force = self.df["close"].diff() * self.df["volume"]
            return force.cumsum()
        except KeyError as e:
            self.logger.error(f"{NEON_RED}FVE calculation error: {e}{RESET}")
            return pd.Series(dtype="float64")

    def calculate_bollinger_bands(self, window: int = 40, num_std_dev: float = 2.0) -> pd.DataFrame:
        try:
            rolling_mean = self.df["close"].rolling(window=window).mean()
            rolling_std = self.df["close"].rolling(window=window).std()
            upper_band = rolling_mean + (rolling_std * num_std_dev)
            lower_band = rolling_mean - (rolling_std * num_std_dev)
            return pd.DataFrame({"upper_band": upper_band, "middle_band": rolling_mean, "lower_band": lower_band}, index=self.df.index)
        except KeyError as e:
            self.logger.error(f"{NEON_RED}Bollinger Bands calculation error: {e}{RESET}")
            return pd.DataFrame()

    def analyze_orderbook_levels(self, orderbook: dict, current_price: Decimal) -> str:
        if not orderbook:
            return f"{NEON_YELLOW}Orderbook data not available.{RESET}"
        bids = pd.DataFrame(orderbook['bids'], columns=['price', 'size'], dtype=float)
        asks = pd.DataFrame(orderbook['asks'], columns=['price', 'size'], dtype=float)

        bids['price'] = bids['price'].astype(float)
        bids['size'] = bids['size'].astype(float)
        asks['price'] = asks['price'].astype(float)
        asks['size'] = asks['size'].astype(float)

        analysis_output = ""

        def check_cluster_at_level(level_name, level_price, bids_df, asks_df, threshold=self.config["orderbook_cluster_threshold"]):
            bid_volume_at_level = bids_df[(bids_df['price'] <= level_price + (level_price * 0.0005)) & (bids_df['price'] >= level_price - (level_price * 0.0005))]['size'].sum()
            ask_volume_at_level = asks_df[(asks_df['price'] <= level_price + (level_price * 0.0005)) & (asks_df['price'] >= level_price - (level_price * 0.0005))]['size'].sum()

            if bid_volume_at_level > threshold:
                return f"Significant bid volume ({bid_volume_at_level:.0f}) near {level_name} ${level_price:.2f}."
            if ask_volume_at_level > threshold:
                return f"Significant ask volume ({ask_volume_at_level:.0f}) near {level_name} ${level_price:.2f}."
            return None

        for level_type, levels in self.levels.get("Support", {}).items():
            cluster_analysis = check_cluster_at_level(f"Support {level_type}", levels, bids, asks)
            if cluster_analysis:
                analysis_output += f"  {NEON_GREEN}{cluster_analysis}{RESET}\n"
        for level_type, levels in self.levels.get("Resistance", {}).items():
            cluster_analysis = check_cluster_at_level(f"Resistance {level_type}", levels, bids, asks)
            if cluster_analysis:
                analysis_output += f"  {NEON_RED}{cluster_analysis}{RESET}\n"
        for level_name, level_price in self.levels.items():
            if isinstance(level_price, (float, int)):
                cluster_analysis = check_cluster_at_level(f"Pivot {level_name}", level_price, bids, asks)
                if cluster_analysis:
                    analysis_output += f"  {NEON_BLUE}{cluster_analysis}{RESET}\n"

        if not analysis_output:
            return "  No significant orderbook clusters detected near Fibonacci/Pivot levels."
        return analysis_output.strip()


    def analyze(self, current_price: Decimal, timestamp: str):
        high = self.df["high"].max()
        low = self.df["low"].min()
        close = self.df["close"].iloc[-1]
        self.calculate_fibonacci_retracement(high, low, float(current_price))
        self.calculate_pivot_points(high, low, close)
        nearest_supports, nearest_resistances = self.find_nearest_levels(float(current_price))

        trend_data = self.determine_trend_momentum()
        trend = trend_data.get("trend", "Unknown")
        strength = trend_data.get("strength", 0)
        atr = self.calculate_atr()
        obv = self.calculate_obv()
        rsi_20 = self.calculate_rsi(window=20)
        rsi_100 = self.calculate_rsi(window=100)
        mfi = self.calculate_mfi()
        cci = self.calculate_cci()
        wr = self.calculate_williams_r()
        adx = self.calculate_adx()
        adi = self.calculate_adi()
        sma = self.calculate_sma(10)
        psar = self.calculate_psar()
        fve = self.calculate_fve()
        macd_df = self.calculate_macd()
        bollinger_bands_df = self.calculate_bollinger_bands(
            window=self.config["bollinger_bands_period"],
            num_std_dev=self.config["bollinger_bands_std_dev"]
        )

        orderbook_data = fetch_orderbook(self.symbol, self.config["orderbook_limit"], self.logger)
        orderbook_analysis_str = self.analyze_orderbook_levels(orderbook_data, current_price)


        indicator_values = {
            "obv": obv.tail(3).tolist(),
            "rsi_20": rsi_20.tail(3).tolist(),
            "rsi_100": rsi_100.tail(3).tolist(),
            "mfi": mfi.tail(3).tolist(),
            "cci": cci.tail(3).tolist(),
            "wr": wr.tail(3).tolist(),
            "adx": [adx] * 3,
            "adi": adi.tail(3).tolist(),
            "mom": [trend_data] * 3,
            "sma": [self.df["close"].iloc[-1]],
            "psar": psar.tail(3).tolist(),
            "fve": fve.tail(3).tolist(),
            "macd": macd_df.tail(3).values.tolist() if not macd_df.empty else [],
            "bollinger_bands": bollinger_bands_df.tail(3).values.tolist() if not bollinger_bands_df.empty else [],
        }

        output = f"""
{NEON_BLUE}Exchange:{RESET} Bybit
{NEON_BLUE}Symbol:{RESET} {self.symbol}
{NEON_BLUE}Interval:{RESET} {self.interval}
{NEON_BLUE}Timestamp:{RESET} {timestamp}
{NEON_BLUE}Price:{RESET}   {self.df['close'].iloc[-3]:.2f} | {self.df['close'].iloc[-2]:.2f} | {self.df['close'].iloc[-1]:.2f}
{NEON_BLUE}Vol:{RESET}   {self.df['volume'].iloc[-3]:,} | {self.df['volume'].iloc[-2]:,} | {self.df['volume'].iloc[-1]:,}
{NEON_BLUE}Current Price:{RESET} {current_price:.2f}
{NEON_BLUE}ATR:{RESET} {atr.iloc[-1]:.4f}
{NEON_BLUE}Trend:{RESET} {trend} (Strength: {strength:.2f})
"""
        rsi_20_val = indicator_values["rsi_20"][-1]
        rsi_100_val = indicator_values["rsi_100"][-1]
        if rsi_20_val > rsi_100_val and indicator_values["rsi_20"][-2] <= indicator_values["rsi_100"][-2]:
            rsi_cross = f"{NEON_GREEN}RSI 20/100:{RESET} RSI 20 crossed ABOVE RSI 100"
        elif rsi_20_val < rsi_100_val and indicator_values["rsi_20"][-2] >= indicator_values["rsi_100"][-2]:
            rsi_cross = f"{NEON_RED}RSI 20/100:{RESET} RSI 20 crossed BELOW RSI 100"
        else:
            rsi_cross = f"{NEON_YELLOW}RSI 20/100:{RESET} No recent cross"

        output += rsi_cross + "\n"

        for indicator_name, values in indicator_values.items():
            if indicator_name not in ["rsi_20", "rsi_100"]:
                output += self.interpret_indicator(self.logger, indicator_name, values) + "\n"

        output += self.interpret_indicator(self.logger, "rsi_20", indicator_values["rsi_20"]) + "\n"
        output += self.interpret_indicator(self.logger, "rsi_100", indicator_values["rsi_100"]) + "\n"


        output += f"""
{NEON_BLUE}Support and Resistance Levels:{RESET}
"""
        for s in nearest_supports:
            output += f"S: {s[0]} ${s[1]:.3f}\n"
        for r in nearest_resistances:
            output += f"R: {r[0]} ${r[1]:.3f}\n"

        output += f"""
{NEON_BLUE}Orderbook Analysis near Fibonacci/Pivot Levels:{RESET}
{orderbook_analysis_str}
"""

        print(output)
        self.logger.info(output)

    def interpret_indicator(self, logger: logging.Logger, indicator_name: str, values: List[float]) -> Union[str, None]:
        if values is None or not values:
            return f"{indicator_name.upper()}: No data or calculation error."
        try:
            if indicator_name == "rsi_20":
                if values[-1] > 70:
                    return f"{NEON_RED}RSI 20:{RESET} Overbought ({values[-1]:.2f})"
                elif values[-1] < 30:
                    return f"{NEON_GREEN}RSI 20:{RESET} Oversold ({values[-1]:.2f})"
                else:
                    return f"{NEON_YELLOW}RSI 20:{RESET} Neutral ({values[-1]:.2f})"
            elif indicator_name == "rsi_100":
                if values[-1] > 70:
                    return f"{NEON_RED}RSI 100:{RESET} Overbought ({values[-1]:.2f})"
                elif values[-1] < 30:
                    return f"{NEON_GREEN}RSI 100:{RESET} Oversold ({values[-1]:.2f})"
                else:
                    return f"{NEON_YELLOW}RSI 100:{RESET} Neutral ({values[-1]:.2f})"
            elif indicator_name == "rsi":
                if values[-1] > 70:
                    return f"{NEON_RED}RSI:{RESET} Overbought ({values[-1]:.2f})"
                elif values[-1] < 30:
                    return f"{NEON_GREEN}RSI:{RESET} Oversold ({values[-1]:.2f})"
                else:
                    return f"{NEON_YELLOW}RSI:{RESET} Neutral ({values[-1]:.2f})"
            elif indicator_name == "mfi":
                if values[-1] > 80:
                    return f"{NEON_RED}MFI:{RESET} Overbought ({values[-1]:.2f})"
                elif values[-1] < 20:
                    return f"{NEON_GREEN}MFI:{RESET} Oversold ({values[-1]:.2f})"
                else:
                    return f"{NEON_YELLOW}MFI:{RESET} Neutral ({values[-1]:.2f})"
            elif indicator_name == "cci":
                if values[-1] > 100:
                    return f"{NEON_RED}CCI:{RESET} Overbought ({values[-1]:.2f})"
                elif values[-1] < -100:
                    return f"{NEON_GREEN}CCI:{RESET} Oversold ({values[-1]:.2f})"
                else:
                    return f"{NEON_YELLOW}CCI:{RESET} Neutral ({values[-1]:.2f})"
            elif indicator_name == "wr":
                if values[-1] < -80:
                    return f"{NEON_GREEN}Williams %R:{RESET} Oversold ({values[-1]:.2f})"
                elif values[-1] > -20:
                    return f"{NEON_RED}Williams %R:{RESET} Overbought ({values[-1]:.2f})"
                else:
                    return f"{NEON_YELLOW}Williams %R:{RESET} Neutral ({values[-1]:.2f})"
            elif indicator_name == "adx":
                if values[0] > 25:
                    return f"{NEON_GREEN}ADX:{RESET} Trending ({values[0]:.2f})"
                else:
                    return f"{NEON_YELLOW}ADX:{RESET} Ranging ({values[0]:.2f})"
            elif indicator_name == "obv":
                return f"{NEON_BLUE}OBV:{RESET} {'Bullish' if values[-1] > values[-2] else 'Bearish' if values[-1] < values[-2] else 'Neutral'}"
            elif indicator_name == "adi":
                return f"{NEON_BLUE}ADI:{RESET} {'Accumulation' if values[-1] > values[-2] else 'Distribution' if values[-1] < values[-2] else 'Neutral'}"
            elif indicator_name == "mom":
                trend = values[0]["trend"]
                strength = values[0]["strength"]
                return f"{NEON_PURPLE}Momentum:{RESET} {trend} (Strength: {strength:.2f})"
            elif indicator_name == "sma":
                return f"{NEON_YELLOW}SMA (10):{RESET} {values[0]:.2f}"
            elif indicator_name == "psar":
                return f"{NEON_BLUE}PSAR:{RESET} {values[-1]:.4f} (Last Value)"
            elif indicator_name == "fve":
                return f"{NEON_BLUE}FVE:{RESET} {values[-1]:.0f} (Last Value)"
            elif indicator_name == "macd":
                macd_values = values[-1]
                if len(macd_values) == 3:
                    macd_line, signal_line, histogram = macd_values[0], macd_values[1], macd_values[2]
                    return (
                        f"{NEON_GREEN}MACD:{RESET} MACD={macd_line:.2f}, Signal={signal_line:.2f}, Histogram={histogram:.2f}"
                    )
                else:
                    return f"{NEON_RED}MACD:{RESET} Calculation issue."
            elif indicator_name == "bollinger_bands":
                bb_values = values[-1]
                if len(bb_values) == 3:
                    upper_band, middle_band, lower_band = bb_values[0], bb_values[1], bb_values[2]
                    if self.df["close"].iloc[-1] > upper_band:
                        return f"{NEON_RED}Bollinger Bands:{RESET} Price above Upper Band ({upper_band:.2f})"
                    elif self.df["close"].iloc[-1] < lower_band:
                        return f"{NEON_GREEN}Bollinger Bands:{RESET} Price below Lower Band ({lower_band:.2f})"
                    else:
                        return f"{NEON_YELLOW}Bollinger Bands:{RESET} Price within Bands (Upper={upper_band:.2f}, Middle={middle_band:.2f}, Lower={lower_band:.2f})"
                else:
                    return f"{NEON_RED}Bollinger Bands:{RESET} Calculation issue."
            else:
                return None
        except (TypeError, IndexError) as e:
            logger.error(f"Error interpreting {indicator_name}: {e}")
            return f"{indicator_name.upper()}: Interpretation error."
        except Exception as e:
            logger.error(f"Unexpected error interpreting {indicator_name}: {e}")
            return f"{indicator_name.upper()}: Unexpected error."

def main():
    symbol = ""
    while True:
        symbol = input(f"{NEON_BLUE}Enter trading symbol (e.g., BTCUSDT): {RESET}").upper().strip()
        if symbol:
            break
        print(f"{NEON_RED}Symbol cannot be empty.{RESET}")

    interval = ""
    while True:
        interval = input(f"{NEON_BLUE}Enter timeframe (e.g., {', '.join(VALID_INTERVALS)}): {RESET}").strip()
        if not interval:
            interval = CONFIG["interval"]
            print(f"{NEON_YELLOW}No interval provided.  Using default of {interval}{RESET}")
            break
        if interval in VALID_INTERVALS:
            break
        print(f"{NEON_RED}Invalid interval: {interval}{RESET}")

    logger = setup_logger(symbol)
    analysis_interval = CONFIG["analysis_interval"]
    retry_delay = CONFIG["retry_delay"]

    while True:
        try:
            current_price = fetch_current_price(symbol, logger)
            if current_price is None:
                logger.error(f"{NEON_RED}Failed to fetch current price. Retrying in {retry_delay} seconds...{RESET}")
                time.sleep(retry_delay)
                continue

            df = fetch_klines(symbol, interval, logger=logger)
            if df.empty:
                logger.error(f"{NEON_RED}Failed to fetch kline data. Retrying in {retry_delay} seconds...{RESET}")
                time.sleep(retry_delay)
                continue

            analyzer = TradingAnalyzer(df, logger, CONFIG, symbol, interval)
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            analyzer.analyze(current_price, timestamp)
            time.sleep(analysis_interval)
        except requests.exceptions.RequestException as e:
            logger.error(f"{NEON_RED}Network error: {e}. Retrying in {retry_delay} seconds...{RESET}")
            time.sleep(retry_delay)
        except KeyboardInterrupt:
            logger.info(f"{NEON_YELLOW}Analysis stopped by user.{RESET}")
            break
        except Exception as e:
            logger.exception(f"{NEON_RED}An unexpected error occurred: {e}. Retrying in {retry_delay} seconds...{RESET}")
            time.sleep(retry_delay)

if __name__ == "__main__":
    main()

