import logging
import os
import time
from typing import Any, Dict

import ccxt
import yaml
import numpy as np
import pandas as pd
from colorama import Fore, Style, init as colorama_init
from dotenv import load_dotenv
from ta.trend import EMAIndicator, MACD
from ta.momentum import RSIIndicator, StochasticRSIIndicator
from ta.volatility import AverageTrueRange

colorama_init(autoreset=True)
load_dotenv()

NEON_GREEN = Fore.GREEN
NEON_CYAN = Fore.CYAN
NEON_YELLOW = Fore.YELLOW
NEON_MAGENTA = Fore.MAGENTA
NEON_RED = Fore.RED
RESET_COLOR = Style.RESET_ALL

logger = logging.getLogger("EnhancedTradingBot")
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter(
    f"{NEON_CYAN}%(asctime)s - {NEON_YELLOW}%(levelname)s - {NEON_GREEN}%(message)s{RESET_COLOR}"
)
console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)
file_handler = logging.FileHandler("enhanced_trading_bot.log")
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)


def retry_api_call(max_retries=3, initial_delay=1):
    """Decorator to retry API calls with exponential backoff."""

    def decorator(func):
        def wrapper(*args, **kwargs):
            retries = 0
            delay = initial_delay
            while retries <= max_retries:
                try:
                    return func(*args, **kwargs)
                except ccxt.RateLimitExceeded as e:
                    logger.warning(
                        f"{Fore.YELLOW}Rate limit exceeded, retrying in {delay} seconds... "
                        f"(Retry {retries + 1}/{max_retries}){Style.RESET_ALL}"
                    )
                    time.sleep(delay)
                    delay *= 2  # Exponential backoff
                    retries += 1
                except ccxt.NetworkError as e:
                    logger.error(
                        f"{Fore.RED}Network error during API call: {e}. "
                        f"Retrying in {delay} seconds... (Retry {retries + 1}/{max_retries}){Style.RESET_ALL}"
                    )
                    time.sleep(delay)
                    delay *= 2
                    retries += 1
                except ccxt.ExchangeError as e:
                    logger.error(
                        f"{Fore.RED}Exchange error during API call: {e}. "
                        f"(Retry {retries + 1}/{max_retries}) {e}{Style.RESET_ALL}"
                    )
                    if "Order does not exist" in str(
                        e
                    ):  # Specific handling for non-critical order cancel errors.
                        return None  # Indicate non-critical failure
                    else:
                        time.sleep(delay)
                        delay *= 2
                        retries += 1
                except Exception as e:
                    logger.error(
                        f"{Fore.RED}Unexpected error during API call: {e}. "
                        f"(Retry {retries + 1}/{max_retries}){Style.RESET_ALL}"
                    )
                    time.sleep(delay)
                    delay *= 2
                    retries += 1
            logger.error(
                f"{Fore.RED}Max retries reached for API call. Aborting.{Style.RESET_ALL}"
            )
            return None  # Indicate failure

        return wrapper

    return decorator


class EnhancedTradingBot:
    """
    Enhanced Trading Bot with configurable signals and integrated Bybit trading via ccxt.
    """

    def __init__(self, symbol, config_file="config.yaml"):
        """
        Initializes the EnhancedTradingBot with configurations and exchange setup.

        Args:
            symbol (str): Trading symbol (e.g., 'BTC/USDT').
            config_file (str): Path to the configuration YAML file.
        """
        self.load_config(config_file)
        logger.info("Initializing EnhancedTradingBot...")

        # --- Exchange and API Configuration ---
        self.exchange_id = self.config["exchange"]["exchange_id"]
        self.api_key = os.getenv("BYBIT_API_KEY")
        self.api_secret = os.getenv("BYBIT_API_SECRET")
        self.simulation_mode = self.config["trading"]["simulation_mode"]

        # --- Trading Parameters ---
        self.symbol = symbol.upper()
        self.order_size_percentage = self.config["risk_management"][
            "order_size_percentage"
        ]
        self.take_profit_pct = self.config["risk_management"]["take_profit_percentage"]
        self.stop_loss_pct = self.config["risk_management"]["stop_loss_percentage"]

        # --- Technical Indicator Parameters ---
        self.ema_period = self.config["indicators"]["ema_period"]
        self.rsi_period = self.config["indicators"]["rsi_period"]
        self.macd_short_period = self.config["indicators"]["macd_short_period"]
        self.macd_long_period = self.config["indicators"]["macd_long_period"]
        self.macd_signal_period = self.config["indicators"]["macd_signal_period"]
        self.stoch_rsi_period = self.config["indicators"]["stoch_rsi_period"]
        self.stoch_rsi_k_period = self.config["indicators"]["stoch_rsi_k_period"]
        self.stoch_rsi_d_period = self.config["indicators"]["stoch_rsi_d_period"]
        self.volatility_window = self.config["indicators"]["volatility_window"]
        self.volatility_multiplier = self.config["indicators"]["volatility_multiplier"]

        # --- Order Book Analysis Parameters ---
        self.order_book_depth = self.config["order_book"]["depth"]
        self.imbalance_threshold = self.config["order_book"]["imbalance_threshold"]
        self.volume_cluster_threshold = self.config["order_book"][
            "volume_cluster_threshold"
        ]
        self.ob_delta_lookback = self.config["order_book"]["ob_delta_lookback"]
        self.cluster_proximity_threshold_pct = self.config["order_book"][
            "cluster_proximity_threshold_pct"
        ]

        # --- Trailing Stop Loss Parameters ---
        self.trailing_stop_loss_active = self.config["trailing_stop"][
            "trailing_stop_active"
        ]
        self.trailing_stop_callback = self.config["trailing_stop"][
            "trailing_stop_callback"
        ]
        self.high_since_entry = -np.inf
        self.low_since_entry = np.inf

        # --- Signal Weights ---
        self.ema_weight = self.config["signal_weights"]["ema_weight"]
        self.rsi_weight = self.config["signal_weights"]["rsi_weight"]
        self.macd_weight = self.config["signal_weights"]["macd_weight"]
        self.stoch_rsi_weight = self.config["signal_weights"]["stoch_rsi_weight"]
        self.imbalance_weight = self.config["signal_weights"]["imbalance_weight"]
        self.ob_delta_change_weight = self.config["signal_weights"][
            "ob_delta_change_weight"
        ]
        self.spread_weight = self.config["signal_weights"]["spread_weight"]
        self.cluster_proximity_weight = self.config["signal_weights"][
            "cluster_proximity_weight"
        ]

        # --- Position Tracking ---
        self.position = None
        self.entry_price = None
        self.order_amount = None
        self.trade_count = 0
        self.last_ob_delta = None
        self.last_spread = None
        self.bot_running_flag = True
        self.open_positions = []  # Track open positions

        # Initialize exchange connection
        self.exchange = self._initialize_exchange()

        logger.info(f"EnhancedTradingBot initialized for symbol: {self.symbol}")
        logger.info("EnhancedTradingBot initialization complete.")

    def load_config(self, config_file):
        """Loads configuration from YAML file."""
        try:
            with open(config_file, "r") as f:
                self.config = yaml.safe_load(f)
            logger.info(f"{Fore.GREEN}Configuration loaded from {config_file}{Style.RESET_ALL}")
        except FileNotFoundError:
            logger.error(
                f"{Fore.RED}
