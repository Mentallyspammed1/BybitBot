#!/usr/bin/env python3
import logging
import os
import time
import random
import threading
from typing import Any, Dict

import ccxt
import yaml
import numpy as np
from colorama import Fore, Style, init as colorama_init
from dotenv import load_dotenv
from ta.trend import EMAIndicator, MACD
from ta.momentum import RSIIndicator, StochRSIIndicator
from ta.volatility import AverageTrueRange

# Initialize colorama and load environment variables
colorama_init(autoreset=True)
load_dotenv()

# Color definitions for logging
NEON_GREEN = Fore.GREEN
NEON_CYAN = Fore.CYAN
NEON_YELLOW = Fore.YELLOW
NEON_MAGENTA = Fore.MAGENTA
NEON_RED = Fore.RED
RESET_COLOR = Style.RESET_ALL

# Setup base logger
logger = logging.getLogger("EnhancedTradingBot")
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter(f"{NEON_CYAN}%(asctime)s - {NEON_YELLOW}%(levelname)s - {NEON_GREEN}%(message)s{RESET_COLOR}")
console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)
file_handler = logging.FileHandler("enhanced_trading_bot.log")
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)


def retry_api_call(max_retries=3, initial_delay=1):
    """Decorator to retry API calls with exponential backoff and randomized jitter."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            retries = 0
            delay = initial_delay
            while retries <= max_retries:
                try:
                    return func(*args, **kwargs)
                except ccxt.RateLimitExceeded as e:
                    logger.warning(f"{Fore.YELLOW}Rate limit exceeded, retrying in {delay:.2f} seconds... (Retry {retries + 1}/{max_retries}){Style.RESET_ALL}")
                except ccxt.NetworkError as e:
                    logger.error(f"{Fore.RED}Network error during API call: {e}. Retrying in {delay:.2f} seconds... (Retry {retries + 1}/{max_retries}){Style.RESET_ALL}")
                except ccxt.ExchangeError as e:
                    logger.error(f"{Fore.RED}Exchange error during API call: {e}. (Retry {retries + 1}/{max_retries}){Style.RESET_ALL}")
                    if 'Order does not exist' in str(e):
                        return None  # Specific handling for non-critical order cancel errors.
                except Exception as e:
                    logger.error(f"{Fore.RED}Unexpected error during API call: {e}. (Retry {retries + 1}/{max_retries}){Style.RESET_ALL}")
                time.sleep(delay + random.uniform(0, 1))
                delay *= 2  # Exponential backoff
                retries += 1
            logger.error(f"{Fore.RED}Max retries reached for API call. Aborting.{Style.RESET_ALL}")
            return None  # Indicate failure
        return wrapper
    return decorator


class EnhancedTradingBot:
    """
    Enhanced Trading Bot with configurable signals, integrated Bybit trading via ccxt,
    and historical order flow analysis threaded into the signal generation logic.
    """
    def __init__(self, symbol, config_file='config.yaml'):
        self.load_config(config_file)
        self._validate_config()  # Ensure all config sections are present
        self._setup_logging()    # Setup logging as per config
        logger.info("Initializing EnhancedTradingBot...")

        # --- Exchange and API Configuration ---
        self.exchange_id = self.config['exchange']['exchange_id']
        self.api_key = os.getenv('BYBIT_API_KEY')
        self.api_secret = os.getenv('BYBIT_API_SECRET')
        self.simulation_mode = self.config['trading']['simulation_mode']

        # --- Trading Parameters ---
        self.symbol = symbol.upper()
        self.order_size_percentage = self.config['risk_management']['order_size_percentage']
        self.take_profit_pct = self.config['risk_management']['take_profit_percentage']
        self.stop_loss_pct = self.config['risk_management']['stop_loss_percentage']
        self.dynamic_position_sizing = self.config['trading'].get('dynamic_position_sizing', False)

        # --- Technical Indicator Parameters ---
        self.ema_period = self.config['indicators']['ema_period']
        self.rsi_period = self.config['indicators']['rsi_period']
        self.macd_short_period = self.config['indicators']['macd_short_period']
        self.macd_long_period = self.config['indicators']['macd_long_period']
        self.macd_signal_period = self.config['indicators']['macd_signal_period']
        self.stoch_rsi_period = self.config['indicators']['stoch_rsi_period']
        self.stoch_rsi_k_period = self.config['indicators']['stoch_rsi_k_period']
        self.stoch_rsi_d_period = self.config['indicators']['stoch_rsi_d_period']
        self.volatility_window = self.config['indicators']['volatility_window']
        self.volatility_multiplier = self.config['indicators']['volatility_multiplier']

        # --- Order Book Analysis Parameters ---
        self.order_book_depth = self.config['order_book']['depth']
        self.imbalance_threshold = self.config['order_book']['imbalance_threshold']
        self.volume_cluster_threshold = self.config['order_book']['volume_cluster_threshold']
        self.ob_delta_lookback = self.config['order_book']['ob_delta_lookback']
        self.cluster_proximity_threshold_pct = self.config['order_book']['cluster_proximity_threshold_pct']

        # --- Trailing Stop Loss Parameters ---
        self.trailing_stop_loss_active = self.config['trailing_stop']['trailing_stop_active']
        self.trailing_stop_callback = self.config['trailing_stop']['trailing_stop_callback']
        self.high_since_entry = -np.inf
        self.low_since_entry = np.inf

        # --- Signal Weights ---
        self.ema_weight = self.config['signal_weights']['ema_weight']
        self.rsi_weight = self.config['signal_weights']['rsi_weight']
        self.macd_weight = self.config['signal_weights']['macd_weight']
        self.stoch_rsi_weight = self.config['signal_weights']['stoch_rsi_weight']
        self.imbalance_weight = self.config['signal_weights']['imbalance_weight']
        self.ob_delta_change_weight = self.config['signal_weights']['ob_delta_change_weight']
        self.spread_weight = self.config['signal_weights']['spread_weight']
        self.cluster_proximity_weight = self.config['signal_weights']['cluster_proximity_weight']
        # New: Historical order flow signal weight
        self.historical_order_flow_weight = self.config['signal_weights'].get('historical_order_flow_weight', 0)

        # --- Position Tracking ---
        self.open_positions = []  # Track open positions
        self.last_ob_delta = None
        self.last_spread = None
        self.last_atr = None   # Updated with latest volatility measurement

        # --- Historical Order Flow Tracking ---
        self.order_flow_history = []  # List of recent order flow deltas
        self.order_flow_lock = threading.Lock()
        self.historical_window = self.config['trading'].get('historical_order_flow_window', 30)  # seconds

        # Start background thread for historical order flow data if enabled
        if self.config['trading'].get('historical_order_flow_enabled', False):
            threading.Thread(target=self._update_order_flow_history, daemon=True).start()
            logger.info("Historical order flow thread started.")

        # Initialize exchange connection
        self.exchange = self._initialize_exchange()

        logger.info(f"EnhancedTradingBot initialized for symbol: {self.symbol}")
        logger.info("EnhancedTradingBot initialization complete.")

    def load_config(self, config_file):
        """Loads configuration from YAML file."""
        try:
            with open(config_file, 'r') as f:
                self.config = yaml.safe_load(f)
            logger.info(f"{Fore.GREEN}Configuration loaded from {config_file}{Style.RESET_ALL}")
        except FileNotFoundError:
            logger.error(f"{Fore.RED}Configuration file {config_file} not found. Exiting.{Style.RESET_ALL}")
            exit()
        except yaml.YAMLError as e:
            logger.error(f"{Fore.RED}Error parsing configuration file {config_file}: {e}. Exiting.{Style.RESET_ALL}")
            exit()

    def _validate_config(self):
        """Validates that the configuration file contains all necessary parameters."""
        required_sections = ['exchange', 'trading', 'risk_management', 'indicators', 'order_book', 'trailing_stop', 'signal_weights', 'trading_thresholds', 'logging']
        for section in required_sections:
            if section not in self.config:
                logger.error(f"{Fore.RED}Configuration section '{section}' is missing in config.yaml. Exiting.{Style.RESET_ALL}")
                exit()
        logger.debug("Configuration sections validated.")

    def _setup_logging(self):
        """Sets up logging level based on configuration."""
        log_level_str = self.config['logging'].get('level', 'DEBUG').upper()
        log_level = getattr(logging, log_level_str, logging.DEBUG)
        logger.setLevel(log_level)
        logger.info(f"Logging level set to: {logging.getLevelName(log_level)}")

    def _initialize_exchange(self):
        """Initializes the exchange connection using ccxt."""
        logger.info(f"Initializing exchange: {self.exchange_id.upper()}...")
        try:
            exchange = getattr(ccxt, self.exchange_id)({
                'apiKey': self.api_key,
                'secret': self.api_secret,
                'options': {'defaultType': 'future'},
                'recvWindow': 60000
            })
            exchange.load_markets()
            logger.info(f"{Fore.GREEN}Connected to {self.exchange_id.upper()} successfully.{Style.RESET_ALL}")
            return exchange
        except Exception as e:
            logger.error(f"{Fore.RED}Error initializing exchange: {e}{Style.RESET_ALL}")
            exit()

    @retry_api_call()
    def fetch_market_price(self):
        """Fetches the current market price for the trading symbol."""
        ticker = self.exchange.fetch_ticker(self.symbol)
        if ticker and 'last' in ticker:
            price = ticker['last']
            logger.debug(f"Fetched market price: {price:.2f}")
            return price
        else:
            logger.warning(f"{Fore.YELLOW}Market price unavailable.{Style.RESET_ALL}")
            return None

    @retry_api_call()
    def fetch_order_book(self):
        """Fetches the order book for the symbol."""
        try:
            orderbook = self.exchange.fetch_order_book(self.symbol, limit=self.order_book_depth)
            logger.debug("Order book fetched successfully.")
            return orderbook
        except Exception as e:
            logger.error(f"{Fore.RED}Error fetching order book: {e}{Style.RESET_ALL}")
            return None

    def calculate_ema(self, prices):
        """Calculates Exponential Moving Average."""
        ema_indicator = EMAIndicator(prices, window=self.ema_period, fillna=True)
        return ema_indicator.ema().iloc[-1]

    def calculate_rsi(self, prices):
        """Calculates Relative Strength Index."""
        rsi_indicator = RSIIndicator(prices, window=self.rsi_period, fillna=True)
        return rsi_indicator.rsi().iloc[-1]

    def calculate_macd(self, prices):
        """Calculates Moving Average Convergence Divergence."""
        macd_indicator = MACD(prices,
                              window_fast=self.macd_short_period,
                              window_slow=self.macd_long_period,
                              window_sign=self.macd_signal_period, fillna=True)
        macd_line = macd_indicator.macd().iloc[-1]
        signal_line = macd_indicator.macd_signal().iloc[-1]
        histogram = macd_indicator.macd_diff().iloc[-1]
        return macd_line, signal_line, histogram

    def calculate_stoch_rsi(self, prices):
        """Calculates Stochastic RSI."""
        stoch_rsi_indicator = StochRSIIndicator(prices,
                                                 window=self.stoch_rsi_period,
                                                 smooth_k=self.stoch_rsi_k_period,
                                                 smooth_d=self.stoch_rsi_d_period, fillna=True)
        stoch_rsi_k = stoch_rsi_indicator.stochrsi_k().iloc[-1]
        stoch_rsi_d = stoch_rsi_indicator.stochrsi_d().iloc[-1]
        return stoch_rsi_k, stoch_rsi_d

    def calculate_atr(self, high_prices, low_prices, close_prices):
        """Calculates Average True Range."""
        atr_indicator = AverageTrueRange(high_prices, low_prices, close_prices, window=self.volatility_window, fillna=True)
        return atr_indicator.average_true_range().iloc[-1]

    def calculate_volatility(self, prices):
        """Calculates volatility using ATR as a measure."""
        # If prices is a numpy array, use it directly
        high_prices = np.array(prices).rolling(window=self.volatility_window).max() if hasattr(prices, "rolling") else np.maximum.accumulate(prices)
        low_prices = np.array(prices).rolling(window=self.volatility_window).min() if hasattr(prices, "rolling") else np.minimum.accumulate(prices)
        close_prices = prices
        atr = self.calculate_atr(high_prices, low_prices, close_prices)
        return atr

    def calculate_indicators(self, ohlcv):
        """Calculates technical indicators from OHLCV data."""
        # Extract prices from OHLCV data (assumes [timestamp, open, high, low, close, volume])
        close_prices = np.array([candle[4] for candle in ohlcv])
        high_prices = np.array([candle[2] for candle in ohlcv])
        low_prices = np.array([candle[3] for candle in ohlcv])

        ema = self.calculate_ema(close_prices)
        rsi = self.calculate_rsi(close_prices)
        macd_line, signal_line, histogram = self.calculate_macd(close_prices)
        stoch_rsi_k, stoch_rsi_d = self.calculate_stoch_rsi(close_prices)
        volatility = self.calculate_volatility(close_prices)
        self.last_atr = volatility  # update ATR for dynamic sizing

        logger.debug(f"Calculated Indicators - EMA: {ema:.2f}, RSI: {rsi:.2f}, MACD: {macd_line:.2f}/{signal_line:.2f}, StochRSI: {stoch_rsi_k:.2f}/{stoch_rsi_d:.2f}, ATR: {volatility:.4f}")
        return {
            'ema': ema,
            'rsi': rsi,
            'macd_line': macd_line,
            'macd_signal': signal_line,
            'macd_histogram': histogram,
            'stoch_rsi_k': stoch_rsi_k,
            'stoch_rsi_d': stoch_rsi_d,
            'volatility': volatility
        }

    def analyze_order_book_imbalance(self, orderbook):
        """Analyzes order book imbalance."""
        if not orderbook or not orderbook.get('bids') or not orderbook.get('asks'):
            return 0  # Neutral if insufficient data

        bid_volume = sum(bid[1] for bid in orderbook['bids'])
        ask_volume = sum(ask[1] for ask in orderbook['asks'])
        imbalance = (bid_volume - ask_volume) / (bid_volume + ask_volume) if (bid_volume + ask_volume) > 0 else 0
        logger.debug(f"Order Book Imbalance: {imbalance:.2f}")
        return imbalance

    def analyze_volume_clusters(self, orderbook):
        """Analyzes volume clusters in the order book."""
        if not orderbook or not orderbook.get('bids') or not orderbook.get('asks'):
            return None, None

        bid_clusters = sorted(orderbook['bids'], key=lambda x: x[1], reverse=True)[:3]
        ask_clusters = sorted(orderbook['asks'], key=lambda x: x[1], reverse=True)[:3]

        top_bid_cluster_price = bid_clusters[0][0] if bid_clusters else None
        top_ask_cluster_price = ask_clusters[0][0] if ask_clusters else None

        logger.debug(f"Top Bid Cluster Price: {top_bid_cluster_price}, Top Ask Cluster Price: {top_ask_cluster_price}")
        return top_bid_cluster_price, top_ask_cluster_price

    def analyze_order_book_delta(self, orderbook):
        """Calculates order book delta change."""
        if not orderbook or not orderbook.get('bids') or not orderbook.get('asks'):
            return 0

        current_ob_delta = sum(bid[1] for bid in orderbook['bids']) - sum(ask[1] for ask in orderbook['asks'])
        delta_change = current_ob_delta - self.last_ob_delta if self.last_ob_delta is not None else 0
        self.last_ob_delta = current_ob_delta
        logger.debug(f"Order Book Delta Change: {delta_change:.2f}")
        return delta_change

    def analyze_spread(self, orderbook):
        """Analyzes the volume-weighted spread between bid and ask prices."""
        if not orderbook or not orderbook.get('bids') or not orderbook.get('asks'):
            return None

        bids = orderbook['bids']
        asks = orderbook['asks']
        total_bid_vol = sum(bid[1] for bid in bids)
        total_ask_vol = sum(ask[1] for ask in asks)
        if total_bid_vol == 0 or total_ask_vol == 0:
            return None

        weighted_bid = sum(bid[0] * bid[1] for bid in bids) / total_bid_vol
        weighted_ask = sum(ask[0] * ask[1] for ask in asks) / total_ask_vol
        spread = weighted_ask - weighted_bid
        spread_percentage = (spread / weighted_bid) * 100 if weighted_bid > 0 else 0
        self.last_spread = spread_percentage
        logger.debug(f"Volume Weighted Spread: {spread:.4f}, Spread Percentage: {spread_percentage:.2f}%")
        return spread_percentage

    # ---------------------------
    # Modular signal generation
    # ---------------------------
    def _get_ema_signal(self, indicators):
        current_price = self.fetch_market_price()
        if current_price is None or indicators['ema'] is None:
            return 0
        if current_price > indicators['ema']:
            logger.debug(f"EMA Bullish Signal: +{self.ema_weight}")
            return self.ema_weight
        elif current_price < indicators['ema']:
            logger.debug(f"EMA Bearish Signal: -{self.ema_weight}")
            return -self.ema_weight
        return 0

    def _get_rsi_signal(self, indicators):
        if indicators['rsi'] is None:
            return 0
        if indicators['rsi'] < 30:
            logger.debug(f"RSI Oversold Bullish Signal: +{self.rsi_weight}")
            return self.rsi_weight
        elif indicators['rsi'] > 70:
            logger.debug(f"RSI Overbought Bearish Signal: -{self.rsi_weight}")
            return -self.rsi_weight
        return 0

    def _get_macd_signal(self, indicators):
        if indicators['macd_line'] is None or indicators['macd_signal'] is None:
            return 0
        if indicators['macd_line'] > indicators['macd_signal']:
            logger.debug(f"MACD Bullish Crossover Signal: +{self.macd_weight}")
            return self.macd_weight
        elif indicators['macd_line'] < indicators['macd_signal']:
            logger.debug(f"MACD Bearish Crossover Signal: -{self.macd_weight}")
            return -self.macd_weight
        return 0

    def _get_stoch_rsi_signal(self, indicators):
        if indicators['stoch_rsi_k'] is None or indicators['stoch_rsi_d'] is None:
            return 0
        if indicators['stoch_rsi_k'] < 20 and indicators['stoch_rsi_d'] < 20 and indicators['stoch_rsi_k'] > indicators['stoch_rsi_d']:
            logger.debug(f"Stoch RSI Bullish Signal: +{self.stoch_rsi_weight}")
            return self.stoch_rsi_weight
        elif indicators['stoch_rsi_k'] > 80 and indicators['stoch_rsi_d'] > 80 and indicators['stoch_rsi_k'] < indicators['stoch_rsi_d']:
            logger.debug(f"Stoch RSI Bearish Signal: -{self.stoch_rsi_weight}")
            return -self.stoch_rsi_weight
        return 0

    def _get_order_book_imbalance_signal(self, orderbook):
        imbalance = self.analyze_order_book_imbalance(orderbook)
        signal_value = imbalance * self.imbalance_weight
        logger.debug(f"Order Book Imbalance Signal: {signal_value:.2f}")
        return signal_value

    def _get_ob_delta_change_signal(self, orderbook):
        ob_delta_change = self.analyze_order_book_delta(orderbook)
        signal_value = ob_delta_change * self.ob_delta_change_weight
        logger.debug(f"Order Book Delta Change Signal: {signal_value:.2f}")
        return signal_value

    def _get_spread_signal(self, orderbook):
        spread_percentage = self.analyze_spread(orderbook)
        if spread_percentage is not None:
            signal_value = -spread_percentage * self.spread_weight  # Wider spread is negative
            logger.debug(f"Spread Signal: {signal_value:.2f}")
            return signal_value
        return 0

    def _get_cluster_proximity_signal(self, orderbook):
        current_price = self.fetch_market_price()
        bid_cluster_price, ask_cluster_price = self.analyze_volume_clusters(orderbook)
        if current_price and bid_cluster_price and ask_cluster_price:
            bid_proximity = abs(current_price - bid_cluster_price) / current_price if current_price > 0 else 0
            ask_proximity = abs(current_price - ask_cluster_price) / current_price if current_price > 0 else 0
            if bid_proximity <= (self.cluster_proximity_threshold_pct / 100):
                logger.debug(f"Bid Cluster Proximity Bullish Signal: +{self.cluster_proximity_weight}")
                return self.cluster_proximity_weight
            if ask_proximity <= (self.cluster_proximity_threshold_pct / 100):
                logger.debug(f"Ask Cluster Proximity Bearish Signal: -{self.cluster_proximity_weight}")
                return -self.cluster_proximity_weight
        return 0

    def _get_historical_order_flow_signal(self):
        """
        Computes an additional signal based on historical order flow.
        It averages the collected order flow deltas over the historical window.
        """
        with self.order_flow_lock:
            if not self.order_flow_history:
                return 0
            # Calculate the average order flow delta
            avg_flow = sum(flow for timestamp, flow in self.order_flow_history) / len(self.order_flow_history)
        # Multiply by the configured weight
        logger.debug(f"Historical Order Flow Signal (avg flow): {avg_flow:.2f}, Weighted: {avg_flow * self.historical_order_flow_weight:.2f}")
        return avg_flow * self.historical_order_flow_weight

    def generate_trading_signal(self, indicators, orderbook):
        """Generates a composite trading signal based on all indicators and order book analysis."""
        signal_score = 0
        signal_score += self._get_ema_signal(indicators)
        signal_score += self._get_rsi_signal(indicators)
        signal_score += self._get_macd_signal(indicators)
        signal_score += self._get_stoch_rsi_signal(indicators)
        signal_score += self._get_order_book_imbalance_signal(orderbook)
        signal_score += self._get_ob_delta_change_signal(orderbook)
        signal_score += self._get_spread_signal(orderbook)
        signal_score += self._get_cluster_proximity_signal(orderbook)
        # Incorporate historical order flow signal if enabled
        if self.config['trading'].get('historical_order_flow_enabled', False):
            signal_score += self._get_historical_order_flow_signal()

        logger.info(f"Composite Signal Score: {signal_score:.2f}")
        if signal_score > self.config['trading_thresholds']['buy_threshold']:
            return 'BUY'
        elif signal_score < self.config['trading_thresholds']['sell_threshold']:
            return 'SELL'
        else:
            return 'NEUTRAL'

    def calculate_order_amount(self):
        """
        Calculates the order amount.
        If dynamic position sizing is enabled, use ATR-based risk adjustment:
            position_size = risk_amount / (ATR * volatility_multiplier)
        Otherwise, use the basic method (risk_amount / market_price).
        """
        balance = self.get_account_balance()
        if balance is None:
            return None

        risk_amount = balance * (self.order_size_percentage / 100)
        market_price = self.fetch_market_price()
        if market_price is None or market_price == 0:
            return None

        if self.dynamic_position_sizing and self.last_atr and self.last_atr > 0:
            # The risk per unit is set as ATR * volatility_multiplier
            risk_per_unit = self.last_atr * self.volatility_multiplier
            order_amount = risk_amount / risk_per_unit
            logger.debug(f"Dynamic Order Sizing: risk_amount={risk_amount:.2f}, ATR={self.last_atr:.4f}, order_amount={order_amount:.4f}")
        else:
            order_amount = risk_amount / market_price
            logger.debug(f"Static Order Sizing: risk_amount={risk_amount:.2f}, market_price={market_price:.2f}, order_amount={order_amount:.4f}")
        return order_amount

    def calculate_take_profit(self, entry_price, side):
        """Calculates take profit price based on entry price and percentage."""
        if not entry_price:
            return None
        if side == 'BUY':
            tp_price = entry_price * (1 + (self.take_profit_pct / 100))
        elif side == 'SELL':
            tp_price = entry_price * (1 - (self.take_profit_pct / 100))
        else:
            return None
        logger.debug(f"Calculated Take Profit Price: {tp_price:.2f}")
        return tp_price

    def calculate_stop_loss(self, entry_price, side):
        """Calculates stop loss price based on entry price and percentage."""
        if not entry_price:
            return None
        if side == 'BUY':
            sl_price = entry_price * (1 - (self.stop_loss_pct / 100))
        elif side == 'SELL':
            sl_price = entry_price * (1 + (self.stop_loss_pct / 100))
        else:
            return None
        logger.debug(f"Calculated Stop Loss Price: {sl_price:.2f}")
        return sl_price

    @retry_api_call()
    def place_order(self, side, amount, take_profit_price=None, stop_loss_price=None):
        """Places a market order with optional take profit and stop loss."""
        if self.simulation_mode:
            logger.info(f"{NEON_YELLOW}[SIMULATION MODE] - Placing {side} order for {amount:.4f} {self.symbol}{RESET_COLOR}")
            return {'orderId': 'SIMULATION_ORDER_ID', 'status': 'open', 'side': side, 'amount': amount, 'price': self.fetch_market_price()}

        try:
            order = self.exchange.create_market_order(self.symbol, side, amount)
            logger.info(f"{NEON_GREEN}Placed {side} order for {amount:.4f} {self.symbol} at market price.{RESET_COLOR} Order ID: {order['id']}")
            if order and order.get('status') == 'open':
                position = {
                    'symbol': self.symbol,
                    'side': side,
                    'order_id': order['id'],
                    'amount': amount,
                    'entry_price': order.get('price', self.fetch_market_price()),
                    'take_profit_price': take_profit_price,
                    'stop_loss_price': stop_loss_price,
                    'trailing_stop_active': self.trailing_stop_loss_active,
                    'trailing_stop_callback': self.trailing_stop_callback,
                    'high_since_entry': -np.inf if side == 'BUY' else np.inf,
                    'low_since_entry': np.inf if side == 'SELL' else -np.inf,
                    'timestamp': time.time()
                }
                self.open_positions.append(position)
                logger.info(f"Open positions count: {len(self.open_positions)}")
                return order
            else:
                logger.error(f"{Fore.RED}Order placement failed or order not open. Details: {order}{Style.RESET_ALL}")
                return None
        except Exception as e:
            logger.error(f"{Fore.RED}Error placing order: {e}{Style.RESET_ALL}")
            return None

    @retry_api_call()
    def cancel_order(self, order_id):
        """Cancels an open order."""
        if self.simulation_mode:
            logger.info(f"{NEON_YELLOW}[SIMULATION MODE] - Cancelling order with ID: {order_id}{RESET_COLOR}")
            return True
        try:
            self.exchange.cancel_order(order_id, self.symbol)
            logger.info(f"{NEON_GREEN}Order {order_id} cancelled successfully.{RESET_COLOR}")
            return True
        except ccxt.OrderNotFound:
            logger.warning(f"{Fore.YELLOW}Order {order_id} not found, possibly already closed.{Style.RESET_ALL}")
            return True
        except Exception as e:
            logger.error(f"{Fore.RED}Error cancelling order {order_id}: {e}{Style.RESET_ALL}")
            return False

    @retry_api_call()
    def fetch_open_positions(self):
        """Fetches current open positions from the exchange."""
        try:
            positions = self.exchange.fetch_positions([self.symbol])
            open_positions = [p for p in positions if p.get('contracts', 0) > 0]
            logger.debug(f"Fetched open positions: {open_positions}")
            return open_positions
        except Exception as e:
            logger.error(f"{Fore.RED}Error fetching open positions: {e}{Style.RESET_ALL}")
            return None

    @retry_api_call()
    def get_account_balance(self):
        """Fetches account balance."""
        try:
            balance = self.exchange.fetch_balance()
            if balance and 'USDT' in balance and 'total' in balance['USDT']:
                usdt_balance = balance['USDT']['total']
                logger.debug(f"Account Balance: USDT {usdt_balance:.2f}")
                return usdt_balance
            else:
                logger.warning(f"{Fore.YELLOW}USDT balance not found in account balance data.{Style.RESET_ALL}")
                return None
        except Exception as e:
            logger.error(f"{Fore.RED}Error fetching account balance: {e}{Style.RESET_ALL}")
            return None

    @retry_api_call()
    def get_leverage(self):
        """Fetches current leverage for the trading symbol."""
        try:
            leverage_info = self.exchange.fetch_leverage(self.symbol)
            leverage = leverage_info.get('leverage', None)
            logger.debug(f"Current Leverage: {leverage}x")
            return leverage
        except Exception as e:
            logger.error(f"{Fore.RED}Error fetching leverage: {e}{Style.RESET_ALL}")
            return None

    @retry_api_call()
    def set_leverage(self, leverage_value):
        """Sets leverage for the trading symbol."""
        if self.simulation_mode:
            logger.info(f"{NEON_YELLOW}[SIMULATION MODE] - Setting leverage to {leverage_value}x{RESET_COLOR}")
            return True
        try:
            self.exchange.set_leverage(leverage_value, self.symbol)
            logger.info(f"{NEON_GREEN}Leverage set to {leverage_value}x successfully.{RESET_COLOR}")
            return True
        except Exception as e:
            logger.error(f"{Fore.RED}Error setting leverage to {leverage_value}x: {e}{Style.RESET_ALL}")
            return False

    def check_trailing_stop_loss(self, position, current_price):
        """Checks and adjusts trailing stop loss if conditions are met."""
        if not position['trailing_stop_active']:
            return False

        side = position['side']
        trailing_stop_callback = position['trailing_stop_callback']

        if side == 'BUY':
            if current_price > position['high_since_entry']:
                position['high_since_entry'] = current_price
                new_trailing_stop = position['high_since_entry'] * (1 - trailing_stop_callback / 100)
                if new_trailing_stop > position.get('stop_loss_price', 0):
                    position['stop_loss_price'] = new_trailing_stop
                    logger.info(f"Trailing stop loss adjusted upwards to {position['stop_loss_price']:.2f} for BUY position.")
                    return True
        elif side == 'SELL':
            if current_price < position['low_since_entry']:
                position['low_since_entry'] = current_price
                new_trailing_stop = position['low_since_entry'] * (1 + trailing_stop_callback / 100)
                if new_trailing_stop < position.get('stop_loss_price', np.inf):
                    position['stop_loss_price'] = new_trailing_stop
                    logger.info(f"Trailing stop loss adjusted downwards to {position['stop_loss_price']:.2f} for SELL position.")
                    return True
        return False

    def adjust_trailing_stop_loss(self, position, current_price):
        """Adjusts the stop loss based on trailing stop logic."""
        if not position['trailing_stop_active']:
            return False

        side = position['side']
        trailing_stop_callback = position['trailing_stop_callback']

        if side == 'BUY' and current_price > position['high_since_entry']:
            position['high_since_entry'] = current_price
            new_stop_loss = position['high_since_entry'] * (1 - trailing_stop_callback / 100)
            position['stop_loss_price'] = new_stop_loss
            logger.info(f"Trailing stop loss adjusted upwards to {position['stop_loss_price']:.2f} for BUY position.")
            return True
        elif side == 'SELL' and current_price < position['low_since_entry']:
            position['low_since_entry'] = current_price
            new_stop_loss = position['low_since_entry'] * (1 + trailing_stop_callback / 100)
            position['stop_loss_price'] = new_stop_loss
            logger.info(f"Trailing stop loss adjusted downwards to {position['stop_loss_price']:.2f} for SELL position.")
            return True
        return False

    def monitor_positions(self):
        """Monitors open positions for take profit, stop loss, and trailing stop adjustments."""
        positions_to_remove = []
        for position in self.open_positions:
            current_price = self.fetch_market_price()
            if current_price is None:
                continue  # Skip if unable to fetch price

            # Adjust trailing stop if enabled
            if self.trailing_stop_loss_active:
                self.check_trailing_stop_loss(position, current_price)

            tp_hit = False
            sl_hit = False

            if position['side'] == 'BUY':
                if position.get('take_profit_price') and current_price >= position['take_profit_price']:
                    tp_hit = True
                elif position.get('stop_loss_price') and current_price <= position['stop_loss_price']:
                    sl_hit = True
            elif position['side'] == 'SELL':
                if position.get('take_profit_price') and current_price <= position['take_profit_price']:
                    tp_hit = True
                elif position.get('stop_loss_price') and current_price >= position['stop_loss_price']:
                    sl_hit = True

            if tp_hit:
                logger.info(f"{NEON_GREEN}Take profit hit for {position['side']} position. Closing position.{RESET_COLOR}")
                self.close_position(position, 'take_profit')
                positions_to_remove.append(position)
            elif sl_hit:
                logger.info(f"{NEON_RED}Stop loss hit for {position['side']} position. Closing position.{RESET_COLOR}")
                self.close_position(position, 'stop_loss')
                positions_to_remove.append(position)

        # Remove closed positions from the open list
        for pos in positions_to_remove:
            if pos in self.open_positions:
                self.open_positions.remove(pos)

    def close_position(self, position, reason):
        """Closes an open position based on the provided reason (take_profit or stop_loss)."""
        side = 'SELL' if position['side'] == 'BUY' else 'BUY'
        amount = position['amount']
        logger.info(f"Closing position for {position['symbol']} due to {reason}. Executing {side} order for {amount:.4f}.")
        order = self.place_order(side, amount)
        if order:
            logger.info(f"Position closed. Order ID: {order.get('id', 'N/A')}")
        else:
            logger.error("Failed to close position.")
        return order

    def _update_order_flow_history(self):
        """
        Background thread that fetches order book snapshots at a high frequency,
        computes the order flow delta (bid volume minus ask volume), and stores
        the data for historical analysis.
        """
        while True:
            orderbook = self.fetch_order_book()
            if orderbook and orderbook.get('bids') and orderbook.get('asks'):
                bid_volume = sum(bid[1] for bid in orderbook['bids'])
                ask_volume = sum(ask[1] for ask in orderbook['asks'])
                flow_delta = bid_volume - ask_volume
                timestamp = time.time()
                with self.order_flow_lock:
                    self.order_flow_history.append((timestamp, flow_delta))
                    # Remove entries older than the historical window (in seconds)
                    self.order_flow_history = [(ts, flow) for ts, flow in self.order_flow_history if timestamp - ts <= self.historical_window]
            time.sleep(1)  # Adjust frequency as needed

    def run(self):
        """Main loop to run the trading bot."""
        logger.info("Starting trading bot loop...")
        while True:
            try:
                # Fetch market data
                ohlcv = self.exchange.fetch_ohlcv(self.symbol, timeframe='1m', limit=100)
                indicators = self.calculate_indicators(ohlcv)
                orderbook = self.fetch_order_book()

                # Generate trading signal
                signal = self.generate_trading_signal(indicators, orderbook)
                logger.info(f"Trading Signal: {signal}")

                # If signal is BUY or SELL, attempt to place an order
                if signal in ['BUY', 'SELL']:
                    amount = self.calculate_order_amount()
                    if amount:
                        entry_price = self.fetch_market_price()
                        tp = self.calculate_take_profit(entry_price, signal)
                        sl = self.calculate_stop_loss(entry_price, signal)
                        self.place_order(signal, amount, take_profit_price=tp, stop_loss_price=sl)

                # Monitor open positions for exit conditions
                self.monitor_positions()

                time.sleep(60)  # Delay between iterations; adjust as necessary
            except Exception as e:
                logger.error(f"{Fore.RED}Error in main loop: {e}{Style.RESET_ALL}")
                time.sleep(60)


if __name__ == '__main__':
    # Replace 'BTC/USDT' with your desired trading pair and ensure config.yaml is properly configured.
    bot = EnhancedTradingBot('BTC/USDT')
    bot.run()
