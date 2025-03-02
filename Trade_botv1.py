--- START OF FILE trade_v3.py ---
import ccxt
import logging
import os
import time

import numpy as np
import pandas as pd
import yaml
from colorama import Fore, Style, init as colorama_init
from dotenv import load_dotenv

colorama_init(autoreset=True)

logger = logging.getLogger("ScalpingBot")
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)
file_handler = logging.FileHandler("scalping_bot_v3.log")
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

load_dotenv()


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
                    delay *= 2
                    retries += 1
                except ccxt.NetworkError as e:
                    logger.error(
                        f"{Fore.RED}Network error during API call: {e}. Retrying in {delay} seconds... "
                        f"(Retry {retries + 1}/{max_retries}){Style.RESET_ALL}"
                    )
                    time.sleep(delay)
                    delay *= 2
                    retries += 1
                except ccxt.ExchangeError as e:
                    logger.error(
                        f"{Fore.RED}Exchange error during API call: {e}. "
                        f"(Retry {retries + 1}/{max_retries}) {e}{Style.RESET_ALL}"
                    )
                    if 'Order does not exist' in str(e):
                        return None
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
            logger.error(f"{Fore.RED}Max retries reached for API call. Aborting.{Style.RESET_ALL}")
            return None
        return wrapper
    return decorator


class ScalpingBot:
    """A cryptocurrency scalping bot."""

    def __init__(self, config_file='config.yaml'):
        """Initialize the ScalpingBot."""
        self.load_config(config_file)
        self.validate_config()
        self.api_key = os.getenv('BYBIT_API_KEY')
        self.api_secret = os.getenv('BYBIT_API_SECRET')
        self.exchange_id = self.config['exchange']['exchange_id']
        self.symbol = self.config['trading']['symbol']
        self.simulation_mode = self.config['trading']['simulation_mode']
        self.entry_order_type = self.config['trading']['entry_order_type']
        self.limit_order_offset_buy = self.config['trading']['limit_order_offset_buy']
        self.limit_order_offset_sell = self.config['trading']['limit_order_offset_sell']

        self.order_book_depth = self.config['order_book']['depth']
        self.imbalance_threshold = self.config['order_book']['imbalance_threshold']

        self.volatility_window = self.config['indicators']['volatility_window']
        self.volatility_multiplier = self.config['indicators']['volatility_multiplier']
        self.ema_period = self.config['indicators']['ema_period']
        self.rsi_period = self.config['indicators']['rsi_period']
        self.macd_short_period = self.config['indicators']['macd_short_period']
        self.macd_long_period = self.config['indicators']['macd_long_period']
        self.macd_signal_period = self.config['indicators']['macd_signal_period']
        self.stoch_rsi_period = self.config['indicators']['stoch_rsi_period']

        self.base_stop_loss_pct = self.config['risk_management']['stop_loss_percentage']  # Base SL
        self.base_take_profit_pct = self.config['risk_management']['take_profit_percentage']  # Base TP
        self.max_open_positions = self.config['risk_management']['max_open_positions']
        self.time_based_exit_minutes = self.config['risk_management']['time_based_exit_minutes']
        self.trailing_stop_loss_percentage = self.config['risk_management']['trailing_stop_loss_percentage']

        self.order_size_percentage = self.config['risk_management']['order_size_percentage']

        self.entry_signals_config = self.config['entry_signals']
        self.exit_signals_config = self.config['exit_signals']

        self.iteration = 0
        self.daily_pnl = 0.0
        self.open_positions = []

        if 'logging_level' in self.config:
            log_level = self.config['logging_level'].upper()
            if log_level in ['DEBUG', 'INFO', 'WARNING', 'ERROR']:
                logger.setLevel(getattr(logging, log_level))
            else:
                logger.warning(
                    f"{Fore.YELLOW}Invalid logging level '{log_level}' in config. "
                    f"Using default (DEBUG).{Style.RESET_ALL}"
                )

        self.exchange = self._initialize_exchange()

    def load_config(self, config_file):
        """Load configuration from YAML file."""
        try:
            with open(config_file, 'r') as f:
                self.config = yaml.safe_load(f)
            logger.info(f"{Fore.GREEN}Configuration loaded from {config_file}{Style.RESET_ALL}")
        except FileNotFoundError:
            logger.error(f"{Fore.RED}Configuration file {config_file} not found. Creating default...{Style.RESET_ALL}")
            self.create_default_config(config_file)
            logger.info(f"{Fore.YELLOW}Please review and modify '{config_file}' then run the bot again.{Style.RESET_ALL}")
            exit()
        except yaml.YAMLError as e:
            logger.error(
                f"{Fore.RED}Error parsing configuration file {config_file}: {e}. Exiting.{Style.RESET_ALL}"
            )
            exit()

    def create_default_config(self, config_file):
        """Create a default configuration file."""
        default_config = {
            'logging_level': 'DEBUG',
            'exchange': {
                'exchange_id': os.getenv('EXCHANGE_ID', 'bybit'),
            },
            'trading': {
                'symbol': input("Enter the trading symbol (e.g., BTC/USDT): ").strip().upper(),
                'simulation_mode': os.getenv('SIMULATION_MODE', 'True').lower() in ('true', '1', 'yes'),
                'entry_order_type': os.getenv('ENTRY_ORDER_TYPE', 'limit').lower(),
                'limit_order_offset_buy': float(os.getenv('LIMIT_ORDER_OFFSET_BUY', 0.001)),
                'limit_order_offset_sell': float(os.getenv('LIMIT_ORDER_OFFSET_SELL', 0.001)),
            },
            'order_book': {
                'depth': int(os.getenv('ORDER_BOOK_DEPTH', 10)),
                'imbalance_threshold': float(os.getenv('IMBALANCE_THRESHOLD', 1.5)),
            },
            'indicators': {
                'volatility_window': int(os.getenv('VOLATILITY_WINDOW', 5)),
                'volatility_multiplier': float(os.getenv('VOLATILITY_MULTIPLIER', 0.02)),
                'ema_period': int(os.getenv('EMA_PERIOD', 10)),
                'rsi_period': int(os.getenv('RSI_PERIOD', 14)),
                'macd_short_period': int(os.getenv('MACD_SHORT_PERIOD', 12)),
                'macd_long_period': int(os.getenv('MACD_LONG_PERIOD', 26)),
                'macd_signal_period': int(os.getenv('MACD_SIGNAL_PERIOD', 9)),
                'stoch_rsi_period': int(os.getenv('STOCH_RSI_PERIOD', 14)),
            },
            'risk_management': {
                'order_size_percentage': float(os.getenv('ORDER_SIZE_PERCENTAGE', 0.01)),
                'stop_loss_percentage': float(os.getenv('STOP_LOSS_PERCENTAGE', 0.015)),
                'take_profit_percentage': float(os.getenv('TAKE_PROFIT_PERCENTAGE', 0.03)),
                'trailing_stop_loss_percentage': float(os.getenv('TRAILING_STOP_LOSS_PERCENTAGE', 0.005)),
                'max_open_positions': int(os.getenv('MAX_OPEN_POSITIONS', 1)),
                'time_based_exit_minutes': int(os.getenv('TIME_BASED_EXIT_MINUTES', 15)),
            },
            'entry_signals': { # Example Entry Signals - Customize these in config.yaml!
                'signal_ema_cross_bullish': {
                    'name': 'EMA Crossover Bullish',
                    'enabled': True,
                    'required_indicators': ['ema', 'macd', 'rsi', 'orderbook_imbalance'], # List of indicators needed for this signal
                    'stop_loss_percentage': 0.01, # SL specific to this signal
                    'confidence_level': 7, # Confidence level for this signal
                },
                'signal_rsi_oversold': {
                    'name': 'RSI Oversold',
                    'enabled': True,
                    'required_indicators': ['rsi', 'stoch_rsi', 'orderbook_imbalance'],
                    'stop_loss_percentage': 0.012,
                    'confidence_level': 6,
                },
                'signal_orderbook_bid_ask_imbalance': {
                    'name': 'Orderbook Imbalance Bid Ask',
                    'enabled': False, # Example: Disabled by default
                    'required_indicators': ['orderbook_imbalance'],
                    'stop_loss_percentage': 0.015,
                    'confidence_level': 5,
                },
                'signal_macd_histogram_divergence': { # Example - You'll need to implement divergence logic
                    'name': 'MACD Histogram Divergence',
                    'enabled': False,
                    'required_indicators': ['macd', 'price'], # Hypothetical - Price might be needed for divergence
                    'stop_loss_percentage': 0.013,
                    'confidence_level': 6,
                },
                'signal_stoch_rsi_k_cross_d': { # Example - Stochastic RSI K crossing D
                    'name': 'Stoch RSI K Cross D',
                    'enabled': False,
                    'required_indicators': ['stoch_rsi'],
                    'stop_loss_percentage': 0.011,
                    'confidence_level': 5,
                }
            },
            'exit_signals': { # Example Exit Signals - Customize these in config.yaml!
                'exit_signal_rsi_overbought': {
                    'name': 'RSI Overbought Exit',
                    'enabled': True,
                    'required_indicators': ['rsi'], # List of indicators for exit signal
                },
                'exit_signal_ema_cross_bearish': {
                    'name': 'EMA Cross Bearish Exit',
                    'enabled': True,
                    'required_indicators': ['ema'],
                },
                'exit_signal_profit_level_reached': { # Hypothetical profit-based exit
                    'name': 'Profit Level Exit',
                    'enabled': False,
                    'required_indicators': ['price'], # Current price needed
                    'profit_multiplier': 1.5 # Example: Exit if profit is 1.5x initial TP
                },
                'exit_signal_time_limit_reached': { # Time-based exit - already implemented, can be another signal
                    'name': 'Time Limit Exit',
                    'enabled': False,
                    'required_indicators': [], # No indicators needed - time-based
                },
                'exit_signal_manual_close': { # Hypothetical manual close trigger (not automated)
                    'name': 'Manual Close Signal',
                    'enabled': False,
                    'required_indicators': [], # No indicators needed - manual trigger
                }
            }
        }
        with open(config_file, 'w') as f:
            yaml.dump(default_config, f, indent=4)

    def validate_config(self):
        """Validate the configuration loaded from the YAML file."""
        # ... (rest of validate_config function - same as trade_v2.py) ...
        if 'entry_signals' not in self.config:
            raise ValueError("Missing 'entry_signals' section in config.yaml")
        if not isinstance(self.config['entry_signals'], dict):
            raise ValueError("'entry_signals' must be a dictionary")
        for signal_name, signal_config in self.config['entry_signals'].items():
            if not isinstance(signal_config, dict):
                raise ValueError(f"Entry signal '{signal_name}' config must be a dictionary")
            required_keys = ['name', 'enabled', 'required_indicators', 'stop_loss_percentage', 'confidence_level']
            for key in required_keys:
                if key not in signal_config:
                    raise ValueError(f"Missing key '{key}' in entry signal '{signal_name}' config")
            if not isinstance(signal_config['enabled'], bool):
                raise ValueError(f"'{signal_name}.enabled' must be a boolean")
            if not isinstance(signal_config['name'], str):
                raise ValueError(f"'{signal_name}.name' must be a string")
            if not isinstance(signal_config['required_indicators'], list):
                raise ValueError(f"'{signal_name}.required_indicators' must be a list")
            if not isinstance(signal_config['stop_loss_percentage'], (int, float)) or signal_config['stop_loss_percentage'] <= 0:
                raise ValueError(f"'{signal_name}.stop_loss_percentage' must be a positive number")
            if not isinstance(signal_config['confidence_level'], int):
                raise ValueError(f"'{signal_name}.confidence_level' must be an integer")

        if 'exit_signals' not in self.config:
            raise ValueError("Missing 'exit_signals' section in config.yaml")
        if not isinstance(self.config['exit_signals'], dict):
            raise ValueError("'exit_signals' must be a dictionary")
        for signal_name, signal_config in self.config['exit_signals'].items():
            if not isinstance(signal_config, dict):
                raise ValueError(f"Exit signal '{signal_name}' config must be a dictionary")
            required_keys = ['name', 'enabled', 'required_indicators']
            for key in required_keys:
                if key not in signal_config:
                    raise ValueError(f"Missing key '{key}' in exit signal '{signal_name}' config")
            if not isinstance(signal_config['enabled'], bool):
                raise ValueError(f"'{signal_name}.enabled' must be a boolean")
            if not isinstance(signal_config['name'], str):
                raise ValueError(f"'{signal_name}.name' must be a string")
            if not isinstance(signal_config['required_indicators'], list):
                raise ValueError(f"'{signal_name}.required_indicators' must be a list")


    def _initialize_exchange(self):
        """Initialize and connect to the exchange."""
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
        """Fetch the current market price of the trading symbol."""
        ticker = self.exchange.fetch_ticker(self.symbol)
        if ticker and 'last' in ticker:
            price = ticker['last']
            logger.debug(f"Fetched market price: {price}")
            return price
        else:
            logger.warning(f"{Fore.YELLOW}Market price unavailable.{Style.RESET_ALL}")
            return None

    @retry_api_call()
    def fetch_order_book(self):
        """Fetch the order book and calculate imbalance ratio."""
        orderbook = self.exchange.fetch_order_book(self.symbol, limit=self.order_book_depth)
        bids = orderbook['bids']
        asks = orderbook['asks']
        if bids and asks:
            bid_volume = sum(bid[1] for bid in bids)
            ask_volume = sum(ask[1] for ask in asks)
            imbalance_ratio = ask_volume / bid_volume if bid_volume > 0 else float('inf')
            logger.debug(f"Order Book - Bid Vol: {bid_volume}, Ask Vol: {ask_volume}, Imbalance: {imbalance_ratio:.2f}")
            return imbalance_ratio
        else:
            logger.warning(f"{Fore.YELLOW}Order book data unavailable.{Style.RESET_ALL}")
            return None

    @retry_api_call()
    def fetch_historical_prices(self, limit=None):
        """Fetch historical prices (OHLCV data)."""
        if limit is None:
            limit = max(
                self.volatility_window, self.ema_period, self.rsi_period + 1, self.macd_long_period,
                self.stoch_rsi_period
            ) + 1
        ohlcv = self.exchange.fetch_ohlcv(self.symbol, timeframe='1m', limit=limit)
        if ohlcv:
            prices = [candle[4] for candle in ohlcv]
            if len(prices) < limit:
                logger.warning(
                    f"{Fore.YELLOW}Insufficient historical data. Fetched {len(prices)}, needed {limit}.{Style.RESET_ALL}"
                )
                return []
            logger.debug(f"Historical prices (last 5): {prices[-5:]}")
            return prices
        else:
            logger.warning(f"{Fore.YELLOW}Historical price data unavailable.{Style.RESET_ALL}")
            return []

    def calculate_volatility(self):
        """Calculate volatility using historical prices."""
        prices = self.fetch_historical_prices(limit=self.volatility_window)
        if not prices or len(prices) < self.volatility_window:
            return None
        returns = np.diff(prices) / prices[:-1]
        volatility = np.std(returns)
        logger.debug(f"Calculated volatility: {volatility}")
        return volatility

    def calculate_ema(self, prices, period=None):
        """Calculate Exponential Moving Average (EMA)."""
        if period is None:
            period = self.ema_period
        if not prices or len(prices) < period:
            return None
        weights = np.exp(np.linspace(-1., 0., period))
        weights /= weights.sum()
        ema = np.convolve(prices, weights, mode='valid')[-1]
        logger.debug(f"Calculated EMA: {ema}")
        return ema

    def calculate_rsi(self, prices):
        """Calculate Relative Strength Index (RSI)."""
        if not prices or len(prices) < self.rsi_period + 1:
            return None
        deltas = np.diff(prices)
        gains = np.maximum(deltas, 0)
        losses = -np.minimum(deltas, 0)
        avg_gain = np.mean(gains[-self.rsi_period:])
        avg_loss = np.mean(losses[-self.rsi_period:])
        if avg_loss == 0:
            rsi = 100
        else:
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
        logger.debug(f"Calculated RSI: {rsi}")
        return rsi

    def calculate_macd(self, prices):
        """Calculate Moving Average Convergence Divergence (MACD)."""
        if not prices or len(prices) < self.macd_long_period:
            return None, None, None
        short_ema = self.calculate_ema(prices[-self.macd_short_period:], self.macd_short_period)
        long_ema = self.calculate_ema(prices[-self.macd_long_period:], self.macd_long_period)
        if short_ema is None or long_ema is None:
            return None, None, None
        macd = short_ema - long_ema
        signal = self.calculate_ema([macd], self.macd_signal_period)
        if signal is None:
            return None, None, None
        hist = macd - signal
        logger.debug(f"MACD: {macd}, Signal: {signal}, Histogram: {hist}")
        return macd, signal, hist

    def calculate_stoch_rsi(self, prices, period=None):
        """Calculate Stochastic RSI."""
        if period is None:
            period = self.stoch_rsi_period
        if not prices or len(prices) < period:
            return None, None

        close = pd.Series(prices)
        min_val = close.rolling(window=period).min()
        max_val = close.rolling(window=period).max()
        stoch_rsi = 100 * (close - min_val) / (max_val - min_val)
        k = stoch_rsi.rolling(window=3).mean()
        d = k.rolling(window=3).mean()

        if k.empty or d.empty or pd.isna(k.iloc[-1]) or pd.isna(d.iloc[-1]):
            return None, None

        logger.debug(f"Calculated Stochastic RSI - K: {k.iloc[-1]}, D: {d.iloc[-1]}")
        return k.iloc[-1], d.iloc[-1]

    @retry_api_call()
    def fetch_balance(self):
        """Fetch USDT balance from the exchange."""
        return self.exchange.fetch_balance().get('USDT', {}).get('free', 0)

    def calculate_order_size(self):
        """Calculate order size based on balance and volatility."""
        balance = self.fetch_balance()
        if balance is None:
            logger.warning(f"{Fore.YELLOW}Could not retrieve USDT balance.{Style.RESET_ALL}")
            return 0

        volatility = self.calculate_volatility()
        if volatility is None:
            base_size = balance * self.order_size_percentage
            logger.info(f"{Fore.CYAN}Default order size (no volatility data): {base_size}{Style.RESET_ALL}")
            return base_size

        adjusted_size = balance * self.order_size_percentage * (1 + (volatility * self.volatility_multiplier))
        final_size = min(adjusted_size, balance * 0.05)
        logger.info(
            f"{Fore.CYAN}Calculated order size: {final_size:.2f} (Balance: {balance:.2f}, Volatility: {volatility:.5f}){Style.RESET_ALL}"
        )
        return final_size

    def check_entry_signals(self, price, indicators):
        """Check all enabled entry signals and return triggered signals with confidence."""
        triggered_signals = []
        for signal_key, signal_config in self.entry_signals_config.items():
            if signal_config['enabled']:
                signal_result = self.evaluate_entry_signal(signal_key, signal_config, price, indicators)
                if signal_result:
                    triggered_signals.append(signal_result)
        return triggered_signals

    def evaluate_entry_signal(self, signal_key, signal_config, price, indicators):
        """Evaluate a specific entry signal based on its configuration."""
        signal_name = signal_config['name']
        required_indicators = signal_config['required_indicators']
        confidence_level = signal_config['confidence_level']

        logger.debug(f"Evaluating Entry Signal: {signal_name}")

        # --- Indicator Checks ---
        if 'ema' in required_indicators and indicators['ema'] is None:
            logger.debug(f"Signal '{signal_name}' requires EMA but it's not available.")
            return None
        if 'rsi' in required_indicators and indicators['rsi'] is None:
            logger.debug(f"Signal '{signal_name}' requires RSI but it's not available.")
            return None
        if 'macd' in required_indicators and (indicators['macd'] is None or indicators['macd_signal'] is None):
            logger.debug(f"Signal '{signal_name}' requires MACD but it's not available.")
            return None
        if 'stoch_rsi' in required_indicators and (indicators['stoch_rsi_k'] is None or indicators['stoch_rsi_d'] is None):
            logger.debug(f"Signal '{signal_name}' requires Stochastic RSI but it's not available.")
            return None
        if 'orderbook_imbalance' in required_indicators and indicators['orderbook_imbalance'] is None:
            logger.debug(f"Signal '{signal_name}' requires Order Book Imbalance but it's not available.")
            return None

        # --- Signal Logic ---
        if signal_key == 'signal_ema_cross_bullish': # Example Signal 1: EMA Crossover Bullish
            if price > indicators['ema'] and indicators['macd'] > indicators['macd_signal'] and indicators['rsi'] < 50 and indicators['orderbook_imbalance'] < 1:
                logger.info(f"{Fore.GREEN}Entry Signal Triggered: {signal_name} - Price:{price:.2f}, EMA:{indicators['ema']:.2f}, MACD:{indicators['macd']:.2f}, Signal:{indicators['macd_signal']:.2f}, RSI:{indicators['rsi']:.2f}, Orderbook Imbalance:{indicators['orderbook_imbalance']:.2f}{Style.RESET_ALL}")
                return {'signal_name': signal_name, 'side': 'buy', 'confidence': confidence_level, 'stop_loss_percentage': signal_config['stop_loss_percentage']}

        elif signal_key == 'signal_rsi_oversold': # Example Signal 2: RSI Oversold
            if indicators['rsi'] < 30 and indicators['stoch_rsi_k'] < 30 and indicators['orderbook_imbalance'] < 1.2:
                logger.info(f"{Fore.GREEN}Entry Signal Triggered: {signal_name} - RSI:{indicators['rsi']:.2f}, Stoch RSI K:{indicators['stoch_rsi_k']:.2f}, Orderbook Imbalance:{indicators['orderbook_imbalance']:.2f}{Style.RESET_ALL}")
                return {'signal_name': signal_name, 'side': 'buy', 'confidence': confidence_level, 'stop_loss_percentage': signal_config['stop_loss_percentage']}

        elif signal_key == 'signal_orderbook_bid_ask_imbalance': # Example Signal 3: Orderbook Imbalance
            if indicators['orderbook_imbalance'] < (1 / self.imbalance_threshold): # Strong bid side
                logger.info(f"{Fore.GREEN}Entry Signal Triggered: {signal_name} - Orderbook Imbalance:{indicators['orderbook_imbalance']:.2f}{Style.RESET_ALL}")
                return {'signal_name': signal_name, 'side': 'buy', 'confidence': confidence_level, 'stop_loss_percentage': signal_config['stop_loss_percentage']}
            elif indicators['orderbook_imbalance'] > self.imbalance_threshold: # Strong ask side
                logger.info(f"{Fore.RED}Entry Signal Triggered: {signal_name} - Orderbook Imbalance:{indicators['orderbook_imbalance']:.2f}{Style.RESET_ALL}")
                return {'signal_name': signal_name, 'side': 'sell', 'confidence': confidence_level, 'stop_loss_percentage': signal_config['stop_loss_percentage']}

        # --- Add logic for signal_macd_histogram_divergence and signal_stoch_rsi_k_cross_d here ---
        # ... (Implement logic for other signals as needed) ...

        return None # No signal triggered

    def check_exit_signals(self, position, price, indicators):
        """Check all enabled exit signals for a given position."""
        triggered_exits = []
        for signal_key, signal_config in self.exit_signals_config.items():
            if signal_config['enabled']:
                exit_signal_result = self.evaluate_exit_signal(signal_key, signal_config, position, price, indicators)
                if exit_signal_result:
                    triggered_exits.append(exit_signal_result)
        return triggered_exits

    def evaluate_exit_signal(self, signal_key, signal_config, position, price, indicators):
        """Evaluate a specific exit signal."""
        signal_name = signal_config['name']
        required_indicators = signal_config['required_indicators']

        logger.debug(f"Evaluating Exit Signal: {signal_name} for position {position['side']}")

        # --- Indicator Checks for Exit Signals ---
        if 'rsi' in required_indicators and indicators['rsi'] is None:
            logger.debug(f"Exit Signal '{signal_name}' requires RSI but it's not available.")
            return None
        if 'ema' in required_indicators and indicators['ema'] is None:
            logger.debug(f"Exit Signal '{signal_name}' requires EMA but it's not available.")
            return None
        # ... (Add checks for other required indicators for exit signals) ...

        # --- Exit Signal Logic ---
        if signal_key == 'exit_signal_rsi_overbought': # Example Exit Signal 1: RSI Overbought
            if position['side'] == 'buy' and indicators['rsi'] > 70:
                logger.info(f"{Fore.YELLOW}Exit Signal Triggered: {signal_name} for LONG position - RSI:{indicators['rsi']:.2f}{Style.RESET_ALL}")
                return {'signal_name': signal_name, 'exit_reason': 'rsi_overbought'}
            elif position['side'] == 'sell' and indicators['rsi'] < 30: # RSI oversold for shorts (cover)
                logger.info(f"{Fore.YELLOW}Exit Signal Triggered: {signal_name} for SHORT position - RSI:{indicators['rsi']:.2f}{Style.RESET_ALL}")
                return {'signal_name': signal_name, 'exit_reason': 'rsi_oversold'}

        elif signal_key == 'exit_signal_ema_cross_bearish': # Example Exit Signal 2: EMA Cross Bearish
            if position['side'] == 'buy' and price < indicators['ema']: # Price falls below EMA for longs
                logger.info(f"{Fore.YELLOW}Exit Signal Triggered: {signal_name} for LONG position - Price:{price:.2f}, EMA:{indicators['ema']:.2f}{Style.RESET_ALL}")
                return {'signal_name': signal_name, 'exit_reason': 'ema_cross_bearish'}
            elif position['side'] == 'sell' and price > indicators['ema']: # Price rises above EMA for shorts (cover)
                logger.info(f"{Fore.YELLOW}Exit Signal Triggered: {signal_name} for SHORT position - Price:{price:.2f}, EMA:{indicators['ema']:.2f}{Style.RESET_ALL}")
                return {'signal_name': signal_name, 'exit_reason': 'ema_cross_bearish'}

        # --- Add logic for other exit signals here (profit level, time limit, manual close, etc.) ---
        # ... (Implement logic for other exit signals as needed) ...

        return None # No exit signal triggered


    def scalp_trade(self):
        """Main trading loop for the scalping bot."""
        while True:
            self.iteration += 1
            logger.info(f"\n--- Iteration {self.iteration} ---")

            price = self.fetch_market_price()
            orderbook_imbalance = self.fetch_order_book()
            historical_prices = self.fetch_historical_prices()

            if price is None or orderbook_imbalance is None or not historical_prices:
                logger.warning(f"{Fore.YELLOW}Insufficient data. Retrying in 10 seconds...{Style.RESET_ALL}")
                time.sleep(10)
                continue

            ema = self.calculate_ema(historical_prices)
            rsi = self.calculate_rsi(historical_prices)
            stoch_rsi_k, stoch_rsi_d = self.calculate_stoch_rsi(historical_prices)
            macd, macd_signal, macd_hist = self.calculate_macd(historical_prices)
            volatility = self.calculate_volatility()

            indicators = { # Collect all indicators for signal evaluation
                'price': price,
                'ema': ema,
                'rsi': rsi,
                'stoch_rsi_k': stoch_rsi_k,
                'stoch_rsi_d': stoch_rsi_d,
                'macd': macd,
                'macd_signal': macd_signal,
                'macd_hist': macd_hist,
                'orderbook_imbalance': orderbook_imbalance,
                'volatility': volatility
            }

            logger.info(
                f"Price: {price:.2f} | EMA: {ema if ema is not None else 'N/A'} | RSI: {rsi if rsi is not None else 'N/A'} | "
                f"Stoch RSI K: {stoch_rsi_k if stoch_rsi_k is not None else 'N/A'} | Stoch RSI D: {stoch_rsi_d if stoch_rsi_d is not None else 'N/A'} | "
                f"MACD: {macd if macd is not None else 'N/A'} | Volatility: {volatility if volatility is not None else 'N/A'}"
            )
            logger.info(f"Order Book Imbalance: {orderbook_imbalance:.2f}")

            order_size = self.calculate_order_size()
            if order_size == 0:
                logger.warning(f"{Fore.YELLOW}Order size is 0. Skipping this iteration.{Style.RESET_ALL}")
                time.sleep(10)
                continue

            # --- Check Entry Signals ---
            entry_signals = self.check_entry_signals(price, indicators)

            if len(self.open_positions) < self.max_open_positions:
                for signal in entry_signals: # Process triggered entry signals
                    side = signal['side']
                    confidence_level = signal['confidence']
                    stop_loss_percentage = signal['stop_loss_percentage']

                    take_profit_pct = self.base_take_profit_pct
                    stop_loss_pct = stop_loss_percentage # Use signal-specific SL %

                    stop_loss_price = price * (1 - stop_loss_pct) if side == 'buy' else price * (1 + stop_loss_pct)
                    take_profit_price = price * (1 + take_profit_pct) if side == 'buy' else price * (1 - take_profit_pct)
                    limit_price = price * (1 - self.limit_order_offset_buy) if side == 'buy' else price * (1 + self.limit_order_offset_sell)


                    entry_order = self.place_order(
                        side, order_size, confidence_level, order_type=self.entry_order_type,
                        price=limit_price if self.entry_order_type == 'limit' else None,
                        stop_loss_price=stop_loss_price, take_profit_price=take_profit_price
                    )

                    if entry_order:
                        log_color = Fore.GREEN if side == 'buy' else Fore.RED
                        logger.info(
                            f"{log_color}Entering {side.upper()} position based on signal: {signal['signal_name']}. Confidence: {confidence_level}, "
                            f"SL: {stop_loss_pct * 100:.2f}%, TP: {take_profit_pct * 100:.2f}%{Style.RESET_ALL}"
                        )
                        self.open_positions.append({
                            'side': side,
                            'size': order_size,
                            'entry_price': entry_order['price'] if not self.simulation_mode and 'price' in entry_order
                                           else price if self.entry_order_type == 'market' else limit_price,
                            'entry_time': time.time(),
                            'stop_loss': stop_loss_price,
                            'take_profit': take_profit_price,
                            'confidence': confidence_level,  # Store confidence level
                            'entry_signal': signal['signal_name'] # Store the signal name
                        })
                        break # Enter only one position per iteration, even if multiple signals trigger (adjust as needed)
            else:
                logger.info(
                    f"{Fore.YELLOW}Max open positions reached ({self.max_open_positions}).  "
                    f"Not entering new trades.{Style.RESET_ALL}"
                )

            # --- Manage Positions and Check Exit Signals ---
            for position in list(self.open_positions): # Iterate over a copy to allow removal during loop
                self.manage_positions() # Default SL/TP/Time based exits still managed

                exit_signals = self.check_exit_signals(position, price, indicators)
                for exit_signal in exit_signals:
                    logger.info(f"{Fore.YELLOW}Exiting {position['side'].upper()} position based on exit signal: {exit_signal['signal_name']}{Style.RESET_ALL}")
                    exit_side = 'sell' if position['side'] == 'buy' else 'buy'
                    self.place_order(exit_side, position['size'], position.get('confidence', 0), order_type='market') # Exit with market order
                    self.open_positions.remove(position)
                    break # Exit position based on the first triggered exit signal (adjust as needed)


            if self.iteration % 60 == 0:
                self.cancel_orders()

            time.sleep(10)


if __name__ == "__main__":
    config_file = 'config.yaml'
    bot = ScalpingBot(config_file=config_file)
    bot.scalp_trade()

--- END OF FILE ---
---config.yaml---
logging_level: DEBUG
exchange:
  exchange_id: bybit
trading:
  symbol: BTC/USDT # Or enter your symbol here
  simulation_mode: true
  entry_order_type: limit
  limit_order_offset_buy: 0.001
  limit_order_offset_sell: 0.001
order_book:
  depth: 10
  imbalance_threshold: 1.5
indicators:
  volatility_window: 5
  volatility_multiplier: 0.02
  ema_period: 10
  rsi_period: 14
  macd_short_period: 12
  macd_long_period: 26
  macd_signal_period: 9
  stoch_rsi_period: 14
risk_management:
  order_size_percentage: 0.01
  stop_loss_percentage: 0.015 # Base SL - can be overridden by entry signals
  take_profit_percentage: 0.03
  trailing_stop_loss_percentage: 0.005
  max_open_positions: 1
  time_based_exit_minutes: 15
entry_signals: # Example Entry Signals - Customize these in config.yaml!
  signal_ema_cross_bullish:
    name: EMA Crossover Bullish
    enabled: true
    required_indicators: ['ema', 'macd', 'rsi', 'orderbook_imbalance'] # List of indicators needed for this signal
    stop_loss_percentage: 0.01 # SL specific to this signal
    confidence_level: 7 # Confidence level for this signal
  signal_rsi_oversold:
    name: RSI Oversold
    enabled: true
    required_indicators: ['rsi', 'stoch_rsi', 'orderbook_imbalance']
    stop_loss_percentage: 0.012
    confidence_level: 6
  signal_orderbook_bid_ask_imbalance:
    name: Orderbook Imbalance Bid Ask
    enabled: false # Example: Disabled by default
    required_indicators: ['orderbook_imbalance']
    stop_loss_percentage: 0.015
    confidence_level: 5
  signal_macd_histogram_divergence: # Example - You'll need to implement divergence logic
    name: MACD Histogram Divergence
    enabled: false
    required_indicators: ['macd', 'price'] # Hypothetical - Price might be needed for divergence
    stop_loss_percentage: 0.013
    confidence_level: 6
  signal_stoch_rsi_k_cross_d: # Example - Stochastic RSI K Cross D
    name: Stoch RSI K Cross D
    enabled: false
    required_indicators: ['stoch_rsi']
    stop_loss_percentage: 0.011
    confidence_level: 5
exit_signals: # Example Exit Signals - Customize these in config.yaml!
  exit_signal_rsi_overbought:
    name: RSI Overbought Exit
    enabled: true
    required_indicators: ['rsi'] # List of indicators for exit signal
  exit_signal_ema_cross_bearish:
    name: EMA Cross Bearish Exit
    enabled: true
    required_indicators: ['ema']
  exit_signal_profit_level_reached: # Hypothetical profit-based exit
    name: Profit Level Exit
    enabled: false
    required_indicators: ['price'] # Current price needed
    profit_multiplier: 1.5 # Example: Exit if profit is 1.5x initial TP
  exit_signal_time_limit_reached: # Time-based exit - already implemented, can be another signal
    name: Time Limit Exit
    enabled: false
    required_indicators: [] # No indicators needed - time-based
  exit_signal_manual_close: # Hypothetical manual close trigger (not automated)
    name: Manual Close Signal
    enabled: false
    required_indicators: [] # No indicators needed - manual trigger
