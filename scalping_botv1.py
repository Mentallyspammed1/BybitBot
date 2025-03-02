import ccxt
import time
import os
import numpy as np
import pandas as pd
import logging
import yaml
from dotenv import load_dotenv
from colorama import Fore, Style, init as colorama_init

# Initialize colorama
colorama_init(autoreset=True)

# Setup logging
logger = logging.getLogger("ScalpingBot")
logger.setLevel(logging.DEBUG)  # Default level, can be overridden by config
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

file_handler = logging.FileHandler("scalping_bot.log")
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

# Load environment variables
load_dotenv()


def retry_api_call(max_retries=3, initial_delay=1):
    """Decorator for retrying API calls with exponential backoff."""

    def decorator(func):
        def wrapper(*args, **kwargs):
            retries = 0
            delay = initial_delay
            while retries <= max_retries:
                try:
                    return func(*args, **kwargs)
                except ccxt.RateLimitExceeded as e:
                    logger.warning(
                        f"{Fore.YELLOW}Rate limit exceeded, retrying in {delay} seconds... (Retry {retries + 1}/{max_retries}){Style.RESET_ALL}"
                    )
                    time.sleep(delay)
                    delay *= 2  # Exponential backoff
                    retries += 1
                except ccxt.NetworkError as e:
                    logger.error(
                        f"{Fore.RED}Network error during API call: {e}. Retrying in {delay} seconds... (Retry {retries + 1}/{max_retries}){Style.RESET_ALL}"
                    )
                    time.sleep(delay)
                    delay *= 2
                    retries += 1
                except ccxt.ExchangeError as e:
                    logger.error(
                        f"{Fore.RED}Exchange error during API call: {e}. (Retry {retries + 1}/{max_retries}) {e}{Style.RESET_ALL}"
                    )
                    if 'Order does not exist' in str(e):
                        return None
                    else:
                        time.sleep(delay)
                        delay *= 2
                        retries += 1
                except Exception as e:
                    logger.error(
                        f"{Fore.RED}Unexpected error during API call: {e}. (Retry {retries + 1}/{max_retries}){Style.RESET_ALL}"
                    )
                    time.sleep(delay)
                    delay *= 2
                    retries += 1
            logger.error(f"{Fore.RED}Max retries reached for API call. Aborting.{Style.RESET_ALL}")
            return None  # Return None to indicate failure

        return wrapper

    return decorator


class ScalpingBot:
    """
    A scalping bot that uses order book imbalance, EMA, RSI, MACD, and Stochastic RSI
    to make trading decisions.
    """

    def __init__(self, config_file='config.yaml'):
        """Initializes the ScalpingBot."""
        self.load_config(config_file)
        self.validate_config()  # Validate configuration after loading
        self.api_key = os.getenv('BYBIT_API_KEY')
        self.api_secret = os.getenv('BYBIT_API_SECRET')
        self.exchange_id = self.config['exchange']['exchange_id']
        self.symbol = self.config['trading']['symbol']
        self.simulation_mode = self.config['trading']['simulation_mode']

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

        self.order_size_percentage = self.config['risk_management']['order_size_percentage']
        self.stop_loss_pct = self.config['risk_management']['stop_loss_percentage']
        self.take_profit_pct = self.config['risk_management']['take_profit_percentage']
        self.max_open_positions = self.config['risk_management']['max_open_positions']
        self.time_based_exit_minutes = self.config['risk_management']['time_based_exit_minutes']

        self.iteration = 0
        self.daily_pnl = 0.0
        self.open_positions = []

        # Set logging level from config
        if 'logging_level' in self.config:
            log_level = self.config['logging_level'].upper()
            if log_level in ['DEBUG', 'INFO', 'WARNING', 'ERROR']:
                logger.setLevel(getattr(logging, log_level))
            else:
                logger.warning(f"{Fore.YELLOW}Invalid logging level '{log_level}' in config. Using default (DEBUG).{Style.RESET_ALL}")

        self.exchange = self._initialize_exchange()

    def load_config(self, config_file):
        """Loads the configuration from a YAML file."""
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

    def validate_config(self):
        """Validates the configuration parameters."""

        # --- Trading Section ---
        if 'trading' not in self.config:
            raise ValueError("Missing 'trading' section in config.yaml")

        if 'symbol' not in self.config['trading']:
            raise ValueError("Missing 'symbol' in config.yaml trading section")
        if not isinstance(self.config['trading']['symbol'], str) or not self.config['trading']['symbol']:
            raise ValueError("'symbol' must be a non-empty string")

        if 'simulation_mode' not in self.config['trading']:
            raise ValueError("Missing 'simulation_mode' in config.yaml trading section")
        if not isinstance(self.config['trading']['simulation_mode'], bool):
            raise ValueError("'simulation_mode' must be a boolean")

        # --- Order Book Section ---
        if 'order_book' not in self.config:
            raise ValueError("Missing 'order_book' section in config.yaml")

        if 'depth' not in self.config['order_book']:
            raise ValueError("Missing 'depth' in config.yaml order_book section")
        if not isinstance(self.config['order_book']['depth'], int) or self.config['order_book']['depth'] <= 0:
            raise ValueError("'depth' must be a positive integer")

        if 'imbalance_threshold' not in self.config['order_book']:
            raise ValueError("Missing 'imbalance_threshold' in config.yaml order_book section")
        if not isinstance(self.config['order_book']['imbalance_threshold'], (int, float)) or self.config['order_book']['imbalance_threshold'] <= 0:
            raise ValueError("'imbalance_threshold' must be a positive number")

        # --- Indicators Section ---
        if 'indicators' not in self.config:
            raise ValueError("Missing 'indicators' section in config.yaml")
        indicator_params = [
            ('volatility_window', int, 1, None), ('volatility_multiplier', (int, float), 0, None),
            ('ema_period', int, 1, None), ('rsi_period', int, 1, None),
            ('macd_short_period', int, 1, None), ('macd_long_period', int, 1, None),
            ('macd_signal_period', int, 1, None), ('stoch_rsi_period', int, 1, None)
        ]
        for param, type_, min_val, max_val in indicator_params:
            if param not in self.config['indicators']:
                raise ValueError(f"Missing '{param}' in config.yaml indicators section")
            if not isinstance(self.config['indicators'][param], type_):
                raise ValueError(f"'{param}' must be of type {type_}")
            if min_val is not None and self.config['indicators'][param] < min_val:
                raise ValueError(f"'{param}' must be greater than or equal to {min_val}")
            if max_val is not None and self.config['indicators'][param] > max_val:
                raise ValueError(f"'{param}' must be less than or equal to {max_val}")

        # --- Risk Management Section ---
        if 'risk_management' not in self.config:
           raise ValueError("Missing 'risk_management' section in config.yaml")

        risk_params = [
            ('order_size_percentage', (int, float), 0, 1), ('stop_loss_percentage', (int, float), 0, 1),
            ('take_profit_percentage', (int, float), 0, 1), ('max_open_positions', int, 1, None),
            ('time_based_exit_minutes', int, 1, None)
        ]
        for param, type_, min_val, max_val in risk_params:
            if param not in self.config['risk_management']:
                raise ValueError(f"Missing '{param}' in config.yaml risk_management section")
            if not isinstance(self.config['risk_management'][param], type_):
                raise ValueError(f"'{param}' must be of type {type_}")
            if min_val is not None and self.config['risk_management'][param] < min_val:
                raise ValueError(f"'{param}' must be greater than or equal to {min_val}")
            if max_val is not None and self.config['risk_management'][param] > max_val:
                raise ValueError(f"'{param}' must be less than or equal to {max_val}")

    def _initialize_exchange(self):
        """Initializes the exchange connection."""
        try:
            exchange = getattr(ccxt, self.exchange_id)({
                'apiKey': self.api_key,
                'secret': self.api_secret,
                'options': {'defaultType': 'future'},
                'recvWindow': 60000
            })
            exchange.load_markets()  # Ensure markets are loaded
            logger.info(f"{Fore.GREEN}Connected to {self.exchange_id.upper()} successfully.{Style.RESET_ALL}")
            return exchange
        except Exception as e:
            logger.error(f"{Fore.RED}Error initializing exchange: {e}{Style.RESET_ALL}")
            exit()

    @retry_api_call()
    def fetch_market_price(self):
        """Fetches the current market price of the symbol."""
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
        """Fetches the order book for the symbol."""
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
        """Fetches historical price data (closing prices)."""
        if limit is None:
            limit = max(self.volatility_window, self.ema_period, self.rsi_period + 1, self.macd_long_period,
                        self.stoch_rsi_period) + 1

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
        """Calculates the historical volatility of the asset."""
        prices = self.fetch_historical_prices(limit=self.volatility_window)
        if not prices or len(prices) < self.volatility_window:
            return None
        returns = np.diff(prices) / prices[:-1]
        volatility = np.std(returns)
        logger.debug(f"Calculated volatility: {volatility}")
        return volatility

    def calculate_ema(self, prices, period=None):
        """Calculates the Exponential Moving Average (EMA)."""
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
        """Calculates the Relative Strength Index (RSI)."""
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
        """Calculates the Moving Average Convergence Divergence (MACD)."""
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
        """Calculates the Stochastic RSI."""
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
        """Fetches the available USDT balance."""
        return self.exchange.fetch_balance().get('USDT', {}).get('free', 0)

    def calculate_order_size(self):
        """Calculates the order size based on balance and volatility."""
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
        final_size = min(adjusted_size, balance * 0.05)  # Cap at 5%
        logger.info(
            f"{Fore.CYAN}Calculated order size: {final_size:.2f} (Balance: {balance:.2f}, Volatility: {volatility:.5f}){Style.RESET_ALL}"
        )
        return final_size

    def compute_trade_signal_score(self, price, ema, rsi, orderbook_imbalance):
        """Computes a trade signal score based on multiple indicators."""
        score = 0
        reasons = []

        macd, macd_signal, macd_hist = self.calculate_macd(self.fetch_historical_prices())
        stoch_rsi_k, stoch_rsi_d = self.calculate_stoch_rsi(self.fetch_historical_prices())

        if orderbook_imbalance is not None:
            if orderbook_imbalance < (1 / self.imbalance_threshold):
                score += 1
                reasons.append(f"{Fore.GREEN}Order book: strong bid-side pressure.{Style.RESET_ALL}")
            elif orderbook_imbalance > self.imbalance_threshold:
                score -= 1
                reasons.append(f"{Fore.RED}Order book: strong ask-side pressure.{Style.RESET_ALL}")

        if ema is not None:
            if price > ema:
                score += 1
                reasons.append(f"{Fore.GREEN}Price above EMA (bullish).{Style.RESET_ALL}")
            else:
                score -= 1
                reasons.append(f"{Fore.RED}Price below EMA (bearish).{Style.RESET_ALL}")

        if rsi is not None:
            if rsi < 30:
                score += 1
                reasons.append(f"{Fore.GREEN}RSI < 30 (oversold).{Style.RESET_ALL}")
            elif rsi > 70:
                score -= 1
                reasons.append(f"{Fore.RED}RSI > 70 (overbought).{Style.RESET_ALL}")
            else:
                reasons.append(f"{Fore.YELLOW}RSI neutral.{Style.RESET_ALL}")

        if macd is not None and macd_signal is not None:
            if macd > macd_signal:
                score += 1
                reasons.append(f"{Fore.GREEN}MACD above signal (bullish).{Style.RESET_ALL}")
            else:
                score -= 1
                reasons.append(f"{Fore.RED}MACD below signal (bearish).{Style.RESET_ALL}")

        if stoch_rsi_k is not None and stoch_rsi_d is not None:
            if stoch_rsi_k < 20 and stoch_rsi_d < 20:
                score += 1
                reasons.append(f"{Fore.GREEN}Stochastic RSI < 20 (bullish).{Style.RESET_ALL}")
            elif stoch_rsi_k > 80 and stoch_rsi_d > 80:
                score -= 1
                reasons.append(f"{Fore.RED}Stochastic RSI > 80 (bearish).{Style.RESET_ALL}")
            else:
                reasons.append(f"{Fore.YELLOW}Stochastic RSI neutral.{Style.RESET_ALL}")

        return score, reasons

    @retry_api_call()
    def place_order(self, side, order_size, order_type="market", price=None, stop_loss_price=None,
                    take_profit_price=None):
        """Places an order with the exchange."""
        try:
            params = {}
            if stop_loss_price:
                params['stopLoss'] = f"{stop_loss_price:.2f}"
            if take_profit_price:
                params['takeProfit'] = f"{take_profit_price:.2f}"

            if self.simulation_mode:
                if order_type == "limit" and price is None:
                    logger.error(
                        f"{Fore.RED}[SIMULATION] Limit price required for limit orders.{Style.RESET_ALL}"
                    )
                    return None
                simulated_price = price if price else self.fetch_market_price()
                trade_details = {
                    "status": "simulated",
                    "side": side,
                    "size": order_size,
                    "price": simulated_price,
                    "order_type": order_type,
                    "stop_loss": stop_loss_price,
                    "take_profit": take_profit_price,
                    "timestamp": time.time()
                }
                logger.info(
                    f"{Fore.CYAN}[SIMULATION] {side.upper()} {order_type} order: size {order_size:.2f} at {simulated_price}, SL: {stop_loss_price}, TP: {take_profit_price}.{Style.RESET_ALL}"
                )
                return trade_details
            else:
                if order_type == "market":
                    order = self.exchange.create_market_order(self.symbol, side, order_size, params=params)
                elif order_type == "limit":
                    if price is None:
                        logger.error(f"{Fore.RED}Limit price required for limit orders.{Style.RESET_ALL}")
                        return None
                    order = self.exchange.create_limit_order(self.symbol, side, order_size, price, params=params)
                elif order_type == "stop_market":
                    if not stop_loss_price:
                        logger.error("{Fore.RED} Stop Loss Price is required for stop_market order")
                        return None
                    order = self.exchange.create_market_order(self.symbol, side, order_size, params=params)

                else:
                    logger.error(f"{Fore.RED}Unsupported order type: {order_type}{Style.RESET_ALL}")
                    return None

                logger.info(f"{Fore.CYAN}{side.upper()} {order_type} order placed: {order}{Style.RESET_ALL}")
                return order

        except ccxt.InsufficientFunds as e:
            logger.error(f"{Fore.RED}Insufficient funds to place {side} order: {e}{Style.RESET_ALL}")
            return None
        except ccxt.OrderNotFound as e:  # Catch OrderNotFound
            logger.error(f"{Fore.RED}Order not found (e.g., trying to cancel a non-existent order): {e}{Style.RESET_ALL}")
            return None
        except ccxt.InvalidOrder as e:  # Catch InvalidOrder
            logger.error(f"{Fore.RED}Invalid order parameters: {e}{Style.RESET_ALL}")
            return None
        except Exception as e:
            logger.error(f"{Fore.RED}Error placing {side} order: {e}{Style.RESET_ALL}")
            return None

    def manage_positions(self):
        """Manages open positions (e.g., time-based exit)."""
        for position in list(self.open_positions):
            time_elapsed = (time.time() - position['entry_time']) / 60
            if time_elapsed >= self.time_based_exit_minutes:
                logger.info(
                    f"{Fore.YELLOW}Time-based exit triggered for position: {position}{Style.RESET_ALL}"
                )
                if position['side'] == 'buy':
                    self.place_order('sell', position['size'])  # Exit at market
                else:
                    self.place_order('buy', position['size'])
                self.open_positions.remove(position)

    @retry_api_call()
    def cancel_orders(self):
        """Cancels all open orders for the symbol."""
        open_orders = self.exchange.fetch_open_orders(self.symbol)
        if open_orders:
            logger.info(f"{Fore.MAGENTA}Cancelling Open Orders: {open_orders}{Style.RESET_ALL}")
            for order in open_orders:
                cancelled = self.exchange.cancel_order(order['id'], self.symbol)
                if cancelled:
                    logger.info(f"{Fore.YELLOW}Cancelled order: {order['id']}{Style.RESET_ALL}")
                else:
                    logger.warning(
                        f"{Fore.YELLOW}Failed to cancel order: {order['id']}, might already be filled or cancelled.{Style.RESET_ALL}"
                    )

    def scalp_trade(self):
        """Main trading loop."""
        while True:
            self.iteration += 1
            logger.info(f"\n--- Iteration {self.iteration} ---")

            # Fetch data (handle None values)
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
            volatility = self.calculate_volatility()

            logger.info(
                f"Price: {price:.2f} | EMA: {ema if ema is not None else 'N/A'} | RSI: {rsi if rsi is not None else 'N/A'} | Stoch RSI K: {stoch_rsi_k if stoch_rsi_k is not None else 'N/A'} | Stoch RSI D: {stoch_rsi_d if stoch_rsi_d is not None else 'N/A'} | Volatility: {volatility if volatility is not None else 'N/A'}"
            )
            logger.info(f"Order Book Imbalance: {orderbook_imbalance:.2f}")

            order_size = self.calculate_order_size()
            if order_size == 0:
                logger.warning(f"{Fore.YELLOW}Order size is 0. Skipping this iteration.{Style.RESET_ALL}")
                time.sleep(10)
                continue

            signal_score, reasons = self.compute_trade_signal_score(price, ema, rsi, orderbook_imbalance)
            logger.info(f"Trade Signal Score: {signal_score}")
            for reason in reasons:
                logger.info(f"Reason: {reason}")

            # Entry logic (limit orders, stop-loss, take-profit)
            if len(self.open_positions) < self.max_open_positions:
                if signal_score >= 2:
                    # Calculate stop-loss and take-profit prices
                    stop_loss_price = price * (1 - self.stop_loss_pct)
                    take_profit_price = price * (1 + self.take_profit_pct)

                    # Place buy limit order
                    entry_order = self.place_order('buy', order_size, order_type="limit", price=price * 0.999,
                                                  stop_loss_price=stop_loss_price, take_profit_price=take_profit_price)

                    if entry_order:
                        logger.info(f"{Fore.GREEN}Entering LONG position.{Style.RESET_ALL}")
                        self.open_positions.append({
                            'side': 'buy',
                            'size': order_size,
                            'entry_price': entry_order['price'] if not self.simulation_mode else entry_order['price'],
                            'entry_time': time.time(),
                            'stop_loss': stop_loss_price,  # Store SL and TP
                            'take_profit': take_profit_price
                        })

                elif signal_score <= -2:
                    # Calculate stop-loss and take-profit prices
                    stop_loss_price = price * (1 + self.stop_loss_pct)
                    take_profit_price = price * (1 - self.take_profit_pct)

                    # Place sell limit order
                    entry_order = self.place_order('sell', order_size, order_type="limit", price=price * 1.001,
                                                  stop_loss_price=stop_loss_price, take_profit_price=take_profit_price)

                    if entry_order:
                        logger.info(f"{Fore.RED}Entering SHORT position.{Style.RESET_ALL}")
                        self.open_positions.append({
                            'side': 'sell',
                            'size': order_size,
                            'entry_price': entry_order['price'] if not self.simulation_mode else entry_order['price'],
                            'entry_time': time.time(),
                            'stop_loss': stop_loss_price,
                            'take_profit': take_profit_price
                        })
            else:
                logger.info(
                    f"{Fore.YELLOW}Max open positions reached ({self.max_open_positions}).  Not entering new trades.{Style.RESET_ALL}"
                )

            # Position management (check for SL/TP hits, time-based exit)
            self.manage_positions()

            # --- Exit logic (check for stop-loss and take-profit hits) ---
            for position in list(self.open_positions):  # Iterate over copy
                if position['side'] == 'buy':
                    if price <= position['stop_loss']:
                        logger.info(f"{Fore.RED}Stop-loss triggered for LONG position.{Style.RESET_ALL}")
                        self.place_order('sell', position['size'])  # Exit at market
                        self.open_positions.remove(position)
                    elif price >= position['take_profit']:
                        logger.info(f"{Fore.GREEN}Take-profit triggered for LONG position.{Style.RESET_ALL}")
                        self.place_order('sell', position['size'])
                        self.open_positions.remove(position)
                elif position['side'] == 'sell':
                    if price >= position['stop_loss']:
                        logger.info(f"{Fore.RED}Stop-loss triggered for SHORT position.{Style.RESET_ALL}")
                        self.place_order('buy', position['size'])
                        self.open_positions.remove(position)
                    elif price <= position['take_profit']:
                        logger.info(f"{Fore.GREEN}Take-profit triggered for SHORT position.{Style.RESET_ALL}")
                        self.place_order('buy', position['size'])
                        self.open_positions.remove(position)

            # Cancel all the orders every 60 iterations
            if self.iteration % 60 == 0:
                self.cancel_orders()

            time.sleep(10)


if __name__ == "__main__":
    config_file = 'config.yaml'
    if not os.path.exists(config_file):
        default_config = {
            'logging_level': 'DEBUG',  # Can be DEBUG, INFO, WARNING, ERROR
            'exchange': {
                'exchange_id': os.getenv('EXCHANGE_ID', 'bybit'),
            },
            'trading': {
                'symbol': input("Enter the trading symbol (e.g., BTC/USDT): ").strip().upper(),
                'simulation_mode': os.getenv('SIMULATION_MODE', 'True').lower() in ('true', '1', 'yes'),
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
                'max_open_positions': int(os.getenv('MAX_OPEN_POSITIONS', 1)),
                'time_based_exit_minutes': int(os.getenv('TIME_BASED_EXIT_MINUTES', 15)),
            }
        }
        with open(config_file, 'w') as f:
            yaml.dump(default_config, f, indent=4)
        print(f"{Fore.YELLOW}Default config.yaml created. Please review and run again.{Style.RESET_ALL}")
    else:
        bot = ScalpingBot(config_file=config_file)
        bot.scalp_trade()
