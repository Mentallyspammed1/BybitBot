import ccxt
import time
import os
import numpy as np
import pandas as pd
import logging
import yaml
from dotenv import load_dotenv
from colorama import Fore, Style, init as colorama_init

colorama_init(autoreset=True)

NEON_GREEN = Fore.GREEN
NEON_CYAN = Fore.CYAN
NEON_YELLOW = Fore.YELLOW
NEON_MAGENTA = Fore.MAGENTA
NEON_RED = Fore.RED
RESET_COLOR = Style.RESET_ALL

logger = logging.getLogger("EnhancedTradingBot")
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter(f"{NEON_CYAN}%(asctime)s - {NEON_YELLOW}%(levelname)s - {NEON_GREEN}%(message)s{RESET_COLOR}")
console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)
file_handler = logging.FileHandler("enhanced_trading_bot.log")
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

load_dotenv()

def retry_api_call(max_retries=3, initial_delay=1):
    def decorator(func):
        def wrapper(*args, **kwargs):
            retries = 0
            delay = initial_delay
            while retries <= max_retries:
                try:
                    return func(*args, **kwargs)
                except ccxt.RateLimitExceeded as e:
                    logger.warning(f"{Fore.YELLOW}Rate limit exceeded, retrying in {delay} seconds... (Retry {retries + 1}/{max_retries}){Style.RESET_ALL}")
                    time.sleep(delay)
                    delay *= 2  # Exponential backoff
                    retries += 1
                except ccxt.NetworkError as e:
                    logger.error(f"{Fore.RED}Network error during API call: {e}. Retrying in {delay} seconds... (Retry {retries + 1}/{max_retries}){Style.RESET_ALL}")
                    time.sleep(delay)
                    delay *= 2
                    retries += 1
                except ccxt.ExchangeError as e:
                    logger.error(f"{Fore.RED}Exchange error during API call: {e}. (Retry {retries + 1}/{max_retries}) {e}{Style.RESET_ALL}")
                    if 'Order does not exist' in str(e): #Specific handling for non critical order cancel errors.
                        return None # Return None, let caller handle.
                    else: # For other exchange errors, retry
                        time.sleep(delay)
                        delay *= 2
                        retries += 1
                except Exception as e:
                    logger.error(f"{Fore.RED}Unexpected error during API call: {e}. (Retry {retries + 1}/{max_retries}){Style.RESET_ALL}")
                    time.sleep(delay)
                    delay *= 2
                    retries += 1
            logger.error(f"{Fore.RED}Max retries reached for API call. Aborting.{Style.RESET_ALL}")
            return None  # Return None to indicate failure
        return wrapper
    return decorator

class EnhancedTradingBot:
    """
    Enhanced Trading Bot with configurable signals and Flask integration.
    """

    def __init__(self, symbol, config_file='config.yaml'):
        self.load_config(config_file)
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

        # --- Signal Weights (Configurable via config.yaml) ---
        self.ema_weight = self.config['signal_weights']['ema_weight']
        self.rsi_weight = self.config['signal_weights']['rsi_weight']
        self.macd_weight = self.config['signal_weights']['macd_weight']
        self.stoch_rsi_weight = self.config['signal_weights']['stoch_rsi_weight']
        self.imbalance_weight = self.config['signal_weights']['imbalance_weight']
        self.ob_delta_change_weight = self.config['signal_weights']['ob_delta_change_weight']
        self.spread_weight = self.config['signal_weights']['spread_weight']
        self.cluster_proximity_weight = self.config['signal_weights']['cluster_proximity_weight']

        # --- Position Tracking ---
        self.position = None
        self.entry_price = None
        self.order_amount = None
        self.trade_count = 0
        self.last_ob_delta = None
        self.last_spread = None
        self.bot_running_flag = True

        # Initialize exchange connection
        self.exchange = self._initialize_exchange()

        logger.info(f"EnhancedTradingBot initialized for symbol: {self.symbol}")
        logger.info("EnhancedTradingBot initialization complete.")

    def load_config(self, config_file):
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

    def _initialize_exchange(self):
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
        if limit is None:
            limit = max(self.volatility_window, self.ema_period, self.rsi_period + 1, self.macd_long_period, self.stoch_rsi_period) + 1
        ohlcv = self.exchange.fetch_ohlcv(self.symbol, timeframe='1m', limit=limit)
        if ohlcv:
            prices = [candle[4] for candle in ohlcv]
            if len(prices) < limit:
                logger.warning(f"{Fore.YELLOW}Insufficient historical data. Fetched {len(prices)}, needed {limit}.{Style.RESET_ALL}")
                return []
            logger.debug(f"Historical prices (last 5): {prices[-5:]}")
            return prices
        else:
            logger.warning(f"{Fore.YELLOW}Historical price data unavailable.{Style.RESET_ALL}")
            return []

    def calculate_volatility(self):
        prices = self.fetch_historical_prices(limit=self.volatility_window)
        if not prices or len(prices) < self.volatility_window:
            return None
        returns = np.diff(prices) / prices[:-1]
        volatility = np.std(returns)
        logger.debug(f"Calculated volatility: {volatility}")
        return volatility

    def calculate_ema(self, prices, period=None):
        if period is None:
            period = self.ema_period
        if not prices or len(prices) < period:
            return None
        weights = np.exp(np.linspace(-1., 0., period))
        weights /= weights.sum()
        ema = np.convolve(prices, weights, mode='valid')[-1]
        logger.debug(f"Calculated EMA: {ema:.2f}")
        return ema

    def calculate_rsi(self, prices):
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
        logger.debug(f"Calculated RSI: {rsi:.2f}")
        return rsi

    def calculate_macd(self, prices):
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
        logger.debug(f"MACD: {macd:.2f}, Signal: {signal:.2f}, Histogram: {hist:.2f}")
        return macd, signal, hist

    def calculate_stoch_rsi(self, prices):
        period = self.stoch_rsi_period
        if not prices or len(prices) < period:
            return None, None

        close_series = pd.Series(prices)
        min_val = close_series.rolling(window=period).min()
        max_val = close_series.rolling(window=period).max()
        stoch_rsi = 100 * (close_series - min_val) / (max_val - min_val)
        k_line = stoch_rsi.rolling(window=self.stoch_rsi_k_period).mean()
        d_line = k_line.rolling(window=self.stoch_rsi_d_period).mean()

        if k_line.empty or d_line.empty or pd.isna(k_line.iloc[-1]) or pd.isna(d_line.iloc[-1]):
            return None, None

        k_val = k_line.iloc[-1]
        d_val = d_line.iloc[-1]
        logger.debug(f"Stoch RSI: K = {k_val:.2f}, D = {d_val:.2f}")
        return k_val, d_val

    def calculate_order_size(self):
        try:
            balance = self.fetch_account_balance()
            if balance is None:
                return 0
        except Exception as e:
            logger.error(f"Error fetching balance for order size calculation: {e}")
            return 0

        order_size_usd = balance * self.order_size_percentage
        last_price = self.fetch_market_price()
        if last_price is None:
            logger.warning("Could not fetch market price to calculate order amount.")
            return 0

        order_amount = order_size_usd / last_price if last_price > 0 else 0
        logger.info(f"Calculated order size: {order_size_usd:.2f} USDT, Amount: {order_amount:.4f} {self.symbol.split('/')[0]} (Balance: {balance:.2f} USDT)")
        return order_amount

    def compute_trade_signal_score(self):
        df = self.fetch_historical_prices(limit=100)
        if not df: # Check for empty list which indicates data fetch failure
            logger.warning("No historical data available for computing trade signal.")
            return 0, []
        closes = df
        ema = self.calculate_ema(closes)
        rsi = self.calculate_rsi(closes)
        macd, macd_signal, _ = self.calculate_macd(closes)
        stoch_k, stoch_d = self.calculate_stoch_rsi(closes)
        imbalance_ratio = self.fetch_order_book()
        current_price = self.fetch_market_price()

        score = 0
        reasons = []

        if ema is not None:
            if closes[-1] > ema:
                score += self.ema_weight
                reasons.append(f"Price is above EMA (bullish) [Weight: {self.ema_weight}].")
            else:
                score -= self.ema_weight
                reasons.append(f"Price is below EMA (bearish) [Weight: {self.ema_weight}].")
        else:
            reasons.append("EMA data unavailable.")

        if rsi is not None:
            if rsi < 30:
                score += self.rsi_weight
                reasons.append(f"RSI indicates oversold conditions [Weight: {self.rsi_weight}].")
            elif rsi > 70:
                score -= self.rsi_weight
                reasons.append(f"RSI indicates overbought conditions [Weight: {self.rsi_weight}].")
            else:
                reasons.append("RSI is neutral.")
        else:
            reasons.append("RSI data unavailable.")

        if macd is not None and macd_signal is not None:
            if macd > macd_signal:
                score += self.macd_weight
                reasons.append(f"MACD is bullish [Weight: {self.macd_weight}].")
            else:
                score -= self.macd_weight
                reasons.append(f"MACD is bearish [Weight: {self.macd_weight}].")
        else:
            reasons.append("MACD data unavailable.")

        if stoch_k is not None and stoch_d is not None:
            if stoch_k < 0.2 and stoch_d < 0.2:
                score += self.stoch_rsi_weight
                reasons.append(f"Stochastic RSI indicates bullish potential [Weight: {self.stoch_rsi_weight}].")
            elif stoch_k > 0.8 and stoch_d > 0.8:
                score -= self.stoch_rsi_weight
                reasons.append(f"Stochastic RSI indicates bearish potential [Weight: {self.stoch_rsi_weight}].")
            else:
                reasons.append("Stochastic RSI is neutral.")
        else:
            reasons.append("Stochastic RSI data unavailable.")

        if imbalance_ratio is not None:
            if imbalance_ratio < (1 / self.imbalance_threshold):
                score += self.imbalance_weight
                reasons.append(f"Order book indicates strong bid-side pressure [Weight: {self.imbalance_weight}].")
            elif imbalance_ratio > self.imbalance_threshold:
                score -= self.imbalance_weight
                reasons.append(f"Order book indicates strong ask-side pressure [Weight: {self.imbalance_weight}].")

        logger.info(f"Trade Signal Score: {score} | Reasons: {reasons}")
        return score, reasons

    @retry_api_call()
    def place_order(self, side, order_amount):
        if self.simulation_mode:
            current_price = self.fetch_market_price()
            logger.info(f"{Fore.CYAN}[SIMULATION] {side.upper()} order for amount {order_amount:.4f} {self.symbol.split('/')[0]} at price {current_price:.2f} executed.{Style.RESET_ALL}")
            trade_details = {"status": "simulated", "side": side, "amount": order_amount, "price": current_price, "timestamp": time.time()}
            logger.info(f"Trade Details: {trade_details}")
            self.trade_count += 1
            return trade_details
        else:
            order = self.exchange.create_market_order(self.symbol, side, order_amount)
            logger.info(f"Order placed: {order}")
            trade_details = {
                "status": "executed",
                "side": order['side'],
                "amount": order['amount'],
                "price": order['price'],
                "timestamp": order['timestamp'],
                "order_id": order['id']
            }
            logger.info(f"Trade Details: {trade_details}")
            self.trade_count += 1
            return order

    @retry_api_call()
    def manage_trailing_stop_loss(self, current_price):
        if not self.trailing_stop_loss_active or self.simulation_mode or self.position is None or self.order_amount is None:
            return

        open_orders = self.exchange.fetch_open_orders(self.symbol)
        stop_orders = [order for order in open_orders if order['type'] == 'stop_market' and order['reduceOnly']]

        if self.position == 'long':
            self.high_since_entry = max(self.high_since_entry, current_price)
            new_stop_price = self.high_since_entry * (1 - self.trailing_stop_callback)
            if stop_orders:
                existing_stop = stop_orders[0]
                existing_price = float(existing_stop.get('stopPrice', 0))
                if new_stop_price > existing_price and not np.isclose(new_stop_price, existing_price):
                    self.exchange.edit_order(existing_stop['id'], self.symbol, 'stop_market', self.order_amount, new_stop_price, params={'stopPrice': new_stop_price, 'reduce_only': True})
                    logger.info(f"Updated trailing stop-loss for long to {new_stop_price:.2f}")
            else:
                self.exchange.create_order(self.symbol, 'stop_market', 'sell', self.order_amount, new_stop_price, params={'stopPrice': new_stop_price, 'reduce_only': True})
                logger.info(f"Set initial trailing stop-loss for long at {new_stop_price:.2f}")

        elif self.position == 'short':
            self.low_since_entry = min(self.low_since_entry, current_price)
            new_stop_price = self.low_since_entry * (1 + self.trailing_stop_callback)
            if stop_orders:
                existing_stop = stop_orders[0]
                existing_price = float(existing_stop.get('stopPrice', 0))
                if new_stop_price < existing_price and not np.isclose(new_stop_price, existing_price):
                    self.exchange.edit_order(existing_stop['id'], self.symbol, 'stop_market', self.order_amount, new_stop_price, params={'stopPrice': new_stop_price, 'reduce_only': True})
                    logger.info(f"Updated trailing stop-loss for short to {new_stop_price:.2f}")
            else:
                self.exchange.create_order(self.symbol, 'stop_market', 'buy', self.order_amount, new_stop_price, params={'stopPrice': new_stop_price, 'reduce_only': True})
                logger.info(f"Set initial trailing stop-loss for short at {new_stop_price:.2f}")

    @retry_api_call()
    def close_position(self):
        if self.position is None:
            logger.warning("No position to close.")
            return None

        close_price = self.fetch_market_price()
        if close_price is None:
            logger.error("Could not fetch market price for closing position, aborting close.")
            return None

        if self.position == 'long':
            order = self.place_order('sell', self.order_amount)
            logger.info(f"{Fore.YELLOW}Closed long position at {close_price:.2f}{Style.RESET_ALL}")
            pnl = (close_price - self.entry_price) * self.order_amount
        elif self.position == 'short':
            order = self.place_order('buy', self.order_amount)
            logger.info(f"{Fore.YELLOW}Closed short position at {close_price:.2f}{Style.RESET_ALL}")
            pnl = (self.entry_price - close_price) * self.order_amount
        else:
            logger.error("Unknown position type, cannot close.")
            return None

        if order:
            logger.info(f"Trade PnL: {pnl:.4f} USDT")
            self.position = None
            self.entry_price = None
            self.order_amount = None
            self.high_since_entry = -np.inf
            self.low_since_entry = np.inf
            return order
        else:
            logger.error(f"Failed to close position.")
            return None

    @retry_api_call()
    def fetch_account_balance(self):
        balance = self.exchange.fetch_balance()
        usdt_balance = balance.get('USDT', {}).get('free', 0)
        logger.debug(f"Fetched account balance: {usdt_balance:.2f} USDT")
        return usdt_balance

    @retry_api_call()
    def fetch_open_orders(self):
        orders = self.exchange.fetch_open_orders(self.symbol)
        logger.debug(f"Fetched {len(orders)} open orders.")
        return orders

    @retry_api_call()
    def fetch_position_pnl(self):
        if self.position is None:
            return 0.0

        position = self.exchange.fetch_positions([self.symbol])
        if position:
            position_data = next((p for p in position if p['symbol'] == self.symbol), None)
            if position_data:
                pnl = position_data.get('percentage', 0)
                logger.debug(f"Fetched position PnL: {pnl:.2f}%")
                return pnl
            else:
                logger.warning(f"No position data found for symbol {self.symbol}")
                return 0.0
        else:
            logger.warning("No position data returned from exchange.")
            return 0.0

    def trading_loop(self):
        iteration = 0
        logger.info("Starting trading loop...")
        while self.bot_running_flag:
            iteration += 1
            logger.info(f"\n--- Iteration {iteration} ---")
            current_price = self.fetch_market_price()
            if current_price is None:
                time.sleep(10)
                continue

            signal_score, reasons = self.compute_trade_signal_score()
            order_amount = self.calculate_order_size()

            if order_amount == 0:
                logger.warning("Calculated order amount is zero, skipping trade decision.")
            elif self.open_positions and len(self.open_positions) >= self.max_open_positions:
                 logger.info(f"Max open positions reached ({self.max_open_positions}). No new positions will be opened.")
            elif self.position is None:
                if signal_score >= 2: # or (signal_score >= 1 and self.is_oversold()): #Consider oversold condition for long
                    logger.info(f"{Fore.GREEN}Entering LONG position based on signal (Score: {signal_score}).{Style.RESET_ALL}")
                    order = self.place_order('buy', order_amount)
                    if order:
                        self.position = 'long'
                        self.entry_price = current_price
                        self.order_amount = order_amount
                        self.high_since_entry = current_price
                        self.open_positions.append({'symbol': self.symbol, 'side': 'long', 'entry_price': current_price, 'amount': order_amount, 'timestamp': time.time()}) # Track positions
                        if self.trailing_stop_loss_active:
                            self.manage_trailing_stop_loss(current_price)
                elif signal_score <= -2: #or (signal_score <= -1 and self.is_overbought()): #Consider overbought for short
                    logger.info(f"{Fore.RED}Entering SHORT position based on signal (Score: {signal_score}).{Style.RESET_ALL}")
                    order = self.place_order('sell', order_amount)
                    if order:
                        self.position = 'short'
                        self.entry_price = current_price
                        self.order_amount = order_amount
                        self.low_since_entry = current_price
                        self.open_positions.append({'symbol': self.symbol, 'side': 'short', 'entry_price': current_price, 'amount': order_amount, 'timestamp': time.time()}) # Track positions
                        if self.trailing_stop_loss_active:
                            self.manage_trailing_stop_loss(current_price)
                else:
                    logger.info(f"No clear trade signal (Score: {signal_score}), remaining flat.")
            else:
                if self.position == 'long' and current_price >= self.entry_price * (1 + self.take_profit_pct):
                    logger.info(f"{Fore.YELLOW}Exiting LONG position for take profit.{Style.RESET_ALL}")
                    self.close_position()
                    self.remove_closed_position({'symbol': self.symbol, 'side': 'long'}) # Clean up closed positions
                elif self.position == 'short' and current_price <= self.entry_price * (1 - self.take_profit_pct):
                    logger.info(f"{Fore.YELLOW}Exiting SHORT position for take profit.{Style.RESET_ALL}")
                    self.close_position()
                    self.remove_closed_position({'symbol': self.symbol, 'side': 'short'}) # Clean up closed positions
                elif self.time_based_exit_minutes and time.time() - self.open_positions[0]['timestamp'] >= self.time_based_exit_minutes * 60: # Check time-based exit
                    logger.info(f"{Fore.MAGENTA}Time-based exit triggered after {self.time_based_exit_minutes} minutes.{Style.RESET_ALL}")
                    self.close_position()
                    self.remove_closed_position({'symbol': self.symbol, 'side': self.position}) # Clean up closed positions
                else:
                    logger.info("Holding position; exit conditions not met.")
                    if self.trailing_stop_loss_active:
                        self.manage_trailing_stop_loss(current_price)

            logger.info(f"Iteration {iteration}: Price = {current_price:.2f}, Signal Score = {signal_score}")
            for idx, reason in enumerate(reasons, 1):
                logger.info(f"  {idx}. {reason}")
            logger.info(f"Total Trades Executed: {self.trade_count}")
            time.sleep(10)

    def remove_closed_position(self, closed_position):
        """Removes a closed position from the open_positions list."""
        self.open_positions = [pos for pos in self.open_positions if not (pos['symbol'] == closed_position['symbol'] and pos['side'] == closed_position['side'])]
        logger.info(f"Removed closed position: {closed_position}, remaining open positions: {len(self.open_positions)}")

    def run_bot(self):
        if self.exchange:
            self.trading_loop()
        else:
            logger.error("Bot initialization failed. Exiting.")

    def stop_bot_loop(self):
        self.bot_running_flag = False

if __name__ == "__main__":
    bot = EnhancedTradingBot(symbol='TRUMP/USDT', config_file='config.yaml') # Example with config file
    bot.run_bot()
