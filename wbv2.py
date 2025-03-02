import ccxt
import time
import os
import numpy as np
import pandas as pd
import logging
from dotenv import load_dotenv
from colorama import Fore, Style, init as colorama_init

# Initialize colorama for colored log outputs to the console
colorama_init(autoreset=True)

# Setup logging to both console and file
logger = logging.getLogger("EnhancedTradingBot")
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

file_handler = logging.FileHandler("enhanced_trading_bot.log")
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

load_dotenv()

class EnhancedTradingBot:
    """
    Enhanced Trading Bot with configurable signals and Flask integration.
    """

    def __init__(self):
        logger.info("Initializing EnhancedTradingBot...")

        # --- Exchange and API Configuration ---
        self.exchange_id = os.getenv('EXCHANGE_ID', 'bybit')
        self.api_key = os.getenv('BYBIT_API_KEY')
        self.api_secret = os.getenv('BYBIT_API_SECRET')
        self.simulation_mode = os.getenv('SIMULATION_MODE', 'True').lower() in ('true', '1', 'yes')

        # --- Trading Parameters ---
        self.symbol = os.getenv('TRADING_SYMBOL', 'BTC/USDT').upper() # Default symbol from env
        self.order_size_percentage = float(os.getenv('ORDER_SIZE_PERCENTAGE', 0.01))
        self.take_profit_pct = float(os.getenv('TAKE_PROFIT_PERCENTAGE', 0.03))
        self.stop_loss_pct = float(os.getenv('STOP_LOSS_PERCENTAGE', 0.015))

        # --- Technical Indicator Parameters ---
        self.ema_period = int(os.getenv('EMA_PERIOD', 10))
        self.rsi_period = int(os.getenv('RSI_PERIOD', 14))
        self.macd_short_period = 12
        self.macd_long_period = 26
        self.macd_signal_period = 9
        self.stoch_rsi_period = int(os.getenv('STOCH_RSI_PERIOD', 14))
        self.stoch_rsi_k_period = int(os.getenv('STOCH_RSI_K_PERIOD', 3))
        self.stoch_rsi_d_period = int(os.getenv('STOCH_RSI_D_PERIOD', 3))

        # --- Order Book Analysis Parameters ---
        self.order_book_depth = int(os.getenv('ORDER_BOOK_DEPTH', 10))
        self.imbalance_threshold = float(os.getenv('IMBALANCE_THRESHOLD', 1.5))
        self.volume_cluster_threshold = float(os.getenv('VOLUME_CLUSTER_THRESHOLD', 10000))
        self.ob_delta_lookback = int(os.getenv('OB_DELTA_LOOKBACK', 5))
        self.cluster_proximity_threshold_pct = float(os.getenv('CLUSTER_PROXIMITY_THRESHOLD_PCT', 0.005))

        # --- Trailing Stop Loss Parameters ---
        self.trailing_stop_loss_active = os.getenv('TRAILING_STOP_ACTIVE', 'False').lower() in ('true', '1', 'yes')
        self.trailing_stop_callback = float(os.getenv('TRAILING_STOP_CALLBACK', 0.02))
        self.high_since_entry = -np.inf
        self.low_since_entry = np.inf

        # --- Signal Weights (Configurable via .env) ---
        self.ema_weight = float(os.getenv('EMA_WEIGHT', 1.0))
        self.rsi_weight = float(os.getenv('RSI_WEIGHT', 1.0))
        self.macd_weight = float(os.getenv('MACD_WEIGHT', 1.0))
        self.stoch_rsi_weight = float(os.getenv('STOCH_RSI_WEIGHT', 1.0))
        self.imbalance_weight = float(os.getenv('IMBALANCE_WEIGHT', 1.5))
        self.ob_delta_change_weight = float(os.getenv('OB_DELTA_CHANGE_WEIGHT', 0.8))
        self.spread_weight = float(os.getenv('SPREAD_WEIGHT', -0.5)) # Negative weight for wide spread
        self.cluster_proximity_weight = float(os.getenv('CLUSTER_PROXIMITY_WEIGHT', 0.5))

        # --- Position Tracking ---
        self.position = None
        self.entry_price = None
        self.order_amount = None
        self.trade_count = 0
        self.last_ob_delta = None
        self.last_spread = None # Track last spread for change calculation

        # Initialize exchange connection
        self.exchange = self._initialize_exchange()

        logger.info("EnhancedTradingBot initialization complete.")

    def _initialize_exchange(self):
        """Initializes and authenticates the exchange connection."""
        logger.info(f"Initializing exchange: {self.exchange_id.upper()}...")
        try:
            exchange_class = getattr(ccxt, self.exchange_id)
            exchange = exchange_class({
                'apiKey': self.api_key,
                'secret': self.api_secret,
                'enableRateLimit': True,
                'options': {'defaultType': 'future'}
            })
            exchange.load_markets()
            logger.info(f"{Fore.GREEN}Connected to {self.exchange_id.upper()} successfully.{Style.RESET_ALL}")
            return exchange
        except Exception as e:
            logger.error(f"Error initializing exchange: {e}")
            return None

    def fetch_market_price(self):
        """Fetches the current market price."""
        try:
            ticker = self.exchange.fetch_ticker(self.symbol)
            price = ticker['last']
            logger.debug(f"Fetched market price: {price:.2f}")
            return price
        except ccxt.NetworkError as e:
            logger.error(f"Network error fetching market price: {e}")
            return None
        except ccxt.ExchangeError as e:
            logger.error(f"Exchange error fetching market price: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error fetching market price: {e}")
            return None

    def fetch_order_book(self):
        """Fetches and analyzes the order book, including advanced analysis."""
        try:
            order_book = self.exchange.fetch_order_book(self.symbol, limit=self.order_book_depth)
            bid_clusters, ask_clusters = self.detect_volume_clusters(order_book)

            bids = order_book.get('bids', [])
            asks = order_book.get('asks', [])

            if bids and asks:
                bid_volume = sum(bid[1] for bid in bids)
                ask_volume = sum(ask[1] for ask in asks)
                imbalance_ratio = ask_volume / bid_volume if bid_volume > 0 else float('inf')
                ob_delta = bid_volume - ask_volume # Order Book Delta

                # Calculate OB Delta Change
                ob_delta_change = None
                if self.last_ob_delta is not None:
                    ob_delta_change = ob_delta - self.last_ob_delta
                self.last_ob_delta = ob_delta # Update last OB Delta

                spread = asks[0][0] - bids[0][0] if bids and asks else None # Bid-Ask Spread
                spread_change = None
                if self.last_spread is not None and spread is not None:
                    spread_change = spread - self.last_spread
                self.last_spread = spread

                logger.info(f"Order Book: Bid Vol = {bid_volume:.2f}, Ask Vol = {ask_volume:.2f}, Imbalance Ratio = {imbalance_ratio:.2f}, OB Delta = {ob_delta:.2f}, OB Delta Change = {ob_delta_change}, Spread = {spread:.2f}, Spread Change = {spread_change}")
                return order_book, imbalance_ratio, ob_delta, ob_delta_change, spread, spread_change, bid_clusters, ask_clusters
            else:
                logger.warning("Order book data unavailable.")
                return order_book, None, None, None, None, None, [], []

        except ccxt.NetworkError as e:
            logger.error(f"Network error fetching order book: {e}")
            return None, None, None, None, None, None, [], []
        except ccxt.ExchangeError as e:
            logger.error(f"Exchange error fetching order book: {e}")
            return None, None, None, None, None, None, [], []
        except Exception as e:
            logger.error(f"Error fetching order book: {e}")
            return None, None, None, None, None, None, [], []


    def detect_volume_clusters(self, order_book):
        """Detects significant volume clusters in the order book and returns cluster prices."""
        bid_cluster_prices = []
        ask_cluster_prices = []
        bids = np.array(order_book.get('bids', []))
        asks = np.array(order_book.get('asks', []))
        if bids.size:
            bid_clusters = bids[bids[:, 1] > self.volume_cluster_threshold]
            if bid_clusters.size:
                logger.info(f"Significant bid clusters detected: {bid_clusters}")
                bid_cluster_prices = bid_clusters[:, 0].tolist()
        if asks.size:
            ask_clusters = asks[asks[:, 1] > self.volume_cluster_threshold]
            if ask_clusters.size:
                logger.info(f"Significant ask clusters detected: {ask_clusters}")
                ask_cluster_prices = ask_clusters[:, 0].tolist()
        return bid_cluster_prices, ask_cluster_prices

    def get_cluster_proximity_signal(self, current_price, bid_clusters, ask_clusters):
        """
        Assesses price proximity to volume clusters and returns a signal.
        """
        signal = 0 # Neutral signal by default
        proximity_threshold = current_price * self.cluster_proximity_threshold_pct

        for bid_price in bid_clusters:
            if 0 < current_price - bid_price <= proximity_threshold:
                signal += 0.5 # Bullish signal if price is near bid cluster (support)
                logger.info(f"Price is close to bid cluster at {bid_price:.2f} (potential support).")

        for ask_price in ask_clusters:
            if 0 < ask_price - current_price <= proximity_threshold:
                signal -= 0.5 # Bearish signal if price is near ask cluster (resistance)
                logger.info(f"Price is close to ask cluster at {ask_price:.2f} (potential resistance).")
        return signal


    def fetch_historical_prices(self, timeframe='1m', limit=100):
        """Fetches historical OHLCV data."""
        try:
            ohlcv = self.exchange.fetch_ohlcv(self.symbol, timeframe='1m', limit=limit)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            logger.debug(f"Fetched {len(df)} historical price entries.")
            return df
        except ccxt.NetworkError as e:
            logger.error(f"Network error fetching historical prices: {e}")
            return None
        except ccxt.ExchangeError as e:
            logger.error(f"Exchange error fetching historical prices: {e}")
            return None
        except Exception as e:
            logger.error(f"Error fetching historical prices: {e}")
            return None

    def calculate_ema(self, prices, period=None):
        """Calculates Exponential Moving Average."""
        if period is None:
            period = self.ema_period
        if len(prices) < period:
            logger.warning("Not enough data for EMA calculation.")
            return None
        weights = np.exp(np.linspace(-1., 0., period))
        weights /= weights.sum()
        ema = np.convolve(prices, weights, mode='valid')[-1]
        logger.debug(f"Calculated EMA: {ema:.2f}")
        return ema

    def calculate_rsi(self, prices, period=None):
        """Calculates Relative Strength Index."""
        if period is None:
            period = self.rsi_period
        if len(prices) < period + 1:
            logger.warning("Not enough data for RSI calculation.")
            return None
        deltas = np.diff(prices)
        gains = np.maximum(deltas, 0)
        losses = -np.minimum(deltas, 0)
        avg_gain = np.mean(gains[-period:])
        avg_loss = np.mean(losses[-period:])
        if avg_loss == 0:
            rsi = 100
        else:
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
        logger.debug(f"Calculated RSI: {rsi:.2f}")
        return rsi

    def calculate_macd(self, prices):
        """Calculates Moving Average Convergence Divergence."""
        if len(prices) < self.macd_long_period:
            logger.warning("Not enough data for MACD calculation.")
            return None, None, None
        short_ema = self.calculate_ema(prices[-self.macd_short_period:], period=self.macd_short_period)
        long_ema = self.calculate_ema(prices[-self.macd_long_period:], period=self.macd_long_period)
        if short_ema is None or long_ema is None:
            logger.warning("Not enough data for MACD EMAs.")
            return None, None, None
        macd = short_ema - long_ema
        signal = self.calculate_ema([macd], period=self.macd_signal_period)
        if signal is None:
            logger.warning("Not enough data for MACD signal calculation.")
            return macd, None, None
        hist = macd - signal
        logger.debug(f"MACD: {macd:.2f}, Signal: {signal:.2f}, Histogram: {hist:.2f}")
        return macd, signal, hist

    def calculate_stoch_rsi(self, df):
        """Calculates Stochastic RSI indicators (K and D lines)."""
        period = self.stoch_rsi_period
        smooth_k = self.stoch_rsi_k_period
        smooth_d = self.stoch_rsi_d_period

        if len(df) < period:
            logger.warning("Not enough data for Stochastic RSI calculation.")
            return None, None
        delta = df['close'].diff(1)
        gain = delta.clip(lower=0).rolling(window=period).mean()
        loss = (-delta.clip(upper=0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        stoch_rsi = (rsi - rsi.rolling(window=period).min()) / (rsi.rolling(window=period).max() - rsi.rolling(window=period).min())
        k = stoch_rsi.rolling(window=smooth_k).mean()
        d = k.rolling(window=smooth_d).mean()
        k_last, d_last = k.iloc[-1], d.iloc[-1]
        logger.debug(f"Stoch RSI: K = {k_last:.2f}, D = {d_last:.2f}")
        return k_last, d_last

    def calculate_order_size(self):
        """Calculates order size based on balance."""
        try:
            balance = self.exchange.fetch_balance().get('USDT', {}).get('free', 0)
        except ccxt.NetworkError as e:
            logger.error(f"Network error fetching balance: {e}")
            return 0
        except ccxt.ExchangeError as e:
            logger.error(f"Exchange error fetching balance: {e}")
            return 0
        except Exception as e:
            logger.error(f"Error fetching balance: {e}")
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
        """Computes a trade signal score based on indicators and order book analysis."""
        df = self.fetch_historical_prices(limit=100)
        if df is None or df.empty:
            logger.warning("No historical data available for computing trade signal.")
            return 0, []
        closes = df['close'].tolist()
        ema = self.calculate_ema(closes)
        rsi = self.calculate_rsi(closes)
        macd, macd_signal, _ = self.calculate_macd(closes)
        stoch_k, stoch_d = self.calculate_stoch_rsi(df)
        (order_book, imbalance_ratio, ob_delta, ob_delta_change, spread, spread_change, bid_clusters, ask_clusters) = self.fetch_order_book()
        current_price = self.fetch_market_price()
        cluster_proximity_signal = self.get_cluster_proximity_signal(current_price, bid_clusters, ask_clusters)


        score = 0
        reasons = []

        # --- Signal Logic with Configurable Weights ---
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

        if ob_delta_change is not None:
            if ob_delta_change > 0:
                score += self.ob_delta_change_weight
                reasons.append(f"Order Book Delta is increasing (bullish) [Weight: {self.ob_delta_change_weight}].")
            elif ob_delta_change < 0:
                score -= self.ob_delta_change_weight
                reasons.append(f"Order Book Delta is decreasing (bearish) [Weight: {self.ob_delta_change_weight}].")

        if spread is not None:
            if spread > current_price * 0.001: # Example wide spread condition (0.1%)
                score += self.spread_weight # Configurable negative weight
                reasons.append(f"Bid-Ask spread is wide (increased risk) [Weight: {self.spread_weight}].")
            else:
                reasons.append("Spread is normal.")

        score += self.cluster_proximity_weight * cluster_proximity_signal # Apply weight to cluster proximity signal
        if cluster_proximity_signal > 0:
            reasons.append(f"Price is near bid volume clusters (potential support) [Weight: {self.cluster_proximity_weight * cluster_proximity_signal}].")
        elif cluster_proximity_signal < 0:
            reasons.append(f"Price is near ask volume clusters (potential resistance) [Weight: {self.cluster_proximity_weight * cluster_proximity_signal}].")

        logger.info(f"Trade Signal Score: {score} | Reasons: {reasons}")
        return score, reasons

    def place_order(self, side, order_amount):
        """Places a market order or simulates it."""
        try:
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
        except ccxt.NetworkError as e:
            logger.error(f"Network error placing {side} order: {e}")
            return None
        except ccxt.ExchangeError as e:
            logger.error(f"Exchange error placing {side} order: {e}")
            return None
        except Exception as e:
            logger.error(f"Error placing {side} order: {e}")
            return None

    def manage_trailing_stop_loss(self, current_price):
        """Manages trailing stop loss for open positions."""
        if not self.trailing_stop_loss_active or self.simulation_mode or self.position is None or self.order_amount is None:
            return

        try:
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
        except ccxt.NetworkError as e:
            logger.error(f"Network error managing trailing stop loss: {e}")
        except ccxt.ExchangeError as e:
            logger.error(f"Exchange error managing trailing stop loss: {e}")
        except Exception as e:
            logger.error(f"Error managing trailing stop loss: {e}")

    def close_position(self):
        """Closes the current open position and calculates PnL."""
        if self.position is None:
            logger.warning("No position to close.")
            return None

        try:
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
        except ccxt.NetworkError as e:
            logger.error(f"Network error closing position: {e}")
            return None
        except ccxt.ExchangeError as e:
            logger.error(f"Exchange error closing position: {e}")
            return None
        except Exception as e:
            logger.error(f"Error closing position: {e}")
            return None

    def trading_loop(self):
        """Main trading loop."""
        iteration = 0
        logger.info("Starting trading loop...")
        while True:
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
            elif self.position is None:
                if signal_score >= 2:
                    logger.info(f"{Fore.GREEN}Entering LONG position based on signal (Score: {signal_score}).{Style.RESET_ALL}")
                    order = self.place_order('buy', order_amount)
                    if order:
                        self.position = 'long'
                        self.entry_price = current_price
                        self.order_amount = order_amount
                        self.high_since_entry = current_price
                        if self.trailing_stop_loss_active:
                            self.manage_trailing_stop_loss(current_price)
                elif signal_score <= -2:
                    logger.info(f"{Fore.RED}Entering SHORT position based on signal (Score: {signal_score}).{Style.RESET_ALL}")
                    order = self.place_order('sell', order_amount)
                    if order:
                        self.position = 'short'
                        self.entry_price = current_price
                        self.order_amount = order_amount
                        self.low_since_entry = current_price
                        if self.trailing_stop_loss_active:
                            self.manage_trailing_stop_loss(current_price)
                else:
                    logger.info(f"No clear trade signal (Score: {signal_score}), remaining flat.")
            else:
                if self.position == 'long' and current_price >= self.entry_price * (1 + self.take_profit_pct):
                    logger.info(f"{Fore.YELLOW}Exiting LONG position for take profit.{Style.RESET_ALL}")
                    self.close_position()
                elif self.position == 'short' and current_price <= self.entry_price * (1 - self.take_profit_pct):
                    logger.info(f"{Fore.YELLOW}Exiting SHORT position for take profit.{Style.RESET_ALL}")
                    self.close_position()
                else:
                    logger.info("Holding position; exit conditions not met.")
                    if self.trailing_stop_loss_active:
                        self.manage_trailing_stop_loss(current_price)

            logger.info(f"Iteration {iteration}: Price = {current_price:.2f}, Signal Score = {signal_score}")
            for idx, reason in enumerate(reasons, 1):
                logger.info(f"  {idx}. {reason}")
            logger.info(f"Total Trades Executed: {self.trade_count}")
            time.sleep(10)

    def run_bot(self):
        """Starts the trading bot loop."""
        if self.exchange:
            self.trading_loop()
        else:
            logger.error("Bot initialization failed. Exiting.")


if __name__ == "__main__":
    bot = EnhancedTradingBot()
    bot.run_bot()
