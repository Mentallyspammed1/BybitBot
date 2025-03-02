import ccxt
import time
import os
import numpy as np
import logging
from dotenv import load_dotenv
from colorama import Fore, Style, init as colorama_init

# Initialize colorama for colored log outputs
colorama_init(autoreset=True)

# Setup logging to both console and file
logger = logging.getLogger("ScalpingBot")
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

file_handler = logging.FileHandler("scalping_bot.log")
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

# Load environment variables from .env
load_dotenv()

class ScalpingBot:
    def __init__(self):
        # API Credentials
        self.api_key = os.getenv('BYBIT_API_KEY')
        self.api_secret = os.getenv('BYBIT_API_SECRET')
        self.exchange_id = os.getenv('EXCHANGE_ID', 'bybit')

        # Trading parameters
        self.symbol = input("Enter the trading symbol (e.g., BTC/USDT): ").strip().upper()
        self.simulation_mode = os.getenv('SIMULATION_MODE', 'True').lower() in ('true', '1', 'yes')
        
        # Order book parameters
        self.order_book_depth = int(os.getenv('ORDER_BOOK_DEPTH', 10))
        self.imbalance_threshold = float(os.getenv('IMBALANCE_THRESHOLD', 1.5))

        # Volatility & Momentum settings
        self.volatility_window = int(os.getenv('VOLATILITY_WINDOW', 5))  # in minutes
        self.volatility_multiplier = float(os.getenv('VOLATILITY_MULTIPLIER', 0.02))
        self.ema_period = int(os.getenv('EMA_PERIOD', 10))
        self.rsi_period = int(os.getenv('RSI_PERIOD', 14))

        # Order and Risk Management
        self.order_size_percentage = float(os.getenv('ORDER_SIZE_PERCENTAGE', 0.01))
        self.stop_loss_pct = float(os.getenv('STOP_LOSS_PERCENTAGE', 0.015))
        self.take_profit_pct = float(os.getenv('TAKE_PROFIT_PERCENTAGE', 0.03))

        # Additional variables: iteration counter & daily PnL tracker
        self.iteration = 0
        self.daily_pnl = 0.0

        # Initialize exchange connection and position state
        self.exchange = self._initialize_exchange()
        self.current_position = None

    def _initialize_exchange(self):
        try:
            exchange = getattr(ccxt, self.exchange_id)({
                'apiKey': self.api_key,
                'secret': self.api_secret,
                'options': {'defaultType': 'future'},
                'recvWindow': 60000  # set recv_window to 60 seconds
            })
            logger.info(f"{Fore.GREEN}Connected to {self.exchange_id.upper()} successfully.{Style.RESET_ALL}")
            return exchange
        except Exception as e:
            logger.error(f"{Fore.RED}Error initializing exchange: {e}{Style.RESET_ALL}")
            exit()

    def fetch_market_price(self):
        try:
            price = self.exchange.fetch_ticker(self.symbol)['last']
            logger.debug(f"Fetched market price: {price}")
            return price
        except Exception as e:
            logger.error(f"Error fetching market price: {e}")
            return None

    def fetch_order_book(self):
        try:
            orderbook = self.exchange.fetch_order_book(self.symbol, limit=self.order_book_depth)
            bids = orderbook['bids']
            asks = orderbook['asks']
            if bids and asks:
                bid_volume = sum(bid[1] for bid in bids)
                ask_volume = sum(ask[1] for ask in asks)
                imbalance_ratio = ask_volume / bid_volume if bid_volume > 0 else float('inf')
                logger.debug(f"Order Book - Bid Vol: {bid_volume}, Ask Vol: {ask_volume}, Imbalance: {imbalance_ratio:.2f}")
                logger.info(f"Order Book: Bid Vol = {bid_volume:.2f}, Ask Vol = {ask_volume:.2f}, Imbalance Ratio = {imbalance_ratio:.2f}")
                if imbalance_ratio > self.imbalance_threshold:
                    logger.info(f"{Fore.RED}Strong Sell-Side Pressure detected (Imbalance > {self.imbalance_threshold}).{Style.RESET_ALL}")
                elif imbalance_ratio < (1 / self.imbalance_threshold):
                    logger.info(f"{Fore.GREEN}Strong Buy-Side Pressure detected (Imbalance < {1/self.imbalance_threshold}).{Style.RESET_ALL}")
                return imbalance_ratio
            else:
                logger.warning("Order book data unavailable.")
                return None
        except Exception as e:
            logger.error(f"Error fetching order book: {e}")
            return None

    def fetch_historical_prices(self, limit=50):
        try:
            ohlcv = self.exchange.fetch_ohlcv(self.symbol, timeframe='1m', limit=limit)
            prices = [candle[4] for candle in ohlcv]  # use closing prices
            logger.debug(f"Historical prices (last 5): {prices[-5:]}")
            return prices
        except Exception as e:
            logger.error(f"Error fetching historical prices: {e}")
            return []

    def calculate_volatility(self):
        prices = self.fetch_historical_prices(limit=self.volatility_window)
        if len(prices) < self.volatility_window:
            logger.warning("Not enough data to calculate volatility.")
            return None
        returns = np.diff(prices) / prices[:-1]
        volatility = np.std(returns)
        logger.debug(f"Calculated volatility: {volatility}")
        return volatility

    def calculate_ema(self, prices, period=None):
        if period is None:
            period = self.ema_period
        if len(prices) < period:
            logger.warning("Not enough data for EMA calculation.")
            return None
        weights = np.exp(np.linspace(-1., 0., period))
        weights /= weights.sum()
        ema = np.convolve(prices, weights, mode='valid')[-1]
        logger.debug(f"Calculated EMA: {ema}")
        return ema

    def calculate_rsi(self, prices):
        if len(prices) < self.rsi_period + 1:
            logger.warning("Not enough data for RSI calculation.")
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
        if len(prices) < 26:  # Ensure enough data for MACD calculation
            logger.warning("Not enough data for MACD calculation.")
            return None, None, None

        short_ema = self.calculate_ema(prices[-12:], 12)  # 12-period EMA
        long_ema = self.calculate_ema(prices[-26:], 26)   # 26-period EMA
        macd = short_ema - long_ema
        signal = self.calculate_ema([macd], 9)            # 9-period EMA of MACD
        hist = macd - signal

        logger.debug(f"MACD: {macd}, Signal: {signal}, Histogram: {hist}")
        return macd, signal, hist

    def calculate_order_size(self):
        balance = self.exchange.fetch_balance().get('USDT', {}).get('free', 0)
        volatility = self.calculate_volatility()
        if volatility is None:
            base_size = balance * self.order_size_percentage
            logger.info(f"Default order size (no volatility data): {base_size}")
            return base_size
        adjusted_size = balance * self.order_size_percentage * (1 + (volatility * self.volatility_multiplier))
        final_size = min(adjusted_size, balance * 0.05)  # Cap max exposure to 5% of balance
        logger.info(f"Calculated order size: {final_size:.2f} (Balance: {balance:.2f}, Volatility: {volatility:.5f})")
        return final_size

    def compute_trade_signal_score(self, price, ema, rsi, orderbook_imbalance):
        """
        Compute a weighted trade signal score based on EMA, RSI, order book imbalance, and MACD.
        Returns a tuple (score, reasons) where reasons is a list of strings.
        """
        score = 0
        reasons = []

        # Calculate MACD
        macd, macd_signal, macd_hist = self.calculate_macd(self.fetch_historical_prices(limit=50))

        # Order book influence
        if orderbook_imbalance < (1 / self.imbalance_threshold):
            score += 1
            reasons.append("Order book indicates strong bid-side pressure.")
        elif orderbook_imbalance > self.imbalance_threshold:
            score -= 1
            reasons.append("Order book indicates strong ask-side pressure.")

        # EMA influence
        if price > ema:
            score += 1
            reasons.append("Price is above EMA (bullish signal).")
        else:
            score -= 1
            reasons.append("Price is below EMA (bearish signal).")

        # RSI influence
        if rsi < 30:
            score += 1
            reasons.append("RSI below 30 (oversold, bullish potential).")
        elif rsi > 70:
            score -= 1
            reasons.append("RSI above 70 (overbought, bearish potential).")
        else:
            reasons.append("RSI in neutral zone.")

        # MACD influence
        if macd > macd_signal:
            score += 1
            reasons.append("MACD above signal line (bullish).")
        else:
            score -= 1
            reasons.append("MACD below signal line (bearish).")

        return score, reasons

    def place_order(self, side, order_size):
        try:
            if self.simulation_mode:
                logger.info(f"{Fore.CYAN}[SIMULATION] {side.upper()} order of size {order_size:.2f} executed.{Style.RESET_ALL}")
                return {"status": "simulated", "side": side, "size": order_size}
            else:
                order = self.exchange.create_market_order(self.symbol, side, order_size)
                logger.info(f"{side.upper()} order placed: {order}")
                return order
        except Exception as e:
            logger.error(f"Error placing {side} order: {e}")
            return None

    def scalp_trade(self):
        """Main trading loop with detailed reasoning output."""
        while True:
            self.iteration += 1
            logger.info(f"\n--- Iteration {self.iteration} ---")
            price = self.fetch_market_price()
            orderbook_imbalance = self.fetch_order_book()
            historical_prices = self.fetch_historical_prices()
            volatility = self.calculate_volatility()

            if not price or orderbook_imbalance is None or len(historical_prices) < self.ema_period:
                logger.warning("Insufficient data for a decision. Retrying in 10 seconds...")
                time.sleep(10)
                continue

            ema = self.calculate_ema(historical_prices)
            rsi = self.calculate_rsi(historical_prices)
            logger.info(f"Price: {price:.2f} | EMA: {ema:.2f} | RSI: {rsi:.2f} | Volatility: {volatility:.5f}")
            logger.info(f"Order Book Imbalance Ratio: {orderbook_imbalance:.2f}")

            order_size = self.calculate_order_size()
            signal_score, reasons = self.compute_trade_signal_score(price, ema, rsi, orderbook_imbalance)
            logger.info(f"Trade Signal Score: {signal_score}")
            for reason in reasons:
                logger.info(f"Reason: {reason}")

            # Decision making based on computed signal score:
            if self.current_position is None:
                if signal_score >= 2:
                    logger.info(f"{Fore.GREEN}Entering LONG position based on signal.{Style.RESET_ALL}")
                    self.current_position = 'long'
                    self.place_order('buy', order_size)
                elif signal_score <= -2:
                    logger.info(f"{Fore.RED}Entering SHORT position based on signal.{Style.RESET_ALL}")
                    self.current_position = 'short'
                    self.place_order('sell', order_size)
                else:
                    logger.info("No clear trade signal, remaining flat.")
            else:
                # Exit conditions: Use EMA and take profit thresholds as baseline
                if self.current_position == 'long' and price >= ema * (1 + self.take_profit_pct):
                    logger.info(f"{Fore.YELLOW}Exiting LONG position for profit.{Style.RESET_ALL}")
                    self.place_order('sell', order_size)
                    self.current_position = None
                elif self.current_position == 'short' and price <= ema * (1 - self.take_profit_pct):
                    logger.info(f"{Fore.YELLOW}Exiting SHORT position for profit.{Style.RESET_ALL}")
                    self.place_order('buy', order_size)
                    self.current_position = None
                else:
                    logger.info("Holding position; exit conditions not met.")

            # Detailed reasoning summary for the current iteration:
            logger.info("=== Detailed Reasoning Summary ===")
            logger.info(f"Iteration {self.iteration}: Price = {price:.2f}, EMA = {ema:.2f}, RSI = {rsi:.2f}")
            logger.info(f"Order Book Imbalance = {orderbook_imbalance:.2f}, Signal Score = {signal_score}")
            for idx, reason in enumerate(reasons, 1):
                logger.info(f"  {idx}. {reason}")
            logger.info("==================================")

            time.sleep(10)

if __name__ == "__main__":
    bot = ScalpingBot()
    bot.scalp_trade()
