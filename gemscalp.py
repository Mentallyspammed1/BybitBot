
import ccxt
import time
import os
import numpy as np
import pandas as pd
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
        self.leverage = int(os.getenv('LEVERAGE', 10))  # Default leverage

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
        self.max_slippage_pct = float(os.getenv('MAX_SLIPPAGE_PERCENTAGE', 0.005))  # Max slippage tolerance

        # Exit parameters
        self.profit_exit_threshold = float(os.getenv('PROFIT_EXIT_THRESHOLD', 1.0)) # Multiplier for TP exit
        self.stop_loss_exit_threshold = float(os.getenv('STOP_LOSS_EXIT_THRESHOLD', 1.0)) # Multiplier for SL exit

        # Additional variables: iteration counter & daily PnL tracker
        self.iteration = 0
        self.daily_pnl = 0.0
        self.positions = [] # Track multiple positions

        # Initialize exchange connection and position state
        self.exchange = self._initialize_exchange()
        self._set_leverage() # Set leverage at initialization

    def _initialize_exchange(self):
        try:
            exchange = getattr(ccxt, self.exchange_id)({
                'apiKey': self.api_key,
                'secret': self.api_secret,
                'options': {'defaultType': 'future'},
                'recvWindow': 60000  # set recv_window to 60 seconds
            })
            exchange.load_markets()
            logger.info(f"{Fore.GREEN}Connected to {self.exchange_id.upper()} successfully.{Style.RESET_ALL}")
            return exchange
        except Exception as e:
            logger.error(f"{Fore.RED}Error initializing exchange: {e}{Style.RESET_ALL}")
            exit()

    def _set_leverage(self):
        try:
            self.exchange.set_leverage(self.leverage, self.symbol)
            logger.info(f"{Fore.CYAN}Leverage set to {self.leverage}x for {self.symbol}.{Style.RESET_ALL}")
        except Exception as e:
            logger.error(f"{Fore.RED}Error setting leverage: {e}{Style.RESET_ALL}")

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

    def calculate_stoch_rsi(self, prices, period=14):
        if len(prices) < period:
            logger.warning("Not enough data for Stochastic RSI calculation.")
            return None, None
        close = pd.Series(prices)
        min_val = close.rolling(window=period).min()
        max_val = close.rolling(window=period).max()
        stoch_rsi = 100 * (close - min_val) / (max_val - min_val)
        k = stoch_rsi.rolling(window=3).mean()
        d = k.rolling(window=3).mean()
        logger.debug(f"Calculated Stochastic RSI - K: {k.iloc[-1]}, D: {d.iloc[-1]}")
        return k.iloc[-1], d.iloc[-1]

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
        Compute a weighted trade signal score based on EMA, RSI, order book imbalance, MACD, and Stochastic RSI.
        Returns a tuple (score, reasons) where reasons is a list of strings.
        """
        score = 0
        reasons = []

        # Calculate MACD
        macd, macd_signal, macd_hist = self.calculate_macd(self.fetch_historical_prices(limit=50))

        # Calculate Stochastic RSI
        stoch_rsi_k, stoch_rsi_d = self.calculate_stoch_rsi(self.fetch_historical_prices(limit=50))

        # Order book influence
        if orderbook_imbalance < (1 / self.imbalance_threshold):
            score += 1
            reasons.append(f"{Fore.GREEN}Order book indicates strong bid-side pressure.{Style.RESET_ALL}")
        elif orderbook_imbalance > self.imbalance_threshold:
            score -= 1
            reasons.append(f"{Fore.RED}Order book indicates strong ask-side pressure.{Style.RESET_ALL}")

        # EMA influence
        if price > ema:
            score += 1
            reasons.append(f"{Fore.GREEN}Price is above EMA (bullish signal).{Style.RESET_ALL}")
        else:
            score -= 1
            reasons.append(f"{Fore.RED}Price is below EMA (bearish signal).{Style.RESET_ALL}")

        # RSI influence
        if rsi < 30:
            score += 1
            reasons.append(f"{Fore.GREEN}RSI below 30 (oversold, bullish potential).{Style.RESET_ALL}")
        elif rsi > 70:
            score -= 1
            reasons.append(f"{Fore.RED}RSI above 70 (overbought, bearish potential).{Style.RESET_ALL}")
        else:
            reasons.append("RSI in neutral zone.")

        # MACD influence
        if macd > macd_signal:
            score += 1
            reasons.append(f"{Fore.GREEN}MACD above signal line (bullish).{Style.RESET_ALL}")
        else:
            score -= 1
            reasons.append(f"{Fore.RED}MACD below signal line (bearish).{Style.RESET_ALL}")

        # Stochastic RSI influence
        if stoch_rsi_k is not None and stoch_rsi_d is not None:
            if stoch_rsi_k < 20 and stoch_rsi_d < 20:
                score += 1
                reasons.append(f"{Fore.GREEN}Stochastic RSI below 20 (bullish potential).{Style.RESET_ALL}")
            elif stoch_rsi_k > 80 and stoch_rsi_d > 80:
                score -= 1
                reasons.append(f"{Fore.RED}Stochastic RSI above 80 (bearish potential).{Style.RESET_ALL}")
            else:
                reasons.append("Stochastic RSI in neutral zone.")

        return score, reasons

    def place_order(self, side, order_size, price=None, stop_loss=None, take_profit=None):
        """
        Place a market order with stop loss and take profit parameters.
        Adjust price based on max_slippage_pct if price is provided for limit orders (not used in this market order function).
        """
        try:
            if self.simulation_mode:
                trade_details = {
                    "status": "simulated",
                    "side": side,
                    "size": order_size,
                    "price": price if price else self.fetch_market_price(),
                    "stopLoss": stop_loss,
                    "takeProfit": take_profit,
                    "timestamp": time.time()
                }
                logger.info(f"{Fore.CYAN}[SIMULATION] {side.upper()} order of size {order_size:.2f} executed at price {trade_details['price']}. SL={stop_loss}, TP={take_profit}{Style.RESET_ALL}")
                logger.info(f"Trade Details: {trade_details}")
                return trade_details
            else:
                params = {}
                if stop_loss:
                    params['stopLoss'] = stop_loss
                if take_profit:
                    params['takeProfit'] = take_profit

                order = self.exchange.create_market_order(self.symbol, side, order_size, params=params)
                trade_details = {
                    "status": "executed",
                    "orderId": order['id'],
                    "side": order['side'],
                    "size": order['amount'],
                    "price": order['average'] if 'average' in order else price, # Use average fill price if available
                    "stopLoss": stop_loss,
                    "takeProfit": take_profit,
                    "timestamp": order['timestamp']
                }
                logger.info(f"{side.upper()} order placed: {order}")
                logger.info(f"Trade Details: {trade_details}")
                return order
        except ccxt.InsufficientFunds as e:
            logger.error(f"{Fore.RED}Insufficient funds to place {side} order: {e}{Style.RESET_ALL}")
            return None
        except Exception as e:
            logger.error(f"Error placing {side} order: {e}")
            return None

    def manage_positions(self):
        """
        Manage positions: Check for take profit and stop loss conditions, and execute exit orders if necessary.
        """
        updated_positions = [] # To hold positions that are still open
        for position in self.positions:
            if position['status'] == 'open': # Only manage open positions
                current_price = self.fetch_market_price()
                if not current_price:
                    logger.warning("Could not fetch current price for position management.")
                    updated_positions.append(position) # Keep position for next iteration
                    continue

                exit_signal = None
                if position['side'] == 'long':
                    profit_condition = current_price >= position['entry_price'] * (1 + self.take_profit_pct * self.profit_exit_threshold)
                    stop_loss_condition = current_price <= position['entry_price'] * (1 - self.stop_loss_pct * self.stop_loss_exit_threshold)

                    if profit_condition:
                        exit_signal = 'take_profit'
                        exit_price = position['entry_price'] * (1 + self.take_profit_pct * self.profit_exit_threshold)
                    elif stop_loss_condition:
                        exit_signal = 'stop_loss'
                        exit_price = position['entry_price'] * (1 - self.stop_loss_pct * self.stop_loss_exit_threshold)

                elif position['side'] == 'short':
                    profit_condition = current_price <= position['entry_price'] * (1 - self.take_profit_pct * self.profit_exit_threshold)
                    stop_loss_condition = current_price >= position['entry_price'] * (1 + self.stop_loss_pct * self.stop_loss_exit_threshold)

                    if profit_condition:
                        exit_signal = 'take_profit'
                        exit_price = position['entry_price'] * (1 - self.take_profit_pct * self.profit_exit_threshold)
                    elif stop_loss_condition:
                        exit_signal = 'stop_loss'
                        exit_price = position['entry_price'] * (1 + self.stop_loss_pct * self.stop_loss_exit_threshold)

                if exit_signal:
                    logger.info(f"{Fore.YELLOW}Exit signal ({exit_signal}) triggered for {position['side'].upper()} position.{Style.RESET_ALL}")
                    self.place_order(
                        side='sell' if position['side'] == 'long' else 'buy',
                        order_size=position['size'],
                        price=exit_price # Exit at calculated TP/SL price
                    )
                    position['status'] = 'closed' # Mark position as closed
                    position['exit_price'] = exit_price
                    position['exit_reason'] = exit_signal
                    self.positions.remove(position) # Remove closed position

                    pnl = 0 # Calculate P&L
                    if exit_signal == 'take_profit':
                        pnl = (position['take_profit'] - position['entry_price']) * position['size'] if position['side'] == 'long' else (position['entry_price'] - position['take_profit']) * position['size']
                        logger.info(f"{Fore.GREEN}Position closed at take profit, P&L: {pnl:.2f} USDT{Style.RESET_ALL}")
                    elif exit_signal == 'stop_loss':
                        pnl = (position['stop_loss'] - position['entry_price']) * position['size'] if position['side'] == 'long' else (position['entry_price'] - position['stop_loss']) * position['size']
                        logger.info(f"{Fore.RED}Position closed at stop loss, P&L: {pnl:.2f} USDT{Style.RESET_ALL}")
                    self.daily_pnl += pnl # Update daily P&L
                else:
                    updated_positions.append(position) # Keep open position
            else:
                self.positions.remove(position) # Remove closed positions from tracking

        self.positions = updated_positions # Update positions list with only open positions

    def scalp_trade(self):
        """Main trading loop."""
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
            stoch_rsi_k, stoch_rsi_d = self.calculate_stoch_rsi(historical_prices)
            logger.info(f"Price: {price:.2f} | EMA: {ema:.2f} | RSI: {rsi:.2f} | Stoch RSI K: {stoch_rsi_k:.2f} | Stoch RSI D: {stoch_rsi_d:.2f} | Volatility: {volatility:.5f}")
            logger.info(f"Order Book Imbalance Ratio: {orderbook_imbalance:.2f}")

            order_size = self.calculate_order_size()
            signal_score, reasons = self.compute_trade_signal_score(price, ema, rsi, orderbook_imbalance)
            logger.info(f"Trade Signal Score: {signal_score}")
            for reason in reasons:
                logger.info(f"Reason: {reason}")

            # Entry Logic - Open new position if no position is open
            if not self.positions: # Check if positions list is empty
                if signal_score >= 2:
                    stop_loss_price = price * (1 - self.stop_loss_pct) # Stop Loss price calculation
                    take_profit_price = price * (1 + self.take_profit_pct) # Take Profit price calculation

                    order_details = self.place_order(
                        side='buy',
                        order_size=order_size,
                        stop_loss=stop_loss_price,
                        take_profit=take_profit_price
                    )
                    if order_details:
                        position = {
                            'status': 'open', # Mark position as open
                            'side': 'long',
                            'size': order_size,
                            'entry_price': price,
                            'stop_loss': stop_loss_price,
                            'take_profit': take_profit_price,
                            'timestamp': time.time()
                        }
                        self.positions.append(position) # Track new position
                        logger.info(f"{Fore.GREEN}Entering LONG position. SL={stop_loss_price:.2f}, TP={take_profit_price:.2f}{Style.RESET_ALL}")

                elif signal_score <= -2:
                    stop_loss_price = price * (1 + self.stop_loss_pct) # Stop Loss price calculation for short
                    take_profit_price = price * (1 - self.take_profit_pct) # Take Profit price calculation for short

                    order_details = self.place_order(
                        side='sell',
                        order_size=order_size,
                        stop_loss=stop_loss_price,
                        take_profit=take_profit_price
                    )
                    if order_details:
                        position = {
                            'status': 'open', # Mark position as open
                            'side': 'short',
                            'size': order_size,
                            'entry_price': price,
                            'stop_loss': stop_loss_price,
                            'take_profit': take_profit_price,
                            'timestamp': time.time()
                        }
                        self.positions.append(position) # Track new position
                        logger.info(f"{Fore.RED}Entering SHORT position. SL={stop_loss_price:.2f}, TP={take_profit_price:.2f}{Style.RESET_ALL}")
                else:
                    logger.info("No clear trade signal, remaining flat.")
            else:
                logger.info(f"Holding existing {self.positions[0]['side'].upper()} position; checking for exit conditions...")

            # Position Management - Check existing positions for exit signals
            self.manage_positions()

            # PnL Reporting - Log daily P&L at end of each iteration
            logger.info(f"Daily P&L: {self.daily_pnl:.2f} USDT")

            # Detailed reasoning summary for the current iteration:
            logger.info("=== Detailed Reasoning Summary ===")
            logger.info(f"Iteration {self.iteration}: Price = {price:.2f}, EMA = {ema:.2f}, RSI = {rsi:.2f}, Stoch RSI K: {stoch_rsi_k}, Stoch RSI D: {stoch_rsi_d}")
            logger.info(f"Order Book Imbalance = {orderbook_imbalance:.2f}, Signal Score = {signal_score}")
            for idx, reason in enumerate(reasons, 1):
                logger.info(f"  {idx}. {reason}")
            logger.info("==================================")

            time.sleep(10)

if __name__ == "__main__":
    bot = ScalpingBot()
    bot.scalp_trade()
----config+--

BYBIT_API_KEY=YOUR_BYBIT_API_KEY
BYBIT_API_SECRET=YOUR_BYBIT_API_SECRET
EXCHANGE_ID=bybit 
SIMULATION_MODE=True
LEVERAGE=10 # Example leverage setting
ORDER_BOOK_DEPTH=10
IMBALANCE_THRESHOLD=1.5
VOLATILITY_WINDOW=5
VOLATILITY_MULTIPLIER=0.02
EMA_PERIOD=10
RSI_PERIOD=14
ORDER_SIZE_PERCENTAGE=0.01
STOP_LOSS_PERCENTAGE=0.015
TAKE_PROFIT_PERCENTAGE=0.03
MAX_SLIPPAGE_PERCENTAGE=0.005
PROFIT_EXIT_THRESHOLD=1.0 # TP exit sensitivity
STOP_LOSS_EXIT_THRESHOLD=1.0 # SL exit sensitivity
