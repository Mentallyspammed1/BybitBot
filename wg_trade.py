# trading_bot.py
import logging
import os
import time

import ccxt
from colorama import Fore, Style, init as colorama_init

colorama_init(autoreset=True)

# Neon color definitions (consistent with app.py)
NEON_GREEN = Fore.GREEN
NEON_CYAN = Fore.CYAN
NEON_YELLOW = Fore.YELLOW
NEON_MAGENTA = Fore.MAGENTA
NEON_RED = Fore.RED
RESET_COLOR = Style.RESET_ALL


class EnhancedTradingBot:
    """
    Enhanced Trading Bot class implementing grid trading strategy.
    """

    def __init__(
        self,
        symbol,
        exchange_name='binance',
        grid_levels=5,
        grid_multiplier=0.01,
        base_order_size=10,
        safety_order_size=10,
        max_safety_orders=5,
        price_deviation_to_open_safety_order=0.01,
        take_profit_percentage=0.02,
        stop_loss_percentage=0.05,
        leverage=1,
        initial_capital=1000,
        max_capital_usage_percentage=0.5,
        dynamic_grid_adjustment=True,
        atr_period=14,
        atr_deviation_multiplier=2,
        safety_order_volume_scale=1.5,
        safety_order_step_scale=1.1,
        rsi_oversold=30,
        rsi_period=14,
        use_rsi_filter=True,
        tp_sl_ratio=1.5,
        trailing_stop_loss_active_profit_percentage=0.01,
    ):
        """
        Initializes the EnhancedTradingBot with trading parameters and exchange.

        Args:
            symbol (str): Trading symbol (e.g., 'BTC/USDT').
            exchange_name (str): Name of the exchange (default: 'binance').
            grid_levels (int): Number of grid levels.
            grid_multiplier (float): Multiplier for grid spacing.
            base_order_size (float): Size of the initial base order.
            safety_order_size (float): Size of each safety order.
            max_safety_orders (int): Maximum number of safety orders to place.
            price_deviation_to_open_safety_order (float): Price deviation % to open safety order.
            take_profit_percentage (float): Take profit percentage.
            stop_loss_percentage (float): Stop loss percentage.
            leverage (int): Leverage to use for trading.
            initial_capital (float): Initial capital for the bot to use.
            max_capital_usage_percentage (float): Max % of capital to use per trade.
            dynamic_grid_adjustment (bool): Enable dynamic grid spacing with ATR.
            atr_period (int): Period for ATR calculation.
            atr_deviation_multiplier (float): Multiplier for ATR deviation.
            safety_order_volume_scale (float): Scale factor for safety order volume.
            safety_order_step_scale (float): Scale factor for safety order step.
            rsi_oversold (int): RSI oversold level for filter.
            rsi_period (int): Period for RSI calculation.
            use_rsi_filter (bool): Enable RSI filter for buy orders.
            tp_sl_ratio (float): Take profit/Stop loss ratio.
            trailing_stop_loss_active_profit_percentage (float): % profit to activate trailing SL.
        """
        self.symbol = symbol
        self.exchange_name = exchange_name
        self.grid_levels = grid_levels
        self.grid_multiplier = grid_multiplier
        self.base_order_size = base_order_size
        self.safety_order_size = safety_order_size
        self.max_safety_orders = max_safety_orders
        self.price_deviation_to_open_safety_order = price_deviation_to_open_safety_order
        self.take_profit_percentage = take_profit_percentage
        self.stop_loss_percentage = stop_loss_percentage
        self.leverage = leverage
        self.initial_capital = initial_capital
        self.max_capital_usage_percentage = max_capital_usage_percentage

        # --- New parameters ---
        self.dynamic_grid_adjustment = dynamic_grid_adjustment  # Improvement 1
        self.atr_period = atr_period  # Improvement 1
        self.atr_deviation_multiplier = atr_deviation_multiplier  # Improvement 1
        self.safety_order_volume_scale = safety_order_volume_scale  # Improvement 2
        self.safety_order_step_scale = safety_order_step_scale  # Improvement 2
        self.rsi_oversold = rsi_oversold  # Improvement 6
        self.rsi_period = rsi_period  # Improvement 6
        self.use_rsi_filter = use_rsi_filter  # Improvement 6
        self.tp_sl_ratio = tp_sl_ratio  # Improvement 4
        self.trailing_stop_loss_active_profit_percentage = (
            trailing_stop_loss_active_profit_percentage
        )  # Improvement 4

        self.exchange = self._initialize_exchange()
        self.running = False
        self.log_file_path = "enhanced_trading_bot.log"
        self.logger = self._setup_logger()
        self.current_grid = []  # To store current grid levels
        self.active_safety_orders_count = 0  # Track active safety orders
        self.safety_order_prices = []  # Improvement 2: Track safety order prices

    def _initialize_exchange(self):
        """Initializes the exchange."""
        try:
            exchange_class = getattr(ccxt, self.exchange_name)
            exchange = exchange_class(
                {
                    "apiKey": os.environ.get("BINANCE_API_KEY"),
                    "secret": os.environ.get("BINANCE_SECRET_KEY"),
                    "timeout": 30000,  # Improvement 8: Request Timeout
                }
            )
            exchange.load_markets()
            self.logger.info(
                f"Successfully initialized {self.exchange_name} exchange."
            )
            exchange.options["defaultType"] = "future"
            exchange.set_leverage(self.leverage, self.symbol)
            self.logger.info(f"Leverage set to {self.leverage}x for {self.symbol}")
            return exchange
        except AttributeError:
            self.logger.error(f"Exchange {self.exchange_name} not found in ccxt.")
            return None
        except ccxt.AuthenticationError:
            self.logger.error(
                f"Authentication failed for {self.exchange_name}. Check API keys."
            )
            return None
        except ccxt.ExchangeNotAvailable as e:
            self.logger.error(f"Exchange {self.exchange_name} is not available: {e}")
            return None
        except ccxt.RequestTimeout as e:
            self.logger.error(f"Request timeout on {self.exchange_name}: {e}")
            return None
        except Exception as e:
            self.logger.error(f"Error initializing {self.exchange_name} exchange: {e}")
            return None

    def _setup_logger(self):
        """Sets up logging for the trading bot."""
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.DEBUG)

        formatter = logging.Formatter(
            f"{NEON_CYAN}%(asctime)s - {NEON_YELLOW}%(levelname)s - {NEON_GREEN}%(message)s{RESET_COLOR}"
        )

        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        # File handler
        file_handler = logging.FileHandler(self.log_file_path)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        return logger

    def run_bot(self):
        """Main loop for the trading bot, implementing grid trading strategy."""
        if not self.exchange:
            self.logger.error("Exchange not initialized. Bot cannot start.")
            return

        self.running = True
        self.logger.info(
            f"{NEON_GREEN}Trading bot started for symbol: {self.symbol}{RESET_COLOR}"
        )
        self.initialize_grid()

        while self.running:
            try:
                ticker = self.exchange.fetch_ticker(self.symbol)
                current_price = ticker["last"]

                if self.dynamic_grid_adjustment:  # Improvement 1: Dynamic Grid Adjustment
                    atr_value = self.calculate_atr(
                        self.symbol, self.atr_period
                    )  # Improvement 1: ATR Calculation
                    if atr_value:  # Ensure ATR calculation is successful
                        self.adjust_grid_if_needed(
                            current_price, atr_value
                        )  # Pass ATR to adjust_grid_if_needed
                    else:
                        self.logger.warning(
                            "ATR calculation failed, using default grid adjustment."
                        )
                        self.adjust_grid_if_needed(
                            current_price
                        )  # Fallback to default if ATR fails
                else:
                    self.adjust_grid_if_needed(
                        current_price
                    )  # Default grid adjustment if dynamic is off

                if self.use_rsi_filter:  # Improvement 6: RSI Filter
                    rsi_value = self.calculate_rsi(
                        self.symbol, self.rsi_period
                    )  # Improvement 6: RSI Calculation
                    if (
                        rsi_value and rsi_value <= self.rsi_oversold
                    ):  # Check RSI condition
                        self.logger.info(
                            f"RSI ({rsi_value:.2f}) is oversold (<= {self.rsi_oversold}), "
                            "executing grid orders."
                        )
                        self.execute_grid_orders(current_price)
                    else:
                        self.logger.debug(
                            f"RSI ({rsi_value:.2f}) is not oversold (>{self.rsi_oversold}), "
                            "skipping grid order execution."
                        )
                else:
                    self.execute_grid_orders(
                        current_price
                    )  # Execute grid orders without RSI filter

                self.check_and_update_take_profit_stop_loss(
                    current_price
                )  # Improvement 4: Check and update TP/SL

                self.update_account_data_in_log()

                time.sleep(10)

            except ccxt.NetworkError as e:
                self.logger.error(f"Network error occurred: {e}")
                time.sleep(30)
            except ccxt.ExchangeError as e:
                self.logger.error(f"Exchange error occurred: {e}")
                time.sleep(30)
            except Exception as e:
                self.logger.error(f"Unexpected error in bot loop: {e}")
                self.logger.exception(e)
                self.stop_bot_loop()
                break

        self.logger.info(
            f"{NEON_YELLOW}Trading bot stopped for symbol: {self.symbol}{RESET_COLOR}"
        )

    def stop_bot_loop(self):
        """Sets the flag to stop the main bot loop and cancels all open orders."""
        self.running = False
        self.logger.info("Stopping bot loop signaled, canceling all open orders...")
        self.cancel_all_open_orders()  # Improvement 3: Cancel Orders
        self.logger.info("Bot loop stopped and open orders cancelled.")

    def cancel_all_open_orders(self):
        """Cancels all open orders for the trading symbol."""
        try:
            if self.exchange:
                orders = self.fetch_open_orders()
                if orders:
                    self.logger.info(f"Cancelling {len(orders)} open orders...")
                    for order in orders:
                        self.exchange.cancel_order(
                            order["id"], self.symbol
                        )  # Cancel each order
                    self.logger.info("All open orders cancelled.")
                else:
                    self.logger.info("No open orders to cancel.")
            else:
                self.logger.warning("Exchange not initialized, cannot cancel orders.")
        except Exception as e:
            self.logger.error(f"Error cancelling open orders: {e}")

    def fetch_account_balance(self):
        """Fetches and returns account balance."""
        try:
            if self.exchange:
                balance = self.exchange.fetch_balance()
                if balance and "free" in balance:
                    return balance["free"]["USDT"] if "USDT" in balance["free"] else balance["free"]  # Improvement 5: USDT Balance Specific
                else:
                    self.logger.warning(
                        "Could not retrieve free balance from exchange response."
                    )
                    return None
            else:
                self.logger.warning("Exchange not initialized.")
                return None
        except Exception as e:
            self.logger.error(f"Error fetching account balance: {e}")
            return None

    def fetch_open_orders(self):
        """Fetches and returns open orders."""
        try:
            if self.exchange:
                orders = self.exchange.fetch_open_orders(self.symbol)
                return orders
            else:
                self.logger.warning("Exchange not initialized.")
                return []
        except Exception as e:
            self.logger.error(f"Error fetching open orders: {e}")
            return []

    def fetch_position_pnl(self):
        """Fetches and returns position PnL (Profit and Loss)."""
        try:
            if self.exchange and hasattr(self.exchange, "fetch_positions"):
                positions = self.exchange.fetch_positions([self.symbol])
                if positions:
                    position = next(
                        (p for p in positions if p["symbol"] == self.symbol), None
                    )
                    if position and position["unrealizedPnl"] is not None:
                        return position["unrealizedPnl"]
                    else:
                        return 0.0
                else:
                    return 0.0
            else:
                if not self.exchange:
                    self.logger.warning("Exchange not initialized.")
                elif not hasattr(self.exchange, "fetch_positions"):
                    self.logger.warning(
                        f"Exchange {self.exchange_name} does not support fetch_positions endpoint."
                    )
                return None
        except Exception as e:
            self.logger.error(f"Error fetching position PnL: {e}")
            return None

    def update_account_data_in_log(self):
        """Periodically fetches and logs account balance, open orders, and PnL."""
        if time.time() % 600 < 10:
            balance = self.fetch_account_balance()
            pnl = self.fetch_position_pnl()
            orders = self.fetch_open_orders()
            self.logger.info(f"\n--- Account Data Update ---")
            self.logger.info(f"Balance: {balance} USDT")
            self.logger.info(f"Position PnL: {pnl}%")
            if orders:
                self.logger.info("Open Orders:")
                for order in orders:
                    self.logger.info(
                        "  - %s %s %s @ %s (ID: %s)",
                        order["side"],
                        order["amount"],
                        order["symbol"],
                        order["price"] or "Market Price",
                        order["id"],  # Improvement 9: Log Order ID
                    )
            else:
                self.logger.info("No Open Orders.")
            self.logger.info("--- End Account Data Update ---\n")

    def initialize_grid(self):
        """Initializes the trading grid based on current market price."""
        if not self.exchange:
            self.logger.error("Exchange not initialized, cannot initialize grid.")
            return

        try:
            ticker = self.exchange.fetch_ticker(self.symbol)
            current_price = ticker["last"]
            if self.dynamic_grid_adjustment:  # Improvement 1: Dynamic Grid Spacing
                atr_value = self.calculate_atr(
                    self.symbol, self.atr_period
                )  # Improvement 1: ATR Calculation
                if atr_value:
                    grid_step = (
                        atr_value * self.atr_deviation_multiplier
                    )  # Grid step based on ATR
                    self.logger.info(
                        f"Using ATR ({atr_value:.4f}) for dynamic grid spacing. "
                        f"Grid step: {grid_step:.4f}"
                    )
                else:
                    grid_step = (
                        current_price * self.grid_multiplier
                    )  # Fallback to % multiplier if ATR fails
                    self.logger.warning(
                        "ATR calculation failed, falling back to percentage-based grid spacing."
                    )
            else:
                grid_step = (
                    current_price * self.grid_multiplier
                )  # Default %-based grid spacing

            self.current_grid = []
            for i in range(self.grid_levels):
                grid_price = current_price - (grid_step * i)
                self.current_grid.append(
                    {
                        "price": grid_price,
                        "order_id": None,
                        "type": "buy",
                        "amount": self.base_order_size,  # Improvement 10: Store Amount
                    }
                )

            self.logger.info(f"Trading grid initialized around price: {current_price}")
            self.log_grid_status()

        except Exception as e:
            self.logger.error(f"Error initializing trading grid: {e}")

    def adjust_grid_if_needed(self, current_price, atr_value=None):
        """Adjusts the grid if current price moves significantly."""
        if not self.current_grid:
            self.logger.warning("Trading grid not yet initialized.")
            return

        grid_center_price = self.current_grid[0]["price"]
        price_drift_percentage = abs(current_price - grid_center_price) / grid_center_price

        if price_drift_percentage > 0.05:
            self.logger.info(
                f"Price drifted significantly ({price_drift_percentage*100:.2f}%). "
                "Re-adjusting grid."
            )
            self.initialize_grid()  # Re-initialize grid around the new price
        else:
            self.logger.debug(
                "Price within acceptable range, grid adjustment not needed."
            )

    def calculate_atr(self, symbol, period=14):
        """Calculates Average True Range (ATR) for dynamic grid adjustment."""
        try:
            if self.exchange and hasattr(self.exchange, "fetch_ohlcv"):
                ohlcv = self.exchange.fetch_ohlcv(
                    symbol, timeframe="1h", limit=period + 1
                )  # Fetch OHLCV data for ATR
                if len(ohlcv) < period + 1:
                    self.logger.warning(
                        f"Insufficient OHLCV data to calculate ATR for {symbol}."
                    )
                    return None

                true_range = []
                for i in range(1, len(ohlcv)):
                    high_low = ohlcv[i][2] - ohlcv[i][3]  # high - low
                    high_close_prev = abs(ohlcv[i][2] - ohlcv[i - 1][4])  # abs(high - close_prev)
                    low_close_prev = abs(ohlcv[i][3] - ohlcv[i - 1][4])  # abs(low - close_prev)
                    tr = max(high_low, high_close_prev, low_close_prev)  # True Range
                    true_range.append(tr)

                atr = sum(true_range) / period  # Average True Range
                return atr
            else:
                self.logger.warning(
                    f"Exchange {self.exchange_name} does not support fetch_ohlcv "
                    "endpoint or exchange not initialized for ATR calculation."
                )
                return None
        except Exception as e:
            self.logger.error(f"Error calculating ATR for {symbol}: {e}")
            return None

    def calculate_rsi(self, symbol, period=14):
        """Calculates Relative Strength Index (RSI) for entry filter."""
        try:
            if self.exchange and hasattr(self.exchange, "fetch_ohlcv"):
                ohlcv = self.exchange.fetch_ohlcv(
                    symbol, timeframe="1h", limit=period + 1
                )  # Fetch OHLCV data for RSI
                if len(ohlcv) < period + 1:
                    self.logger.warning(
                        f"Insufficient OHLCV data to calculate RSI for {symbol}."
                    )
                    return None

                prices = [candle[4] for candle in ohlcv[1:]]  # Closing prices
                if not prices:
                    self.logger.warning(
                        f"No price data available to calculate RSI for {symbol}."
                    )
                    return None

                diffs = [
                    prices[i] - prices[i - 1] for i in range(1, len(prices))
                ]  # Price differences
                if not diffs:
                    self.logger.warning(
                        f"No price differences to calculate RSI for {symbol}."
                    )
                    return None

                avg_gain = sum(d for d in diffs if d > 0) / period  # Average gain
                avg_loss = abs(
                    sum(d for d in diffs if d < 0) / period
                )  # Average loss (absolute value)

                if avg_loss == 0:  # Avoid division by zero
                    return 100  # RSI is 100 if no losses

                rs = avg_gain / avg_loss  # Relative Strength
                rsi = 100 - (100 / (1 + rs))  # Relative Strength Index
                return rsi
            else:
                self.logger.warning(
                    f"Exchange {self.exchange_name} does not support fetch_ohlcv "
                    "endpoint or exchange not initialized for RSI calculation."
                )
                return None
        except Exception as e:
            self.logger.error(f"Error calculating RSI for {symbol}: {e}")
            return None

    def execute_grid_orders(self, current_price):
        """Executes trading logic based on current price and trading grid."""
        if not self.current_grid:
            self.logger.warning("Trading grid not initialized, cannot execute orders.")
            return

        for level in self.current_grid:
            if level["type"] == "buy" and current_price <= level["price"]:
                if not level["order_id"]:
                    order = self.place_limit_buy_order(
                        level["price"], level["amount"]
                    )  # Use amount from grid level
                    if order:  # Improvement 3: Order placement check
                        level["order_id"] = order["id"]  # Improvement 3: Store order ID
                        level["type"] = "sell"
                        self.active_safety_orders_count = 0  # Reset safety order count
                        self.safety_order_prices = []  # Clear safety order prices
                        self.log_grid_status()
                    else:
                        self.logger.error(
                            f"Failed to place buy order at price {level['price']}. "
                            "Check logs."
                        )

            elif level["type"] == "sell" and current_price >= level["price"]:
                if not level["order_id"]:
                    order = self.place_limit_sell_order(
                        level["price"], level["amount"]
                    )  # Use amount from grid level
                    if order:  # Improvement 3: Order placement check
                        level["order_id"] = order["id"]  # Improvement 3: Store order ID
                        level["type"] = "buy"
                        self.adjust_take_profit_and_stop_loss(
                            current_price
                        )  # Improvement 4: TP/SL Adjustment
                        self.log_grid_status()
                    else:
                        self.logger.error(
                            f"Failed to place sell order at price {level['price']}. "
                            "Check logs."
                        )
            elif (
                level["type"] == "buy"
                and current_price > level["price"]
                and self.active_safety_orders_count < self.max_safety_orders
                and level["price"] not in self.safety_order_prices
                and abs(current_price - level["price"]) / current_price
                >= self.price_deviation_to_open_safety_order
            ):  # Improvement 2: Safety Order Logic
                if not level["order_id"]:
                    safety_order_amount = self.safety_order_size * (
                        self.safety_order_volume_scale ** self.active_safety_orders_count
                    )  # Scale safety order volume
                    order = self.place_limit_buy_order(
                        level["price"], safety_order_amount
                    )  # Place safety buy order
                    if order:  # Improvement 3: Order placement check
                        level["order_id"] = order["id"]  # Improvement 3: Store order ID
                        level["type"] = "sell"  # Type becomes sell after buy
                        self.active_safety_orders_count += 1  # Increment safety order count
                        self.safety_order_prices.append(
                            level["price"]
                        )  # Track safety order price
                        self.log_grid_status()
                    else:
                        self.logger.error(
                            f"Failed to place SAFETY buy order at price {level['price']}. "
                            "Check logs."
                        )

    def place_limit_buy_order(self, price, amount):
        """Places a limit buy order at the specified price."""
        try:
            if self.exchange:
                balance = self.fetch_account_balance()  # Improvement 5: Balance Check
                if (
                    balance is not None
                    and balance > price * amount * self.max_capital_usage_percentage
                ):  # Improvement 5: Balance Check
                    order = self.exchange.create_limit_buy_order(
                        self.symbol, amount, price
                    )
                    self.logger.info(
                        "Limit Buy Order placed: price=%s, amount=%s, order_id=%s",
                        price,
                        amount,
                        order["id"],  # Improvement 9: Log Order ID
                    )
                    return order
                else:
                    self.logger.warning(
                        "Insufficient balance to place buy order at price %s, amount %s."
                        " Available balance: %s USDT.",
                        price,
                        amount,
                        balance,
                    )  # Improvement 5: Balance Warning
                    return None  # Indicate order not placed due to balance
            else:
                self.logger.warning("Exchange not initialized, cannot place buy order.")
                return None
        except Exception as e:
            self.logger.error(f"Error placing limit buy order at price {price}: {e}")
            return None

    def place_limit_sell_order(self, price, amount):
        """Places a limit sell order at the specified price."""
        try:
            if self.exchange:
                order = self.exchange.create_limit_sell_order(
                    self.symbol, amount, price
                )
                self.logger.info(
                    "Limit Sell Order placed: price=%s, amount=%s, order_id=%s",
                    price,
                    amount,
                    order["id"],  # Improvement 9: Log Order ID
                )
                return order
            else:
                self.logger.warning("Exchange not initialized, cannot place sell order.")
                return None
        except Exception as e:
            self.logger.error(f"Error placing limit sell order at price {price}: {e}")
            return None

    def adjust_take_profit_and_stop_loss(self, current_price):
        """Adjusts take profit and stop loss dynamically based on price."""
        if not self.current_grid or not self.current_grid[0]:
            self.logger.warning("Trading grid not initialized, cannot adjust TP/SL.")
            return

        initial_buy_price = self.current_grid[0]["price"]  # Top grid level is initial buy
        take_profit_price = initial_buy_price * (
            1 + self.take_profit_percentage
        )  # TP based on initial buy
        stop_loss_price = initial_buy_price * (
            1 - self.stop_loss_percentage
        )  # SL based on initial buy

        # Improvement 4: Trailing Stop Loss (Example - Simple)
        if current_price > initial_buy_price * (
            1 + self.trailing_stop_loss_active_profit_percentage
        ):  # Activate trailing SL after profit
            stop_loss_price = max(
                stop_loss_price, current_price * (1 - self.stop_loss_percentage)
            )  # Raise SL

        self.logger.info(
            f"Adjusting Take Profit to {take_profit_price:.2f}, "
            f"Stop Loss to {stop_loss_price:.2f}"
        )
        # Implement actual TP/SL order placement or update logic if exchange supports it

    def get_grid_status(self):
        """Returns the current status of the trading grid for monitoring."""
        status = []
        for level in self.current_grid:
            status.append(
                {
                    "price": level["price"],
                    "type": level["type"],
                    "order_status": "Open" if level["order_id"] else "Inactive",  # Improved
                    "order_id": level["order_id"],  # Improvement 3: Include Order ID
                    "amount": level["amount"],  # Improvement 10: Include Amount
                }
            )
        return status

    def log_grid_status(self):
        """Logs the current status of the trading grid."""
        grid_status = self.get_grid_status()
        self.logger.info("\n--- Trading Grid Status ---")
        for level_status in grid_status:
            self.logger.info(
                "  Price: %s, Type: %s, Status: %s, Order ID: %s, Amount: %s",
                level_status["price"],
                level_status["type"],
                level_status["order_status"],
                level_status["order_id"],  # Improvement 9 & 10: Log Order ID and Amount
                level_status["amount"],
            )
        self.logger.info(
            f"Active Safety Orders Count: {self.active_safety_orders_count}"
        )  # Improvement 2: Log Safety Order Count
        self.logger.info("--- End Grid Status ---\n")


if __name__ == "__main__":
    # Example usage if you want to run the bot directly from trading_bot.py for testing
    bot = EnhancedTradingBot(
        symbol="BTC/USDT", dynamic_grid_adjustment=True, use_rsi_filter=False
    )  # Example symbol, dynamic grid enabled, RSI filter disabled
    if bot.exchange:
        bot.run_bot()
```
