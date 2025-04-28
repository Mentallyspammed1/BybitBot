import os
import sys
import time
import logging
import pandas as pd
import numpy as np
from pybit.unified_trading import HTTP

# --- Configuration ---
# API Credentials (Fetch from environment variables for security)
API_KEY = os.environ.get("BYBIT_API_KEY")
API_SECRET = os.environ.get("BYBIT_API_SECRET")
# Use testnet? (Set environment variable BYBIT_TESTNET=true or false)
USE_TESTNET = os.environ.get("BYBIT_TESTNET", 'false').lower() == 'true'

# Trading Parameters
SYMBOL = "BTCUSDT"          # Symbol to trade (e.g., BTCUSDT, ETHUSDT)
TIMEFRAME = "15"            # Kline interval (e.g., "1", "5", "15", "60", "240", "D")
LEVERAGE = 5                # Desired leverage (e.g., 5x, 10x) - Ensure your account supports this

# Strategy Parameters
PRICE_SMA_PERIOD = 20       # Period for Price Simple Moving Average
VOLUME_SMA_PERIOD = 20      # Period for Volume Simple Moving Average
KLINE_LIMIT = PRICE_SMA_PERIOD + 5 # Number of candles to fetch (needs enough for longest MA)

# Risk Management Parameters
RISK_PER_TRADE_PERCENT = 1.0  # Risk 1% of available balance per trade
STOP_LOSS_PERCENT = 1.0     # Stop loss percentage from entry price (e.g., 1.0 means 1%)
RISK_REWARD_RATIO = 1.5     # Desired Risk/Reward ratio (TP distance = SL distance * R/R)
TAKE_PROFIT_PERCENT = STOP_LOSS_PERCENT * RISK_REWARD_RATIO

# Bot Operation Parameters
SLEEP_INTERVAL_SECONDS = 60 # Check for signals every 60 seconds

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("trading_bot.log"),
        logging.StreamHandler(sys.stdout) # Also print logs to console
    ]
)
logger = logging.getLogger("VolumaticTrader")

# --- Helper Functions ---
def calculate_indicators(df, price_sma_period, volume_sma_period):
    """Calculates SMAs for price and volume."""
    if df.empty or len(df) < max(price_sma_period, volume_sma_period):
        logger.warning("Not enough data to calculate indicators.")
        return df
    df['price_sma'] = df['close'].rolling(window=price_sma_period).mean()
    df['volume_sma'] = df['volume'].rolling(window=volume_sma_period).mean()
    return df

def get_available_balance(session, coin="USDT"):
    """Gets the available USDT balance for derivatives."""
    try:
        # For Unified Trading Account (UTA)
        balance_info = session.get_wallet_balance(accountType="UNIFIED", coin=coin)
        # Find the specific coin balance details
        if balance_info and balance_info.get('retCode') == 0:
            coin_balance = next((item for item in balance_info['result']['list'][0]['coin'] if item['coin'] == coin), None)
            if coin_balance:
                  # Use availableToWithdraw or walletBalance depending on your needs/account setup
                  # availableToWithdraw considers margin requirements, walletBalance is total
                  # Using walletBalance for risk calculation base might be simpler
                  # Adjust if needed based on Bybit API response structure
                  return float(coin_balance.get('availableToWithdraw', coin_balance.get('walletBalance', 0))) # Prefer available, fallback to wallet
            else:
                logger.warning(f"{coin} balance not found in UNIFIED account.")
                return 0.0
        else:
            logger.error(f"Error fetching wallet balance: {balance_info.get('retMsg') if balance_info else 'No response'}")
            # Fallback attempt for non-UTA or different structure (adjust if necessary)
            # balance_info = session.get_wallet_balance(accountType="CONTRACT", coin=coin) # Example
            return 0.0

    except Exception as e:
        logger.error(f"Exception getting wallet balance: {e}")
        return 0.0

def calculate_position_size(balance, risk_percent, sl_percent, entry_price, leverage, min_order_qty):
    """Calculates position size based on risk parameters."""
    if entry_price <= 0 or sl_percent <= 0:
        return 0

    # Calculate risk amount in USDT
    risk_amount_usdt = balance * (risk_percent / 100.0)

    # Calculate how much the price needs to move for the SL to trigger (in USDT)
    sl_distance_usdt_per_unit = entry_price * (sl_percent / 100.0)

    # Calculate position size in base currency units (e.g., BTC) BEFORE leverage
    # position_size_units = risk_amount_usdt / sl_distance_usdt_per_unit

    # More direct calculation considering leverage impact on required margin vs loss:
    # The actual loss is position_size_units * sl_distance_usdt_per_unit
    # We want this loss to equal risk_amount_usdt
    # So, position_size_units = risk_amount_usdt / sl_distance_usdt_per_unit

    # However, Bybit order quantity is usually in the base currency (e.g., BTC amount).
    # Let Q be the quantity in base currency (e.g. BTC)
    # Position Value = Q * entry_price
    # Loss = Q * (entry_price * sl_percent / 100)
    # We want Loss = risk_amount_usdt
    # Q * entry_price * sl_percent / 100 = risk_amount_usdt
    # Q = (risk_amount_usdt * 100) / (entry_price * sl_percent)

    position_size_units = (risk_amount_usdt * 100.0) / (entry_price * sl_percent)


    # Ensure size meets minimum order quantity requirement
    if position_size_units < min_order_qty:
         logger.warning(f"Calculated size {position_size_units:.8f} is below minimum {min_order_qty:.8f}. Adjusting to minimum.")
         position_size_units = min_order_qty
         # Optional: Recalculate risk if forced to use min size, or just proceed cautiously
         actual_risk_amount = position_size_units * entry_price * sl_percent / 100.0
         logger.info(f"Using min order qty. Actual USDT risk approx: {actual_risk_amount:.2f}")


    # Optional: Add check for maximum order size if needed
    # max_order_qty = ...
    # if position_size_units > max_order_qty:
    #     position_size_units = max_order_qty
    #     logger.warning(f"Calculated size exceeds maximum. Capping at {max_order_qty}.")

    # Bybit requires quantity precision, round appropriately (needs instrument info)
    # Assuming BTC precision of 0.001 for example - MUST CHECK FOR ACTUAL SYMBOL
    # Fetching instrument info is better, but hardcoding for simplicity here
    precision = 3 # Example for BTCUSDT qty precision
    position_size_units = round(position_size_units, precision)

    # Final check if rounded size is still above minimum
    if position_size_units < min_order_qty:
        logger.error(f"Rounded size {position_size_units} is still below minimum {min_order_qty}. Cannot place trade.")
        return 0

    logger.info(f"Calculated position size: {position_size_units} {SYMBOL[:-4]} based on Balance=${balance:.2f}, Risk={risk_percent}%, SL={sl_percent}%, Entry=${entry_price:.2f}")
    return position_size_units


# --- Main Trading Class ---
class VolumaticTrader:
    def __init__(self, api_key, api_secret, symbol, timeframe, leverage, use_testnet=False):
        self.symbol = symbol
        self.timeframe = timeframe
        self.leverage = leverage
        self.min_order_qty = 0.0 # Will be fetched later

        try:
            self.session = HTTP(
                testnet=use_testnet,
                api_key=api_key,
                api_secret=api_secret,
            )
            logger.info(f"Bybit session initialized. Testnet: {use_testnet}")
            self._initial_setup()
            self._fetch_instrument_info() # Fetch min order size etc.
        except Exception as e:
            logger.error(f"Failed to initialize Bybit session: {e}")
            sys.exit(1) # Exit if connection fails critically

    def _initial_setup(self):
        """Set leverage and margin mode (if needed)."""
        try:
            # --- Set Leverage ---
            # Category needs to be 'linear' for USDT perpetuals
            self.session.set_leverage(
                category="linear",
                symbol=self.symbol,
                buyLeverage=str(self.leverage),
                sellLeverage=str(self.leverage),
            )
            logger.info(f"Leverage for {self.symbol} set to {self.leverage}x")

            # --- Set Margin Mode (Optional but Recommended: ISOLATED) ---
            # Check current mode first if needed, otherwise just set it
            # 0: cross margin, 1: isolated margin
            # Note: Switching might fail if positions/orders exist. Handle this in real bot.
            try:
                 current_pos = self.get_position()
                 if not current_pos or float(current_pos.get('size', 0)) == 0:
                      self.session.switch_margin_mode(
                          category="linear",
                          symbol=self.symbol,
                          tradeMode=1 # 1 for Isolated
                      )
                      logger.info(f"Margin mode for {self.symbol} set to ISOLATED.")
                 else:
                      logger.warning("Cannot switch margin mode while position is open. Keeping current mode.")

            except Exception as e:
                 logger.warning(f"Could not set margin mode (may already be set or other issue): {e}")


        except Exception as e:
            logger.error(f"Error during initial setup (leverage/margin): {e}")
            # Depending on severity, you might want to exit or just log warning

    def _fetch_instrument_info(self):
       """Fetches instrument details like min order size and qty step."""
       try:
           info = self.session.get_instruments_info(category="linear", symbol=self.symbol)
           if info and info['retCode'] == 0 and info['result']['list']:
               instrument = info['result']['list'][0]
               self.min_order_qty = float(instrument['lotSizeFilter']['minOrderQty'])
               # Determine qty precision from qtyStep
               qty_step_str = instrument['lotSizeFilter']['qtyStep']
               if '.' in qty_step_str:
                   self.qty_precision = len(qty_step_str.split('.')[-1])
               else:
                   self.qty_precision = 0 # Whole numbers

               logger.info(f"Instrument info for {self.symbol}: Min Qty={self.min_order_qty}, Qty Precision={self.qty_precision}")
           else:
               logger.error(f"Could not fetch instrument info: {info.get('retMsg') if info else 'No response'}. Using default minimum 0.001.")
               # Fallback defaults - adjust if needed!
               self.min_order_qty = 0.001
               self.qty_precision = 3

       except Exception as e:
           logger.error(f"Exception fetching instrument info: {e}. Using default minimum 0.001 / precision 3.")
           self.min_order_qty = 0.001
           self.qty_precision = 3


    def get_ohlcv(self, limit=KLINE_LIMIT):
        """Fetches OHLCV data and calculates indicators."""
        try:
            # Category 'linear' for USDT perpetuals
            response = self.session.get_kline(
                category="linear",
                symbol=self.symbol,
                interval=self.timeframe,
                limit=limit
            )

            if response and response['retCode'] == 0:
                data = response['result']['list']
                if not data:
                    logger.warning("No kline data received.")
                    return None

                # Bybit returns data oldest first, reverse it and name columns
                df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'turnover'])
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                df.set_index('timestamp', inplace=True)
                df = df.iloc[::-1].copy() # Reverse to have newest data last

                # Convert columns to numeric
                for col in ['open', 'high', 'low', 'close', 'volume']:
                    df[col] = pd.to_numeric(df[col])

                # Calculate indicators
                df = calculate_indicators(df, PRICE_SMA_PERIOD, VOLUME_SMA_PERIOD)
                return df
            else:
                logger.error(f"Error fetching kline data: {response.get('retMsg') if response else 'No response'}")
                return None
        except Exception as e:
            logger.error(f"Exception fetching kline data: {e}")
            return None

    def get_position(self):
        """Gets current position information for the symbol."""
        try:
            # Category 'linear' for USDT perpetuals
            response = self.session.get_positions(
                category="linear",
                symbol=self.symbol,
            )
            if response and response['retCode'] == 0 and response['result']['list']:
                # Bybit API v5 returns a list, usually the first item is the relevant one for a symbol
                position_data = response['result']['list'][0]
                # Check if size is non-zero to determine if a position exists
                if float(position_data.get('size', 0)) != 0:
                     logger.debug(f"Current position data: {position_data}")
                     return position_data
                else:
                     logger.debug("No active position found.")
                     return None # No position open
            elif response and response['retCode'] == 0:
                 logger.debug("Position list is empty.")
                 return None
            else:
                logger.error(f"Error fetching position data: {response.get('retMsg') if response else 'No response'}")
                return None
        except Exception as e:
            logger.error(f"Exception fetching position data: {e}")
            return None

    def place_order(self, side, qty, stop_loss_price, take_profit_price):
        """Places a market order with SL and TP."""
        try:
            logger.info(f"Attempting to place {side} order for {qty} {self.symbol}...")
            response = self.session.place_order(
                category="linear",
                symbol=self.symbol,
                side=side,  # "Buy" or "Sell"
                orderType="Market",
                qty=str(qty),
                stopLoss=f"{stop_loss_price:.{self._get_price_precision()}f}", # Format price to required precision
                takeProfit=f"{take_profit_price:.{self._get_price_precision()}f}",
                slTriggerBy="LastPrice", # Or MarkPrice, IndexPrice
                tpTriggerBy="LastPrice", # Or MarkPrice, IndexPrice
                # timeInForce="GoodTillCancel", # GTC is default for Market orders usually
                # reduce_only=False # Ensure this is false for entry orders
            )
            if response and response['retCode'] == 0:
                order_id = response['result'].get('orderId')
                logger.info(f"Market {side} order placed successfully! Order ID: {order_id}. Qty: {qty}, SL: {stop_loss_price}, TP: {take_profit_price}")
                return order_id
            else:
                logger.error(f"Failed to place {side} order: {response.get('retMsg') if response else 'No response'}")
                return None
        except Exception as e:
            logger.error(f"Exception placing {side} order: {e}")
            return None

    def close_position(self, position_data):
        """Closes the current position with a market order."""
        if not position_data:
            logger.warning("Attempted to close position, but no position data provided.")
            return False

        side = position_data.get('side') # "Buy" or "Sell" (representing the existing position)
        size = float(position_data.get('size', 0))

        if size == 0:
            logger.warning("Attempted to close position, but size is zero.")
            return False

        # To close a position, place an order on the opposite side
        close_side = "Sell" if side == "Buy" else "Buy"
        logger.info(f"Attempting to close {side} position of size {size} with a {close_side} market order.")

        try:
            response = self.session.place_order(
                category="linear",
                symbol=self.symbol,
                side=close_side,
                orderType="Market",
                qty=str(size),
                reduce_only=True # IMPORTANT: Ensures this order only closes a position
            )
            if response and response['retCode'] == 0:
                order_id = response['result'].get('orderId')
                logger.info(f"Position close order placed successfully! Order ID: {order_id}. Side: {close_side}, Qty: {size}")
                return True
            else:
                logger.error(f"Failed to place position close order: {response.get('retMsg') if response else 'No response'}")
                return False
        except Exception as e:
            logger.error(f"Exception closing position: {e}")
            return False

    def _get_price_precision(self):
        """Helper to get price precision based on symbol (basic)."""
        # This should ideally be fetched from instrument info
        # Hardcoding common precisions as fallback
        if "BTC" in self.symbol: return 1
        if "ETH" in self.symbol: return 2
        return 4 # Default guess

    def run(self):
        """Main trading loop."""
        logger.info("Starting trading loop...")
        while True:
            try:
                # 1. Get Data and Calculate Indicators
                df = self.get_ohlcv()
                if df is None or df.empty or 'price_sma' not in df.columns or df['price_sma'].isna().all():
                    logger.warning("Could not get valid data or indicators. Waiting...")
                    time.sleep(SLEEP_INTERVAL_SECONDS)
                    continue

                # Get the most recent *closed* candle data [-2] and the one before [-3] for crossover check
                # Use [-1] for current (incomplete) candle's price if needed for faster checks, but signals based on closed candles [-2] are safer
                last_closed = df.iloc[-2]
                prev_closed = df.iloc[-3]
                # current_price = df.iloc[-1]['close'] # More up-to-date price for calcs if needed


                # 2. Check Current Position and Balance
                position = self.get_position()
                balance = get_available_balance(self.session, "USDT")
                if balance <= 0:
                    logger.error("Available balance is zero or could not be fetched. Stopping.")
                    break # Stop if no balance

                # 3. Check Exit Conditions (if position exists)
                if position:
                    pos_side = position.get('side') # 'Buy' or 'Sell'
                    pos_size = float(position.get('size', 0))
                    entry_price = float(position.get('avgPrice', 0))

                    logger.info(f"Holding {pos_side} position. Size: {pos_size}, Entry: {entry_price:.{self._get_price_precision()}f}")

                    # Exit Condition 1: Price crosses back over SMA against the position
                    exit_signal = False
                    if pos_side == "Buy" and last_closed['close'] < last_closed['price_sma']:
                        logger.info(f"Exit Signal: Price ({last_closed['close']}) crossed below SMA ({last_closed['price_sma']:.{self._get_price_precision()}f}). Closing Long.")
                        exit_signal = True
                    elif pos_side == "Sell" and last_closed['close'] > last_closed['price_sma']:
                        logger.info(f"Exit Signal: Price ({last_closed['close']}) crossed above SMA ({last_closed['price_sma']:.{self._get_price_precision()}f}). Closing Short.")
                        exit_signal = True

                    if exit_signal:
                        # Close position via market order (SL/TP set previously might still exist, closing manually overrides)
                        # It might be better to *only* rely on SL/TP set during entry and remove this SMA cross exit.
                        # Keeping it here as requested for an explicit exit signal besides SL/TP.
                        if self.close_position(position):
                             logger.info("Position closed based on SMA cross signal.")
                        else:
                             logger.error("Failed to close position based on SMA cross signal.")
                        # Wait before potentially re-entering on the next loop
                        time.sleep(SLEEP_INTERVAL_SECONDS)
                        continue # Skip entry logic for this cycle

                    # Exit Condition 2: SL/TP (Handled by Bybit server-side order placed at entry)
                    # No explicit check needed here if SL/TP were set correctly with the entry order.
                    # The `get_position` call will return None if SL/TP was hit.

                # 4. Check Entry Conditions (if no position exists)
                else: # No position currently open
                    logger.info("No position open. Checking for entry signals...")
                    entry_price_approx = last_closed['close'] # Use last close as approximate entry

                    # Long Entry Signal:
                    # Price crossed *above* SMA on the last closed candle
                    # Volume on that candle was above Volume SMA
                    long_signal = (prev_closed['close'] < prev_closed['price_sma'] and
                                   last_closed['close'] > last_closed['price_sma'] and
                                   last_closed['volume'] > last_closed['volume_sma'])

                    # Short Entry Signal:
                    # Price crossed *below* SMA on the last closed candle
                    # Volume on that candle was above Volume SMA
                    short_signal = (prev_closed['close'] > prev_closed['price_sma'] and
                                    last_closed['close'] < last_closed['price_sma'] and
                                    last_closed['volume'] > last_closed['volume_sma'])

                    if long_signal:
                        logger.info(f"Long Entry Signal detected at ~${entry_price_approx:.{self._get_price_precision()}f}")
                        # Calculate SL/TP
                        sl_price = entry_price_approx * (1 - STOP_LOSS_PERCENT / 100.0)
                        tp_price = entry_price_approx * (1 + TAKE_PROFIT_PERCENT / 100.0)
                        # Calculate position size
                        qty = calculate_position_size(balance, RISK_PER_TRADE_PERCENT, STOP_LOSS_PERCENT, entry_price_approx, self.leverage, self.min_order_qty)

                        if qty > 0:
                             # Place Buy order
                             self.place_order("Buy", qty, sl_price, tp_price)
                        else:
                             logger.warning("Could not enter long trade due to zero quantity calculation.")


                    elif short_signal:
                        logger.info(f"Short Entry Signal detected at ~${entry_price_approx:.{self._get_price_precision()}f}")
                        # Calculate SL/TP
                        sl_price = entry_price_approx * (1 + STOP_LOSS_PERCENT / 100.0)
                        tp_price = entry_price_approx * (1 - TAKE_PROFIT_PERCENT / 100.0)
                        # Calculate position size
                        qty = calculate_position_size(balance, RISK_PER_TRADE_PERCENT, STOP_LOSS_PERCENT, entry_price_approx, self.leverage, self.min_order_qty)

                        if qty > 0:
                             # Place Sell order
                             self.place_order("Sell", qty, sl_price, tp_price)
                        else:
                             logger.warning("Could not enter short trade due to zero quantity calculation.")

                    else:
                        logger.info("No entry signal detected.")

                # 5. Wait for the next interval
                logger.info(f"Waiting for {SLEEP_INTERVAL_SECONDS} seconds...")
                time.sleep(SLEEP_INTERVAL_SECONDS)

            except KeyboardInterrupt:
                logger.info("Keyboard interrupt detected. Stopping bot...")
                # Optional: Attempt to close any open position before exiting
                position = self.get_position()
                if position:
                    logger.info("Attempting to close open position on exit...")
                    self.close_position(position)
                break
            except Exception as e:
                logger.error(f"An unexpected error occurred in the main loop: {e}", exc_info=True)
                # Implement more robust error handling here (e.g., retry logic, specific error checks)
                logger.info(f"Pausing for {SLEEP_INTERVAL_SECONDS*2} seconds after error...")
                time.sleep(SLEEP_INTERVAL_SECONDS * 2) # Longer pause after error


# --- Script Execution ---
if __name__ == "__main__":
    if not API_KEY or not API_SECRET:
        logger.error("API Key or Secret not found in environment variables (BYBIT_API_KEY, BYBIT_API_SECRET). Exiting.")
        sys.exit(1)

    # Validate essential parameters
    if RISK_PER_TRADE_PERCENT <= 0 or STOP_LOSS_PERCENT <= 0 or RISK_REWARD_RATIO <= 0:
        logger.error("Risk management percentages/ratios must be positive. Exiting.")
        sys.exit(1)
    if LEVERAGE <= 0:
        logger.error("Leverage must be positive. Exiting.")
        sys.exit(1)


    trader = VolumaticTrader(
        api_key=API_KEY,
        api_secret=API_SECRET,
        symbol=SYMBOL,
        timeframe=TIMEFRAME,
        leverage=LEVERAGE,
        use_testnet=USE_TESTNET
    )

    # Ensure minimum order quantity was fetched before running
    if trader.min_order_qty == 0.0:
         logger.error("Failed to fetch minimum order quantity during initialization. Exiting.")
         sys.exit(1)

    trader.run()
    logger.info("Bot stopped.")
