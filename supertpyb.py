import os
import time
import logging
from dotenv import load_dotenv
import pandas as pd
import pandas_ta as ta
from pybit.unified_trading import HTTP
import schedule # Using schedule for cleaner loop timing

# --- Configuration ---
load_dotenv() # Load environment variables from .env file

API_KEY = os.getenv("BYBIT_API_KEY")
API_SECRET = os.getenv("BYBIT_API_SECRET")
USE_TESTNET = os.getenv("BYBIT_TESTNET", 'true').lower() == 'true' # Default to testnet if not set

# Trading Parameters
SYMBOL = "BTCUSDT" # Ensure this matches Bybit's symbol format (e.g., BTCUSDT for USDT perpetual)
TIMEFRAME = "15" # Bybit timeframe (1, 3, 5, 15, 30, 60, 120, 240, 360, 720, D, W, M)
CATEGORY = "linear" # Or "inverse" or "spot"

# Strategy Parameters
SUPERTREND_ATR_PERIOD = 10
SUPERTREND_MULTIPLIER = 3.0
ORDER_SIZE_USDT = 100 # Example: Fixed USDT value per trade (adjust based on your risk tolerance)
# --- OR ---
# ORDER_SIZE_PERCENT = 0.01 # Example: 1% of available balance (more complex to implement accurately)

# Risk Management Parameters
STOP_LOSS_PERCENT = 0.015 # 1.5% stop loss from entry price
TAKE_PROFIT_PERCENT = 0.03 # 3.0% take profit from entry price (2:1 R:R)
# --- OR --- Use Supertrend line itself as a trailing stop (more complex logic)

# Logging Configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Global variable to track position state
current_position = None # Can be None, 'LONG', 'SHORT'
last_signal = None # To detect trend changes

# --- Bybit Connection ---
try:
    session = HTTP(
        testnet=USE_TESTNET,
        api_key=API_KEY,
        api_secret=API_SECRET,
    )
    logging.info(f"Connected to Bybit {'Testnet' if USE_TESTNET else 'Mainnet'}")
except Exception as e:
    logging.error(f"Error connecting to Bybit: {e}")
    exit()

# --- Helper Functions ---

def get_balance(quote_currency="USDT"):
    """Gets the available balance for the specified currency."""
    try:
        balance_info = session.get_wallet_balance(accountType="UNIFIED", coin=quote_currency) # Or CONTRACT for older accounts
        if balance_info and balance_info['retCode'] == 0:
            # Find the correct balance entry (handling unified vs contract differences might be needed)
            # This assumes UNIFIED account structure
            if balance_info['result']['list']:
                 # Look for the specific coin balance details
                for item in balance_info['result']['list']:
                    if item['coin'] == quote_currency:
                         # Use availableToWithdraw or walletBalance depending on needs
                         # availableToWithdraw might be safer if margin is used elsewhere
                        return float(item.get('availableToWithdraw', item.get('walletBalance', 0))) # Adjust field based on testing
            return 0.0 # Return 0 if coin not found or list empty
        else:
            logging.error(f"Error fetching balance: {balance_info.get('retMsg')}")
            return None
    except Exception as e:
        logging.error(f"Exception fetching balance: {e}")
        return None

def calculate_quantity(entry_price, order_size_usdt, symbol):
    """Calculates the order quantity in base currency."""
    if entry_price is None or entry_price == 0:
        return None
    try:
        # Get symbol info for precision (tick size, min order size)
        symbol_info = session.get_instruments_info(category=CATEGORY, symbol=symbol)
        min_order_qty = 0
        qty_step = 0 # Lot size step

        if symbol_info and symbol_info['retCode'] == 0 and symbol_info['result']['list']:
            info = symbol_info['result']['list'][0]
            lot_size_filter = info.get('lotSizeFilter', {})
            min_order_qty = float(lot_size_filter.get('minOrderQty', 0))
            qty_step = float(lot_size_filter.get('qtyStep', 0)) # The smallest increment allowed for quantity
        else:
            logging.warning(f"Could not fetch instrument info for {symbol}. Using default precision.")
            # Set reasonable defaults if info fetch fails
            min_order_qty = 0.001 if "BTC" in symbol else 0.1
            qty_step = 0.001 if "BTC" in symbol else 0.1


        # Calculate raw quantity
        raw_qty = order_size_usdt / entry_price

        # Adjust quantity based on step size
        if qty_step > 0:
            adjusted_qty = round(raw_qty / qty_step) * qty_step
            # Ensure quantity is at least the minimum required
            final_qty = max(adjusted_qty, min_order_qty)
            # Format to the required precision (derived from qty_step)
            precision = len(str(qty_step).split('.')[-1]) if '.' in str(qty_step) else 0
            return f"{final_qty:.{precision}f}"
        else:
            # Fallback if step size is zero (shouldn't happen for valid symbols)
             logging.warning(f"Qty step size is zero for {symbol}. Using raw quantity.")
             return str(max(raw_qty, min_order_qty))


    except Exception as e:
        logging.error(f"Error calculating quantity: {e}")
        return None


def get_ohlcv(symbol, interval, limit=200):
    """Fetches OHLCV data from Bybit."""
    try:
        klines = session.get_kline(
            category=CATEGORY,
            symbol=symbol,
            interval=interval,
            limit=limit
        )
        if klines['retCode'] == 0 and klines['result']['list']:
            df = pd.DataFrame(klines['result']['list'], columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'turnover'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            df = df.astype(float)
            df.sort_index(inplace=True) # Ensure data is sorted chronologically
            return df
        else:
            logging.error(f"Error fetching klines: {klines.get('retMsg')}")
            return None
    except Exception as e:
        logging.error(f"Exception fetching klines: {e}")
        return None

def calculate_supertrend(df, length, multiplier):
    """Calculates Supertrend using pandas_ta."""
    if df is None or df.empty:
        return None
    try:
        df.ta.supertrend(length=length, multiplier=multiplier, append=True)
        # Column names might vary slightly based on pandas_ta version, check them
        # Common names: SUPERT_d (direction), SUPERTl (long line), SUPERTs (short line)
        # Let's standardize the direction column name
        direction_col = next((col for col in df.columns if col.startswith('SUPERT_') and col.endswith('_d')), None)
        if direction_col:
            df.rename(columns={direction_col: 'supertrend_direction'}, inplace=True)
        else:
             logging.error("Could not find Supertrend direction column in DataFrame.")
             return None
        return df
    except Exception as e:
        logging.error(f"Error calculating Supertrend: {e}")
        return None

def get_current_position(symbol):
    """Checks if there's an open position for the symbol."""
    global current_position # Use the global state variable
    try:
        position_info = session.get_positions(category=CATEGORY, symbol=symbol)
        if position_info['retCode'] == 0 and position_info['result']['list']:
            pos = position_info['result']['list'][0]
            size = float(pos.get('size', 0))
            side = pos.get('side', 'None') # 'Buy' for long, 'Sell' for short

            if size > 0:
                if side == 'Buy':
                    current_position = 'LONG'
                    return 'LONG', size
                elif side == 'Sell':
                    current_position = 'SHORT'
                    return 'SHORT', size
                else: # Should not happen if size > 0
                     current_position = None
                     return None, 0
            else:
                current_position = None
                return None, 0
        else:
            logging.info(f"No active position or error fetching position for {symbol}: {position_info.get('retMsg')}")
            current_position = None
            return None, 0
    except Exception as e:
        logging.error(f"Exception checking position: {e}")
        current_position = None # Reset state on error
        return None, 0

def place_order(symbol, side, qty, order_type="Market", stop_loss=None, take_profit=None):
    """Places an order on Bybit."""
    logging.info(f"Attempting to place {side} {order_type} order for {qty} {symbol}...")
    params = {
        "category": CATEGORY,
        "symbol": symbol,
        "side": side, # "Buy" or "Sell"
        "orderType": order_type,
        "qty": str(qty),
        "timeInForce": "GTC", # Good 'Til Canceled for Market SL/TP
        # "reduceOnly": False, # Ensure it's not reduce only for entry orders
    }

    # Add SL/TP - Ensure prices are formatted correctly
    # Bybit requires SL/TP prices, not percentages directly in the order call
    if stop_loss is not None:
        params["stopLoss"] = str(stop_loss)
    if take_profit is not None:
        params["takeProfit"] = str(take_profit)

    try:
        order_result = session.place_order(**params)
        logging.debug(f"Order placement raw response: {order_result}") # Debug log

        if order_result['retCode'] == 0:
            order_id = order_result['result'].get('orderId')
            logging.info(f"Successfully placed {side} order for {qty} {symbol}. Order ID: {order_id}. SL: {stop_loss}, TP: {take_profit}")
            return order_id
        else:
            logging.error(f"Failed to place {side} order for {symbol}. Error: {order_result['retMsg']}")
            # Specific error handling can be added here (e.g., insufficient margin)
            if "insufficient balance" in order_result['retMsg'].lower():
                logging.error("Insufficient balance detected.")
            return None
    except Exception as e:
        logging.error(f"Exception placing {side} order for {symbol}: {e}")
        return None

def close_position(symbol, position_side_to_close):
    """Closes the current open position with a market order."""
    pos_side, pos_size = get_current_position(symbol) # Re-check position just before closing

    if pos_side != position_side_to_close or pos_size <= 0:
        logging.info(f"No active {position_side_to_close} position found for {symbol} to close.")
        return True # Consider it successful if no position exists

    close_side = "Sell" if position_side_to_close == "LONG" else "Buy"
    logging.info(f"Attempting to close {position_side_to_close} position for {symbol} ({pos_size}) with a {close_side} order.")

    params = {
        "category": CATEGORY,
        "symbol": symbol,
        "side": close_side,
        "orderType": "Market",
        "qty": str(pos_size),
        "reduceOnly": True # IMPORTANT: Ensures this order only closes position
    }

    try:
        order_result = session.place_order(**params)
        logging.debug(f"Close order placement raw response: {order_result}") # Debug log

        if order_result['retCode'] == 0:
            order_id = order_result['result'].get('orderId')
            logging.info(f"Successfully placed closing {close_side} order for {pos_size} {symbol}. Order ID: {order_id}")
            global current_position
            current_position = None # Update state after successful close order placement
            return True
        else:
            logging.error(f"Failed to place closing {close_side} order for {symbol}. Error: {order_result['retMsg']}")
            # Handle potential issues, e.g., maybe the position closed already due to SL/TP
            if "Order quantity exceeded the current position size" in order_result['retMsg']:
                 logging.warning(f"Position {symbol} might have already closed (SL/TP hit?).")
                 current_position = None # Assume closed
                 return True
            return False # Failed to place closing order
    except Exception as e:
        logging.error(f"Exception closing {position_side_to_close} position for {symbol}: {e}")
        return False


# --- Main Strategy Logic ---

def run_strategy():
    """The main loop running the trading strategy."""
    global last_signal, current_position

    logging.info(f"Starting strategy for {SYMBOL} on {TIMEFRAME} timeframe...")

    try:
        # 1. Initial check: Get current position state at start
        position_side, position_size = get_current_position(SYMBOL)
        logging.info(f"Initial check: Position Side = {position_side}, Size = {position_size}")

        # 2. Fetch initial data and calculate indicator
        df = get_ohlcv(SYMBOL, TIMEFRAME, limit=200) # Need enough data for indicator calculation
        if df is None or df.empty:
            logging.error("Failed to fetch initial market data. Exiting.")
            return
        df = calculate_supertrend(df, SUPERTREND_ATR_PERIOD, SUPERTREND_MULTIPLIER)
        if df is None or 'supertrend_direction' not in df.columns:
            logging.error("Failed to calculcate initial Supertrend. Exiting.")
            return

        # Get the signal from the *last closed* candle (-2 index)
        # And the signal from the candle before that (-3 index) to check for flips
        if len(df) < 3:
            logging.warning("Not enough historical data yet to determine trend flip.")
            return # Skip first run if not enough data

        last_closed_candle = df.iloc[-2]
        prev_closed_candle = df.iloc[-3]

        current_signal_value = last_closed_candle['supertrend_direction']
        prev_signal_value = prev_closed_candle['supertrend_direction']
        last_price = last_closed_candle['close'] # Use close of last completed candle

        # Initialize last_signal if it's the first proper run
        if last_signal is None:
            last_signal = 'UP' if current_signal_value == 1 else 'DOWN'
            logging.info(f"Initialized last signal based on historical data: {last_signal}")


        logging.info(f"--- Cycle Start ---")
        logging.info(f"Time: {df.index[-1]}, Last Close: {last_price}")
        logging.info(f"Current Position: {current_position}, Last Signal: {last_signal}")
        logging.info(f"Supertrend Signal (Last Closed Candle): {'UP' if current_signal_value == 1 else 'DOWN'}")

        # 3. Decision Logic based on the last *closed* candle's signal
        signal_changed = current_signal_value != prev_signal_value
        current_trend = 'UP' if current_signal_value == 1 else 'DOWN'

        # --- EXIT LOGIC ---
        # Check if we need to exit BEFORE checking for entries
        if current_position == 'LONG' and current_trend == 'DOWN' and signal_changed:
            logging.info(f"Supertrend flipped DOWN. Closing LONG position for {SYMBOL}.")
            close_position(SYMBOL, 'LONG')
            # Reset state after attempting close
            current_position = None
            last_signal = 'DOWN'

        elif current_position == 'SHORT' and current_trend == 'UP' and signal_changed:
            logging.info(f"Supertrend flipped UP. Closing SHORT position for {SYMBOL}.")
            close_position(SYMBOL, 'SHORT')
            # Reset state after attempting close
            current_position = None
            last_signal = 'UP'


        # --- ENTRY LOGIC ---
        # Only enter if flat (no current position) and signal has just flipped
        if current_position is None:
            if current_trend == 'UP' and signal_changed: # Flipped to UP
                logging.info(f"Supertrend flipped UP. Potential LONG entry for {SYMBOL}.")
                qty = calculate_quantity(last_price, ORDER_SIZE_USDT, SYMBOL)
                if qty:
                    # Calculate SL/TP prices
                    sl_price = round(last_price * (1 - STOP_LOSS_PERCENT), 2) # Adjust rounding based on symbol precision
                    tp_price = round(last_price * (1 + TAKE_PROFIT_PERCENT), 2) # Adjust rounding
                    logging.info(f"Calculated LONG entry: Qty={qty}, EntryPrice(approx)={last_price}, SL={sl_price}, TP={tp_price}")

                    order_id = place_order(SYMBOL, "Buy", qty, order_type="Market", stop_loss=sl_price, take_profit=tp_price)
                    if order_id:
                        current_position = 'LONG' # Update state only if order placed successfully
                        last_signal = 'UP'
                else:
                    logging.warning("Could not calculate quantity for LONG entry.")

            elif current_trend == 'DOWN' and signal_changed: # Flipped to DOWN
                logging.info(f"Supertrend flipped DOWN. Potential SHORT entry for {SYMBOL}.")
                qty = calculate_quantity(last_price, ORDER_SIZE_USDT, SYMBOL)
                if qty:
                    # Calculate SL/TP prices
                    sl_price = round(last_price * (1 + STOP_LOSS_PERCENT), 2) # Adjust rounding
                    tp_price = round(last_price * (1 - TAKE_PROFIT_PERCENT), 2) # Adjust rounding
                    logging.info(f"Calculated SHORT entry: Qty={qty}, EntryPrice(approx)={last_price}, SL={sl_price}, TP={tp_price}")

                    order_id = place_order(SYMBOL, "Sell", qty, order_type="Market", stop_loss=sl_price, take_profit=tp_price)
                    if order_id:
                        current_position = 'SHORT' # Update state only if order placed successfully
                        last_signal = 'DOWN'
                else:
                     logging.warning("Could not calculate quantity for SHORT entry.")
            else:
                 # If no flip, update last_signal state if needed (e.g., after manual close or SL/TP hit)
                 last_signal = current_trend # Keep track of the ongoing trend even if no entry


        else: # Already in a position
             logging.info(f"Holding {current_position} position. No new entry actions.")
             # Optional: Implement trailing stop logic here if desired, e.g., adjust SL based on Supertrend line


        logging.info(f"--- Cycle End ---")


    except Exception as e:
        logging.error(f"An error occurred in the main strategy loop: {e}", exc_info=True)
        # Consider adding more robust error handling, e.g., pausing the script


# --- Scheduling ---
def job():
    """Function wrapper for the scheduler."""
    logging.info(f"Running scheduled job at {time.ctime()}")
    run_strategy()

# Calculate the schedule interval based on the timeframe
# This runs the check slightly after the candle close
if TIMEFRAME == '1': schedule.every().minute.at(":05").do(job) # Run 5 seconds past the minute
elif TIMEFRAME == '5': schedule.every(5).minutes.at(":05").do(job)
elif TIMEFRAME == '15': schedule.every(15).minutes.at(":05").do(job)
elif TIMEFRAME == '30': schedule.every(30).minutes.at(":05").do(job)
elif TIMEFRAME == '60': schedule.every().hour.at(":01:00").do(job) # Run 1 min past the hour
# Add more timeframe schedules as needed (e.g., H4, D)
else:
    logging.error(f"Unsupported timeframe '{TIMEFRAME}' for scheduling. Exiting.")
    exit()

# --- Main Execution ---
if __name__ == "__main__":
    logging.info("Starting Trading Bot...")
    # Run once immediately at start
    job()
    # Then run based on the schedule
    while True:
        schedule.run_pending()
        time.sleep(1) # Sleep for a second to avoid high CPU usage
