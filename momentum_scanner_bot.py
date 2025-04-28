import os
import sys
import time
import logging
import pandas as pd
import numpy as np
import pandas_ta as ta
from pybit.unified_trading import HTTP
from datetime import datetime, timedelta

# --- Configuration ---
# API Credentials
API_KEY = os.environ.get("BYBIT_API_KEY")
API_SECRET = os.environ.get("BYBIT_API_SECRET")
USE_TESTNET = os.environ.get("BYBIT_TESTNET", 'false').lower() == 'true'

# Trading Parameters
SYMBOL = "BTCUSDT"
TIMEFRAME = "15"  # Kline interval (e.g., "1", "5", "15", "60", "D")
LEVERAGE = 5

# --- Strategy Parameters (Based on Advanced Momentum Scanner) ---
# EMA Settings
ULTRA_FAST_EMA_PERIOD = 5
FAST_EMA_PERIOD = 10
MID_EMA_PERIOD = 30
SLOW_EMA_PERIOD = 100
PRICE_SOURCE = 'close' # 'close', 'hl2', 'hlc3', etc.

# Momentum Settings
ROC_PERIOD = 5
MOM_THRESHOLD_PERCENT = 0.1 # Pine script uses 0.5, but crypto might need lower? TEST THIS!

# Sensitivity Settings
SENSITIVITY_MODE = 'Balanced' # 'Conservative', 'Balanced', 'Aggressive'
ENABLE_EARLY_SIGNALS = True # Use early signals in addition to confirmed ones?

# Other Indicator Settings (Keep consistent or tune)
ATR_PERIOD_RISK = 14      # ATR period specifically for SL/TP calculation
ATR_PERIOD_VOLATILITY = 14 # ATR period for volatility checks (can be same as risk)
RSI_PERIOD = 10
RSI_HIGH = 70
RSI_LOW = 30
# Volatility Threshold Factor (Adjusts Pine Script's dyn_thresh concept)
# Higher = needs more volatility; Lower = needs less
VOLATILITY_THRESHOLD_FACTOR = 1.0 # Adjusted based on mode later
# Volume Check (Optional - Pine Script uses it, can be disabled)
USE_VOLUME_CONFIRMATION = True
VOLUME_SMA_PERIOD = 20
VOLUME_FACTOR = 1.5 # volume > sma * factor

KLINE_LIMIT = max(ULTRA_FAST_EMA_PERIOD, FAST_EMA_PERIOD, MID_EMA_PERIOD, SLOW_EMA_PERIOD,
                  ROC_PERIOD, ATR_PERIOD_RISK, ATR_PERIOD_VOLATILITY, RSI_PERIOD, VOLUME_SMA_PERIOD) + 20 # Ensure enough data

# Risk Management Parameters
RISK_PER_TRADE_PERCENT = 1.0
ATR_MULTIPLIER_SL = 1.5
ATR_MULTIPLIER_TP = 2.0  # R:R = TP_Mult / SL_Mult

# Bot Operation Parameters
SLEEP_INTERVAL_SECONDS = 60
COOLDOWN_PERIOD_SECONDS = 300 # 5 minutes

# --- State Variables ---
last_exit_time = None # Timestamp of the last position exit

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("momentum_scanner_bot.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("MomentumScannerTrader")

# --- Helper Functions ---
def calculate_indicators_momentum(df):
    """Calculates indicators needed for the Advanced Momentum Scanner strategy."""
    if df.empty or len(df) < KLINE_LIMIT:
        logger.warning("Not enough data to calculate indicators.")
        return None
    try:
        # EMAs
        df.ta.ema(length=ULTRA_FAST_EMA_PERIOD, close=PRICE_SOURCE, append=True, col_names=('ema_ultra'))
        df.ta.ema(length=FAST_EMA_PERIOD, close=PRICE_SOURCE, append=True, col_names=('ema_fast'))
        df.ta.ema(length=MID_EMA_PERIOD, close=PRICE_SOURCE, append=True, col_names=('ema_mid'))
        df.ta.ema(length=SLOW_EMA_PERIOD, close=PRICE_SOURCE, append=True, col_names=('ema_slow'))

        # EMA Slopes (Percentage Change)
        df['slope_ultra'] = df['ema_ultra'].pct_change() * 100
        df['slope_fast'] = df['ema_fast'].pct_change() * 100
        df['slope_mid'] = df['ema_mid'].pct_change() * 100

        # Rate of Change (RoC)
        df.ta.roc(length=ROC_PERIOD, close=PRICE_SOURCE, append=True, col_names=('roc'))
        df.ta.sma(close='roc', length=3, append=True, col_names=('roc_smooth')) # Smoothed RoC

        # ATR (for Risk and Volatility)
        df.ta.atr(length=ATR_PERIOD_RISK, append=True, col_names=('atr_risk'))
        df.ta.atr(length=ATR_PERIOD_VOLATILITY, append=True, col_names=('atr_volatility'))
        df['norm_atr'] = (df['atr_volatility'] / df['close']) * 100 # Normalized ATR %

        # RSI
        df.ta.rsi(length=RSI_PERIOD, close=PRICE_SOURCE, append=True, col_names=('rsi'))

        # Volume
        if USE_VOLUME_CONFIRMATION:
            df.ta.sma(close='volume', length=VOLUME_SMA_PERIOD, append=True, col_names=('volume_sma'))
            df['high_volume'] = df['volume'] > (df['volume_sma'] * VOLUME_FACTOR)
        else:
            df['high_volume'] = True # Always true if volume check disabled

        # Dynamic Volatility Threshold (adapting Pine Script logic)
        base_vol_thresh = df['norm_atr'].rolling(ATR_PERIOD_VOLATILITY).mean() # Avg NORM ATR
        dyn_thresh_factor = VOLATILITY_THRESHOLD_FACTOR
        if SENSITIVITY_MODE == 'Conservative': dyn_thresh_factor *= 1.2
        elif SENSITIVITY_MODE == 'Aggressive': dyn_thresh_factor *= 0.8
        df['volatility_threshold'] = base_vol_thresh * dyn_thresh_factor
        df['is_volatile'] = df['norm_atr'] > df['volatility_threshold']

        # --- Calculate Signals based on Pine Script Logic ---

        # Early Trend
        df['early_up'] = (df['ema_ultra'] > df['ema_fast']) & (df['slope_fast'] > 0) & (df['roc_smooth'] > MOM_THRESHOLD_PERCENT)
        df['early_down'] = (df['ema_ultra'] < df['ema_fast']) & (df['slope_fast'] < 0) & (df['roc_smooth'] < -MOM_THRESHOLD_PERCENT)

        # Confirmed Trend
        df['confirm_up'] = (df['ema_fast'] > df['ema_mid']) & (df['ema_mid'] > df['ema_slow']) & (df['close'] > df['ema_fast'])
        df['confirm_down'] = (df['ema_fast'] < df['ema_mid']) & (df['ema_mid'] < df['ema_slow']) & (df['close'] < df['ema_fast'])

        # Final Trend Direction (based on mode)
        df['trend_up'] = False
        df['trend_down'] = False
        if SENSITIVITY_MODE == 'Conservative':
            df['trend_up'] = df['confirm_up']
            df['trend_down'] = df['confirm_down']
        elif SENSITIVITY_MODE == 'Aggressive':
            df['trend_up'] = df['early_up'] | ((df['ema_ultra'] > df['ema_fast']) & df['confirm_up'])
            df['trend_down'] = df['early_down'] | ((df['ema_ultra'] < df['ema_fast']) & df['confirm_down'])
        else: # Balanced
            df['trend_up'] = (df['early_up'] & (df['ema_fast'] > df['ema_mid'])) | df['confirm_up']
            df['trend_down'] = (df['early_down'] & (df['ema_fast'] < df['ema_mid'])) | df['confirm_down']
        df['trend_neutral'] = ~df['trend_up'] & ~df['trend_down']

        # Early Entry Signals (More sensitive)
        # Pine: rsi > 45 and rsi < rsi_high - 5 (Long)
        # Pine: rsi < 55 and rsi > rsi_low + 5 (Short)
        df['early_long_entry'] = df['early_up'] & ~df['trend_down'].shift(1) & (df['roc_smooth'] > MOM_THRESHOLD_PERCENT) & (df['rsi'] > 45) & (df['rsi'] < (RSI_HIGH - 5))
        df['early_short_entry'] = df['early_down'] & ~df['trend_up'].shift(1) & (df['roc_smooth'] < -MOM_THRESHOLD_PERCENT) & (df['rsi'] < 55) & (df['rsi'] > (RSI_LOW + 5))

        # Confirmed Entry Signals (More reliable)
        # Pine: rsi > 50 and rsi < rsi_high (Long)
        # Pine: rsi < 50 and rsi > rsi_low (Short)
        df['confirm_long_entry'] = df['trend_up'] & (df['rsi'] > 50) & (df['rsi'] < RSI_HIGH) & df['is_volatile'] & df['high_volume']
        df['confirm_short_entry'] = df['trend_down'] & (df['rsi'] < 50) & (df['rsi'] > RSI_LOW) & df['is_volatile'] & df['high_volume']

        # Final Entry Signals (based on ENABLE_EARLY_SIGNALS and mode)
        df['long_signal'] = False
        df['short_signal'] = False
        if ENABLE_EARLY_SIGNALS:
            # Simplified combination: Use confirmed if available, else use early if mode allows/conditions met
            # Note: Pine script has more nuanced logic combining early signal strength here. Keeping it simpler.
            if SENSITIVITY_MODE == 'Conservative':
                 df['long_signal'] = df['confirm_long_entry'] | (df['early_long_entry'] & (df['ema_fast'] > df['ema_mid'])) # Early only if mid-term confirms
                 df['short_signal'] = df['confirm_short_entry'] | (df['early_short_entry'] & (df['ema_fast'] < df['ema_mid']))
            elif SENSITIVITY_MODE == 'Aggressive':
                 df['long_signal'] = df['confirm_long_entry'] | df['early_long_entry']
                 df['short_signal'] = df['confirm_short_entry'] | df['early_short_entry']
            else: # Balanced
                 df['long_signal'] = df['confirm_long_entry'] | (df['early_long_entry'] & df['trend_up']) # Early only if overall trend aligns
                 df['short_signal'] = df['confirm_short_entry'] | (df['early_short_entry'] & df['trend_down'])
        else: # Only use confirmed signals
            df['long_signal'] = df['confirm_long_entry']
            df['short_signal'] = df['confirm_short_entry']


        # Exit Signals
        # Pine: confirm_down OR (ema_ultra < ema_fast AND ema_fast < ema_mid) OR rsi > rsi_high
        df['exit_long_signal'] = df['confirm_down'] | ((df['ema_ultra'] < df['ema_fast']) & (df['ema_fast'] < df['ema_mid'])) | (df['rsi'] > RSI_HIGH)
        # Pine: confirm_up OR (ema_ultra > ema_fast AND ema_fast > ema_mid) OR rsi < rsi_low
        df['exit_short_signal'] = df['confirm_up'] | ((df['ema_ultra'] > df['ema_fast']) & (df['ema_fast'] > df['ema_mid'])) | (df['rsi'] < RSI_LOW)

        # Ensure all calculated columns exist
        required_cols = ['ema_ultra', 'ema_fast', 'ema_mid', 'ema_slow', 'slope_fast', 'roc_smooth',
                         'atr_risk', 'rsi', 'high_volume', 'is_volatile', 'trend_up', 'trend_down',
                         'long_signal', 'short_signal', 'exit_long_signal', 'exit_short_signal']
        if not all(col in df.columns for col in required_cols):
            missing = [col for col in required_cols if col not in df.columns]
            logger.error(f"Failed to calculate all indicators. Missing: {missing}")
            return None # Indicate failure

        logger.debug("Momentum Scanner indicators calculated.")
        return df

    except Exception as e:
        logger.error(f"Error calculating indicators: {e}", exc_info=True)
        return None

# --- Existing Helper Functions (get_available_balance, calculate_position_size_atr) ---
# (Keep these exactly as they were in the previous "advanced_volumatic_trader.py")
def get_available_balance(session, coin="USDT"):
    """Gets the available USDT balance for derivatives."""
    try:
        balance_info = session.get_wallet_balance(accountType="UNIFIED", coin=coin)
        if balance_info and balance_info.get('retCode') == 0:
            coin_balance = next((item for item in balance_info['result']['list'][0]['coin'] if item['coin'] == coin), None)
            if coin_balance:
                  # Example: Using availableToWithdraw. Consider 'equity' or 'availableBalance' based on needs.
                  balance = float(coin_balance.get('availableToWithdraw', 0)) # Or 'walletBalance', 'availableBalance'
                  logger.debug(f"Available balance ({coin}): {balance}")
                  return balance
            else:
                logger.warning(f"{coin} balance not found in UNIFIED account.")
                return 0.0
        else:
            logger.error(f"Error fetching wallet balance: {balance_info.get('retMsg') if balance_info else 'No response'}")
            return 0.0
    except Exception as e:
        logger.error(f"Exception getting wallet balance: {e}")
        return 0.0

def calculate_position_size_atr(balance, risk_percent, sl_distance_price, entry_price, min_order_qty, qty_precision):
    """Calculates position size based on risk % and ATR-based SL distance."""
    if entry_price <= 0 or sl_distance_price <= 0 or balance <= 0:
        logger.warning(f"Invalid input for position size calc: entry={entry_price}, sl_dist={sl_distance_price}, bal={balance}")
        return 0

    risk_amount_usdt = balance * (risk_percent / 100.0)
    position_size_units = risk_amount_usdt / sl_distance_price

    if position_size_units < min_order_qty:
         logger.warning(f"Calculated size {position_size_units:.8f} is below minimum {min_order_qty:.8f}. Adjusting to minimum.")
         position_size_units = min_order_qty
         actual_risk_amount = position_size_units * sl_distance_price
         logger.info(f"Using min order qty. Actual USDT risk approx: {actual_risk_amount:.2f}")


    position_size_units = round(position_size_units, qty_precision)

    if position_size_units < min_order_qty:
        logger.error(f"Rounded size {position_size_units} is still below minimum {min_order_qty}. Cannot place trade.")
        return 0

    logger.info(f"Calculated position size: {position_size_units} based on Balance=${balance:.2f}, Risk={risk_percent}%, SL Distance=${sl_distance_price:.4f}, Entry=${entry_price:.2f}")
    return position_size_units


# --- Main Trading Class (Updated for Momentum Scanner Logic) ---
class MomentumScannerTrader:
    def __init__(self, api_key, api_secret, symbol, timeframe, leverage, use_testnet=False):
        self.symbol = symbol
        self.timeframe = timeframe
        self.leverage = leverage
        self.min_order_qty = 0.0
        self.qty_precision = 3
        self.price_precision = 2
        self.last_exit_time = None # Instance variable for cooldown

        try:
            self.session = HTTP(
                testnet=use_testnet,
                api_key=api_key,
                api_secret=api_secret,
            )
            logger.info(f"Bybit session initialized. Testnet: {use_testnet}")
            self._initial_setup()
            self._fetch_instrument_info()
        except Exception as e:
            logger.error(f"Failed to initialize Bybit session: {e}", exc_info=True)
            sys.exit(1)

    # --- _initial_setup, _fetch_instrument_info, get_position, place_order, close_position ---
    # (Keep these methods exactly as they were in "advanced_volumatic_trader.py")
    # Note: Ensure close_position updates self.last_exit_time = datetime.utcnow()
    def _initial_setup(self):
        """Set leverage and margin mode."""
        try:
            self.session.set_leverage(
                category="linear", symbol=self.symbol,
                buyLeverage=str(self.leverage), sellLeverage=str(self.leverage)
            )
            logger.info(f"Leverage for {self.symbol} set to {self.leverage}x")
            try:
                 pos = self.get_position(log_error=False)
                 if not pos or float(pos.get('size', 0)) == 0:
                     mode_resp = self.session.switch_margin_mode(category="linear", symbol=self.symbol, tradeMode=1) # 1 for Isolated
                     if mode_resp and mode_resp.get('retCode') == 0:
                         logger.info(f"Margin mode for {self.symbol} set to ISOLATED.")
                     else:
                         logger.warning(f"Could not switch margin mode (retCode={mode_resp.get('retCode')}, msg='{mode_resp.get('retMsg')}'). May already be Isolated or orders exist.")
                 else:
                     logger.warning("Cannot switch margin mode while position is open. Keeping current mode.")
            except Exception as e:
                 logger.warning(f"Could not set margin mode (exception): {e}")
        except Exception as e:
            logger.error(f"Error during initial setup (leverage/margin): {e}", exc_info=True)

    def _fetch_instrument_info(self):
       """Fetches instrument details like min order size, qty step, price precision."""
       try:
           info = self.session.get_instruments_info(category="linear", symbol=self.symbol)
           if info and info['retCode'] == 0 and info['result']['list']:
               instrument = info['result']['list'][0]
               self.min_order_qty = float(instrument['lotSizeFilter']['minOrderQty'])
               qty_step_str = instrument['lotSizeFilter']['qtyStep']
               self.qty_precision = len(qty_step_str.split('.')[-1]) if '.' in qty_step_str else 0

               price_tick_str = instrument['priceFilter']['tickSize']
               self.price_precision = len(price_tick_str.split('.')[-1]) if '.' in price_tick_str else 0

               logger.info(f"Instrument info for {self.symbol}: Min Qty={self.min_order_qty}, Qty Precision={self.qty_precision}, Price Precision={self.price_precision}")
           else:
               logger.error(f"Could not fetch instrument info: {info.get('retMsg') if info else 'No response'}. Using defaults.")
       except Exception as e:
           logger.error(f"Exception fetching instrument info: {e}. Using defaults.")

    def get_ohlcv(self, limit=KLINE_LIMIT):
        """Fetches OHLCV data and calculates Momentum Scanner indicators."""
        try:
            response = self.session.get_kline(
                category="linear", symbol=self.symbol, interval=self.timeframe, limit=limit
            )
            if response and response['retCode'] == 0:
                data = response['result']['list']
                if not data:
                    logger.warning("No kline data received.")
                    return None
                df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'turnover'])
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                df.set_index('timestamp', inplace=True)
                df = df.iloc[::-1].copy() # Reverse to have newest data last
                for col in ['open', 'high', 'low', 'close', 'volume']: df[col] = pd.to_numeric(df[col])

                # Calculate indicators using the new function
                df = calculate_indicators_momentum(df) # <--- Use the new indicator function
                return df
            else:
                logger.error(f"Error fetching kline data: {response.get('retMsg') if response else 'No response'}")
                return None
        except Exception as e:
            logger.error(f"Exception fetching kline data: {e}", exc_info=True)
            return None

    def get_position(self, log_error=True):
        """Gets current position information."""
        try:
            response = self.session.get_positions(category="linear", symbol=self.symbol)
            if response and response['retCode'] == 0 and response['result']['list']:
                position_data = response['result']['list'][0]
                if float(position_data.get('size', 0)) != 0:
                     # logger.debug(f"Current position data: {position_data}") # Reduce log noise
                     return position_data
                else:
                     return None
            elif response and response['retCode'] == 0:
                 return None
            else:
                if log_error: logger.error(f"Error fetching position data: {response.get('retMsg') if response else 'No response'}")
                return None
        except Exception as e:
            if log_error: logger.error(f"Exception fetching position data: {e}", exc_info=True)
            return None

    def place_order(self, side, qty, stop_loss_price, take_profit_price):
        """Places a market order with SL and TP."""
        try:
            sl_price_str = f"{stop_loss_price:.{self.price_precision}f}"
            tp_price_str = f"{take_profit_price:.{self.price_precision}f}"
            qty_str = f"{qty:.{self.qty_precision}f}"

            logger.info(f"Attempting to place {side} order: Qty={qty_str}, SL={sl_price_str}, TP={tp_price_str}")

            response = self.session.place_order(
                category="linear",
                symbol=self.symbol,
                side=side,
                orderType="Market",
                qty=qty_str,
                stopLoss=sl_price_str,
                takeProfit=tp_price_str,
                slTriggerBy="LastPrice",
                tpTriggerBy="LastPrice",
            )
            if response and response['retCode'] == 0:
                order_id = response['result'].get('orderId')
                logger.info(f"Market {side} order placed successfully! Order ID: {order_id}")
                return order_id
            else:
                error_msg = response.get('retMsg', 'No error message')
                ret_code = response.get('retCode', 'N/A')
                logger.error(f"Failed to place {side} order. Code: {ret_code}, Message: '{error_msg}'")
                logger.debug(f"Order details attempted: Qty={qty_str}, SL={sl_price_str}, TP={tp_price_str}")
                if "insufficient available balance" in error_msg.lower():
                     logger.critical("Order failed due to insufficient balance!")
                return None
        except Exception as e:
            logger.error(f"Exception placing {side} order: {e}", exc_info=True)
            return None

    def close_position(self, position_data):
        """Closes the current position with a market order."""
        if not position_data:
            logger.warning("Attempted to close position, but no position data provided.")
            return False
        side = position_data.get('side')
        size = float(position_data.get('size', 0))
        if size == 0: return False

        close_side = "Sell" if side == "Buy" else "Buy"
        qty_str = f"{size:.{self.qty_precision}f}"
        logger.info(f"Attempting to close {side} position of size {size} with a {close_side} market order.")
        try:
            response = self.session.place_order(
                category="linear", symbol=self.symbol, side=close_side,
                orderType="Market", qty=qty_str, reduce_only=True # Ensure reduce_only
            )
            if response and response['retCode'] == 0:
                order_id = response['result'].get('orderId')
                logger.info(f"Position close order placed successfully! Order ID: {order_id}. Side: {close_side}, Qty: {qty_str}")
                self.last_exit_time = datetime.utcnow() # Record exit time for cooldown
                logger.info(f"Cooldown period started. Next entry possible after {self.last_exit_time + timedelta(seconds=COOLDOWN_PERIOD_SECONDS)}")
                return True
            else:
                logger.error(f"Failed to place position close order: {response.get('retMsg') if response else 'No response'}")
                return False
        except Exception as e:
            logger.error(f"Exception closing position: {e}", exc_info=True)
            return False


    def run(self):
        """Main trading loop using Momentum Scanner logic."""
        logger.info(f"Starting Momentum Scanner trading loop for {self.symbol}...")
        logger.info(f"Mode: {SENSITIVITY_MODE}, Early Signals: {ENABLE_EARLY_SIGNALS}, Volume Confirm: {USE_VOLUME_CONFIRMATION}")
        logger.info(f"Risk: {RISK_PER_TRADE_PERCENT}%, SL: {ATR_MULTIPLIER_SL}*ATR, TP: {ATR_MULTIPLIER_TP}*ATR")

        while True:
            try:
                # --- Cooldown Check ---
                if self.last_exit_time:
                    cooldown_end_time = self.last_exit_time + timedelta(seconds=COOLDOWN_PERIOD_SECONDS)
                    if datetime.utcnow() < cooldown_end_time:
                        wait_time = (cooldown_end_time - datetime.utcnow()).total_seconds()
                        logger.info(f"In cooldown period. Waiting for {wait_time:.0f} more seconds...")
                        time.sleep(min(wait_time + 1, SLEEP_INTERVAL_SECONDS))
                        continue

                # 1. Get Data and Calculate Indicators
                df = self.get_ohlcv()
                if df is None or df.empty or df.isnull().any().any(): # More robust NaN check
                    logger.warning("Could not get valid data or indicators contain NaNs. Waiting...")
                    time.sleep(SLEEP_INTERVAL_SECONDS)
                    continue

                # Get latest closed candle data [-2]
                if len(df) < 3: # Need at least 3 for pct_change checks etc.
                     logger.warning("Not enough historical data points yet for signal generation. Waiting...")
                     time.sleep(SLEEP_INTERVAL_SECONDS)
                     continue

                last_closed = df.iloc[-2] # Use the last fully closed candle for signals

                # Check essential values are present
                if pd.isna(last_closed['atr_risk']) or last_closed['atr_risk'] <= 0:
                    logger.warning(f"ATR_Risk is invalid ({last_closed['atr_risk']}). Skipping cycle.")
                    time.sleep(SLEEP_INTERVAL_SECONDS)
                    continue
                if pd.isna(last_closed['long_signal']) or pd.isna(last_closed['short_signal']) or \
                   pd.isna(last_closed['exit_long_signal']) or pd.isna(last_closed['exit_short_signal']):
                     logger.warning("Signal values missing or NaN on last closed candle. Waiting...")
                     time.sleep(SLEEP_INTERVAL_SECONDS)
                     continue


                # 2. Check Current Position and Balance
                position = self.get_position()
                balance = get_available_balance(self.session, "USDT")
                if balance <= 0:
                    logger.error("Available balance is zero or negative. Stopping.")
                    break

                # 3. Check Exit Conditions (if position exists)
                if position:
                    pos_side = position.get('side') # 'Buy' or 'Sell'
                    pos_size = float(position.get('size', 0))
                    entry_price = float(position.get('avgPrice', 0))
                    logger.info(f"Holding {pos_side} position. Size: {pos_size}, Entry: {entry_price:.{self.price_precision}f}")

                    # Exit Condition 1: Strategy Exit Signal
                    exit_signal = False
                    if pos_side == "Buy" and last_closed['exit_long_signal']:
                        logger.info(f"Exit Signal: Strategy exit long condition met. Closing Long.")
                        exit_signal = True
                    elif pos_side == "Sell" and last_closed['exit_short_signal']:
                        logger.info(f"Exit Signal: Strategy exit short condition met. Closing Short.")
                        exit_signal = True

                    if exit_signal:
                        if self.close_position(position):
                            logger.info("Position closed based on strategy exit signal.")
                        else:
                            logger.error("Failed to close position based on strategy exit signal.")
                        time.sleep(SLEEP_INTERVAL_SECONDS) # Wait after closing
                        continue

                    # Exit Condition 2: SL/TP (Handled by Bybit)

                # 4. Check Entry Conditions (if no position exists and not in cooldown)
                else: # No position open
                    logger.debug("No position open. Checking for entry signals...")
                    entry_price_approx = last_closed['close'] # Use last close as approx entry for calculations
                    current_atr = last_closed['atr_risk']

                    # Check for Long Entry Signal
                    if last_closed['long_signal']:
                        logger.info(f"Long Entry Signal detected at ~${entry_price_approx:.{self.price_precision}f}, ATR={current_atr:.{self.price_precision}f}")
                        sl_distance = current_atr * ATR_MULTIPLIER_SL
                        tp_distance = current_atr * ATR_MULTIPLIER_TP
                        sl_price = entry_price_approx - sl_distance
                        tp_price = entry_price_approx + tp_distance

                        qty = calculate_position_size_atr(balance, RISK_PER_TRADE_PERCENT, sl_distance, entry_price_approx, self.min_order_qty, self.qty_precision)

                        if qty > 0:
                            logger.info(f"Placing LONG order: Qty={qty}, SL={sl_price:.{self.price_precision}f}, TP={tp_price:.{self.price_precision}f}")
                            self.place_order("Buy", qty, sl_price, tp_price)
                        else:
                            logger.warning("Could not enter long trade due to zero quantity calculation.")
                        time.sleep(SLEEP_INTERVAL_SECONDS) # Wait after attempting entry
                        continue # Skip short check if long was triggered


                    # Check for Short Entry Signal
                    elif last_closed['short_signal']:
                        logger.info(f"Short Entry Signal detected at ~${entry_price_approx:.{self.price_precision}f}, ATR={current_atr:.{self.price_precision}f}")
                        sl_distance = current_atr * ATR_MULTIPLIER_SL
                        tp_distance = current_atr * ATR_MULTIPLIER_TP
                        sl_price = entry_price_approx + sl_distance
                        tp_price = entry_price_approx - tp_distance

                        qty = calculate_position_size_atr(balance, RISK_PER_TRADE_PERCENT, sl_distance, entry_price_approx, self.min_order_qty, self.qty_precision)

                        if qty > 0:
                            logger.info(f"Placing SHORT order: Qty={qty}, SL={sl_price:.{self.price_precision}f}, TP={tp_price:.{self.price_precision}f}")
                            self.place_order("Sell", qty, sl_price, tp_price)
                        else:
                            logger.warning("Could not enter short trade due to zero quantity calculation.")
                        time.sleep(SLEEP_INTERVAL_SECONDS) # Wait after attempting entry
                        continue

                    else:
                        logger.debug("No entry signal detected this cycle.")

                # 5. Wait for the next interval
                logger.debug(f"Loop end. Waiting for {SLEEP_INTERVAL_SECONDS} seconds...")
                time.sleep(SLEEP_INTERVAL_SECONDS)

            except KeyboardInterrupt:
                logger.info("Keyboard interrupt detected. Attempting graceful shutdown...")
                position = self.get_position()
                if position:
                    logger.warning("Closing open position due to script termination...")
                    self.close_position(position)
                logger.info("Bot stopped.")
                break
            except Exception as e:
                logger.error(f"An unexpected error occurred in the main loop: {e}", exc_info=True)
                logger.info(f"Pausing for {SLEEP_INTERVAL_SECONDS*2} seconds after error...")
                time.sleep(SLEEP_INTERVAL_SECONDS * 2) # Longer pause after error

# --- Script Execution ---
if __name__ == "__main__":
    if not API_KEY or not API_SECRET:
        logger.error("API Key or Secret not found in environment variables. Exiting.")
        sys.exit(1)

    # Validate essential parameters
    if RISK_PER_TRADE_PERCENT <= 0 or ATR_MULTIPLIER_SL <= 0 or ATR_MULTIPLIER_TP <= 0:
        logger.error("Risk management percentages/multipliers must be positive. Exiting.")
        sys.exit(1)
    if LEVERAGE <= 0: logger.error("Leverage must be positive. Exiting."); sys.exit(1)
    if SENSITIVITY_MODE not in ['Conservative', 'Balanced', 'Aggressive']:
        logger.error(f"Invalid SENSITIVITY_MODE: {SENSITIVITY_MODE}. Must be 'Conservative', 'Balanced', or 'Aggressive'. Exiting.")
        sys.exit(1)
    if MOM_THRESHOLD_PERCENT <= 0: logger.warning("MOM_THRESHOLD_PERCENT is zero or negative. This might affect signal generation.")
    if COOLDOWN_PERIOD_SECONDS < 0: logger.warning("Cooldown period is negative, disabling cooldown."); COOLDOWN_PERIOD_SECONDS = 0

    trader = MomentumScannerTrader(
        api_key=API_KEY,
        api_secret=API_SECRET,
        symbol=SYMBOL,
        timeframe=TIMEFRAME,
        leverage=LEVERAGE,
        use_testnet=USE_TESTNET
    )

    if trader.min_order_qty == 0.0:
         logger.warning("Minimum order quantity is 0.0 (fetch might have failed). Check instrument manually. Proceeding with caution.")

    trader.run()
    logger.info("Bot stopped.")
