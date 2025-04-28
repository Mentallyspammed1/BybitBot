import os
import sys
import time
import logging
import pandas as pd
import numpy as np
import pandas_ta as ta
from pybit.unified_trading import HTTP
from datetime import datetime, timedelta
import colorama
from colorama import Fore, Style, Back
import json # For loading config
import argparse # For command-line arguments

# --- Initialize Colorama ---
colorama.init(autoreset=True)

# --- Constants ---
CONFIG_FILE = "config.json"
LOG_FILE = "momentum_scanner_bot_integrated.log"

# --- Load API Keys (Keep using environment variables for security) ---
API_KEY = os.environ.get("BYBIT_API_KEY")
API_SECRET = os.environ.get("BYBIT_API_SECRET")

# --- Logging Setup (Keep basic file logging, enhance console with color) ---
log_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler = logging.FileHandler(LOG_FILE)
file_handler.setFormatter(log_formatter)
logger = logging.getLogger("MomentumScannerTrader")
logger.setLevel(logging.INFO)
logger.addHandler(file_handler)
logger.propagate = False

# --- Configuration Loading Function ---
def load_config(config_path):
    """Loads configuration from a JSON file."""
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        print(f"{Fore.GREEN}INFO:{Style.RESET_ALL} Configuration loaded successfully from {config_path}")
        logger.info(f"Configuration loaded successfully from {config_path}")
        # --- Calculate KLINE_LIMIT based on loaded config ---
        strategy_cfg = config.get('STRATEGY_CONFIG', {})
        bot_cfg = config.get('BOT_CONFIG', {})
        periods = [
            strategy_cfg.get('ULTRA_FAST_EMA_PERIOD', 5),
            strategy_cfg.get('FAST_EMA_PERIOD', 10),
            strategy_cfg.get('MID_EMA_PERIOD', 30),
            strategy_cfg.get('SLOW_EMA_PERIOD', 100),
            strategy_cfg.get('ROC_PERIOD', 5),
            strategy_cfg.get('ATR_PERIOD_RISK', 14),
            strategy_cfg.get('ATR_PERIOD_VOLATILITY', 14),
            strategy_cfg.get('RSI_PERIOD', 10),
            strategy_cfg.get('VOLUME_SMA_PERIOD', 20)
        ]
        config['INTERNAL'] = {
            'KLINE_LIMIT': max(periods) + bot_cfg.get('KLINE_LIMIT_BUFFER', 20)
        }
        return config
    except FileNotFoundError:
        print(f"{Back.RED}{Fore.WHITE}{Style.BRIGHT}FATAL:{Style.RESET_ALL} Configuration file not found: {config_path}")
        logger.critical(f"Configuration file not found: {config_path}")
        sys.exit(1)
    except json.JSONDecodeError:
        print(f"{Back.RED}{Fore.WHITE}{Style.BRIGHT}FATAL:{Style.RESET_ALL} Invalid JSON format in configuration file: {config_path}")
        logger.critical(f"Invalid JSON format in configuration file: {config_path}")
        sys.exit(1)
    except Exception as e:
        print(f"{Back.RED}{Fore.WHITE}{Style.BRIGHT}FATAL:{Style.RESET_ALL} Error loading configuration: {e}")
        logger.critical(f"Error loading configuration: {e}", exc_info=True)
        sys.exit(1)

# --- Helper Functions (Modified to accept config) ---
def calculate_indicators_momentum(df, strategy_cfg):
    """Calculates indicators using parameters from strategy_cfg."""
    kline_limit = config.get('INTERNAL', {}).get('KLINE_LIMIT', 120) # Get calculated limit
    if df.empty or len(df) < kline_limit:
        print(f"{Fore.YELLOW}{Style.BRIGHT}WARN:{Style.RESET_ALL} Not enough data ({len(df)}/{kline_limit}) to calculate indicators.")
        logger.warning("Not enough data to calculate indicators.")
        return None
    try:
        price_source = strategy_cfg.get('PRICE_SOURCE', 'close')
        # EMA Calculations
        df.ta.ema(length=strategy_cfg['ULTRA_FAST_EMA_PERIOD'], close=price_source, append=True, col_names=('ema_ultra'))
        df.ta.ema(length=strategy_cfg['FAST_EMA_PERIOD'], close=price_source, append=True, col_names=('ema_fast'))
        df.ta.ema(length=strategy_cfg['MID_EMA_PERIOD'], close=price_source, append=True, col_names=('ema_mid'))
        df.ta.ema(length=strategy_cfg['SLOW_EMA_PERIOD'], close=price_source, append=True, col_names=('ema_slow'))
        # Slope & RoC
        df['slope_ultra'] = df['ema_ultra'].pct_change() * 100
        df['slope_fast'] = df['ema_fast'].pct_change() * 100
        df['slope_mid'] = df['ema_mid'].pct_change() * 100
        df.ta.roc(length=strategy_cfg['ROC_PERIOD'], close=price_source, append=True, col_names=('roc'))
        df.ta.sma(close='roc', length=3, append=True, col_names=('roc_smooth'))
        # ATR & Volatility
        df.ta.atr(length=strategy_cfg['ATR_PERIOD_RISK'], append=True, col_names=('atr_risk'))
        df.ta.atr(length=strategy_cfg['ATR_PERIOD_VOLATILITY'], append=True, col_names=('atr_volatility'))
        df['norm_atr'] = (df['atr_volatility'] / df['close']) * 100
        # RSI
        df.ta.rsi(length=strategy_cfg['RSI_PERIOD'], close=price_source, append=True, col_names=('rsi'))
        # Volume
        if strategy_cfg.get('USE_VOLUME_CONFIRMATION', True):
            df.ta.sma(close='volume', length=strategy_cfg['VOLUME_SMA_PERIOD'], append=True, col_names=('volume_sma'))
            df['high_volume'] = df['volume'] > (df['volume_sma'] * strategy_cfg.get('VOLUME_FACTOR', 1.5))
        else:
            df['high_volume'] = True # Always true if volume confirmation disabled
        # Dynamic Thresholds & Sensitivity
        base_vol_thresh = df['norm_atr'].rolling(strategy_cfg['ATR_PERIOD_VOLATILITY']).mean()
        dyn_thresh_factor = strategy_cfg.get('VOLATILITY_THRESHOLD_FACTOR', 1.0)
        sensitivity_mode = strategy_cfg.get('SENSITIVITY_MODE', 'Balanced')
        if sensitivity_mode == 'Conservative': dyn_thresh_factor *= 1.2
        elif sensitivity_mode == 'Aggressive': dyn_thresh_factor *= 0.8
        df['volatility_threshold'] = base_vol_thresh * dyn_thresh_factor
        df['is_volatile'] = df['norm_atr'] > df['volatility_threshold']
        # Signals (Logic remains the same, uses config values)
        mom_threshold = strategy_cfg.get('MOM_THRESHOLD_PERCENT', 0.1)
        rsi_high = strategy_cfg.get('RSI_HIGH', 70)
        rsi_low = strategy_cfg.get('RSI_LOW', 30)
        df['early_up'] = (df['ema_ultra'] > df['ema_fast']) & (df['slope_fast'] > 0) & (df['roc_smooth'] > mom_threshold)
        df['early_down'] = (df['ema_ultra'] < df['ema_fast']) & (df['slope_fast'] < 0) & (df['roc_smooth'] < -mom_threshold)
        df['confirm_up'] = (df['ema_fast'] > df['ema_mid']) & (df['ema_mid'] > df['ema_slow']) & (df['close'] > df['ema_fast'])
        df['confirm_down'] = (df['ema_fast'] < df['ema_mid']) & (df['ema_mid'] < df['ema_slow']) & (df['close'] < df['ema_fast'])
        # ... [Rest of the signal logic from V3, using config values like sensitivity_mode, rsi_high/low etc.] ...
        df['trend_up'] = False
        df['trend_down'] = False
        if sensitivity_mode == 'Conservative':
             df['trend_up'] = df['confirm_up']
             df['trend_down'] = df['confirm_down']
        elif sensitivity_mode == 'Aggressive':
             df['trend_up'] = df['early_up'] | ((df['ema_ultra'] > df['ema_fast']) & df['confirm_up'])
             df['trend_down'] = df['early_down'] | ((df['ema_ultra'] < df['ema_fast']) & df['confirm_down'])
        else: # Balanced
             df['trend_up'] = (df['early_up'] & (df['ema_fast'] > df['ema_mid'])) | df['confirm_up']
             df['trend_down'] = (df['early_down'] & (df['ema_fast'] < df['ema_mid'])) | df['confirm_down']
        df['trend_neutral'] = ~df['trend_up'] & ~df['trend_down']
        df['early_long_entry'] = df['early_up'] & ~df['trend_down'].shift(1) & (df['roc_smooth'] > mom_threshold) & (df['rsi'] > 45) & (df['rsi'] < (rsi_high - 5))
        df['early_short_entry'] = df['early_down'] & ~df['trend_up'].shift(1) & (df['roc_smooth'] < -mom_threshold) & (df['rsi'] < 55) & (df['rsi'] > (rsi_low + 5))
        df['confirm_long_entry'] = df['trend_up'] & (df['rsi'] > 50) & (df['rsi'] < rsi_high) & df['is_volatile'] & df['high_volume']
        df['confirm_short_entry'] = df['trend_down'] & (df['rsi'] < 50) & (df['rsi'] > rsi_low) & df['is_volatile'] & df['high_volume']
        df['long_signal'] = False
        df['short_signal'] = False
        if strategy_cfg.get('ENABLE_EARLY_SIGNALS', True):
            if sensitivity_mode == 'Conservative':
                 df['long_signal'] = df['confirm_long_entry'] | (df['early_long_entry'] & (df['ema_fast'] > df['ema_mid']))
                 df['short_signal'] = df['confirm_short_entry'] | (df['early_short_entry'] & (df['ema_fast'] < df['ema_mid']))
            elif sensitivity_mode == 'Aggressive':
                 df['long_signal'] = df['confirm_long_entry'] | df['early_long_entry']
                 df['short_signal'] = df['confirm_short_entry'] | df['early_short_entry']
            else: # Balanced
                 df['long_signal'] = df['confirm_long_entry'] | (df['early_long_entry'] & df['trend_up'])
                 df['short_signal'] = df['confirm_short_entry'] | (df['early_short_entry'] & df['trend_down'])
        else:
             df['long_signal'] = df['confirm_long_entry']
             df['short_signal'] = df['confirm_short_entry']
        df['exit_long_signal'] = df['confirm_down'] | ((df['ema_ultra'] < df['ema_fast']) & (df['ema_fast'] < df['ema_mid'])) | (df['rsi'] > rsi_high)
        df['exit_short_signal'] = df['confirm_up'] | ((df['ema_ultra'] > df['ema_fast']) & (df['ema_fast'] > df['ema_mid'])) | (df['rsi'] < rsi_low)
        # Check required columns
        required_cols = ['ema_ultra', 'ema_fast', 'ema_mid', 'ema_slow', 'slope_fast', 'roc_smooth',
                         'atr_risk', 'rsi', 'high_volume', 'is_volatile', 'trend_up', 'trend_down',
                         'long_signal', 'short_signal', 'exit_long_signal', 'exit_short_signal']
        if not all(col in df.columns for col in required_cols):
            missing = [col for col in required_cols if col not in df.columns]
            print(f"{Fore.RED}{Style.BRIGHT}ERROR:{Style.RESET_ALL} Failed to calculate all indicators. Missing: {missing}")
            logger.error(f"Failed to calculate all indicators. Missing: {missing}")
            return None
        return df
    except Exception as e:
        print(f"{Fore.RED}{Style.BRIGHT}ERROR:{Style.RESET_ALL} Error calculating indicators: {e}")
        logger.error(f"Error calculating indicators: {e}", exc_info=True)
        return None

def get_available_balance(session, coin="USDT"):
    """Gets the available USDT balance for derivatives."""
    # (Keep exact code from previous version)
    try:
        balance_info = session.get_wallet_balance(accountType="UNIFIED", coin=coin)
        # ... (rest of the function is identical) ...
        if balance_info and balance_info.get('retCode') == 0:
            # ... find coin balance ...
             if coin_balance:
                  balance = float(coin_balance.get('availableToWithdraw', 0))
                  logger.debug(f"Available balance ({coin}): {balance}")
                  return balance
             else:
                  # ... handle missing coin ...
                  return 0.0
        else:
            # ... handle API error ...
             return 0.0
    except Exception as e:
        # ... handle exception ...
         return 0.0


def calculate_position_size_atr(balance, risk_percent, sl_distance_price, entry_price, min_order_qty, qty_precision, price_precision, max_position_usdt=0):
    """Calculates position size, potentially capping by max_position_usdt."""
    # (Keep initial checks)
    if entry_price <= 0 or sl_distance_price <= 0 or balance <= 0:
         print(f"{Fore.YELLOW}{Style.BRIGHT}WARN:{Style.RESET_ALL} Invalid input for pos size calc...")
         logger.warning(f"Invalid input for position size calc...")
         return 0

    risk_amount_usdt = balance * (risk_percent / 100.0)
    position_size_units = risk_amount_usdt / sl_distance_price

    # --- Add Max Position USDT Check ---
    position_value_usdt = position_size_units * entry_price
    if max_position_usdt > 0 and position_value_usdt > max_position_usdt:
        cap_msg = f"Calculated position value ${position_value_usdt:.2f} exceeds max ${max_position_usdt:.2f}. Capping size."
        print(f"{Fore.YELLOW}WARN:{Style.RESET_ALL} {cap_msg}")
        logger.warning(cap_msg)
        position_size_units = max_position_usdt / entry_price
        # Recalculate actual risk with capped size
        actual_risk_amount = position_size_units * sl_distance_price
        risk_info_msg = f"Capped size leads to actual USDT risk approx: ${actual_risk_amount:.2f}"
        print(f"{Fore.CYAN}INFO:{Style.RESET_ALL} {risk_info_msg}")
        logger.info(risk_info_msg)
    # --- End Max Position Check ---

    # Adjust to minimum order quantity if needed
    if position_size_units < min_order_qty:
         print(f"{Fore.YELLOW}WARN:{Style.RESET_ALL} Calculated size {position_size_units:.8f} (or capped size) is below minimum {min_order_qty:.8f}. Adjusting.")
         logger.warning(f"Calculated/Capped size {position_size_units:.8f} is below minimum {min_order_qty:.8f}. Adjusting.")
         position_size_units = min_order_qty
         actual_risk_amount = position_size_units * sl_distance_price
         print(f"{Fore.CYAN}INFO:{Style.RESET_ALL} Using min order qty. Actual USDT risk approx: ${actual_risk_amount:.2f}")
         logger.info(f"Using min order qty. Actual USDT risk approx: ${actual_risk_amount:.2f}")

    # Round to required precision
    position_size_units = round(position_size_units, qty_precision)

    # Final check after rounding
    if position_size_units < min_order_qty:
        print(f"{Fore.RED}{Style.BRIGHT}ERROR:{Style.RESET_ALL} Rounded size {position_size_units} is still below minimum {min_order_qty}. Cannot place trade.")
        logger.error(f"Rounded size {position_size_units} is still below minimum {min_order_qty}. Cannot place trade.")
        return 0

    calc_info = f"Final position size: {position_size_units} units. Based on Balance=${balance:.2f}, Risk={risk_percent}%, SL Distance=${sl_distance_price:.{price_precision}f}, Entry=${entry_price:.{price_precision}f}"
    print(f"{Fore.CYAN}INFO:{Style.RESET_ALL} {calc_info}")
    logger.info(calc_info)
    return position_size_units


# --- Main Trading Class (Updated for Config and Dry Run) ---
class MomentumScannerTrader:
    def __init__(self, api_key, api_secret, config, dry_run=False):
        self.config = config
        self.dry_run = dry_run
        self.bybit_cfg = config.get('BYBIT_CONFIG', {})
        self.strategy_cfg = config.get('STRATEGY_CONFIG', {})
        self.risk_cfg = config.get('RISK_CONFIG', {})
        self.bot_cfg = config.get('BOT_CONFIG', {})
        self.internal_cfg = config.get('INTERNAL', {})

        self.symbol = self.bybit_cfg.get('SYMBOL', 'BTCUSDT')
        self.timeframe = self.bybit_cfg.get('TIMEFRAME', '15')
        self.leverage = self.bybit_cfg.get('LEVERAGE', 5)
        self.use_testnet = self.bybit_cfg.get('USE_TESTNET', True)

        self.min_order_qty = 0.0
        self.qty_precision = 3
        self.price_precision = 2
        self.last_exit_time = None

        if self.dry_run:
            dry_run_msg = "DRY RUN MODE ENABLED. No real orders will be placed."
            print(f"{Back.YELLOW}{Fore.BLACK}{Style.BRIGHT}***** {dry_run_msg} *****{Style.RESET_ALL}")
            logger.warning(dry_run_msg)

        try:
            self.session = HTTP(
                testnet=self.use_testnet,
                api_key=api_key,
                api_secret=api_secret,
            )
            init_msg = f"Bybit session initialized. Testnet: {self.use_testnet}"
            print(f"{Fore.GREEN}{Style.BRIGHT}INIT:{Style.RESET_ALL} {init_msg}")
            logger.info(init_msg)
            self._fetch_instrument_info() # Fetch info first
            if not self.dry_run: # Only setup leverage/margin if not dry run
                self._initial_setup()
            else:
                print(f"{Fore.YELLOW}DRY RUN:{Style.RESET_ALL} Skipping leverage/margin setup.")
                logger.info("Dry Run: Skipping leverage/margin setup.")

        except Exception as e:
            err_msg = f"Failed to initialize Bybit session: {e}"
            print(f"{Back.RED}{Fore.WHITE}{Style.BRIGHT}FATAL:{Style.RESET_ALL} {err_msg}")
            logger.error(err_msg, exc_info=True)
            sys.exit(1)

    def _initial_setup(self):
        """Set leverage and margin mode (only if not dry_run)."""
        if self.dry_run: return
        try:
            # Set Leverage
            self.session.set_leverage(
                category="linear", symbol=self.symbol,
                buyLeverage=str(self.leverage), sellLeverage=str(self.leverage)
            )
            lev_msg = f"Leverage for {self.symbol} set to {self.leverage}x"
            print(f"{Fore.CYAN}SETUP:{Style.RESET_ALL} {lev_msg}")
            logger.info(lev_msg)

            # Set Margin Mode (Isolated) - Best effort
            try:
                 pos = self.get_position(log_error=False)
                 if not pos or float(pos.get('size', 0)) == 0:
                     mode_resp = self.session.switch_margin_mode(category="linear", symbol=self.symbol, tradeMode=1) # 1 for Isolated
                     if mode_resp and mode_resp.get('retCode') == 0:
                         margin_msg = f"Margin mode for {self.symbol} set to ISOLATED."
                         print(f"{Fore.CYAN}SETUP:{Style.RESET_ALL} {margin_msg}")
                         logger.info(margin_msg)
                     else:
                         warn_msg = f"Could not switch margin mode (retCode={mode_resp.get('retCode')}, msg='{mode_resp.get('retMsg')}'). May already be Isolated or orders exist."
                         print(f"{Fore.YELLOW}SETUP WARN:{Style.RESET_ALL} {warn_msg}")
                         logger.warning(warn_msg)
                 else:
                     warn_msg = "Cannot switch margin mode while position is open. Keeping current mode."
                     print(f"{Fore.YELLOW}SETUP WARN:{Style.RESET_ALL} {warn_msg}")
                     logger.warning(warn_msg)
            except Exception as e:
                 warn_msg = f"Could not set margin mode (exception): {e}"
                 print(f"{Fore.YELLOW}SETUP WARN:{Style.RESET_ALL} {warn_msg}")
                 logger.warning(warn_msg)

        except Exception as e:
            err_msg = f"Error during initial setup (leverage/margin): {e}"
            print(f"{Fore.RED}SETUP ERROR:{Style.RESET_ALL} {err_msg}")
            logger.error(err_msg, exc_info=True)
            # Continue running, but log the error

    def _fetch_instrument_info(self):
       """Fetches instrument details (always runs)."""
       try:
           info = self.session.get_instruments_info(category="linear", symbol=self.symbol)
           # ... (rest of the function is identical, parsing min_order_qty, qty_precision, price_precision) ...
           if info and info['retCode'] == 0 and info['result']['list']:
                instrument = info['result']['list'][0]
                self.min_order_qty = float(instrument['lotSizeFilter']['minOrderQty'])
                qty_step_str = instrument['lotSizeFilter']['qtyStep']
                self.qty_precision = len(qty_step_str.split('.')[-1]) if '.' in qty_step_str else 0
                price_tick_str = instrument['priceFilter']['tickSize']
                self.price_precision = len(price_tick_str.split('.')[-1]) if '.' in price_tick_str else 0
                info_msg = f"Instrument info for {self.symbol}: Min Qty={self.min_order_qty}, Qty Precision={self.qty_precision}, Price Precision={self.price_precision}"
                print(f"{Fore.CYAN}SETUP:{Style.RESET_ALL} {info_msg}")
                logger.info(info_msg)
           else:
               # ... Handle error ...
                err_msg = f"Could not fetch instrument info: {info.get('retMsg') if info else 'No response'}. Using defaults (QtyPrec={self.qty_precision}, PricePrec={self.price_precision})."
                print(f"{Fore.RED}SETUP ERROR:{Style.RESET_ALL} {err_msg}")
                logger.error(err_msg)
       except Exception as e:
           # ... Handle exception ...
            err_msg = f"Exception fetching instrument info: {e}. Using defaults (QtyPrec={self.qty_precision}, PricePrec={self.price_precision})."
            print(f"{Fore.RED}SETUP ERROR:{Style.RESET_ALL} {err_msg}")
            logger.error(err_msg)


    def get_ohlcv(self):
        """Fetches OHLCV data and calculates indicators using config."""
        limit = self.internal_cfg.get('KLINE_LIMIT', 120)
        try:
            response = self.session.get_kline(
                category="linear", symbol=self.symbol, interval=self.timeframe, limit=limit
            )
            if response and response['retCode'] == 0:
                # ... (data processing identical to previous version) ...
                 df = pd.DataFrame(response['result']['list'], columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'turnover'])
                 # ... convert types, reverse order ...
                 df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                 df.set_index('timestamp', inplace=True)
                 df = df.iloc[::-1].copy()
                 for col in ['open', 'high', 'low', 'close', 'volume']: df[col] = pd.to_numeric(df[col])
                 # Pass strategy config to indicator calculation
                 df = calculate_indicators_momentum(df, self.strategy_cfg)
                 return df
            else:
                # ... handle API error ...
                 return None
        except Exception as e:
            # ... handle exception ...
             return None

    def get_position(self, log_error=True):
        """Gets current position information (simulates if dry_run)."""
        if self.dry_run:
            # In dry run, we don't have a real position. We could potentially simulate one
            # based on logged actions, but for simplicity, we'll assume no position.
            # A more advanced dry run would track simulated state.
            return None # Assume no position in dry run for now

        # If not dry run, proceed as before
        try:
            response = self.session.get_positions(category="linear", symbol=self.symbol)
            # ... (rest of the function is identical) ...
            if response and response['retCode'] == 0 and response['result']['list']:
                 # ... return position data or None ...
            elif response and response['retCode'] == 0:
                  return None
            else:
                 # ... log error if needed ...
                  return None
        except Exception as e:
             # ...log error if needed ...
              return None

    def place_order(self, side, qty, stop_loss_price, take_profit_price):
        """Places a market order with SL and TP, or simulates if dry_run."""
        sl_price_str = f"{stop_loss_price:.{self.price_precision}f}"
        tp_price_str = f"{take_profit_price:.{self.price_precision}f}"
        qty_str = f"{qty:.{self.qty_precision}f}"
        side_color = Fore.GREEN if side == "Buy" else Fore.RED
        log_prefix = f"{Fore.YELLOW}[DRY RUN]{Style.RESET_ALL} " if self.dry_run else ""
        log_msg = f"{log_prefix}Attempting to place {side_color}{side.upper()}{Style.RESET_ALL} order: Qty={qty_str}, SL={sl_price_str}, TP={tp_price_str}"

        print(f"{Fore.MAGENTA}{Style.BRIGHT}ACTION:{Style.RESET_ALL} {log_msg}")
        logger.info(f"{'DRY RUN: ' if self.dry_run else ''}Attempting {side} order: Qty={qty_str}, SL={sl_price_str}, TP={tp_price_str}")

        if self.dry_run:
            print(f"{Fore.YELLOW}DRY RUN:{Style.RESET_ALL} Simulated order placement.")
            logger.info("Dry Run: Simulated order placement.")
            return "dry_run_order_id" # Return a dummy ID for simulation consistency

        # --- Actual Order Placement (Not Dry Run) ---
        try:
            response = self.session.place_order(
                category="linear", symbol=self.symbol, side=side, orderType="Market",
                qty=qty_str, stopLoss=sl_price_str, takeProfit=tp_price_str,
                slTriggerBy="LastPrice", tpTriggerBy="LastPrice",
            )
            if response and response['retCode'] == 0:
                order_id = response['result'].get('orderId')
                success_msg = f"Market {side_color}{side.upper()}{Style.RESET_ALL} order placed successfully! Order ID: {order_id}"
                print(f"{Fore.GREEN}{Style.BRIGHT}SUCCESS:{Style.RESET_ALL} {success_msg}")
                logger.info(f"Market {side} order placed successfully! Order ID: {order_id}")
                return order_id
            else:
                # ... Handle API failure (identical error logging) ...
                return None
        except Exception as e:
            # ... Handle Exception (identical error logging) ...
             return None

    def close_position(self, position_data):
        """Closes the current position, or simulates if dry_run."""
        if not position_data and not self.dry_run: # No real position data needed for dry run simulation
            warn_msg = "Attempted to close position, but no position data provided."
            print(f"{Fore.YELLOW}WARN:{Style.RESET_ALL} {warn_msg}")
            logger.warning(warn_msg)
            return False

        # Simulate close if dry run
        if self.dry_run:
            # Need to know which side to simulate closing
            # For simplicity, let's assume we need context or log a generic close
            sim_side = "LONG" # Assume long for example, needs better state tracking for accuracy
            log_prefix = f"{Fore.YELLOW}[DRY RUN]{Style.RESET_ALL} "
            action_msg = f"{log_prefix}Attempting to close simulated {sim_side} position."
            print(f"{Fore.MAGENTA}{Style.BRIGHT}ACTION:{Style.RESET_ALL} {action_msg}")
            logger.info(f"DRY RUN: Simulating close of {sim_side} position.")
            self.last_exit_time = datetime.utcnow() # Still simulate cooldown start
            cooldown_end = self.last_exit_time + timedelta(seconds=self.bot_cfg.get('COOLDOWN_PERIOD_SECONDS', 300))
            cooldown_msg = f"Cooldown period started (simulated). Next entry possible after {cooldown_end.strftime('%Y-%m-%d %H:%M:%S')} UTC"
            print(f"{Fore.CYAN}INFO:{Style.RESET_ALL} {cooldown_msg}")
            logger.info(cooldown_msg)
            return True

        # --- Actual Position Close (Not Dry Run) ---
        side = position_data.get('side')
        size = float(position_data.get('size', 0))
        if size == 0: return False

        close_side = "Sell" if side == "Buy" else "Buy"
        qty_str = f"{size:.{self.qty_precision}f}"
        orig_side_color = Fore.GREEN if side == "Buy" else Fore.RED
        close_side_color = Fore.RED if close_side == "Sell" else Fore.GREEN
        action_msg = f"Attempting to close {orig_side_color}{side}{Style.RESET_ALL} position of size {size} with a {close_side_color}{close_side.upper()}{Style.RESET_ALL} market order."
        print(f"{Fore.MAGENTA}{Style.BRIGHT}ACTION:{Style.RESET_ALL} {action_msg}")
        logger.info(f"Attempting to close {side} position of size {size} with a {close_side} market order.")

        try:
            response = self.session.place_order(
                category="linear", symbol=self.symbol, side=close_side,
                orderType="Market", qty=qty_str, reduce_only=True
            )
            if response and response['retCode'] == 0:
                # ... Handle success (identical logging, set last_exit_time) ...
                 self.last_exit_time = datetime.utcnow()
                 # ... Log cooldown ...
                 return True
            else:
                # ... Handle API failure (idential logging) ...
                 return False
        except Exception as e:
            # ... Handle exception (idential logging) ...
             return False

    def run(self):
        """Main trading loop using config, dry run, and colors."""
        start_msg = f"Starting Momentum Scanner loop for {self.symbol} (TF: {self.timeframe}). Dry Run: {self.dry_run}"
        print(f"\n{Fore.BLUE}{Style.BRIGHT}===== {start_msg} ====={Style.RESET_ALL}")
        logger.info(start_msg)

        # Print key settings
        settings_msg = f"Mode: {self.strategy_cfg.get('SENSITIVITY_MODE')}, Early Signals: {self.strategy_cfg.get('ENABLE_EARLY_SIGNALS')}, Vol Confirm: {self.strategy_cfg.get('USE_VOLUME_CONFIRMATION')}, Regime Filter: {self.strategy_cfg.get('ENABLE_MARKET_REGIME_FILTER')}"
        print(f"{Fore.CYAN}CONFIG:{Style.RESET_ALL} {settings_msg}")
        logger.info(settings_msg)
        risk_msg = f"Risk: {self.risk_cfg.get('RISK_PER_TRADE_PERCENT')}%, SL: {self.risk_cfg.get('ATR_MULTIPLIER_SL')}*ATR, TP: {self.risk_cfg.get('ATR_MULTIPLIER_TP')}*ATR, Max Pos USDT: {self.risk_cfg.get('MAX_POSITION_USDT', 'N/A')}"
        print(f"{Fore.CYAN}CONFIG:{Style.RESET_ALL} {risk_msg}")
        logger.info(risk_msg)

        sleep_interval = self.bot_cfg.get('SLEEP_INTERVAL_SECONDS', 60)
        cooldown_seconds = self.bot_cfg.get('COOLDOWN_PERIOD_SECONDS', 300)

        while True:
            try:
                # --- Cooldown Check ---
                if self.last_exit_time and cooldown_seconds > 0:
                    cooldown_end_time = self.last_exit_time + timedelta(seconds=cooldown_seconds)
                    if datetime.utcnow() < cooldown_end_time:
                        # ... (Cooldown logic identical, uses cooldown_seconds) ...
                        wait_time = (cooldown_end_time - datetime.utcnow()).total_seconds()
                        cd_msg = f"In cooldown period. Waiting for {Fore.YELLOW}{wait_time:.0f}{Style.RESET_ALL} more seconds..."
                        print(f"{Fore.CYAN}STATUS:{Style.RESET_ALL} {cd_msg}")
                        logger.info(f"In cooldown period. Waiting for {wait_time:.0f} more seconds...")
                        sleep_duration = min(wait_time + 1, sleep_interval)
                        time.sleep(sleep_duration)
                        continue

                # 1. Get Data and Indicators
                print(f"\n{Fore.BLUE}CYCLE:{Style.RESET_ALL} Checking data/signals at {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC")
                df = self.get_ohlcv()
                if df is None or df.empty or len(df) < 2: # Need at least 2 rows for iloc[-2]
                     print(f"{Fore.YELLOW}WARN:{Style.RESET_ALL} No valid KLine data/indicators. Waiting...")
                     logger.warning("No valid KLine data/indicators. Waiting...")
                     time.sleep(sleep_interval)
                     continue

                # Use last fully closed candle
                last_closed = df.iloc[-2]

                # Check for NaNs in critical calculated columns
                critical_cols = ['atr_risk', 'ema_slow', 'long_signal', 'short_signal', 'exit_long_signal', 'exit_short_signal']
                if last_closed[critical_cols].isnull().any():
                     nan_cols = last_closed[critical_cols].index[last_closed[critical_cols].isnull()].tolist()
                     print(f"{Fore.YELLOW}{Style.BRIGHT}WARN:{Style.RESET_ALL} Indicators contain NaNs ({', '.join(nan_cols)}) on last closed candle. Waiting...")
                     logger.warning(f"Indicators contain NaNs ({', '.join(nan_cols)}) on last closed candle. Waiting...")
                     time.sleep(sleep_interval)
                     continue

                if pd.isna(last_closed['atr_risk']) or last_closed['atr_risk'] <= 0:
                    # ... (Handle invalid ATR identical) ...
                    time.sleep(sleep_interval)
                    continue

                # --- Market Regime Filter ---
                current_regime = 'neutral'
                if not pd.isna(last_closed['close']) and not pd.isna(last_closed['ema_slow']):
                    if last_closed['close'] > last_closed['ema_slow']:
                        current_regime = 'bull'
                    elif last_closed['close'] < last_closed['ema_slow']:
                        current_regime = 'bear'

                regime_filter_enabled = self.strategy_cfg.get('ENABLE_MARKET_REGIME_FILTER', False)
                if regime_filter_enabled:
                    regime_color = Fore.GREEN if current_regime == 'bull' else Fore.RED if current_regime == 'bear' else Fore.YELLOW
                    regime_msg = f"Market Regime ({self.symbol} {self.timeframe} TF): {regime_color}{current_regime.upper()}{Style.RESET_ALL} (based on Close vs EMA {self.strategy_cfg.get('SLOW_EMA_PERIOD')})"
                    print(f"{Fore.CYAN}INFO:{Style.RESET_ALL} {regime_msg}")
                    logger.info(f"Market Regime: {current_regime.upper()}")
                # --- End Market Regime ---


                # 2. Check Position and Balance
                position = self.get_position(log_error=True) # Log errors if getting real position fails
                balance = get_available_balance(self.session, "USDT") # Always check real balance
                if balance <= 0 and not self.dry_run: # Critical only if not dry run
                    crit_msg = "Available balance is zero or negative. Stopping."
                    print(f"{Back.RED}{Fore.WHITE}{Style.BRIGHT}CRITICAL:{Style.RESET_ALL} {crit_msg}")
                    logger.critical(crit_msg)
                    break
                elif balance <=0 and self.dry_run:
                    balance = 10000 # Assign dummy balance for dry run size calculation
                    print(f"{Fore.YELLOW}DRY RUN:{Style.RESET_ALL} Balance is zero, using dummy balance ${balance} for calculations.")


                # 3. Check Exit Conditions
                if position: # Only possible if not Dry Run (get_position returns None in dry run)
                    # ... (Position status printing identical) ...
                    pos_side = position.get('side')
                    # ...

                    # Check Strategy Exit Signal
                    exit_signal = False
                    if pos_side == "Buy" and last_closed['exit_long_signal']:
                        exit_reason = "Strategy exit long condition met."
                        print(f"{Fore.MAGENTA}SIGNAL:{Style.RESET_ALL} {exit_reason} Closing Long.")
                        logger.info(f"Exit Signal: {exit_reason}")
                        exit_signal = True
                    elif pos_side == "Sell" and last_closed['exit_short_signal']:
                        exit_reason = "Strategy exit short condition met."
                        print(f"{Fore.MAGENTA}SIGNAL:{Style.RESET_ALL} {exit_reason} Closing Short.")
                        logger.info(f"Exit Signal: {exit_reason}")
                        exit_signal = True

                    if exit_signal:
                        if self.close_position(position): # Will use actual close logic
                            print(f"{Fore.GREEN}INFO:{Style.RESET_ALL} Position closed based on strategy exit signal.")
                            logger.info("Position closed based on strategy exit signal.")
                        else:
                            print(f"{Fore.RED}{Style.BRIGHT}ERROR:{Style.RESET_ALL} Failed to close position on strategy exit signal.")
                            logger.error("Failed to close position on strategy exit signal.")
                        time.sleep(sleep_interval)
                        continue

                # 4. Check Entry Conditions
                elif not position: # No real position open (or always true in simple dry run)
                    status_prefix = f"{Fore.YELLOW}[DRY RUN]{Style.RESET_ALL} " if self.dry_run else ""
                    print(f"{Fore.CYAN}STATUS:{Style.RESET_ALL} {status_prefix}No position open. Checking for entry signals...")
                    logger.debug(f"{'DRY RUN: ' if self.dry_run else ''}No position open. Checking for entry signals...")

                    entry_price_approx = last_closed['close']
                    current_atr = last_closed['atr_risk']
                    risk_percent = self.risk_cfg.get('RISK_PER_TRADE_PERCENT', 1.0)
                    atr_sl_mult = self.risk_cfg.get('ATR_MULTIPLIER_SL', 1.5)
                    atr_tp_mult = self.risk_cfg.get('ATR_MULTIPLIER_TP', 2.0)
                    max_pos_usdt = self.risk_cfg.get('MAX_POSITION_USDT', 0)

                    # Check Long Entry
                    if last_closed['long_signal']:
                        signal_msg = f"LONG Entry Signal detected at ~${entry_price_approx:.{self.price_precision}f}, ATR={current_atr:.{self.price_precision}f}"
                        print(f"{Fore.GREEN}{Style.BRIGHT}SIGNAL:{Style.RESET_ALL} {signal_msg}")
                        logger.info(signal_msg)

                        # Apply Regime Filter if enabled
                        if regime_filter_enabled and current_regime != 'bull':
                            filter_msg = f"Regime Filter ACTIVE. Ignoring LONG signal due to non-bullish market regime ({current_regime.upper()})."
                            print(f"{Fore.YELLOW}FILTER:{Style.RESET_ALL} {filter_msg}")
                            logger.info(filter_msg)
                        else:
                            # Calculate SL/TP/Size
                            sl_distance = current_atr * atr_sl_mult
                            tp_distance = current_atr * atr_tp_mult
                            sl_price = entry_price_approx - sl_distance
                            tp_price = entry_price_approx + tp_distance

                            qty = calculate_position_size_atr(balance, risk_percent, sl_distance, entry_price_approx, self.min_order_qty, self.qty_precision, self.price_precision, max_pos_usdt)

                            if qty > 0:
                                self.place_order("Buy", qty, sl_price, tp_price) # Will simulate if dry_run is True
                            else:
                                warn_msg = "Could not enter long trade due to zero quantity calculation."
                                print(f"{Fore.YELLOW}{Style.BRIGHT}WARN:{Style.RESET_ALL} {warn_msg}")
                                logger.warning(warn_msg)
                            time.sleep(sleep_interval) # Pause after action/attempt
                            continue

                    # Check Short Entry
                    elif last_closed['short_signal']:
                        signal_msg = f"SHORT Entry Signal detected at ~${entry_price_approx:.{self.price_precision}f}, ATR={current_atr:.{self.price_precision}f}"
                        print(f"{Fore.RED}{Style.BRIGHT}SIGNAL:{Style.RESET_ALL} {signal_msg}")
                        logger.info(signal_msg)

                        # Apply Regime Filter if enabled
                        if regime_filter_enabled and current_regime != 'bear':
                            filter_msg = f"Regime Filter ACTIVE. Ignoring SHORT signal due to non-bearish market regime ({current_regime.upper()})."
                            print(f"{Fore.YELLOW}FILTER:{Style.RESET_ALL} {filter_msg}")
                            logger.info(filter_msg)
                        else:
                            # Calculate SL/TP/Size
                            sl_distance = current_atr * atr_sl_mult
                            tp_distance = current_atr * atr_tp_mult
                            sl_price = entry_price_approx + sl_distance
                            tp_price = entry_price_approx - tp_distance

                            qty = calculate_position_size_atr(balance, risk_percent, sl_distance, entry_price_approx, self.min_order_qty, self.qty_precision, self.price_precision, max_pos_usdt)

                            if qty > 0:
                                self.place_order("Sell", qty, sl_price, tp_price) # Will simulate if dry_run is True
                            else:
                                warn_msg = "Could not enter short trade due to zero quantity calculation."
                                print(f"{Fore.YELLOW}{Style.BRIGHT}WARN:{Style.RESET_ALL} {warn_msg}")
                                logger.warning(warn_msg)
                            time.sleep(sleep_interval) # Pause after action/attempt
                            continue

                    else: # No signal detected
                        print(f"{Fore.BLUE}INFO:{Style.RESET_ALL} No entry signal detected this cycle.")
                        logger.debug("No entry signal detected this cycle.")

                # 5. Wait
                wait_msg = f"Loop end. Waiting for {sleep_interval} seconds..."
                print(f"{Fore.BLUE}INFO:{Style.RESET_ALL} {wait_msg}")
                logger.debug(wait_msg)
                time.sleep(sleep_interval)

            except KeyboardInterrupt:
                stop_msg = "Keyboard interrupt detected. Shutting down..."
                print(f"\n{Fore.YELLOW}{Style.BRIGHT}STOP:{Style.RESET_ALL} {stop_msg}")
                logger.info(stop_msg)
                if not self.dry_run: # Only try to close real positions
                    position = self.get_position(log_error=False) # Don't spam errors on shutdown
                    if position:
                        close_msg = "Closing open position due to script termination..."
                        print(f"{Fore.YELLOW}STOP:{Style.RESET_ALL} {close_msg}")
                        logger.warning(close_msg)
                        self.close_position(position) # Use the existing close method
                final_msg = "Bot stopped."
                print(f"{Fore.BLUE}{Style.BRIGHT}===== {final_msg} ====={Style.RESET_ALL}")
                logger.info(final_msg)
                break
            except Exception as e:
                # ... (Identical main loop exception handling) ...
                exc_msg = f"An unexpected error occurred in the main loop: {e}"
                print(f"{Back.RED}{Fore.WHITE}{Style.BRIGHT}FATAL LOOP ERROR:{Style.RESET_ALL} {exc_msg}")
                logger.error(exc_msg, exc_info=True)
                pause_msg = f"Pausing for {sleep_interval*2} seconds after error..."
                print(f"{Fore.YELLOW}INFO:{Style.RESET_ALL} {pause_msg}")
                logger.info(pause_msg)
                time.sleep(sleep_interval * 2)


# --- Script Execution ---
if __name__ == "__main__":
    # --- Argument Parser ---
    parser = argparse.ArgumentParser(description="Bybit Momentum Scanner Trading Bot")
    parser.add_argument(
        "--config", type=str, default=CONFIG_FILE, help=f"Path to the configuration file (default: {CONFIG_FILE})"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Enable dry run mode (no orders placed)"
    )
    args = parser.parse_args()

    # --- Load Configuration ---
    config = load_config(args.config) # Exits if loading fails

    # --- Validate API Keys ---
    if not API_KEY or not API_SECRET:
        print(f"{Back.RED}{Fore.WHITE}{Style.BRIGHT}FATAL:{Style.RESET_ALL} API Key or Secret not found in environment variables (BYBIT_API_KEY, BYBIT_API_SECRET). Exiting.")
        logger.critical("API Key or Secret not found in environment variables. Exiting.")
        sys.exit(1)

    # --- Validate Critical Config Parameters ---
    # (Add more validation as needed for required fields)
    valid = True
    risk_cfg = config.get('RISK_CONFIG', {})
    if risk_cfg.get('RISK_PER_TRADE_PERCENT', -1) <= 0:
        print(f"{Fore.RED}CONFIG ERROR:{Style.RESET_ALL} RISK_PER_TRADE_PERCENT must be positive.")
        logger.error("RISK_PER_TRADE_PERCENT must be positive.")
        valid = False
    if risk_cfg.get('ATR_MULTIPLIER_SL', -1) <= 0:
        print(f"{Fore.RED}CONFIG ERROR:{Style.RESET_ALL} ATR_MULTIPLIER_SL must be positive.")
        logger.error("ATR_MULTIPLIER_SL must be positive.")
        valid = False
    if risk_cfg.get('ATR_MULTIPLIER_TP', -1) <= 0:
        print(f"{Fore.RED}CONFIG ERROR:{Style.RESET_ALL} ATR_MULTIPLIER_TP must be positive.")
        logger.error("ATR_MULTIPLIER_TP must be positive.")
        valid = False
    # Add checks for strategy params if needed

    if not valid:
        print(f"{Fore.RED}{Style.BRIGHT}Exiting due to critical configuration errors.{Style.RESET_ALL}")
        sys.exit(1)

    # --- Instantiate and Run Trader ---
    try:
        trader = MomentumScannerTrader(
            api_key=API_KEY,
            api_secret=API_SECRET,
            config=config,
            dry_run=args.dry_run
        )

        if trader.min_order_qty == 0.0:
             warn_msg = "Initial minimum order quantity is 0.0 (fetch might have failed). Check instrument on Bybit. Proceeding with caution."
             print(f"{Fore.YELLOW}{Style.BRIGHT}WARN:{Style.RESET_ALL} {warn_msg}")
             logger.warning(warn_msg)

        trader.run()

    except Exception as e: # Catch errors during trader initialization specifically
        init_err_msg = f"Failed to initialize or run the trader: {e}"
        print(f"{Back.RED}{Fore.WHITE}{Style.BRIGHT}FATAL:{Style.RESET_ALL} {init_err_msg}")
        logger.critical(init_err_msg, exc_info=True)
        sys.exit(1)

    final_msg = "Bot execution finished."
    print(f"{Fore.BLUE}{Style.BRIGHT}===== {final_msg} ====={Style.RESET_ALL}")
    logger.info(final_msg)
