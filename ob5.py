# -*- coding: utf-8 -*-
# ==============================================================================
# Pyrmethus's Arcane Market Analyzer v1.2
# Woven with the threads of ccxt, colored by Colorama's magic.
# Corrected error handling scope.
# ==============================================================================
import ccxt
import time
import os
import sys
import decimal
import subprocess # For Termux spells
from dotenv import load_dotenv
from colorama import init, Fore, Style, Back

# ==============================================================================
# Initialize Arcane Environment
# ==============================================================================
init(autoreset=True) # Let the colors flow freely!
decimal.getcontext().prec = 30 # Precision for potent calculations

# ==============================================================================
# Load Scrolls of Configuration (.env) & Define Constants
# ==============================================================================
load_dotenv()
print(f"{Fore.CYAN}{Style.DIM}# Loading ancient scrolls (.env)...{Style.RESET_ALL}")

CONFIG = {
    # --- API Secrets (Keep these guarded!) ---
    "API_KEY": os.environ.get("BYBIT_API_KEY"),
    "API_SECRET": os.environ.get("BYBIT_API_SECRET"),

    # --- Market & Order Book ---
    "VOLUME_THRESHOLDS": {'high': decimal.Decimal('10'), 'medium': decimal.Decimal('2')}, # Thresholds for highlighting volume
    "REFRESH_INTERVAL": 9, # Seconds between market whispers
    "MAX_ORDERBOOK_DEPTH_DISPLAY": 50, # How deep the order book vision extends
    "ORDER_FETCH_LIMIT": 200, # How many orders to summon from the book
    "DEFAULT_EXCHANGE_TYPE": 'linear', # 'linear' or 'inverse' contracts
    "CONNECT_TIMEOUT": 30000, # Milliseconds to wait for the exchange spirit
    "RETRY_DELAY_NETWORK_ERROR": 10, # Seconds to pause after network static
    "RETRY_DELAY_RATE_LIMIT": 60, # Seconds to pause when the exchange demands patience

    # --- Indicators Configuration ---
    "INDICATOR_TIMEFRAME": '15m', # Candle interval for most indicators
    "SMA_PERIOD": 9,
    "SMA2_PERIOD": 20,
    "EMA1_PERIOD": 12,
    "EMA2_PERIOD": 34,
    "MOMENTUM_PERIOD": 10,
    "RSI_PERIOD": 14,
    "STOCH_K_PERIOD": 14,
    "STOCH_D_PERIOD": 3,
    "STOCH_RSI_OVERSOLD": decimal.Decimal('20'),
    "STOCH_RSI_OVERBOUGHT": decimal.Decimal('80'),

    # --- Display & Formatting ---
    "PIVOT_TIMEFRAME": '30m', # Candle interval for pivot points (usually daily)
    "PNL_PRECISION": 2, # Decimal places for Profit/Loss
    "MIN_PRICE_DISPLAY_PRECISION": 3, # Minimum decimal places for prices
    "STOCH_RSI_DISPLAY_PRECISION": 3, # Decimal places for Stoch RSI values
    "VOLUME_DISPLAY_PRECISION": 0, # Decimal places for volume figures
    "BALANCE_DISPLAY_PRECISION": 2, # Decimal places for balance display

    # --- Trading ---
    "FETCH_BALANCE_ASSET": "USDT", # Asset to display balance for
}

# Fibonacci Ratios - Universal constants
FIB_RATIOS = {
    'r3': decimal.Decimal('1.000'), 'r2': decimal.Decimal('0.618'), 'r1': decimal.Decimal('0.382'),
    's1': decimal.Decimal('0.382'), 's2': decimal.Decimal('0.618'), 's3': decimal.Decimal('1.000'),
}

# ==============================================================================
# Utility Glyphs (Helper Functions)
# ==============================================================================

def print_color(text, color=Fore.WHITE, style=Style.NORMAL, end='\n', **kwargs):
    """Prints text imbued with Colorama's magic."""
    print(f"{style}{color}{text}{Style.RESET_ALL}", end=end, **kwargs)

def termux_toast(message, duration="short"):
    """Sends a fleeting message to the Termux ether."""
    try:
        # Ensure message is a string
        message_str = str(message)
        # Basic sanitization: remove potential command injection risks (simple approach)
        safe_message = ''.join(c for c in message_str if c.isalnum() or c in ' .,!?-:')[:100] # Limit length
        subprocess.run(['termux-toast', '-d', duration, safe_message], check=True, capture_output=True, timeout=5)
        # print_color(f"# Toast sent: {safe_message}", color=Fore.MAGENTA, style=Style.DIM) # Optional: uncomment for debugging toasts
    except FileNotFoundError:
        print_color("# termux-toast spell not found. Install termux-api?", color=Fore.YELLOW, style=Style.DIM)
    except subprocess.TimeoutExpired:
         print_color("# termux-toast spell timed out.", color=Fore.YELLOW, style=Style.DIM)
    except Exception as e:
        # Avoid printing the original potentially complex error object 'e' directly in the failure message
        print_color(f"# Toast failed (subprocess error).", color=Fore.YELLOW, style=Style.DIM)


def format_decimal(value, reported_precision, min_display_precision=None):
    """Formats decimal numbers with arcane precision for display."""
    if value is None: return "N/A"
    if not isinstance(value, decimal.Decimal):
        try: value = decimal.Decimal(str(value))
        except (decimal.InvalidOperation, TypeError, ValueError): return str(value) # Return as string if conversion fails

    try:
        # Determine the display precision, respecting minimum if provided
        display_precision = int(reported_precision)
        if min_display_precision is not None:
            display_precision = max(display_precision, int(min_display_precision))
        if display_precision < 0: display_precision = 0 # Precision cannot be negative

        # Create the quantizer based on display precision
        quantizer = decimal.Decimal('1') / (decimal.Decimal('10') ** display_precision)

        # Round the value
        rounded_value = value.quantize(quantizer, rounding=decimal.ROUND_HALF_UP)

        # Format as string
        if quantizer == 1: # Integer precision
             formatted_str = str(rounded_value.to_integral_value(rounding=decimal.ROUND_HALF_UP))
        else:
            formatted_str = str(rounded_value.normalize()) # normalize() removes trailing zeros

        # Ensure minimum decimal places are shown even if zero
        if '.' not in formatted_str and display_precision > 0:
            formatted_str += '.' + '0' * display_precision
        elif '.' in formatted_str:
            integer_part, decimal_part = formatted_str.split('.')
            if len(decimal_part) < display_precision:
                 formatted_str += '0' * (display_precision - len(decimal_part))

        return formatted_str
    except (ValueError, TypeError, decimal.InvalidOperation) as e:
        print_color(f"# FormatDecimalError ({value}): {e}", color=Fore.YELLOW, style=Style.DIM)
        return str(value) # Fallback to simple string conversion
    except Exception as e:
        print_color(f"# FormatDecimal Unexpected Error ({value}): {e}", color=Fore.YELLOW, style=Style.DIM)
        return str(value) # Fallback

def get_market_info(exchange, symbol):
    """Queries the exchange spirits for market runes (precision, limits)."""
    try:
        print_color(f"{Fore.CYAN}# Querying market runes for {symbol}...", style=Style.DIM, end='\r')
        # Ensure markets are loaded
        if not exchange.markets or symbol not in exchange.markets:
            print_color(f"{Fore.CYAN}# Summoning market list...", style=Style.DIM, end='\r')
            exchange.load_markets(True)
            sys.stdout.write("\033[K") # Clear line after loading

        market = exchange.market(symbol)
        sys.stdout.write("\033[K") # Clear "Querying..." message

        # Extract precision and limits, providing defaults
        price_prec_raw = market.get('precision', {}).get('price')
        amount_prec_raw = market.get('precision', {}).get('amount')
        min_amount_raw = market.get('limits', {}).get('amount', {}).get('min')

        # Safely convert to appropriate types using Decimal's log10 for precision digits
        try: price_prec = int(decimal.Decimal(str(price_prec_raw)).log10() * -1) if price_prec_raw is not None else 8
        except: price_prec = 8 # Default precision if conversion fails
        try: amount_prec = int(decimal.Decimal(str(amount_prec_raw)).log10() * -1) if amount_prec_raw is not None else 8
        except: amount_prec = 8 # Default precision
        try: min_amount = decimal.Decimal(str(min_amount_raw)) if min_amount_raw is not None else decimal.Decimal('0')
        except: min_amount = decimal.Decimal('0') # Default min amount

        # Calculate tick sizes (useful for some operations, though not heavily used here yet)
        price_tick_size = decimal.Decimal('1') / (decimal.Decimal('10') ** price_prec) if price_prec >= 0 else decimal.Decimal('1')
        amount_step = decimal.Decimal('1') / (decimal.Decimal('10') ** amount_prec) if amount_prec >= 0 else decimal.Decimal('1')


        return {
            'price_precision': price_prec,
            'amount_precision': amount_prec,
            'min_amount': min_amount,
            'price_tick_size': price_tick_size,
            'amount_step': amount_step,
            'symbol': symbol # Store symbol within info
        }
    except ccxt.BadSymbol:
        sys.stdout.write("\033[K")
        print_color(f"Symbol '{symbol}' is but an illusion (not found).", color=Fore.RED, style=Style.BRIGHT)
        return None
    except ccxt.NetworkError as e:
        sys.stdout.write("\033[K")
        print_color(f"Network ether disturbed (Market Info): {e}", color=Fore.YELLOW)
        return None # Indicate error but allow retry potentially
    except Exception as e:
        sys.stdout.write("\033[K")
        print_color(f"An arcane error occurred fetching market info: {e}", color=Fore.RED)
        return None # Indicate error

# ==============================================================================
# Indicator Incantations
# ==============================================================================

def calculate_sma(data, period):
    """Calculates the Simple Moving Average."""
    if not data or len(data) < period: return None
    try:
        decimal_data = [decimal.Decimal(str(p)) for p in data[-period:]]
        return sum(decimal_data) / decimal.Decimal(period)
    except (decimal.InvalidOperation, TypeError, ValueError) as e:
        print_color(f"# SMA Calc Error: {e}", color=Fore.YELLOW, style=Style.DIM); return None
    except Exception as e:
        print_color(f"# SMA Unexpected Calc Error: {e}", color=Fore.YELLOW, style=Style.DIM); return None

def calculate_ema(data, period):
    """Calculates the Exponential Moving Average."""
    if not data or len(data) < period: return None
    try:
        decimal_data = [decimal.Decimal(str(p)) for p in data]
        ema_values = [None] * len(decimal_data)
        multiplier = decimal.Decimal(2) / (decimal.Decimal(period) + 1)

        # Initial SMA for the first EMA value
        sma_initial = sum(decimal_data[:period]) / decimal.Decimal(period)
        ema_values[period - 1] = sma_initial

        # Calculate subsequent EMA values
        for i in range(period, len(decimal_data)):
            ema_values[i] = (decimal_data[i] - ema_values[i - 1]) * multiplier + ema_values[i - 1]

        # Return only the valid EMA values (from period-1 onwards)
        return [ema for ema in ema_values if ema is not None]
    except (decimal.InvalidOperation, TypeError, ValueError) as e:
        print_color(f"# EMA Calc Error: {e}", color=Fore.YELLOW, style=Style.DIM); return None
    except Exception as e:
        print_color(f"# EMA Unexpected Calc Error: {e}", color=Fore.YELLOW, style=Style.DIM); return None


def calculate_momentum(data, period):
    """Calculates the Momentum indicator."""
    if not data or len(data) <= period: return None # Need at least period+1 points
    try:
        current_price = decimal.Decimal(str(data[-1]))
        past_price = decimal.Decimal(str(data[-period -1])) # Price 'period' candles ago
        return current_price - past_price
    except (decimal.InvalidOperation, TypeError, ValueError) as e:
        print_color(f"# Momentum Calc Error: {e}", color=Fore.YELLOW, style=Style.DIM); return None
    except Exception as e:
        print_color(f"# Momentum Unexpected Calc Error: {e}", color=Fore.YELLOW, style=Style.DIM); return None


def calculate_fib_pivots(high, low, close):
    """Calculates Fibonacci Pivot Points for the *next* period."""
    if None in [high, low, close]: return None
    try:
        h, l, c = decimal.Decimal(str(high)), decimal.Decimal(str(low)), decimal.Decimal(str(close))
        if h <= 0 or l <= 0 or c <= 0 or h < l: return None # Basic validation

        pp = (h + l + c) / 3 # Pivot Point (PP)
        range_hl = h - l # High-Low Range

        return {
            'R3': pp + (range_hl * FIB_RATIOS['r3']), 'R2': pp + (range_hl * FIB_RATIOS['r2']),
            'R1': pp + (range_hl * FIB_RATIOS['r1']), 'PP': pp,
            'S1': pp - (range_hl * FIB_RATIOS['s1']), 'S2': pp - (range_hl * FIB_RATIOS['s2']),
            'S3': pp - (range_hl * FIB_RATIOS['s3'])
        }
    except (decimal.InvalidOperation, TypeError, ValueError) as e:
        print_color(f"# Pivot Calc Error: {e}", color=Fore.YELLOW, style=Style.DIM); return None
    except Exception as e:
        print_color(f"# Pivot Unexpected Calc Error: {e}", color=Fore.YELLOW, style=Style.DIM); return None

def calculate_rsi_manual(close_prices_list, period=14):
    """Calculates RSI manually using Wilder's Smoothing Method."""
    if not close_prices_list or len(close_prices_list) <= period:
        return None, "Insufficient data for RSI"

    prices = []
    for p in close_prices_list:
        try: prices.append(decimal.Decimal(str(p)))
        except (decimal.InvalidOperation, TypeError, ValueError): return None, "Invalid price data for RSI"

    deltas = [prices[i] - prices[i-1] for i in range(1, len(prices))]

    gains = [d if d > 0 else decimal.Decimal('0') for d in deltas]
    losses = [-d if d < 0 else decimal.Decimal('0') for d in deltas]

    if len(gains) < period: return None, "Insufficient deltas for RSI"

    # Calculate initial average gain/loss using SMA
    avg_gain = sum(gains[:period]) / decimal.Decimal(period)
    avg_loss = sum(losses[:period]) / decimal.Decimal(period)

    rsi_values = [decimal.Decimal('NaN')] * period # Pad with NaN for initial period

    # Calculate first RSI value
    if avg_loss == 0: rs = decimal.Decimal('inf') # Avoid division by zero
    else: rs = avg_gain / avg_loss
    first_rsi = 100 - (100 / (1 + rs)) if rs.is_finite() else decimal.Decimal('100')
    rsi_values.append(first_rsi)

    # Apply Wilder's Smoothing for subsequent values
    for i in range(period, len(gains)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / decimal.Decimal(period)
        avg_loss = (avg_loss * (period - 1) + losses[i]) / decimal.Decimal(period)

        if avg_loss == 0: rs = decimal.Decimal('inf')
        else: rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs)) if rs.is_finite() else decimal.Decimal('100')
        rsi_values.append(rsi)

    # Return only valid RSI values (excluding initial NaNs)
    return [r for r in rsi_values if not r.is_nan()], None

def calculate_stoch_rsi_manual(rsi_values, k_period=14, d_period=3):
    """Calculates Stochastic RSI %K and %D manually from RSI values."""
    if not rsi_values or len(rsi_values) < k_period:
        return None, None, "Insufficient RSI values for Stoch %K"

    valid_rsi = [r for r in rsi_values if r is not None and r.is_finite()]
    if len(valid_rsi) < k_period:
        return None, None, "Insufficient valid RSI values for Stoch %K"

    stoch_k_values = []
    for i in range(k_period - 1, len(valid_rsi)):
        rsi_window = valid_rsi[i - k_period + 1 : i + 1]
        current_rsi = rsi_window[-1]
        min_rsi = min(rsi_window)
        max_rsi = max(rsi_window)

        # Handle division by zero if max == min
        if max_rsi == min_rsi:
            stoch_k = decimal.Decimal('50') # Often considered neutral or 50 in this case
        else:
            stoch_k = ((current_rsi - min_rsi) / (max_rsi - min_rsi)) * 100
        stoch_k_values.append(stoch_k)

    if not stoch_k_values:
        return None, None, "Could not calculate any %K values"

    # Calculate %D (SMA of %K)
    if len(stoch_k_values) < d_period:
        # Return latest K if D cannot be calculated yet
        return stoch_k_values[-1], None, "Insufficient %K values for Stoch %D"

    stoch_d_values = []
    for i in range(d_period - 1, len(stoch_k_values)):
        k_window = stoch_k_values[i - d_period + 1 : i + 1]
        stoch_d = sum(k_window) / decimal.Decimal(d_period)
        stoch_d_values.append(stoch_d)

    latest_k = stoch_k_values[-1] if stoch_k_values else None
    latest_d = stoch_d_values[-1] if stoch_d_values else None

    return latest_k, latest_d, None # Success

# ==============================================================================
# Data Fetching Conjurations
# ==============================================================================

def fetch_market_data(exchange, symbol, config): # Takes lowercase config param
    """Fetches Ticker, OHLCV, Positions, and Balance data."""
    results = {"ticker": None, "indicator_ohlcv": None, "pivot_ohlcv": None, "positions": [], "balance": None}
    error_occurred = False
    # Access values using the 'config' parameter
    rate_limit_wait = config["RETRY_DELAY_RATE_LIMIT"]
    network_wait = config["RETRY_DELAY_NETWORK_ERROR"]

    # Determine max history needed based on all indicators
    indicator_history_needed = max(
        config['SMA_PERIOD'], config['SMA2_PERIOD'],
        config['EMA1_PERIOD'], config['EMA2_PERIOD'],
        config['MOMENTUM_PERIOD'] + 1, # Momentum needs one extra point
        config['RSI_PERIOD'] + config['STOCH_K_PERIOD'] + config['STOCH_D_PERIOD'] # Chain dependency
    ) + 5 # Add buffer

    api_calls = [
        {"func": exchange.fetch_ticker, "args": [symbol], "desc": "ticker"},
        {"func": exchange.fetch_ohlcv, "args": [symbol, config['INDICATOR_TIMEFRAME'], None, indicator_history_needed], "desc": "Indicator OHLCV"},
        {"func": exchange.fetch_ohlcv, "args": [symbol, config['PIVOT_TIMEFRAME'], None, 2], "desc": "Pivot OHLCV"}, # Need previous candle
        {"func": exchange.fetch_positions, "args": [[symbol]], "desc": "positions"},
        {"func": exchange.fetch_balance, "args": [], "desc": "balance"},
    ]

    print_color(f"{Fore.CYAN}# Contacting exchange spirits...", style=Style.DIM, end='\r')
    for call in api_calls:
        try:
            data = call["func"](*call["args"])
            if call["desc"] == "positions":
                # Filter positions for the relevant symbol and non-zero contracts
                results[call["desc"]] = [p for p in data if p.get('symbol') == symbol and p.get('contracts') is not None and decimal.Decimal(str(p['contracts'])) != 0]
            elif call["desc"] == "balance":
                 # Extract the specific asset balance
                 asset = config["FETCH_BALANCE_ASSET"] # Use config parameter
                 results[call["desc"]] = data.get('total', {}).get(asset) # Store total balance for the asset
            else:
                results[call["desc"]] = data
            time.sleep(exchange.rateLimit / 1000) # Respect basic rate limit

        except ccxt.RateLimitExceeded as e:
            print_color(f"Rate Limit ({call['desc']}). Pausing {rate_limit_wait}s.", color=Fore.YELLOW, style=Style.DIM)
            time.sleep(rate_limit_wait)
            error_occurred = True; break # Stop fetching this cycle
        except ccxt.NetworkError as e:
            print_color(f"Network Error ({call['desc']}). Pausing {network_wait}s.", color=Fore.YELLOW, style=Style.DIM)
            time.sleep(network_wait)
            error_occurred = True # Mark error, might retry next cycle
            # Don't break here, maybe other calls succeed
        except ccxt.AuthenticationError as e:
            print_color(f"Auth Error ({call['desc']}). Check API scrolls!", color=Fore.RED, style=Style.BRIGHT)
            error_occurred = True; raise e # Fatal error, stop the script
        except Exception as e:
            # Only show error for critical data, positions/balance failing is less critical for display
            if call['desc'] in ["ticker", "Indicator OHLCV", "Pivot OHLCV"]:
                 print_color(f"Error ({call['desc']}): {e}", color=Fore.RED, style=Style.DIM)
            results[call["desc"]] = None # Ensure failed calls result in None
            error_occurred = True # Mark error

    sys.stdout.write("\033[K") # Clear "Contacting..." message
    return results, error_occurred


def analyze_orderbook_volume(exchange, symbol, market_info, config): # Takes lowercase config param
    """Summons and analyzes the order book spirits."""
    print_color(f"{Fore.CYAN}# Summoning order book spirits...", style=Style.DIM, end='\r')
    try:
        # Access values using the 'config' parameter
        orderbook = exchange.fetch_order_book(symbol, limit=config["ORDER_FETCH_LIMIT"])
        sys.stdout.write("\033[K")
    except ccxt.RateLimitExceeded as e:
        sys.stdout.write("\033[K"); print_color(f"Rate Limit (OB). Pausing...", color=Fore.YELLOW); time.sleep(config["RETRY_DELAY_RATE_LIMIT"]); return None, True # Error, needs retry
    except ccxt.NetworkError as e:
        sys.stdout.write("\033[K"); print_color(f"Network Error (OB). Retrying...", color=Fore.YELLOW); time.sleep(config["RETRY_DELAY_NETWORK_ERROR"]); return None, True # Error, needs retry
    except ccxt.ExchangeError as e:
        sys.stdout.write("\033[K"); print_color(f"Exchange Error (OB): {e}", color=Fore.RED); return None, False # Error, likely won't resolve soon
    except Exception as e:
        sys.stdout.write("\033[K"); print_color(f"Unexpected Error (OB): {e}", color=Fore.RED); return None, False # Error

    if not orderbook or not orderbook.get('bids') or not orderbook.get('asks'):
        print_color(f"Order book for {symbol} is empty or unavailable.", color=Fore.YELLOW); return None, False # No data, but not necessarily an API error

    price_prec = market_info['price_precision']
    amount_prec = market_info['amount_precision']
    # Access values using the 'config' parameter
    vol_disp_prec = config["VOLUME_DISPLAY_PRECISION"]
    volume_thresholds = config["VOLUME_THRESHOLDS"]

    analyzed_orderbook = {
        'symbol': symbol, 'timestamp': exchange.iso8601(exchange.milliseconds()),
        'asks': [], 'bids': [],
        'ask_total_volume': decimal.Decimal('0'), 'bid_total_volume': decimal.Decimal('0'),
        'ask_weighted_price': decimal.Decimal('0'), 'bid_weighted_price': decimal.Decimal('0'),
        'volume_imbalance_ratio': decimal.Decimal('0'),
        'cumulative_ask_volume': decimal.Decimal('0'), # Within displayed depth
        'cumulative_bid_volume': decimal.Decimal('0'), # Within displayed depth
    }

    # --- Process Asks (Offers to Sell) ---
    ask_volume_times_price = decimal.Decimal('0')
    cumulative_ask_volume = decimal.Decimal('0')
    for i, ask in enumerate(orderbook['asks']):
        if i >= config["MAX_ORDERBOOK_DEPTH_DISPLAY"]: break # Use config parameter
        try:
            price, volume = decimal.Decimal(str(ask[0])), decimal.Decimal(str(ask[1]))
        except (decimal.InvalidOperation, TypeError, ValueError): continue # Skip malformed entry

        volume_str = format_decimal(volume, amount_prec, vol_disp_prec)
        cumulative_ask_volume += volume

        # Determine color based on volume thresholds
        highlight_color, highlight_style = Fore.WHITE, Style.NORMAL
        if volume >= volume_thresholds.get('high', decimal.Decimal('inf')):
            highlight_color, highlight_style = Fore.LIGHTRED_EX, Style.BRIGHT
        elif volume >= volume_thresholds.get('medium', decimal.Decimal('0')):
            highlight_color, highlight_style = Fore.RED, Style.NORMAL

        analyzed_orderbook['asks'].append({
            'price': price, 'volume': volume, 'volume_str': volume_str,
            'color': highlight_color, 'style': highlight_style,
            'cumulative_volume': format_decimal(cumulative_ask_volume, amount_prec, vol_disp_prec)
        })
        analyzed_orderbook['ask_total_volume'] += volume
        ask_volume_times_price += price * volume

    # --- Process Bids (Offers to Buy) ---
    bid_volume_times_price = decimal.Decimal('0')
    cumulative_bid_volume = decimal.Decimal('0')
    for i, bid in enumerate(orderbook['bids']):
        if i >= config["MAX_ORDERBOOK_DEPTH_DISPLAY"]: break # Use config parameter
        try:
            price, volume = decimal.Decimal(str(bid[0])), decimal.Decimal(str(bid[1]))
        except (decimal.InvalidOperation, TypeError, ValueError): continue # Skip malformed entry

        volume_str = format_decimal(volume, amount_prec, vol_disp_prec)
        cumulative_bid_volume += volume

        # Determine color based on volume thresholds
        highlight_color, highlight_style = Fore.WHITE, Style.NORMAL
        if volume >= volume_thresholds.get('high', decimal.Decimal('inf')):
            highlight_color, highlight_style = Fore.LIGHTGREEN_EX, Style.BRIGHT
        elif volume >= volume_thresholds.get('medium', decimal.Decimal('0')):
            highlight_color, highlight_style = Fore.GREEN, Style.NORMAL

        analyzed_orderbook['bids'].append({
            'price': price, 'volume': volume, 'volume_str': volume_str,
            'color': highlight_color, 'style': highlight_style,
            'cumulative_volume': format_decimal(cumulative_bid_volume, amount_prec, vol_disp_prec)
        })
        analyzed_orderbook['bid_total_volume'] += volume
        bid_volume_times_price += price * volume

    # Store cumulative volumes within displayed depth
    analyzed_orderbook['cumulative_ask_volume'] = cumulative_ask_volume
    analyzed_orderbook['cumulative_bid_volume'] = cumulative_bid_volume

    # --- Calculate VWAP and Imbalance ---
    ask_total_vol = analyzed_orderbook['ask_total_volume']
    bid_total_vol = analyzed_orderbook['bid_total_volume']

    if ask_total_vol > 0:
        analyzed_orderbook['ask_weighted_price'] = ask_volume_times_price / ask_total_vol
        # Calculate imbalance (Bid/Ask ratio)
        analyzed_orderbook['volume_imbalance_ratio'] = bid_total_vol / ask_total_vol
    else:
        analyzed_orderbook['ask_weighted_price'] = decimal.Decimal('0')
        # Handle zero ask volume case for imbalance
        analyzed_orderbook['volume_imbalance_ratio'] = (decimal.Decimal('inf') if bid_total_vol > 0 else decimal.Decimal('0'))

    if bid_total_vol > 0:
        analyzed_orderbook['bid_weighted_price'] = bid_volume_times_price / bid_total_vol
    else:
        analyzed_orderbook['bid_weighted_price'] = decimal.Decimal('0')

    return analyzed_orderbook, False # Success, no retry needed

# ==============================================================================
# Display Enchantments (Breaking down the monolith)
# ==============================================================================

def display_header(symbol, timestamp, balance_info, config): # Takes lowercase config param
    """Displays the main header and balance."""
    print_color("=" * 80, color=Fore.CYAN)
    print_color(f"📜 Pyrmethus's Market Vision: {Fore.MAGENTA}{Style.BRIGHT}{symbol}{Style.RESET_ALL} @ {Fore.YELLOW}{timestamp}", color=Fore.CYAN)

    balance_str = f"{Fore.YELLOW}N/A{Style.RESET_ALL}"
    # Access values using the 'config' parameter
    asset = config["FETCH_BALANCE_ASSET"]
    bal_disp_prec = config["BALANCE_DISPLAY_PRECISION"]
    if balance_info is not None:
        try:
            balance_val = decimal.Decimal(str(balance_info))
            balance_str = f"{Fore.GREEN}{format_decimal(balance_val, bal_disp_prec, bal_disp_prec)} {asset}{Style.RESET_ALL}"
        except (decimal.InvalidOperation, TypeError, ValueError):
            balance_str = f"{Fore.YELLOW}Error{Style.RESET_ALL}"

    print_color(f"💰 Available Balance ({asset}): {balance_str}")
    print_color("-" * 80, color=Fore.CYAN)


def display_ticker_and_trend(ticker_info, indicators_info, config, market_info): # Takes lowercase config param
    """Displays the last price and primary trend indicator (SMA1)."""
    price_prec = market_info['price_precision']
    # Access values using the 'config' parameter
    min_disp_prec = config["MIN_PRICE_DISPLAY_PRECISION"]

    last_price, current_price_str, price_color = None, f"{Fore.YELLOW}N/A{Style.RESET_ALL}", Fore.WHITE
    if ticker_info and ticker_info.get('last') is not None:
        try:
            last_price = decimal.Decimal(str(ticker_info['last']))
            current_price_str_fmt = format_decimal(last_price, price_prec, min_disp_prec)

            # Color based on primary SMA
            sma1 = indicators_info.get('sma1', {}).get('value')
            if sma1 is not None:
                if last_price > sma1: price_color = Fore.GREEN
                elif last_price < sma1: price_color = Fore.RED
                else: price_color = Fore.YELLOW
            current_price_str = f"{price_color}{Style.BRIGHT}{current_price_str_fmt}{Style.RESET_ALL}"
        except (decimal.InvalidOperation, TypeError, ValueError):
            current_price_str = f"{Fore.YELLOW}Error{Style.RESET_ALL}"
            last_price = None # Ensure last_price is None if formatting fails

    # --- Primary Trend (SMA1) ---
    sma1_val = indicators_info.get('sma1', {}).get('value')
    sma1_error = indicators_info.get('sma1', {}).get('error')
    # Access values using the 'config' parameter
    sma1_period = config['SMA_PERIOD']
    tf = config['INDICATOR_TIMEFRAME']
    trend_str, trend_color = f"Trend ({sma1_period}@{tf}): -", Fore.YELLOW

    if sma1_error:
        trend_str, trend_color = f"Trend ({sma1_period}@{tf}): SMA Error", Fore.YELLOW
    elif sma1_val is not None and last_price is not None:
        sma1_str_fmt = format_decimal(sma1_val, price_prec, min_disp_prec)
        if last_price > sma1_val: trend_str, trend_color = f"Above {sma1_period}@{tf} SMA ({sma1_str_fmt})", Fore.GREEN
        elif last_price < sma1_val: trend_str, trend_color = f"Below {sma1_period}@{tf} SMA ({sma1_str_fmt})", Fore.RED
        else: trend_str, trend_color = f"On {sma1_period}@{tf} SMA ({sma1_str_fmt})", Fore.YELLOW
    else:
        trend_str = f"Trend ({sma1_period}@{tf}): SMA unavailable"

    print_color(f"  Last Price: {current_price_str} | {trend_color}{trend_str}{Style.RESET_ALL}")
    return last_price # Return parsed last price for other displays


def display_indicators(indicators_info, config, market_info, last_price): # Takes lowercase config param
    """Displays SMA2, EMAs, Momentum, and Stoch RSI."""
    price_prec = market_info['price_precision']
    # Access values using the 'config' parameter
    min_disp_prec = config["MIN_PRICE_DISPLAY_PRECISION"]
    stoch_disp_prec = config["STOCH_RSI_DISPLAY_PRECISION"]
    tf = config['INDICATOR_TIMEFRAME']

    # --- Display Second SMA (SMA2) ---
    sma2_val = indicators_info.get('sma2', {}).get('value')
    sma2_error = indicators_info.get('sma2', {}).get('error')
    sma2_period = config['SMA2_PERIOD'] # Use config parameter
    sma2_str = f"{Fore.YELLOW}N/A{Style.RESET_ALL}"
    sma2_trend_str, sma2_trend_color = f"SMA ({sma2_period}@{tf}): -", Fore.YELLOW

    if sma2_error:
        sma2_trend_str, sma2_trend_color = f"SMA ({sma2_period}@{tf}): Error", Fore.YELLOW
    elif sma2_val is not None:
        sma2_str_fmt = format_decimal(sma2_val, price_prec, min_disp_prec)
        sma2_str = f"{Fore.YELLOW}{sma2_str_fmt}{Style.RESET_ALL}" # Value itself isn't colored by trend
        if last_price is not None:
            if last_price > sma2_val: sma2_trend_str, sma2_trend_color = f"Above {sma2_period}@{tf} SMA", Fore.GREEN
            elif last_price < sma2_val: sma2_trend_str, sma2_trend_color = f"Below {sma2_period}@{tf} SMA", Fore.RED
            else: sma2_trend_str, sma2_trend_color = f"On {sma2_period}@{tf} SMA", Fore.YELLOW
        else:
             sma2_trend_str, sma2_trend_color = f"SMA ({sma2_period}@{tf}): OK", Fore.WHITE # Have value, no price context
    else:
        sma2_trend_str = f"SMA ({sma2_period}@{tf}): unavailable"
    print_color(f"  SMA ({sma2_period}@{tf}): {sma2_str} | {sma2_trend_color}{sma2_trend_str}{Style.RESET_ALL}")

    # --- Display EMAs ---
    ema1_val = indicators_info.get('ema1', {}).get('value')
    ema2_val = indicators_info.get('ema2', {}).get('value')
    ema_error = indicators_info.get('ema1', {}).get('error') or indicators_info.get('ema2', {}).get('error') # Error if either failed
    ema1_period, ema2_period = config['EMA1_PERIOD'], config['EMA2_PERIOD'] # Use config parameter
    ema1_str, ema2_str = f"{Fore.YELLOW}N/A{Style.RESET_ALL}", f"{Fore.YELLOW}N/A{Style.RESET_ALL}"
    ema_trend_str, ema_trend_color, ema_trend_signal = f"EMA ({ema1_period}/{ema2_period}@{tf}): -", Fore.YELLOW, ""

    if ema_error:
         ema_trend_str, ema_trend_color = f"EMA ({ema1_period}/{ema2_period}@{tf}): Error", Fore.YELLOW
    elif ema1_val is not None and ema2_val is not None:
        ema1_str_fmt = format_decimal(ema1_val, price_prec, min_disp_prec)
        ema2_str_fmt = format_decimal(ema2_val, price_prec, min_disp_prec)
        ema1_str, ema2_str = f"{Fore.YELLOW}{ema1_str_fmt}{Style.RESET_ALL}", f"{Fore.YELLOW}{ema2_str_fmt}{Style.RESET_ALL}"
        if last_price is not None:
            if ema1_val > ema2_val and last_price > ema1_val: ema_trend_color, ema_trend_signal = Fore.GREEN, "(Bullish)"
            elif ema1_val < ema2_val and last_price < ema1_val: ema_trend_color, ema_trend_signal = Fore.RED, "(Bearish)"
            elif ema1_val > ema2_val: ema_trend_color, ema_trend_signal = Fore.LIGHTGREEN_EX, "(Neutral-Bullish)"
            elif ema1_val < ema2_val: ema_trend_color, ema_trend_signal = Fore.LIGHTRED_EX, "(Neutral-Bearish)"
            else: ema_trend_color, ema_trend_signal = Fore.YELLOW, "(Neutral)"
        else: # Have values, no price context
            if ema1_val > ema2_val: ema_trend_color, ema_trend_signal = Fore.LIGHTGREEN_EX, "(Crossed Up)"
            elif ema1_val < ema2_val: ema_trend_color, ema_trend_signal = Fore.LIGHTRED_EX, "(Crossed Down)"
            else: ema_trend_color, ema_trend_signal = Fore.YELLOW, "(Neutral)"
        ema_trend_str = f"EMA ({ema1_period}/{ema2_period}@{tf})"
    else:
        ema_trend_str = f"EMA ({ema1_period}/{ema2_period}@{tf}): unavailable"
    print_color(f"  EMAs: {ema1_str} ({ema1_period}), {ema2_str} ({ema2_period}) | {ema_trend_color}{ema_trend_str} {ema_trend_signal}{Style.RESET_ALL}")

    # --- Display Momentum ---
    mom_val = indicators_info.get('momentum', {}).get('value')
    mom_error = indicators_info.get('momentum', {}).get('error')
    mom_period = config['MOMENTUM_PERIOD'] # Use config parameter
    momentum_str = f"{Fore.YELLOW}N/A{Style.RESET_ALL}"
    momentum_trend_str, momentum_color = f"Momentum ({mom_period}@{tf}): -", Fore.YELLOW

    if mom_error:
        momentum_trend_str, momentum_color = f"Momentum ({mom_period}@{tf}): Error", Fore.YELLOW
    elif mom_val is not None:
        momentum_str_fmt = format_decimal(mom_val, price_prec, min_disp_prec)
        momentum_str = f"{Fore.YELLOW}{momentum_str_fmt}{Style.RESET_ALL}" # Value itself not colored
        if mom_val > 0: momentum_color, momentum_trend_str = Fore.GREEN, f"Momentum ({mom_period}@{tf}): Positive"
        elif mom_val < 0: momentum_color, momentum_trend_str = Fore.RED, f"Momentum ({mom_period}@{tf}): Negative"
        else: momentum_color, momentum_trend_str = Fore.YELLOW, f"Momentum ({mom_period}@{tf}): Neutral"
    else:
         momentum_trend_str = f"Momentum ({mom_period}@{tf}): unavailable"
    print_color(f"  {momentum_color}{momentum_trend_str}: {momentum_str}{Style.RESET_ALL}")

    # --- Display Stoch RSI ---
    stoch_k_val = indicators_info.get('stoch_rsi', {}).get('k')
    stoch_d_val = indicators_info.get('stoch_rsi', {}).get('d')
    stoch_error = indicators_info.get('stoch_rsi', {}).get('error')
    # Use config parameter
    rsi_p, k_p, d_p = config['RSI_PERIOD'], config['STOCH_K_PERIOD'], config['STOCH_D_PERIOD']
    stoch_k_str, stoch_d_str, stoch_color, stoch_signal = "N/A", "N/A", Fore.WHITE, ""

    if stoch_error:
        stoch_k_str, stoch_d_str = "Error", "Error"; stoch_color = Fore.YELLOW
        stoch_signal = f"({stoch_error})" if isinstance(stoch_error, str) else "(Error)"
    elif stoch_k_val is not None: # D might be None initially
        stoch_k_str = format_decimal(stoch_k_val, stoch_disp_prec)
        stoch_d_str = format_decimal(stoch_d_val, stoch_disp_prec) if stoch_d_val is not None else "N/A"

        # Use config parameter
        oversold_lim = config['STOCH_RSI_OVERSOLD']
        overbought_lim = config['STOCH_RSI_OVERBOUGHT']

        # Determine signal and color
        is_oversold = stoch_k_val < oversold_lim and (stoch_d_val is None or stoch_d_val < oversold_lim)
        is_overbought = stoch_k_val > overbought_lim and (stoch_d_val is None or stoch_d_val > overbought_lim)

        if is_oversold: stoch_color, stoch_signal = Fore.GREEN, "(Oversold)"
        elif is_overbought: stoch_color, stoch_signal = Fore.RED, "(Overbought)"
        elif stoch_d_val is not None: # Check cross only if D exists
            if stoch_k_val > stoch_d_val: stoch_color = Fore.LIGHTGREEN_EX
            elif stoch_k_val < stoch_d_val: stoch_color = Fore.LIGHTRED_EX
        else: # K exists but D doesn't, neutral color
             stoch_color = Fore.WHITE
    # else: K is None, default "N/A" strings are used

    print_color(f"  Stoch RSI ({rsi_p},{k_p},{d_p}@{tf}): {stoch_color}%K={stoch_k_str}, %D={stoch_d_str} {stoch_signal}{Style.RESET_ALL}")


def display_position(position_info, ticker_info, market_info, config): # Takes lowercase config param
    """Displays current position information."""
    # Use config parameter
    pnl_prec = config["PNL_PRECISION"]
    price_prec = market_info['price_precision']
    amount_prec = market_info['amount_precision']
    min_disp_prec = config["MIN_PRICE_DISPLAY_PRECISION"] # Use config parameter

    pnl_str = f"{Fore.LIGHTBLACK_EX}Position: None or Fetch Failed{Style.RESET_ALL}"
    if position_info.get('has_position'):
        pos = position_info['position']
        side = pos.get('side', 'N/A').capitalize()
        size_str = pos.get('contracts', '0')
        entry_price_str = pos.get('entryPrice', '0')
        quote_asset = pos.get('quoteAsset', 'USDT') # Assuming quote asset if available
        pnl_val = position_info.get('unrealizedPnl') # Already Decimal or None

        try:
            size = decimal.Decimal(size_str)
            entry_price = decimal.Decimal(entry_price_str)

            size_fmt = format_decimal(size, amount_prec)
            entry_fmt = format_decimal(entry_price, price_prec, min_disp_prec)

            side_color = Fore.GREEN if side.lower() == 'long' else Fore.RED if side.lower() == 'short' else Fore.WHITE

            # Calculate PNL if not provided by exchange or if it was None
            if pnl_val is None and ticker_info and ticker_info.get('last') is not None and entry_price > 0 and size != 0:
                last_price_for_pnl = decimal.Decimal(str(ticker_info['last']))
                if side.lower() == 'long':
                    pnl_val = (last_price_for_pnl - entry_price) * size
                else: # short
                    pnl_val = (entry_price - last_price_for_pnl) * size

            # Format PNL
            pnl_val_str = "N/A"
            pnl_color = Fore.WHITE
            if pnl_val is not None:
                 pnl_val_str = format_decimal(pnl_val, pnl_prec)
                 pnl_color = Fore.GREEN if pnl_val > 0 else Fore.RED if pnl_val < 0 else Fore.WHITE

            pnl_str = (f"Position: {side_color}{side} {size_fmt}{Style.RESET_ALL} | "
                       f"Entry: {Fore.YELLOW}{entry_fmt}{Style.RESET_ALL} | "
                       f"uPNL: {pnl_color}{pnl_val_str} {quote_asset}{Style.RESET_ALL}")

        except (decimal.InvalidOperation, TypeError, ValueError) as e:
            pnl_str = f"{Fore.YELLOW}Position: Error parsing data ({e}){Style.RESET_ALL}"

    print_color(f"  {pnl_str}")


def display_pivots(pivots_info, last_price, market_info, config): # Takes lowercase config param
    """Displays Fibonacci Pivot Points."""
    print_color("--- Fibonacci Pivots (Prev Day) ---", color=Fore.BLUE)
    price_prec = market_info['price_precision']
    min_disp_prec = config["MIN_PRICE_DISPLAY_PRECISION"] # Use config parameter
    pivot_width = max(10, price_prec + 6) # Dynamic width based on precision

    if pivots_info:
        pivot_lines = {}
        levels = ['R3', 'R2', 'R1', 'PP', 'S1', 'S2', 'S3']
        for level in levels:
            value = pivots_info.get(level)
            if value is not None:
                level_str = f"{level}:".ljust(4)
                value_str = format_decimal(value, price_prec, min_disp_prec)
                # Ensure value_str is aligned
                value_str_aligned = value_str.rjust(pivot_width)

                level_color = Fore.RED if 'R' in level else Fore.GREEN if 'S' in level else Fore.YELLOW
                highlight = ""
                # Highlight if price is near the pivot level
                if last_price:
                    try:
                        diff_ratio = abs(last_price - value) / last_price if last_price else decimal.Decimal('inf')
                        # Highlight if within 0.1%
                        if diff_ratio < decimal.Decimal('0.001'):
                            highlight = Back.LIGHTBLACK_EX + Fore.WHITE + Style.BRIGHT + " *NEAR* " + Style.RESET_ALL
                    except (decimal.InvalidOperation, TypeError, ValueError): pass # Ignore errors during highlight calc

                pivot_lines[level] = f"  {level_color}{level_str}{value_str_aligned}{Style.RESET_ALL}{highlight}"
            else:
                # Display N/A aligned
                pivot_lines[level] = f"  {level}:".ljust(6) + f"{'N/A':>{pivot_width}}"

        # Print pivots in a structured way
        print(f"{pivot_lines.get('R3','')}")
        print(f"{pivot_lines.get('R2','').ljust(pivot_width + 15)} {pivot_lines.get('S2','')}") # Adjust spacing as needed
        print(f"{pivot_lines.get('R1','').ljust(pivot_width + 15)} {pivot_lines.get('S1','')}")
        print(f"{pivot_lines.get('PP','')}")
        print(f"{pivot_lines.get('S3','')}")
    else:
        print_color(f"  {Fore.YELLOW}Pivot data unavailable.{Style.RESET_ALL}")


def display_orderbook(analyzed_orderbook, market_info, config): # Takes lowercase config param
    """Displays the analyzed order book."""
    print_color("--- Order Book Depths ---", color=Fore.BLUE)
    if not analyzed_orderbook: # Handle case where OB fetch failed completely
        print_color(f"  {Fore.YELLOW}Order book data unavailable.{Style.RESET_ALL}")
        return

    price_prec = market_info['price_precision']
    amount_prec = market_info['amount_precision']
    # Use config parameter
    min_disp_prec = config["MIN_PRICE_DISPLAY_PRECISION"]
    vol_disp_prec = config["VOLUME_DISPLAY_PRECISION"]

    # Dynamic column widths
    price_width = max(10, price_prec + 4)
    volume_width = max(12, vol_disp_prec + 6) # Width for individual volume
    cum_volume_width = max(15, vol_disp_prec + 10) # Width for cumulative volume display part

    ask_lines, bid_lines = [], []

    # Prepare Ask lines (reversed for typical display: lowest ask at bottom)
    for ask in reversed(analyzed_orderbook['asks']):
        price_str = format_decimal(ask['price'], price_prec, min_disp_prec)
        cum_vol_str = f"{Fore.LIGHTBLACK_EX}(Cum: {ask['cumulative_volume']}){Style.RESET_ALL}"
        ask_lines.append(
            f"{Style.NORMAL}{Fore.WHITE}{price_str:<{price_width}}"
            f"{ask['style']}{ask['color']}{ask['volume_str']:<{volume_width}} "
            f"{cum_vol_str}"
        )

    # Prepare Bid lines
    for bid in analyzed_orderbook['bids']:
        price_str = format_decimal(bid['price'], price_prec, min_disp_prec)
        cum_vol_str = f"{Fore.LIGHTBLACK_EX}(Cum: {bid['cumulative_volume']}){Style.RESET_ALL}"
        bid_lines.append(
            f"{Style.NORMAL}{Fore.WHITE}{price_str:<{price_width}}"
            f"{bid['style']}{bid['color']}{bid['volume_str']:<{volume_width}} "
            f"{cum_vol_str}"
        )

    # Print header
    col_width_total = price_width + volume_width + len("(Cum: )") + cum_volume_width # More accurate width estimate
    print_color(f"{'Asks':^{col_width_total}}{'Bids':^{col_width_total}}", color=Fore.LIGHTBLACK_EX)
    print_color(f"{'-'*col_width_total:<{col_width_total}} {'-'*col_width_total:<{col_width_total}}", color=Fore.LIGHTBLACK_EX)


    # Print rows side-by-side
    max_rows = max(len(ask_lines), len(bid_lines))
    for i in range(max_rows):
        # Estimate actual displayed length (crude, ignores color codes)
        ask_part = ask_lines[i] if i < len(ask_lines) else ''
        bid_part = bid_lines[i] if i < len(bid_lines) else ''
        # Pad dynamically based on estimated width (might not be perfect with colors)
        print(f"{ask_part:<{col_width_total}}  {bid_part}")


    # Calculate and display spread
    best_ask = analyzed_orderbook['asks'][-1]['price'] if analyzed_orderbook['asks'] else decimal.Decimal('NaN')
    best_bid = analyzed_orderbook['bids'][0]['price'] if analyzed_orderbook['bids'] else decimal.Decimal('NaN')
    spread = best_ask - best_bid if best_ask.is_finite() and best_bid.is_finite() else decimal.Decimal('NaN')
    spread_str = format_decimal(spread, price_prec, min_disp_prec) if spread.is_finite() else "N/A"
    print_color(f"\n--- Spread: {spread_str} ---", color=Fore.MAGENTA, style=Style.DIM)


def display_volume_analysis(analyzed_orderbook, market_info, config): # Takes lowercase config param
    """Displays summary volume analysis and pressure reading."""
    if not analyzed_orderbook: return # Nothing to display if OB failed

    amount_prec = market_info['amount_precision']
    price_prec = market_info['price_precision']
    # Use config parameter
    vol_disp_prec = config["VOLUME_DISPLAY_PRECISION"]
    min_disp_prec = config["MIN_PRICE_DISPLAY_PRECISION"]

    print_color("\n--- Volume Analysis (Displayed Depth) ---", color=Fore.BLUE)
    total_ask_vol = analyzed_orderbook['ask_total_volume']
    total_bid_vol = analyzed_orderbook['bid_total_volume']
    cum_ask_vol = analyzed_orderbook['cumulative_ask_volume'] # Already calculated for displayed depth
    cum_bid_vol = analyzed_orderbook['cumulative_bid_volume'] # Already calculated for displayed depth

    print_color(f"  Total Ask : {Fore.RED}{format_decimal(total_ask_vol, amount_prec, vol_disp_prec)}{Style.RESET_ALL} | "
                f"Total Bid : {Fore.GREEN}{format_decimal(total_bid_vol, amount_prec, vol_disp_prec)}{Style.RESET_ALL}")
    print_color(f"  Cumul. Ask: {Fore.RED}{format_decimal(cum_ask_vol, amount_prec, vol_disp_prec)}{Style.RESET_ALL} | "
                f"Cumul. Bid: {Fore.GREEN}{format_decimal(cum_bid_vol, amount_prec, vol_disp_prec)}{Style.RESET_ALL}")

    imbalance_ratio = analyzed_orderbook['volume_imbalance_ratio']
    imbalance_color = Fore.WHITE
    imbalance_str = "N/A"

    if imbalance_ratio.is_infinite():
        imbalance_color = Fore.LIGHTGREEN_EX; imbalance_str = "inf"
    elif imbalance_ratio.is_finite():
        imbalance_str = format_decimal(imbalance_ratio, 2) # Format to 2 decimal places
        if imbalance_ratio > decimal.Decimal('1.5'): imbalance_color = Fore.GREEN
        elif imbalance_ratio < decimal.Decimal('0.67') and not imbalance_ratio.is_zero(): imbalance_color = Fore.RED
        elif imbalance_ratio.is_zero() and analyzed_orderbook['ask_total_volume'] > 0: imbalance_color = Fore.LIGHTRED_EX # Zero bids, some asks
        # else: stays white for balanced ratio

    ask_vwap_str = format_decimal(analyzed_orderbook['ask_weighted_price'], price_prec, min_disp_prec)
    bid_vwap_str = format_decimal(analyzed_orderbook['bid_weighted_price'], price_prec, min_disp_prec)

    print_color(f"  Imbalance (B/A): {imbalance_color}{imbalance_str}{Style.RESET_ALL} | "
                f"Ask VWAP: {Fore.YELLOW}{ask_vwap_str}{Style.RESET_ALL} | "
                f"Bid VWAP: {Fore.YELLOW}{bid_vwap_str}{Style.RESET_ALL}")

    # --- Pressure Reading ---
    print_color("--- Pressure Reading ---", color=Fore.BLUE)
    if imbalance_ratio.is_infinite():
        print_color("  Extreme Bid Dominance (Infinite B/A)", color=Fore.LIGHTYELLOW_EX)
    elif imbalance_ratio.is_zero() and analyzed_orderbook['ask_total_volume'] > 0:
         print_color("  Extreme Ask Dominance (Zero B/A)", color=Fore.LIGHTYELLOW_EX)
    elif imbalance_ratio > decimal.Decimal('1.5'):
        print_color("  Strong Buy Pressure", color=Fore.GREEN, style=Style.BRIGHT)
    elif imbalance_ratio < decimal.Decimal('0.67'): # Already checked for non-zero
        print_color("  Strong Sell Pressure", color=Fore.RED, style=Style.BRIGHT)
    else: # Covers balanced and zero total volume cases
        print_color("  Volume Relatively Balanced", color=Fore.WHITE)

    print_color("=" * 80, color=Fore.CYAN)


def display_combined_analysis(analysis_data, market_info, config): # Takes lowercase config param
    """Orchestrates the display of all analyzed data."""
    # Extract data for clarity
    analyzed_orderbook = analysis_data['orderbook']
    ticker_info = analysis_data['ticker']
    indicators_info = analysis_data['indicators']
    position_info = analysis_data['position']
    pivots_info = analysis_data['pivots']
    balance_info = analysis_data['balance']
    timestamp = analysis_data.get('timestamp', 'N/A') # Use collected timestamp

    symbol = market_info['symbol']

    # --- Clear Screen (Optional, can be disruptive) ---
    # print("\033[H\033[J", end="") # Clears screen - use if preferred

    # --- Display Sections ---
    # Pass the 'config' parameter down consistently
    display_header(symbol, timestamp, balance_info, config)
    last_price = display_ticker_and_trend(ticker_info, indicators_info, config, market_info)
    display_indicators(indicators_info, config, market_info, last_price)
    display_position(position_info, ticker_info, market_info, config)
    display_pivots(pivots_info, last_price, market_info, config)
    display_orderbook(analyzed_orderbook, market_info, config) # Handles None case internally
    display_volume_analysis(analyzed_orderbook, market_info, config) # Handles None case internally


# ==============================================================================
# Trading Spells
# ==============================================================================

def place_market_order(exchange, symbol, side, amount_str, market_info): # Doesn't need config directly
    """Places a market order after casting confirmation charms."""
    print_color(f"{Fore.CYAN}# Preparing {side.upper()} market order spell...{Style.RESET_ALL}")
    try:
        amount = decimal.Decimal(amount_str)
        min_amount = market_info.get('min_amount', decimal.Decimal('0'))
        amount_prec = market_info['amount_precision']
        # Use amount_step if available for rounding, otherwise use precision
        amount_step = market_info.get('amount_step', decimal.Decimal('1') / (decimal.Decimal('10') ** amount_prec))

        if amount <= 0:
            print_color("Amount must be a positive number.", color=Fore.YELLOW); return

        # Check minimum amount
        if min_amount > 0 and amount < min_amount:
            min_amount_fmt = format_decimal(min_amount, amount_prec)
            print_color(f"Amount {amount_str} is below the minimum required ({min_amount_fmt}).", color=Fore.YELLOW); return

        # Check and potentially round based on amount step/precision
        if amount_step > 0 and (amount % amount_step) != 0:
            # Round down to the nearest valid step
            rounded_amount = (amount // amount_step) * amount_step
            rounded_amount_str = format_decimal(rounded_amount, amount_prec)

            # Ensure rounded amount still meets minimum
            if min_amount > 0 and rounded_amount < min_amount:
                print_color(f"Amount step is invalid. Rounding down to {rounded_amount_str} would be below minimum.", color=Fore.YELLOW); return

            round_confirm = input(f"{Fore.YELLOW}Amount step is invalid. Round down to {rounded_amount_str}? ({Fore.GREEN}yes{Fore.YELLOW}/{Fore.RED}no{Fore.YELLOW}): {Style.RESET_ALL}").strip().lower()
            if round_confirm == 'yes':
                amount, amount_str = rounded_amount, rounded_amount_str
                print_color(f"Using rounded amount: {amount_str}", color=Fore.CYAN)
            else:
                print_color("Order cancelled due to invalid amount step.", color=Fore.YELLOW); return
        # If no rounding needed, format the original amount for confirmation
        else:
             amount_str = format_decimal(amount, amount_prec)


    except (decimal.InvalidOperation, TypeError, ValueError) as e:
        print_color(f"Invalid amount entered: {e}", color=Fore.YELLOW); return
    except KeyError as e:
         print_color(f"Market info missing for validation: {e}", color=Fore.YELLOW); return

    # --- Confirmation ---
    side_color = Fore.GREEN if side == 'buy' else Fore.RED
    prompt = (f"{Style.BRIGHT}Confirm MARKET {side_color}{side.upper()}{Style.RESET_ALL} order: "
              f"{Fore.YELLOW}{amount_str}{Style.RESET_ALL} {Fore.MAGENTA}{symbol}{Style.RESET_ALL} "
              f"({Fore.GREEN}yes{Style.RESET_ALL}/{Fore.RED}no{Style.RESET_ALL}): {Style.RESET_ALL}")

    try:
        confirm = input(prompt).strip().lower()
    except (EOFError, KeyboardInterrupt):
        print_color("\nOrder cancelled by user.", color=Fore.YELLOW); return

    if confirm == 'yes':
        print_color(f"{Fore.CYAN}# Transmitting order to the exchange ether...", style=Style.DIM, end='\r')
        try:
            # CCXT expects float for amount, ensure conversion happens correctly
            amount_float = float(amount)
            # Bybit specific params (may need adjustment based on account mode - Hedge/One-Way)
            # For Hedge mode, you might need 'positionIdx': 1 for Buy/Long, 2 for Sell/Short
            # For One-Way mode, 'positionIdx': 0 or omit it. Assume One-Way for simplicity here.
            params = {
                # 'positionIdx': 0 # Uncomment or adjust if needed for Hedge Mode
            }

            order = exchange.create_market_order(symbol, side, amount_float, params=params)

            sys.stdout.write("\033[K") # Clear "Transmitting..." message
            order_id = order.get('id', 'N/A')
            avg_price = order.get('average')
            filled_amount = order.get('filled')

            confirmation_msg = f"✅ Market order {side.upper()} [{order_id}] placed!"
            details = []
            if filled_amount: details.append(f"Filled: {format_decimal(filled_amount, amount_prec)}")
            if avg_price: details.append(f"Avg Price: {format_decimal(avg_price, market_info['price_precision'])}")
            if details: confirmation_msg += f" ({', '.join(details)})"

            print_color(f"\n{confirmation_msg}", color=Fore.GREEN, style=Style.BRIGHT)
            termux_toast(f"{symbol} {side.upper()} Order Placed: {order_id}") # Send toast notification

        except ccxt.InsufficientFunds as e:
            sys.stdout.write("\033[K"); print_color(f"\n❌ Insufficient Funds: {e}", color=Fore.RED, style=Style.BRIGHT)
            termux_toast(f"{symbol} Order Failed: Insufficient Funds", duration="long")
        except ccxt.ExchangeError as e:
            sys.stdout.write("\033[K")
            # Check for common Bybit hedge mode error
            if "10001" in str(e) and ("position idx" in str(e).lower() or "position_idx" in str(e).lower()):
                print_color(f"\n❌ Exchange Error: {e}\n"
                            f"{Fore.YELLOW}Suggestion: Your account might be in HEDGE MODE. "
                            f"This script assumes ONE-WAY mode. Adjust 'params' in 'place_market_order' or change account settings.",
                            color=Fore.RED, style=Style.BRIGHT)
                termux_toast(f"{symbol} Order Failed: Hedge Mode Error?", duration="long")
            else:
                print_color(f"\n❌ Exchange Error: {e}", color=Fore.RED, style=Style.BRIGHT)
                termux_toast(f"{symbol} Order Failed: ExchangeError", duration="long") # Simplified toast
        except Exception as e:
            sys.stdout.write("\033[K"); print_color(f"\n❌ Order Placement Error: {e}", color=Fore.RED, style=Style.BRIGHT)
            termux_toast(f"{symbol} Order Failed: PlacementError", duration="long") # Simplified toast
    else:
        print_color("Order cancelled by user.", color=Fore.YELLOW)


# ==============================================================================
# Main Analysis Loop
# ==============================================================================

def run_analysis_cycle(exchange, symbol, market_info, config): # Takes lowercase config param
    """Performs one cycle of fetching, processing, and displaying data."""
    print_color(f"{Fore.CYAN}# Beginning analysis cycle...", style=Style.DIM, end='\r')

    # --- Fetch Data ---
    # Pass the 'config' parameter correctly
    fetched_data, data_error = fetch_market_data(exchange, symbol, config)
    analyzed_orderbook, orderbook_error = analyze_orderbook_volume(exchange, symbol, market_info, config)
    sys.stdout.write("\033[K") # Clear status message

    if data_error and not any(fetched_data.values()): # If all critical fetches failed
         print_color(f"{Fore.YELLOW}Critical data fetch failed, skipping analysis this cycle.{Style.RESET_ALL}")
         return False # Indicate cycle failed, allows main loop to decide on waiting

    # --- Process Indicators ---
    print_color(f"{Fore.CYAN}# Weaving indicator patterns...", style=Style.DIM, end='\r')
    indicators_info = { # Store value and error status separately
        'sma1': {'value': None, 'error': False},
        'sma2': {'value': None, 'error': False},
        'ema1': {'value': None, 'error': False},
        'ema2': {'value': None, 'error': False},
        'momentum': {'value': None, 'error': False},
        'stoch_rsi': {'k': None, 'd': None, 'error': None} # Store k, d, and potential error message
    }
    indicator_ohlcv = fetched_data.get("indicator_ohlcv")
    if indicator_ohlcv:
        close_prices = [candle[4] for candle in indicator_ohlcv] # Index 4 is close price

        # SMA1 - Use config parameter
        sma1 = calculate_sma(close_prices, config["SMA_PERIOD"])
        if sma1 is not None: indicators_info['sma1']['value'] = sma1
        else: indicators_info['sma1']['error'] = True

        # SMA2 - Use config parameter
        sma2 = calculate_sma(close_prices, config["SMA2_PERIOD"])
        if sma2 is not None: indicators_info['sma2']['value'] = sma2
        else: indicators_info['sma2']['error'] = True

        # EMAs - Use config parameter
        ema1_values = calculate_ema(close_prices, config["EMA1_PERIOD"])
        ema2_values = calculate_ema(close_prices, config["EMA2_PERIOD"])
        if ema1_values: indicators_info['ema1']['value'] = ema1_values[-1]
        else: indicators_info['ema1']['error'] = True
        if ema2_values: indicators_info['ema2']['value'] = ema2_values[-1]
        else: indicators_info['ema2']['error'] = True # Mark error if calculation failed

        # Momentum - Use config parameter
        momentum = calculate_momentum(close_prices, config["MOMENTUM_PERIOD"])
        if momentum is not None: indicators_info['momentum']['value'] = momentum
        else: indicators_info['momentum']['error'] = True

        # Stoch RSI - Use config parameter
        rsi_list, rsi_error = calculate_rsi_manual(close_prices, config["RSI_PERIOD"])
        if rsi_error:
            indicators_info['stoch_rsi']['error'] = f"RSI Error: {rsi_error}"
        elif rsi_list:
            stoch_k, stoch_d, stoch_err = calculate_stoch_rsi_manual(rsi_list, config["STOCH_K_PERIOD"], config["STOCH_D_PERIOD"])
            indicators_info['stoch_rsi']['k'] = stoch_k # Can be None if not enough data for K
            indicators_info['stoch_rsi']['d'] = stoch_d # Can be None if not enough data for D
            if stoch_err: indicators_info['stoch_rsi']['error'] = stoch_err # Store error message
        else: # Should not happen if rsi_error is None, but safety check
             indicators_info['stoch_rsi']['error'] = "RSI list empty"

    else: # OHLCV data missing
        indicators_info['sma1']['error'] = True; indicators_info['sma2']['error'] = True
        indicators_info['ema1']['error'] = True; indicators_info['ema2']['error'] = True
        indicators_info['momentum']['error'] = True; indicators_info['stoch_rsi']['error'] = "OHLCV data missing"

    # --- Process Pivots ---
    pivots_info = None
    pivot_ohlcv = fetched_data.get("pivot_ohlcv")
    if pivot_ohlcv and len(pivot_ohlcv) > 0:
        # Use the *first* candle fetched for the pivot timeframe (should be previous day/period)
        prev_day_candle = pivot_ohlcv[0]
        # Indices: 0: time, 1: open, 2: high, 3: low, 4: close, 5: volume
        p_high, p_low, p_close = prev_day_candle[2], prev_day_candle[3], prev_day_candle[4]
        pivots_info = calculate_fib_pivots(p_high, p_low, p_close)


    # --- Process Positions ---
    position_info = {'has_position': False, 'position': None, 'unrealizedPnl': None}
    position_data = fetched_data.get("positions") # Already filtered in fetch
    if position_data:
        current_pos = position_data[0] # Assume only one position per symbol for linear/inverse
        position_info['has_position'] = True
        position_info['position'] = current_pos
        try:
            # Try to get PNL directly from exchange data first
            pnl_raw = current_pos.get('unrealizedPnl')
            if pnl_raw is not None:
                position_info['unrealizedPnl'] = decimal.Decimal(str(pnl_raw))
            # else: PNL will be calculated later in display if needed and possible
        except (decimal.InvalidOperation, TypeError, ValueError) as pnl_e:
            print_color(f"# Warning: PNL parsing error: {pnl_e}", color=Fore.YELLOW, style=Style.DIM)
            position_info['unrealizedPnl'] = None # Mark as unavailable if parsing failed

    sys.stdout.write("\033[K") # Clear indicator message

    # --- Aggregate Data for Display ---
    analysis_data = {
        'ticker': fetched_data.get('ticker'),
        'indicators': indicators_info,
        'pivots': pivots_info,
        'position': position_info,
        'balance': fetched_data.get('balance'),
        'orderbook': analyzed_orderbook, # Can be None
        'timestamp': analyzed_orderbook['timestamp'] if analyzed_orderbook else (fetched_data.get('ticker', {}).get('iso8601', 'N/A'))
    }

    # --- Display ---
    # Pass the 'config' parameter correctly
    display_combined_analysis(analysis_data, market_info, config)

    # Return True if the cycle involved successful data fetches (even if OB failed)
    # Return False only if critical data fetch failed entirely
    return not data_error or any(fetched_data.values())


def main():
    """The main summoning ritual."""
    # Use global CONFIG here
    print_color("*" * 80, color=Fore.RED, style=Style.BRIGHT)
    print_color("   🔥 Pyrmethus's Termux Market Analyzer Activated 🔥", color=Fore.RED, style=Style.BRIGHT)
    print_color("   Use with wisdom. Market forces are potent and volatile.", color=Fore.YELLOW)
    print_color("   MARKET ORDERS CARRY SLIPPAGE RISK. YOU ARE THE MASTER OF YOUR FATE.", color=Fore.YELLOW)
    print_color("*" * 80, color=Fore.RED, style=Style.BRIGHT); print("\n")

    if not CONFIG["API_KEY"] or not CONFIG["API_SECRET"]:
        print_color("API Key/Secret scrolls are missing or unreadable in .env.", color=Fore.RED, style=Style.BRIGHT)
        print_color("Ensure .env file exists and contains BYBIT_API_KEY and BYBIT_API_SECRET.", color=Fore.YELLOW)
        return

    print_color(f"{Fore.CYAN}# Binding to Bybit ({CONFIG['DEFAULT_EXCHANGE_TYPE']}) exchange spirit...{Style.RESET_ALL}", style=Style.DIM)
    try:
        exchange = ccxt.bybit({
            'apiKey': CONFIG["API_KEY"],
            'secret': CONFIG["API_SECRET"],
            'options': {
                'defaultType': CONFIG["DEFAULT_EXCHANGE_TYPE"],
                'adjustForTimeDifference': True, # Crucial for signature timing
            },
            'timeout': CONFIG["CONNECT_TIMEOUT"],
            'enableRateLimit': True # Let ccxt handle basic rate limiting
        })
        # Test connection with a lightweight call
        print_color(f"{Fore.CYAN}# Testing connection conduit...{Style.RESET_ALL}", style=Style.DIM, end='\r')
        exchange.fetch_time()
        sys.stdout.write("\033[K")
        print_color("Connection established. Exchange spirit is responsive.", color=Fore.GREEN)
    except ccxt.AuthenticationError:
         sys.stdout.write("\033[K")
         print_color("Authentication failed! Check your API Key and Secret.", color=Fore.RED, style=Style.BRIGHT)
         return
    except Exception as e:
        sys.stdout.write("\033[K")
        print_color(f"Failed to connect to the exchange spirit: {e}", color=Fore.RED, style=Style.BRIGHT)
        return

    # --- Symbol Selection ---
    symbol, market_info = "", None
    while not market_info:
        try:
            symbol_input = input(f"{Style.BRIGHT}{Fore.BLUE}Which Bybit Futures market shall we observe? (e.g., BTCUSDT): {Style.RESET_ALL}").strip().upper()
            if not symbol_input: continue
            market_info = get_market_info(exchange, symbol_input) # Handles errors internally
            if market_info:
                symbol = symbol_input # Confirmed valid symbol
                print_color(f"Focusing vision on: {Fore.MAGENTA}{symbol}{Fore.CYAN} | "
                            f"Price Precision: {market_info['price_precision']}, Amount Precision: {market_info['amount_precision']}", color=Fore.CYAN)
        except (EOFError, KeyboardInterrupt):
            print_color("\nSummoning interrupted by user.", color=Fore.YELLOW); return
        except Exception as e: # Catch unexpected errors during input/validation
            print_color(f"An unexpected disturbance in symbol selection: {e}", color=Fore.RED)
            time.sleep(1) # Pause before retrying

    print_color(f"\n{Fore.CYAN}Starting continuous analysis for {symbol}. Press Ctrl+C to banish.{Style.RESET_ALL}")
    last_critical_error_msg = ""

    # --- Main Loop ---
    while True:
        cycle_successful = False
        try:
            # Pass the global CONFIG (uppercase) to the cycle function
            cycle_successful = run_analysis_cycle(exchange, symbol, market_info, CONFIG)

            # --- Action Prompt ---
            if cycle_successful: # Only prompt if the analysis cycle provided some data
                action = input(f"\n{Style.BRIGHT}{Fore.BLUE}Action ({Fore.CYAN}refresh{Fore.BLUE}/{Fore.GREEN}buy{Fore.BLUE}/{Fore.RED}sell{Fore.BLUE}/{Fore.YELLOW}exit{Fore.BLUE}): {Style.RESET_ALL}").strip().lower()
                if action == 'buy':
                    qty_str = input(f"{Style.BRIGHT}{Fore.GREEN}BUY Quantity: {Style.RESET_ALL}").strip()
                    place_market_order(exchange, symbol, 'buy', qty_str, market_info)
                    # Pause slightly after order attempt to allow viewing result
                    time.sleep(2)
                elif action == 'sell':
                    qty_str = input(f"{Style.BRIGHT}{Fore.RED}SELL Quantity: {Style.RESET_ALL}").strip()
                    place_market_order(exchange, symbol, 'sell', qty_str, market_info)
                    time.sleep(2)
                elif action == 'refresh' or action == '': # Enter key also refreshes
                    print_color("Refreshing...", color=Fore.CYAN, style=Style.DIM)
                    # No sleep here, loop will continue shortly below
                elif action == 'exit':
                    print_color("Dispelling the vision...", color=Fore.YELLOW); break
                else:
                    print_color("Unknown command whispered.", color=Fore.YELLOW)
                    time.sleep(1) # Brief pause for invalid command

                # Use global CONFIG here for refresh interval
                print_color(f"{Fore.CYAN}# Pausing for {CONFIG['REFRESH_INTERVAL']}s before next cycle...{Style.RESET_ALL}", style=Style.DIM, end='\r')
                time.sleep(CONFIG["REFRESH_INTERVAL"])
                sys.stdout.write("\033[K")


            else: # Cycle failed (likely critical data error)
                print_color("Waiting due to data fetch errors...", color=Fore.YELLOW, style=Style.DIM)
                # Use global CONFIG here for retry delay
                time.sleep(CONFIG["RETRY_DELAY_NETWORK_ERROR"]) # Wait longer if fetches failed

        except KeyboardInterrupt:
            print_color("\nBanished by user input.", color=Fore.YELLOW); break
        except ccxt.AuthenticationError: # Catch auth errors possibly missed earlier
             print_color("\nAuthentication Error! Halting the ritual.", color=Fore.RED, style=Style.BRIGHT)
             break
        except Exception as e:
            # This is the handler that was likely causing the issue
            current_error_msg = str(e)
            print_color(".", color=Fore.RED, end='') # Indicate an error occurred
            # Only print the full error if it's different from the last one to avoid spam
            if current_error_msg != last_critical_error_msg:
                print_color(f"\nCritical Loop Error: {e}", color=Fore.RED, style=Style.BRIGHT)
                # Also send a toast for critical errors
                termux_toast(f"Pyrmethus Error: {current_error_msg}", duration="long") # Pass the error string
                last_critical_error_msg = current_error_msg
            # Wait longer after a critical error
            # !!! CRITICAL FIX: Use uppercase CONFIG here !!!
            try:
                time.sleep(CONFIG["REFRESH_INTERVAL"] * 2)
            except NameError:
                # Fallback if CONFIG is somehow undefined (shouldn't happen normally)
                print_color("\nFATAL: CONFIG not defined in critical error handler!", color=Fore.RED, style=Style.BRIGHT)
                time.sleep(20) # Default long sleep
            except KeyError:
                 # Fallback if REFRESH_INTERVAL is missing
                print_color("\nFATAL: CONFIG['REFRESH_INTERVAL'] missing in critical error handler!", color=Fore.RED, style=Style.BRIGHT)
                time.sleep(20) # Default long sleep


if __name__ == '__main__':
    try:
        main()
    finally:
        # Final farewell message
        print_color("\nWizard Pyrmethus departs. May your analysis illuminate the path!", color=Fore.MAGENTA, style=Style.BRIGHT)
