# -*- coding: utf-8 -*-
import ccxt
import time
import os
import sys
from dotenv import load_dotenv
from colorama import init, Fore, Style, Back
import decimal # Use decimal for precise price/amount calculations
import numpy as np # For SMA calculation
import pandas as pd # For RSI / Stoch RSI calculation

# ==============================================================================
# Initialize Colorama & Decimal Context
# ==============================================================================
init(autoreset=True)
decimal.getcontext().prec = 28 # Set precision for Decimal calculations

# ==============================================================================
# Load Environment Variables & Configuration
# ==============================================================================
load_dotenv()

CONFIG = {
    "API_KEY": os.environ.get("BYBIT_API_KEY"),
    "API_SECRET": os.environ.get("BYBIT_API_SECRET"),
    "VOLUME_THRESHOLDS": {'high': 10, 'medium': 2}, # Adjust based on symbol's base currency value
    "REFRESH_INTERVAL": 9, # Increased slightly for more API calls + calculations
    "MAX_ORDERBOOK_DEPTH_DISPLAY": 10,
    "ORDER_FETCH_LIMIT": 50,
    "DEFAULT_EXCHANGE_TYPE": 'linear',
    "CONNECT_TIMEOUT": 30000,
    "RETRY_DELAY_NETWORK_ERROR": 10,
    "RETRY_DELAY_RATE_LIMIT": 60,
    # Indicators Config
    "SMA_PERIOD": 20,
    "INDICATOR_TIMEFRAME": '5m', # Timeframe for SMA, RSI, StochRSI
    "RSI_PERIOD": 14,
    "STOCH_K_PERIOD": 14, # Stochastic %K period (applied to RSI)
    "STOCH_D_PERIOD": 3,  # Stochastic %D period (SMA of %K)
    "STOCH_RSI_OVERSOLD": 20,
    "STOCH_RSI_OVERBOUGHT": 80,
    # Display Config
    "PIVOT_TIMEFRAME": '1d',
    "PNL_PRECISION": 2, # Decimal places for PNL display (quote currency)
    "MIN_PRICE_DISPLAY_PRECISION": 2, # Minimum decimals for displaying prices/pivots
    "STOCH_RSI_DISPLAY_PRECISION": 2, # Decimals for Stoch RSI K/D display
}

# Fibonacci Ratios as Decimals
FIB_RATIOS = {
    'r3': decimal.Decimal('1.000'), 'r2': decimal.Decimal('0.618'), 'r1': decimal.Decimal('0.382'),
    's1': decimal.Decimal('0.382'), 's2': decimal.Decimal('0.618'), 's3': decimal.Decimal('1.000'),
}

# ==============================================================================
# Helper Functions - Arcane Utilities
# ==============================================================================

def print_color(text, color=Fore.WHITE, style=Style.NORMAL, end='\n', **kwargs):
    """Helper to print colored text."""
    print(f"{style}{color}{text}{Style.RESET_ALL}", end=end, **kwargs)

# --- ENHANCED format_decimal with min_display_precision ---
def format_decimal(value, reported_precision, min_display_precision=None):
    """
    Formats a decimal value to string, respecting reported precision
    but ensuring a minimum display precision if specified.
    """
    if value is None: return "N/A"
    if not isinstance(value, decimal.Decimal):
        try: value = decimal.Decimal(str(value))
        except (decimal.InvalidOperation, TypeError, ValueError): return str(value)

    try:
        # Determine the effective precision for display
        display_precision = int(reported_precision)
        if min_display_precision is not None:
             display_precision = max(display_precision, int(min_display_precision))

        if display_precision < 0: display_precision = 0

        # Use quantize for rounding
        quantizer = decimal.Decimal('1') / (decimal.Decimal('10') ** display_precision)
        rounded_value = value.quantize(quantizer, rounding=decimal.ROUND_HALF_UP)

        # Format and ensure string output
        if quantizer == 1: formatted_str = str(rounded_value.to_integral_value(rounding=decimal.ROUND_HALF_UP))
        else: formatted_str = str(rounded_value.normalize()) # normalize removes trailing zeros

        # If original precision was 0 but min_display > 0, ensure decimals are shown
        if int(reported_precision) == 0 and display_precision > 0 and '.' not in formatted_str:
            formatted_str += '.' + '0' * display_precision

        return formatted_str
    except (ValueError, TypeError): return str(value) # Fallback string
    except Exception: return str(value) # Wider fallback string


def get_market_info(exchange, symbol):
    """Fetches and returns market information including precision and limits."""
    try:
        print_color(f"{Fore.CYAN}# Querying market runes for {symbol}...", style=Style.DIM, end='\r')
        if not exchange.markets or symbol not in exchange.markets: exchange.load_markets(True)
        market = exchange.market(symbol)
        sys.stdout.write("\033[K")

        price_prec_raw = market.get('precision', {}).get('price')
        amount_prec_raw = market.get('precision', {}).get('amount')
        min_amount_raw = market.get('limits', {}).get('amount', {}).get('min')

        try: price_prec = int(price_prec_raw) if price_prec_raw is not None else 8
        except: price_prec = 8
        try: amount_prec = int(amount_prec_raw) if amount_prec_raw is not None else 8
        except: amount_prec = 8
        try: min_amount = decimal.Decimal(str(min_amount_raw)) if min_amount_raw is not None else decimal.Decimal('0')
        except: min_amount = decimal.Decimal('0')

        price_tick_size = decimal.Decimal('1') / (decimal.Decimal('10') ** price_prec)
        amount_step = decimal.Decimal('1') / (decimal.Decimal('10') ** amount_prec)

        return {'price_precision': price_prec, 'amount_precision': amount_prec, 'min_amount': min_amount, 'price_tick_size': price_tick_size, 'amount_step': amount_step}
    except ccxt.BadSymbol: sys.stdout.write("\033[K"); print_color(f"Symbol '{symbol}' not found.", color=Fore.RED, style=Style.BRIGHT); return None
    except ccxt.NetworkError as e: sys.stdout.write("\033[K"); print_color(f"Network Error (Market Info): {e}", color=Fore.YELLOW); return None
    except Exception as e: sys.stdout.write("\033[K"); print_color(f"Error fetching market info: {e}", color=Fore.RED); return None

def calculate_sma(data, period):
    """Calculates Simple Moving Average using Decimal."""
    if not data or len(data) < period: return None
    try:
        decimal_data = [decimal.Decimal(str(p)) for p in data[-period:]]
        return sum(decimal_data) / decimal.Decimal(period)
    except Exception as e: print_color(f"SMA Calc Error: {e}", color=Fore.YELLOW, style=Style.DIM); return None

def calculate_fib_pivots(high, low, close):
    """Calculates Fibonacci Pivot Points using Decimal."""
    if None in [high, low, close]: return None
    try:
        h, l, c = decimal.Decimal(str(high)), decimal.Decimal(str(low)), decimal.Decimal(str(close))
        if h <= 0 or l <= 0 or c <= 0 or h < l: return None
        pp = (h + l + c) / 3; range_hl = h - l
        return {'R3': pp + (range_hl * FIB_RATIOS['r3']), 'R2': pp + (range_hl * FIB_RATIOS['r2']), 'R1': pp + (range_hl * FIB_RATIOS['r1']), 'PP': pp, 'S1': pp - (range_hl * FIB_RATIOS['s1']), 'S2': pp - (range_hl * FIB_RATIOS['s2']), 'S3': pp - (range_hl * FIB_RATIOS['s3'])}
    except Exception as e: print_color(f"Pivot Calc Error: {e}", color=Fore.YELLOW, style=Style.DIM); return None

# --- NEW: Stochastic RSI Calculation ---
def calculate_stoch_rsi(close_prices_list, rsi_period, stoch_k_period, stoch_d_period):
    """Calculates Stochastic RSI %K and %D using pandas."""
    if not close_prices_list or len(close_prices_list) < rsi_period + stoch_k_period:
        return None, None # Not enough data

    try:
        prices = pd.Series(close_prices_list, dtype=float) # Use float for pandas/TA compat

        # Calculate RSI
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=rsi_period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=rsi_period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))

        # Calculate Stochastic RSI %K
        min_rsi = rsi.rolling(window=stoch_k_period).min()
        max_rsi = rsi.rolling(window=stoch_k_period).max()
        stoch_rsi_k = ((rsi - min_rsi) / (max_rsi - min_rsi)) * 100

        # Calculate Stochastic RSI %D (SMA of %K)
        stoch_rsi_d = stoch_rsi_k.rolling(window=stoch_d_period).mean()

        # Return the latest values, handling potential NaNs at the beginning
        latest_k = stoch_rsi_k.iloc[-1]
        latest_d = stoch_rsi_d.iloc[-1]

        # Convert to Decimal for consistency if needed, else return float
        # return decimal.Decimal(str(latest_k)) if not pd.isna(latest_k) else None, \
        #        decimal.Decimal(str(latest_d)) if not pd.isna(latest_d) else None
        return latest_k if not pd.isna(latest_k) else None, \
               latest_d if not pd.isna(latest_d) else None

    except Exception as e:
        print_color(f"StochRSI Calc Error: {e}", color=Fore.YELLOW, style=Style.DIM)
        return None, None

# ==============================================================================
# Core Logic - The Grand Spellbook
# ==============================================================================

def fetch_market_data(exchange, symbol, config):
    """Fetches Ticker, OHLCV (Indicators & Pivots), Positions."""
    ticker, indicator_ohlcv, pivot_ohlcv, positions = None, None, None, None
    error_occurred = False
    rate_limit_wait = config["RETRY_DELAY_RATE_LIMIT"]
    network_wait = config["RETRY_DELAY_NETWORK_ERROR"]

    # Determine required history length for indicators
    indicator_history_needed = config['RSI_PERIOD'] + config['STOCH_K_PERIOD'] + config['STOCH_D_PERIOD'] + 5 # Add buffer

    api_calls = [
        {"func": exchange.fetch_ticker, "args": [symbol], "desc": "ticker"},
        {"func": exchange.fetch_ohlcv, "args": [symbol, config['INDICATOR_TIMEFRAME'], None, indicator_history_needed], "desc": "Indicator OHLCV"},
        {"func": exchange.fetch_ohlcv, "args": [symbol, config['PIVOT_TIMEFRAME'], None, 2], "desc": "Pivot OHLCV"},
        {"func": exchange.fetch_positions, "args": [[symbol]], "desc": "positions"},
    ]
    results = {}
    for call in api_calls:
        try:
            results[call["desc"]] = call["func"](*call["args"])
            if call["desc"] == "positions":
                 results[call["desc"]] = [p for p in results[call["desc"]] if p.get('symbol') == symbol and p.get('contracts') is not None and float(p['contracts']) != 0]
        except ccxt.RateLimitExceeded as e: print_color(f"Rate Limit ({call['desc']}). Wait {rate_limit_wait}s.", color=Fore.YELLOW, style=Style.DIM); time.sleep(rate_limit_wait); error_occurred = True; break
        except ccxt.NetworkError as e: print_color(f"Network Error ({call['desc']}). Wait {network_wait}s.", color=Fore.YELLOW, style=Style.DIM); time.sleep(network_wait); error_occurred = True
        except ccxt.AuthenticationError as e: print_color(f"Auth Error ({call['desc']}). Check API keys.", color=Fore.RED); error_occurred = True; break
        except Exception as e:
            if call['desc'] != "positions": print_color(f"Error ({call['desc']}): {e}", color=Fore.RED, style=Style.DIM)
            results[call["desc"]] = None

    return (results.get("ticker"), results.get("Indicator OHLCV"), results.get("Pivot OHLCV"), results.get("positions", []), error_occurred)


def analyze_orderbook_volume(exchange, symbol, market_info, config):
    """ Fetches and analyzes the order book using Decimal. """
    print_color(f"{Fore.CYAN}# Summoning order book spirits...", style=Style.DIM, end='\r')
    try:
        orderbook = exchange.fetch_order_book(symbol, limit=config["ORDER_FETCH_LIMIT"])
        sys.stdout.write("\033[K")
    except ccxt.RateLimitExceeded as e: sys.stdout.write("\033[K"); print_color(f"Rate Limit (OB). Wait...", color=Fore.YELLOW); time.sleep(config["RETRY_DELAY_RATE_LIMIT"]); return None, True
    except ccxt.NetworkError as e: sys.stdout.write("\033[K"); print_color(f"Network Error (OB). Retry...", color=Fore.YELLOW); time.sleep(config["RETRY_DELAY_NETWORK_ERROR"]); return None, True
    except ccxt.ExchangeError as e: sys.stdout.write("\033[K"); print_color(f"Exchange Error (OB): {e}", color=Fore.RED); return None, False
    except Exception as e: sys.stdout.write("\033[K"); print_color(f"Unexpected Error (OB): {e}", color=Fore.RED); return None, False

    if not orderbook or not orderbook.get('bids') or not orderbook.get('asks'): print_color(f"Order book {symbol} empty/unavailable.", color=Fore.YELLOW); return None, False

    price_prec, amount_prec = market_info['price_precision'], market_info['amount_precision']
    volume_thresholds = config["VOLUME_THRESHOLDS"]
    analyzed_orderbook = {'symbol': symbol, 'timestamp': exchange.iso8601(exchange.milliseconds()), 'asks': [], 'bids': [], 'ask_total_volume': decimal.Decimal('0'), 'bid_total_volume': decimal.Decimal('0'), 'ask_weighted_price': decimal.Decimal('0'), 'bid_weighted_price': decimal.Decimal('0'), 'volume_imbalance_ratio': decimal.Decimal('0')}
    ask_volume_times_price = decimal.Decimal('0')
    for i, ask in enumerate(orderbook['asks']):
        if i >= config["MAX_ORDERBOOK_DEPTH_DISPLAY"]: break
        try: price, volume = decimal.Decimal(str(ask[0])), decimal.Decimal(str(ask[1]))
        except: continue
        volume_str = format_decimal(volume, amount_prec) # Amount uses reported precision only
        highlight_color, highlight_style = Fore.WHITE, Style.NORMAL
        if volume >= decimal.Decimal(str(volume_thresholds.get('high', 'inf'))): highlight_color, highlight_style = Fore.LIGHTRED_EX, Style.BRIGHT
        elif volume >= decimal.Decimal(str(volume_thresholds.get('medium', '0'))): highlight_color, highlight_style = Fore.RED, Style.NORMAL
        analyzed_orderbook['asks'].append({'price': price, 'volume': volume, 'volume_str': volume_str, 'color': highlight_color, 'style': highlight_style})
        analyzed_orderbook['ask_total_volume'] += volume; ask_volume_times_price += price * volume
    bid_volume_times_price = decimal.Decimal('0')
    for i, bid in enumerate(orderbook['bids']):
        if i >= config["MAX_ORDERBOOK_DEPTH_DISPLAY"]: break
        try: price, volume = decimal.Decimal(str(bid[0])), decimal.Decimal(str(bid[1]))
        except: continue
        volume_str = format_decimal(volume, amount_prec) # Amount uses reported precision only
        highlight_color, highlight_style = Fore.WHITE, Style.NORMAL
        if volume >= decimal.Decimal(str(volume_thresholds.get('high', 'inf'))): highlight_color, highlight_style = Fore.LIGHTGREEN_EX, Style.BRIGHT
        elif volume >= decimal.Decimal(str(volume_thresholds.get('medium', '0'))): highlight_color, highlight_style = Fore.GREEN, Style.NORMAL
        analyzed_orderbook['bids'].append({'price': price, 'volume': volume, 'volume_str': volume_str, 'color': highlight_color, 'style': highlight_style})
        analyzed_orderbook['bid_total_volume'] += volume; bid_volume_times_price += price * volume
    ask_total_vol, bid_total_vol = analyzed_orderbook['ask_total_volume'], analyzed_orderbook['bid_total_volume']
    if ask_total_vol > 0: analyzed_orderbook['volume_imbalance_ratio'], analyzed_orderbook['ask_weighted_price'] = bid_total_vol / ask_total_vol, ask_volume_times_price / ask_total_vol
    else: analyzed_orderbook['volume_imbalance_ratio'], analyzed_orderbook['ask_weighted_price'] = (decimal.Decimal('inf') if bid_total_vol > 0 else decimal.Decimal('0')), decimal.Decimal('0')
    if bid_total_vol > 0: analyzed_orderbook['bid_weighted_price'] = bid_volume_times_price / bid_total_vol
    else: analyzed_orderbook['bid_weighted_price'] = decimal.Decimal('0')
    return analyzed_orderbook, False


def display_combined_analysis(analyzed_orderbook, market_info, ticker_info, indicators_info, position_info, pivots_info, config):
    """ Displays Order Book, Ticker, Indicators, PNL, and Pivot Info """
    is_placeholder_ob = False
    if not analyzed_orderbook:
        analyzed_orderbook = {'symbol': market_info.get('symbol', 'N/A'), 'timestamp': ticker_info.get('iso8601', 'N/A') if ticker_info else 'N/A', 'asks':[], 'bids':[], 'ask_total_volume': 0, 'bid_total_volume': 0, 'ask_weighted_price': 0, 'bid_weighted_price': 0, 'volume_imbalance_ratio': 0}
        is_placeholder_ob = True

    symbol, timestamp = analyzed_orderbook['symbol'], analyzed_orderbook['timestamp']
    price_prec, amount_prec = market_info['price_precision'], market_info['amount_precision']
    pnl_prec = config["PNL_PRECISION"]
    min_disp_prec = config["MIN_PRICE_DISPLAY_PRECISION"]
    stoch_disp_prec = config["STOCH_RSI_DISPLAY_PRECISION"]

    print_color(f"\n📊 Market Analysis: {Fore.MAGENTA}{Style.BRIGHT}{symbol}{Style.RESET_ALL} @ {Fore.YELLOW}{timestamp}", color=Fore.CYAN)
    print_color("=" * 80, color=Fore.CYAN) # Wider separator

    # --- Ticker & Trend Info ---
    last_price, current_price_str, price_color = None, f"{Fore.YELLOW}N/A{Style.RESET_ALL}", Fore.WHITE
    if ticker_info and ticker_info.get('last') is not None:
        try:
            last_price = decimal.Decimal(str(ticker_info['last']))
            current_price_str_fmt = format_decimal(last_price, price_prec, min_disp_prec) # Use min display prec
            if indicators_info.get('sma') is not None:
                if last_price > indicators_info['sma']: price_color = Fore.GREEN
                elif last_price < indicators_info['sma']: price_color = Fore.RED
                else: price_color = Fore.YELLOW
            current_price_str = f"{price_color}{Style.BRIGHT}{current_price_str_fmt}{Style.RESET_ALL}"
        except: pass

    sma_str, trend_str, trend_color = f"{Fore.YELLOW}N/A{Style.RESET_ALL}", "Trend: -", Fore.YELLOW
    if indicators_info.get('sma') is not None and last_price is not None:
        sma_val = indicators_info['sma']; sma_str_fmt = format_decimal(sma_val, price_prec, min_disp_prec) # Use min display prec
        if last_price > sma_val: trend_str, trend_color = f"Above {config['SMA_PERIOD']}@{config['INDICATOR_TIMEFRAME']} SMA ({sma_str_fmt})", Fore.GREEN
        elif last_price < sma_val: trend_str, trend_color = f"Below {config['SMA_PERIOD']}@{config['INDICATOR_TIMEFRAME']} SMA ({sma_str_fmt})", Fore.RED
        else: trend_str, trend_color = f"On {config['SMA_PERIOD']}@{config['INDICATOR_TIMEFRAME']} SMA ({sma_str_fmt})", Fore.YELLOW
    elif indicators_info.get('sma_error'): trend_str = f"Trend: SMA Error"
    else: trend_str = f"Trend: SMA ({config['SMA_PERIOD']}@{config['INDICATOR_TIMEFRAME']}) unavailable"
    print_color(f"  Last Price: {current_price_str} | {trend_color}{trend_str}{Style.RESET_ALL}")

    # --- Stochastic RSI ---
    stoch_k_val = indicators_info.get('stoch_k')
    stoch_d_val = indicators_info.get('stoch_d')
    stoch_k_str, stoch_d_str = "N/A", "N/A"
    stoch_color = Fore.WHITE
    stoch_signal = ""
    if stoch_k_val is not None and stoch_d_val is not None:
        stoch_k_str = format_decimal(stoch_k_val, stoch_disp_prec) # Specific precision for Stoch RSI
        stoch_d_str = format_decimal(stoch_d_val, stoch_disp_prec)
        if stoch_k_val < config['STOCH_RSI_OVERSOLD'] and stoch_d_val < config['STOCH_RSI_OVERSOLD']:
            stoch_color = Fore.GREEN; stoch_signal = "(Oversold)"
        elif stoch_k_val > config['STOCH_RSI_OVERBOUGHT'] and stoch_d_val > config['STOCH_RSI_OVERBOUGHT']:
            stoch_color = Fore.RED; stoch_signal = "(Overbought)"
        elif stoch_k_val > stoch_d_val: stoch_color = Fore.LIGHTGREEN_EX # K above D
        elif stoch_k_val < stoch_d_val: stoch_color = Fore.LIGHTRED_EX # K below D

    print_color(f"  Stoch RSI ({config['RSI_PERIOD']},{config['STOCH_K_PERIOD']},{config['STOCH_D_PERIOD']}@{config['INDICATOR_TIMEFRAME']}): "
                f"{stoch_color}%K={stoch_k_str}, %D={stoch_d_str} {stoch_signal}{Style.RESET_ALL}")


    # --- Position & PNL Info ---
    pnl_str = f"{Fore.LIGHTBLACK_EX}Position: None or Fetch Failed{Style.RESET_ALL}" # Updated message
    if position_info.get('has_position'):
        pos = position_info['position']; side = pos.get('side', 'N/A').capitalize()
        size = decimal.Decimal(str(pos.get('contracts', '0'))); entry_price = decimal.Decimal(str(pos.get('entryPrice', '0')))
        pnl = position_info.get('unrealizedPnl', decimal.Decimal('0')); size_str = format_decimal(size, amount_prec) # Amount precision
        entry_str = format_decimal(entry_price, price_prec, min_disp_prec) # Price precision
        pnl_val_str = format_decimal(pnl, pnl_prec) # PNL precision
        side_color = Fore.GREEN if side.lower() == 'long' else Fore.RED if side.lower() == 'short' else Fore.WHITE
        pnl_color = Fore.GREEN if pnl > 0 else Fore.RED if pnl < 0 else Fore.WHITE
        pnl_str = (f"Position: {side_color}{side} {size_str}{Style.RESET_ALL} | Entry: {Fore.YELLOW}{entry_str}{Style.RESET_ALL} | uPNL: {pnl_color}{pnl_val_str} {pos.get('quoteAsset','USDT')}{Style.RESET_ALL}")
    print_color(f"  {pnl_str}")

    # --- Fibonacci Pivots ---
    print_color("--- Fibonacci Pivots (Prev Day) ---", color=Fore.BLUE)
    pivot_width = max(10, price_prec + 6)
    if pivots_info:
        pivot_lines = {}
        for level in ['R3', 'R2', 'R1', 'PP', 'S1', 'S2', 'S3']:
            value = pivots_info.get(level)
            if value is not None:
                level_str = f"{level}:".ljust(4)
                value_str = format_decimal(value, price_prec, min_disp_prec) # Use min display prec
                if isinstance(value_str, str): value_str = value_str.rjust(pivot_width)
                else: value_str = str(value).rjust(pivot_width) # Fallback
                level_color = Fore.RED if 'R' in level else Fore.GREEN if 'S' in level else Fore.YELLOW
                highlight = ""
                if last_price:
                    try:
                        diff_ratio = abs(last_price - value) / last_price if last_price else decimal.Decimal('inf')
                        if diff_ratio < decimal.Decimal('0.001'): highlight = Back.LIGHTBLACK_EX + Fore.WHITE + Style.BRIGHT + " *NEAR* " + Style.RESET_ALL
                    except: pass
                pivot_lines[level] = f"  {level_color}{level_str}{value_str}{Style.RESET_ALL}{highlight}"
            else: pivot_lines[level] = f"  {level}:".ljust(6) + f"{'N/A':>{pivot_width}}"
        # Print Pivots trying to pair R/S levels
        print(f"{pivot_lines.get('R3','')}")
        print(f"{pivot_lines.get('R2','').ljust(pivot_width+10)}{pivot_lines.get('S2','')}")
        print(f"{pivot_lines.get('R1','').ljust(pivot_width+10)}{pivot_lines.get('S1','')}")
        print(f"{pivot_lines.get('PP','')}")
        print(f"{pivot_lines.get('S3','')}")
    else: print_color(f"  {Fore.YELLOW}Pivot data unavailable.{Style.RESET_ALL}")

    # --- Order Book Display ---
    print_color("--- Order Book ---", color=Fore.BLUE)
    if is_placeholder_ob: print_color(f"  {Fore.YELLOW}Order book data unavailable.{Style.RESET_ALL}")
    else:
        price_width, volume_width = max(10, price_prec + 4), max(12, amount_prec + 6)
        ask_lines, bid_lines = [], []
        # Use min display precision for OB prices
        for ask in reversed(analyzed_orderbook['asks']): price_str = format_decimal(ask['price'], price_prec, min_disp_prec); ask_lines.append(f"{Style.NORMAL}{Fore.WHITE}{price_str:<{price_width}}{ask['style']}{ask['color']}{ask['volume_str']:<{volume_width}}{Style.RESET_ALL}")
        for bid in analyzed_orderbook['bids']: price_str = format_decimal(bid['price'], price_prec, min_disp_prec); bid_lines.append(f"{Style.NORMAL}{Fore.WHITE}{price_str:<{price_width}}{bid['style']}{bid['color']}{bid['volume_str']:<{volume_width}}{Style.RESET_ALL}")
        max_rows = max(len(ask_lines), len(bid_lines))
        print_color(f"{'Asks':^{price_width+volume_width}}{'Bids':^{price_width+volume_width}}", color=Fore.LIGHTBLACK_EX)
        print_color(f"{'-'*(price_width+volume_width-1):<{price_width+volume_width}} {'-'*(price_width+volume_width-1):<{price_width+volume_width}}", color=Fore.LIGHTBLACK_EX)
        for i in range(max_rows): print(f"{ask_lines[i] if i < len(ask_lines) else ' ' * (price_width + volume_width)}  {bid_lines[i] if i < len(bid_lines) else ' ' * (price_width + volume_width)}")
        best_ask = analyzed_orderbook['asks'][-1]['price'] if analyzed_orderbook['asks'] else decimal.Decimal('0')
        best_bid = analyzed_orderbook['bids'][0]['price'] if analyzed_orderbook['bids'] else decimal.Decimal('0')
        spread = best_ask - best_bid if best_ask > 0 and best_bid > 0 else decimal.Decimal('0')
        print_color(f"\n--- Spread: {format_decimal(spread, price_prec, min_disp_prec)} ---", color=Fore.MAGENTA, style=Style.DIM) # Use min display prec

        # --- Volume Analysis Footer ---
        print_color("\n--- Volume Analysis ---", color=Fore.BLUE)
        print_color(f"  Total Ask: {Fore.RED}{format_decimal(analyzed_orderbook['ask_total_volume'], amount_prec)}{Style.RESET_ALL} | Total Bid: {Fore.GREEN}{format_decimal(analyzed_orderbook['bid_total_volume'], amount_prec)}{Style.RESET_ALL}") # Amount prec
        imbalance_ratio = analyzed_orderbook['volume_imbalance_ratio']
        imbalance_color = Fore.WHITE
        if imbalance_ratio.is_infinite(): imbalance_color = Fore.LIGHTGREEN_EX
        elif imbalance_ratio > decimal.Decimal('1.5'): imbalance_color = Fore.GREEN
        elif imbalance_ratio < decimal.Decimal('0.67') and not imbalance_ratio.is_zero(): imbalance_color = Fore.RED
        elif imbalance_ratio.is_zero() and analyzed_orderbook['ask_total_volume'] > 0: imbalance_color = Fore.LIGHTRED_EX
        imbalance_str = "inf" if imbalance_ratio.is_infinite() else format_decimal(imbalance_ratio, 2) # Fixed precision for ratio
        # Use min display precision for VWAP
        print_color(f"  Imbalance (B/A): {imbalance_color}{imbalance_str}{Style.RESET_ALL} | Ask VWAP: {Fore.YELLOW}{format_decimal(analyzed_orderbook['ask_weighted_price'], price_prec, min_disp_prec)}{Style.RESET_ALL} | Bid VWAP: {Fore.YELLOW}{format_decimal(analyzed_orderbook['bid_weighted_price'], price_prec, min_disp_prec)}{Style.RESET_ALL}")

        # --- Pressure Reading ---
        print_color("--- Pressure Reading ---", color=Fore.BLUE)
        if imbalance_ratio.is_infinite(): print_color("  Extreme Bid Dominance", color=Fore.LIGHTYELLOW_EX)
        elif imbalance_ratio > decimal.Decimal('1.5'): print_color("  Strong Buy Pressure", color=Fore.GREEN, style=Style.BRIGHT)
        elif imbalance_ratio < decimal.Decimal('0.67') and not imbalance_ratio.is_zero(): print_color("  Strong Sell Pressure", color=Fore.RED, style=Style.BRIGHT)
        elif imbalance_ratio.is_zero() and analyzed_orderbook['ask_total_volume'] > 0: print_color("  Extreme Ask Dominance", color=Fore.LIGHTYELLOW_EX)
        else: print_color("  Volume Relatively Balanced", color=Fore.WHITE)
    print_color("=" * 80, color=Fore.CYAN) # Wider separator


def place_market_order(exchange, symbol, side, amount_str, market_info):
    """ Places a market order after validation and confirmation. """
    try:
        amount = decimal.Decimal(amount_str)
        min_amount = market_info.get('min_amount', decimal.Decimal('0'))
        amount_step = market_info.get('amount_step', decimal.Decimal('1'))
        if amount <= 0: print_color("Amount must be positive.", color=Fore.YELLOW); return
        if min_amount > 0 and amount < min_amount: print_color(f"Amount < minimum ({format_decimal(min_amount, market_info['amount_precision'])}).", color=Fore.YELLOW); return
        if amount_step > 0 and (amount % amount_step) != 0:
            rounded_amount = (amount // amount_step) * amount_step
            if min_amount > 0 and rounded_amount < min_amount: print_color(f"Amount step invalid, rounding makes it < minimum.", color=Fore.YELLOW); return
            round_confirm = input(f"{Fore.YELLOW}Amount step invalid. Round down to {format_decimal(rounded_amount, market_info['amount_precision'])}? (yes/no): {Style.RESET_ALL}").strip().lower()
            if round_confirm == 'yes': amount, amount_str = rounded_amount, format_decimal(rounded_amount, market_info['amount_precision']); print_color(f"Using rounded: {amount_str}", color=Fore.CYAN)
            else: print_color("Order cancelled.", color=Fore.YELLOW); return
    except (decimal.InvalidOperation, TypeError, ValueError) as e: print_color(f"Invalid amount/market data: {e}", color=Fore.YELLOW); return

    side_color = Fore.GREEN if side == 'buy' else Fore.RED
    prompt = (f"{Style.BRIGHT}Confirm MARKET {side_color}{side.upper()}{Style.RESET_ALL} order: {Fore.YELLOW}{amount_str}{Style.RESET_ALL} {Fore.MAGENTA}{symbol}{Style.RESET_ALL} (yes/no): {Style.RESET_ALL}")
    if input(prompt).strip().lower() == 'yes':
        print_color(f"{Fore.CYAN}# Transmitting order...", style=Style.DIM, end='\r')
        try:
            params = {} # Add 'positionIdx' here if using Hedge Mode
            formatted_amount = exchange.amount_to_precision(symbol, float(amount)) # CCXT often expects float here
            order = exchange.create_market_order(symbol, side, formatted_amount, params=params)
            sys.stdout.write("\033[K"); print_color(f"\n✅ Market order placed! ID: {Fore.YELLOW}{order.get('id')}", color=Fore.GREEN, style=Style.BRIGHT)
        except ccxt.InsufficientFunds as e: sys.stdout.write("\033[K"); print_color(f"\n❌ Insufficient Funds: {e}", color=Fore.RED, style=Style.BRIGHT)
        except ccxt.ExchangeError as e:
            sys.stdout.write("\033[K")
            if "10001" in str(e) and "position idx" in str(e): print_color(f"\n❌ Exchange Error: {e}\n{Fore.YELLOW}Suggestion: Account likely in HEDGE MODE. Specify 'positionIdx' or use One-Way mode.", color=Fore.RED, style=Style.BRIGHT)
            else: print_color(f"\n❌ Exchange Error: {e}", color=Fore.RED, style=Style.BRIGHT)
        except Exception as e: sys.stdout.write("\033[K"); print_color(f"\n❌ Order Placement Error: {e}", color=Fore.RED, style=Style.BRIGHT)
    else: print_color("Order cancelled.", color=Fore.YELLOW)

# ==============================================================================
# Main Execution Block - The Ritual Begins
# ==============================================================================

def main():
    """Main execution function."""
    print_color("*" * 80, color=Fore.RED, style=Style.BRIGHT)
    print_color("   DISCLAIMER: ADVANCED TRADING SCRIPT. USE WITH EXTREME CAUTION.", color=Fore.RED, style=Style.BRIGHT)
    print_color("   MARKET ORDERS SLIPPAGE RISK. YOU ARE RESPONSIBLE FOR TRADES.", color=Fore.RED, style=Style.BRIGHT)
    print_color("*" * 80, color=Fore.RED, style=Style.BRIGHT); print("\n")

    if not CONFIG["API_KEY"] or not CONFIG["API_SECRET"]: print_color("API Key/Secret missing.", color=Fore.RED, style=Style.BRIGHT); return

    print_color(f"{Fore.CYAN}# Initializing Bybit ({CONFIG['DEFAULT_EXCHANGE_TYPE']})...", style=Style.DIM)
    try:
        exchange = ccxt.bybit({'apiKey': CONFIG["API_KEY"], 'secret': CONFIG["API_SECRET"], 'options': {'defaultType': CONFIG["DEFAULT_EXCHANGE_TYPE"], 'adjustForTimeDifference': True}, 'timeout': CONFIG["CONNECT_TIMEOUT"], 'enableRateLimit': True})
        print_color("Connection object created.", color=Fore.GREEN)
    except Exception as e: print_color(f"Connection failed: {e}", color=Fore.RED, style=Style.BRIGHT); return

    symbol, market_info = "", None
    while not market_info:
        try:
            symbol_input = input(f"{Style.BRIGHT}{Fore.BLUE}Enter Bybit Futures symbol (e.g., BTCUSDT): {Style.RESET_ALL}").strip().upper()
            if not symbol_input: continue
            market_info = get_market_info(exchange, symbol_input)
            if market_info:
                symbol, market_info['symbol'] = symbol_input, symbol_input
                # Display REPORTED precision here
                print_color(f"Selected: {Fore.MAGENTA}{symbol}{Fore.CYAN} | Reported Prec => Price: {market_info['price_precision']}, Amount: {market_info['amount_precision']}", color=Fore.CYAN)
            # Error/retry handled in get_market_info
        except (EOFError, KeyboardInterrupt): print_color("\nExiting.", color=Fore.YELLOW); return
        except Exception as e: print_color(f"Symbol input error: {e}", color=Fore.RED); time.sleep(1)

    print_color(f"\nStarting analysis for {symbol}. Press Ctrl+C to exit.", color=Fore.CYAN)
    last_error_msg = ""
    while True:
        data_error, orderbook_error = False, False
        try:
            print_color(f"{Fore.CYAN}# Fetching market data...", style=Style.DIM, end='\r')
            ticker_info, indicator_ohlcv, pivot_ohlcv, position_data, data_error = fetch_market_data(exchange, symbol, CONFIG)
            analyzed_orderbook, orderbook_error = analyze_orderbook_volume(exchange, symbol, market_info, CONFIG)
            sys.stdout.write("\033[K")

            # --- Process Indicators ---
            indicators_info = {'sma': None, 'stoch_k': None, 'stoch_d': None, 'sma_error': False, 'stoch_error': False}
            if indicator_ohlcv:
                close_prices = [candle[4] for candle in indicator_ohlcv] # Index 4 is close
                # SMA
                sma = calculate_sma(close_prices, CONFIG["SMA_PERIOD"])
                if sma is not None: indicators_info['sma'] = sma
                elif len(close_prices) < CONFIG["SMA_PERIOD"]: indicators_info['sma_error'] = True
                # Stoch RSI
                stoch_k, stoch_d = calculate_stoch_rsi(close_prices, CONFIG["RSI_PERIOD"], CONFIG["STOCH_K_PERIOD"], CONFIG["STOCH_D_PERIOD"])
                if stoch_k is not None: indicators_info['stoch_k'] = stoch_k
                if stoch_d is not None: indicators_info['stoch_d'] = stoch_d
                if stoch_k is None or stoch_d is None: indicators_info['stoch_error'] = True # Mark error if calculation failed
            else:
                indicators_info['sma_error'] = True; indicators_info['stoch_error'] = True

            # --- Process Pivots ---
            pivots_info = None
            if pivot_ohlcv and len(pivot_ohlcv) > 0:
                prev_day_candle = pivot_ohlcv[0]
                p_high, p_low, p_close = prev_day_candle[2], prev_day_candle[3], prev_day_candle[4]
                pivots_info = calculate_fib_pivots(p_high, p_low, p_close)

            # --- Process Positions ---
            position_info = {'has_position': False, 'position': None, 'unrealizedPnl': decimal.Decimal('0')}
            if position_data:
                current_pos = position_data[0]
                position_info['has_position'] = True; position_info['position'] = current_pos
                try:
                    if current_pos.get('unrealizedPnl') is not None: position_info['unrealizedPnl'] = decimal.Decimal(str(current_pos['unrealizedPnl']))
                except: pass

            # --- Display ---
            display_combined_analysis(analyzed_orderbook, market_info, ticker_info, indicators_info, position_info, pivots_info, CONFIG)

            # --- Action Prompt ---
            if not data_error and not orderbook_error:
                action = input(f"\n{Style.BRIGHT}{Fore.BLUE}Action ({Fore.CYAN}analyze{Fore.BLUE}/{Fore.GREEN}buy{Fore.BLUE}/{Fore.RED}sell{Fore.BLUE}/{Fore.YELLOW}exit{Fore.BLUE}): {Style.RESET_ALL}").strip().lower()
                if action == 'buy': place_market_order(exchange, symbol, 'buy', input(f"{Style.BRIGHT}{Fore.GREEN}BUY Qty: {Style.RESET_ALL}").strip(), market_info)
                elif action == 'sell': place_market_order(exchange, symbol, 'sell', input(f"{Style.BRIGHT}{Fore.RED}SELL Qty: {Style.RESET_ALL}").strip(), market_info)
                elif action == 'analyze': print_color("Refreshing...", color=Fore.CYAN, style=Style.DIM)
                elif action == 'exit': print_color("Exiting.", color=Fore.YELLOW); break
                else: print_color("Invalid action.", color=Fore.YELLOW)
            else: print_color("Waiting due to data fetch errors...", color=Fore.YELLOW, style=Style.DIM); time.sleep(CONFIG["RETRY_DELAY_NETWORK_ERROR"])

            time.sleep(CONFIG["REFRESH_INTERVAL"])

        except KeyboardInterrupt: print_color("\nExiting.", color=Fore.YELLOW); break
        except Exception as e:
            current_error_msg = str(e); print_color(".", color=Fore.RED, end='')
            if current_error_msg != last_error_msg: print_color(f"\nCritical Loop Error: {e}", color=Fore.RED, style=Style.BRIGHT); last_error_msg = current_error_msg
            time.sleep(CONFIG["REFRESH_INTERVAL"] * 2); last_error_msg = current_error_msg

if __name__ == '__main__':
    # Ensure pandas is installed: pip install pandas
    main()
    print_color("\nWizard Pyrmethus departs. May your analysis illuminate the path!", color=Fore.MAGENTA, style=Style.BRIGHT)
