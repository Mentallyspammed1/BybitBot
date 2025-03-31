# -*- coding: utf-8 -*-
import ccxt
import time
import os
import sys
from dotenv import load_dotenv
from colorama import init, Fore, Style, Back
import decimal # Use decimal for precise price/amount calculations
import numpy as np # For SMA calculation

# ==============================================================================
# Initialize Colorama - The Palette of Power!
# ==============================================================================
init(autoreset=True)

# ==============================================================================
# Load Environment Variables & Configuration
# ==============================================================================
load_dotenv()

CONFIG = {
    "API_KEY": os.environ.get("BYBIT_API_KEY"),
    "API_SECRET": os.environ.get("BYBIT_API_SECRET"),
    "VOLUME_THRESHOLDS": {'high': 10, 'medium': 2}, # Thresholds (adjust based on symbol's base currency)
    "REFRESH_INTERVAL": 7, # Seconds between refreshes (increased slightly for more API calls)
    "MAX_ORDERBOOK_DEPTH_DISPLAY": 12, # Limit display depth for clarity
    "ORDER_FETCH_LIMIT": 50, # How many levels to fetch for analysis
    "DEFAULT_EXCHANGE_TYPE": 'linear', # 'linear' for USDT perps, 'inverse' for coin-margined
    "CONNECT_TIMEOUT": 30000, # milliseconds for CCXT connection
    "RETRY_DELAY_NETWORK_ERROR": 10, # seconds
    "RETRY_DELAY_RATE_LIMIT": 60, # seconds
    "SMA_PERIOD": 20, # Period for Simple Moving Average calculation
    "SMA_TIMEFRAME": '5m', # Timeframe for SMA calculation (e.g., '1m', '5m', '15m', '1h')
    "PNL_PRECISION": 2, # Decimal places for PNL display
}

# Set Decimal precision (adjust if needed, default is usually sufficient)
# decimal.getcontext().prec = 28

# ==============================================================================
# Helper Functions - Arcane Utilities
# ==============================================================================

def print_color(text, color=Fore.WHITE, style=Style.NORMAL, end='\n', **kwargs):
    """Helper to print colored text with dynamic formatting."""
    print(f"{style}{color}{text}{Style.RESET_ALL}", end=end, **kwargs)

def format_decimal(value, precision):
    """Formats a decimal value to the specified precision string, handling potential invalid precision values gracefully."""
    if not isinstance(value, decimal.Decimal):
        try: value = decimal.Decimal(str(value))
        except (decimal.InvalidOperation, TypeError, ValueError): return str(value) # Fallback

    try:
        int_precision = int(precision)
        if int_precision < 0: int_precision = 0
        # Use 'normalize()' to remove trailing zeros for cleaner display
        return f"{value:.{int_precision}f}".rstrip('0').rstrip('.') if '.' in f"{value:.{int_precision}f}" else f"{value:.{int_precision}f}"
        # Alternate: return f"{value:.{int_precision}f}" # Keep trailing zeros
    except (ValueError, TypeError): return str(value) # Fallback
    except Exception: return str(value) # Wider fallback

def get_market_info(exchange, symbol):
    """Fetches and returns market information including precision and limits."""
    try:
        print_color(f"{Fore.CYAN}# Querying market runes for {symbol}...", style=Style.DIM, end='\r')
        # Force reload markets if not loaded or perhaps periodically? For now, load once if needed.
        if not exchange.markets or symbol not in exchange.markets:
             exchange.load_markets(True) # Force reload
        market = exchange.market(symbol)
        sys.stdout.write("\033[K")

        price_prec = market.get('precision', {}).get('price')
        amount_prec = market.get('precision', {}).get('amount')
        min_amount = market.get('limits', {}).get('amount', {}).get('min')

        # Validate and convert precision/limits safely
        try: price_prec_int = int(price_prec) if price_prec is not None else 8
        except (ValueError, TypeError): price_prec_int = 8
        try: amount_prec_int = int(amount_prec) if amount_prec is not None else 8
        except (ValueError, TypeError): amount_prec_int = 8
        try: min_amount_dec = decimal.Decimal(str(min_amount)) if min_amount is not None else decimal.Decimal('0')
        except (decimal.InvalidOperation, TypeError, ValueError): min_amount_dec = decimal.Decimal('0')

        # Calculate step sizes based on precision
        price_tick_size = decimal.Decimal('1') / (decimal.Decimal('10') ** price_prec_int)
        amount_step = decimal.Decimal('1') / (decimal.Decimal('10') ** amount_prec_int)

        return {
            'price_precision': price_prec_int,
            'amount_precision': amount_prec_int,
            'min_amount': min_amount_dec,
            'price_tick_size': price_tick_size,
            'amount_step': amount_step
        }
    except ccxt.BadSymbol:
        sys.stdout.write("\033[K")
        print_color(f"Symbol '{symbol}' not found or invalid on Bybit.", color=Fore.RED, style=Style.BRIGHT)
        return None
    except ccxt.NetworkError as e:
         sys.stdout.write("\033[K")
         print_color(f"Network Error fetching market info: {e}", color=Fore.YELLOW)
         return None # Allow retry
    except Exception as e:
        sys.stdout.write("\033[K")
        print_color(f"Error fetching market info for {symbol}: {e}", color=Fore.RED)
        return None

def calculate_sma(data, period):
    """Calculates Simple Moving Average."""
    if len(data) < period:
        return None
    try:
        # Use Decimal for calculation if data contains Decimals, else float
        if isinstance(data[0], decimal.Decimal):
            return sum(data[-period:]) / decimal.Decimal(period)
        else:
             # Convert to float for numpy if needed, handle potential errors
             float_data = [float(x) for x in data]
             return np.mean(float_data[-period:])
    except Exception as e:
        print_color(f"Error calculating SMA: {e}", color=Fore.YELLOW, style=Style.DIM)
        return None

# ==============================================================================
# Core Logic - The Grand Spellbook
# ==============================================================================

def fetch_additional_data(exchange, symbol, sma_timeframe, sma_period):
    """Fetches ticker, OHLCV for SMA, and positions."""
    ticker = None
    ohlcv = None
    positions = None
    error_occurred = False

    # Fetch Ticker
    try:
        ticker = exchange.fetch_ticker(symbol)
    except ccxt.RateLimitExceeded as e:
        print_color(f"Rate Limit fetching ticker: {e}", color=Fore.YELLOW, style=Style.DIM)
        error_occurred = True # Signal to wait longer
    except ccxt.NetworkError as e:
        print_color(f"Network Error fetching ticker: {e}", color=Fore.YELLOW, style=Style.DIM)
        error_occurred = True
    except Exception as e:
        print_color(f"Error fetching ticker: {e}", color=Fore.RED, style=Style.DIM)
        error_occurred = True

    # Fetch OHLCV for SMA
    try:
        # Fetch slightly more data than needed for stability
        limit = sma_period + 5
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe=sma_timeframe, limit=limit)
    except ccxt.RateLimitExceeded as e:
        print_color(f"Rate Limit fetching OHLCV: {e}", color=Fore.YELLOW, style=Style.DIM)
        error_occurred = True
    except ccxt.NetworkError as e:
        print_color(f"Network Error fetching OHLCV: {e}", color=Fore.YELLOW, style=Style.DIM)
        error_occurred = True
    except Exception as e:
        print_color(f"Error fetching OHLCV: {e}", color=Fore.RED, style=Style.DIM)
        # Don't set error_occurred=True here, maybe just SMA fails

    # Fetch Positions
    try:
        # Note: fetch_positions might behave differently based on account type (UTA vs Normal)
        # and might require specific permissions.
        # For Bybit Unified Trading Account (UTA), fetch_positions often works well.
        positions = exchange.fetch_positions([symbol])
        # Filter for potentially relevant positions (non-zero size)
        positions = [p for p in positions if p.get('contracts') is not None and float(p['contracts']) != 0]

    except ccxt.RateLimitExceeded as e:
        print_color(f"Rate Limit fetching positions: {e}", color=Fore.YELLOW, style=Style.DIM)
        error_occurred = True
    except ccxt.NetworkError as e:
        print_color(f"Network Error fetching positions: {e}", color=Fore.YELLOW, style=Style.DIM)
        error_occurred = True
    except ccxt.AuthenticationError as e:
         print_color(f"Authentication Error fetching positions: {e}. Check API key permissions.", color=Fore.RED)
         error_occurred = True # Stop if auth fails
    except Exception as e:
        print_color(f"Error fetching positions: {e}", color=Fore.RED, style=Style.DIM)
        # May fail if no position exists, which isn't necessarily a critical error for display
        positions = [] # Assume no positions if fetch fails non-critically

    return ticker, ohlcv, positions, error_occurred


def analyze_orderbook_volume(exchange, symbol, market_info):
    """ Fetches and analyzes the order book. """
    print_color(f"{Fore.CYAN}# Summoning order book spirits for {symbol}...", style=Style.DIM, end='\r')
    try:
        orderbook = exchange.fetch_order_book(symbol, limit=CONFIG["ORDER_FETCH_LIMIT"])
        sys.stdout.write("\033[K")
    except ccxt.RateLimitExceeded as e:
        sys.stdout.write("\033[K")
        print_color(f"Rate Limit fetching order book: {e}. Waiting...", color=Fore.YELLOW)
        time.sleep(CONFIG["RETRY_DELAY_RATE_LIMIT"])
        return None, True # Indicate error for retry
    except ccxt.NetworkError as e:
        sys.stdout.write("\033[K")
        print_color(f"Network Error fetching order book: {e}. Retrying...", color=Fore.YELLOW)
        time.sleep(CONFIG["RETRY_DELAY_NETWORK_ERROR"])
        return None, True # Indicate error for retry
    except ccxt.ExchangeError as e:
        sys.stdout.write("\033[K")
        print_color(f"Exchange Error fetching order book: {e}", color=Fore.RED, style=Style.BRIGHT)
        return None, False # Don't necessarily retry immediately on exchange error unless network related
    except Exception as e:
        sys.stdout.write("\033[K")
        print_color(f"Unexpected error fetching order book: {e}", color=Fore.RED, style=Style.BRIGHT)
        return None, False

    if not orderbook or not orderbook.get('bids') or not orderbook.get('asks'):
        print_color(f"Order book for {symbol} is empty or unavailable.", color=Fore.YELLOW)
        return None, False

    price_prec = market_info['price_precision']
    amount_prec = market_info['amount_precision']
    volume_thresholds = CONFIG["VOLUME_THRESHOLDS"]

    analyzed_orderbook = {
        'symbol': symbol, 'timestamp': exchange.iso8601(exchange.milliseconds()),
        'asks': [], 'bids': [],
        'ask_total_volume': decimal.Decimal('0'), 'bid_total_volume': decimal.Decimal('0'),
        'ask_weighted_price': decimal.Decimal('0'), 'bid_weighted_price': decimal.Decimal('0'),
        'volume_imbalance_ratio': decimal.Decimal('0')
    }

    # Process Asks & Bids (using Decimal)
    ask_volume_times_price = decimal.Decimal('0')
    for i, ask in enumerate(orderbook['asks']):
        if i >= CONFIG["MAX_ORDERBOOK_DEPTH_DISPLAY"]: break
        try:
            price = decimal.Decimal(str(ask[0]))
            volume = decimal.Decimal(str(ask[1]))
        except: continue # Skip invalid data
        volume_str = format_decimal(volume, amount_prec)
        highlight_color, highlight_style = Fore.WHITE, Style.NORMAL
        if volume >= decimal.Decimal(str(volume_thresholds.get('high', 'inf'))): highlight_color, highlight_style = Fore.LIGHTRED_EX, Style.BRIGHT
        elif volume >= decimal.Decimal(str(volume_thresholds.get('medium', '0'))): highlight_color, highlight_style = Fore.RED, Style.NORMAL
        analyzed_orderbook['asks'].append({'price': price, 'volume': volume, 'volume_str': volume_str, 'color': highlight_color, 'style': highlight_style})
        analyzed_orderbook['ask_total_volume'] += volume
        ask_volume_times_price += price * volume

    bid_volume_times_price = decimal.Decimal('0')
    for i, bid in enumerate(orderbook['bids']):
        if i >= CONFIG["MAX_ORDERBOOK_DEPTH_DISPLAY"]: break
        try:
            price = decimal.Decimal(str(bid[0]))
            volume = decimal.Decimal(str(bid[1]))
        except: continue # Skip invalid data
        volume_str = format_decimal(volume, amount_prec)
        highlight_color, highlight_style = Fore.WHITE, Style.NORMAL
        if volume >= decimal.Decimal(str(volume_thresholds.get('high', 'inf'))): highlight_color, highlight_style = Fore.LIGHTGREEN_EX, Style.BRIGHT
        elif volume >= decimal.Decimal(str(volume_thresholds.get('medium', '0'))): highlight_color, highlight_style = Fore.GREEN, Style.NORMAL
        analyzed_orderbook['bids'].append({'price': price, 'volume': volume, 'volume_str': volume_str, 'color': highlight_color, 'style': highlight_style})
        analyzed_orderbook['bid_total_volume'] += volume
        bid_volume_times_price += price * volume

    # Calculations
    ask_total_vol = analyzed_orderbook['ask_total_volume']
    bid_total_vol = analyzed_orderbook['bid_total_volume']
    if ask_total_vol > 0:
        analyzed_orderbook['volume_imbalance_ratio'] = bid_total_vol / ask_total_vol
        analyzed_orderbook['ask_weighted_price'] = ask_volume_times_price / ask_total_vol
    else:
        analyzed_orderbook['volume_imbalance_ratio'] = decimal.Decimal('inf') if bid_total_vol > 0 else decimal.Decimal('0')
        analyzed_orderbook['ask_weighted_price'] = decimal.Decimal('0')
    if bid_total_vol > 0:
        analyzed_orderbook['bid_weighted_price'] = bid_volume_times_price / bid_total_vol
    else:
        analyzed_orderbook['bid_weighted_price'] = decimal.Decimal('0')

    return analyzed_orderbook, False # Success

def display_combined_analysis(analyzed_orderbook, market_info, ticker_info, trend_info, position_info):
    """ Displays Order Book, Ticker, Trend, and PNL Info """
    if not analyzed_orderbook: return # Don't display if OB fetch failed

    symbol = analyzed_orderbook['symbol']
    timestamp = analyzed_orderbook['timestamp']
    price_prec = market_info['price_precision']
    amount_prec = market_info['amount_precision']
    pnl_prec = CONFIG["PNL_PRECISION"]

    print_color(f"\n📊 Market Analysis for {Fore.MAGENTA}{Style.BRIGHT}{symbol}{Style.RESET_ALL} on Bybit Futures @ {Fore.YELLOW}{timestamp}", color=Fore.CYAN)
    print_color("=" * 75, color=Fore.CYAN)

    # --- Ticker & Trend Info ---
    current_price_str = "N/A"
    price_color = Fore.WHITE
    if ticker_info and ticker_info.get('last') is not None:
        last_price = decimal.Decimal(str(ticker_info['last']))
        current_price_str = format_decimal(last_price, price_prec)
        # Color price based on change (optional, needs 'previousClose' or similar)
        # if ticker_info.get('change') is not None:
        #     if float(ticker_info['change']) > 0: price_color = Fore.GREEN
        #     elif float(ticker_info['change']) < 0: price_color = Fore.RED

        sma_str = "N/A"
        trend_str = "Trend: -"
        trend_color = Fore.WHITE
        if trend_info.get('sma') is not None:
            sma_val = decimal.Decimal(str(trend_info['sma']))
            sma_str = format_decimal(sma_val, price_prec)
            if last_price > sma_val:
                trend_str = f"Trend: Above {CONFIG['SMA_PERIOD']}-SMA ({sma_str})"
                trend_color = Fore.GREEN
            elif last_price < sma_val:
                trend_str = f"Trend: Below {CONFIG['SMA_PERIOD']}-SMA ({sma_str})"
                trend_color = Fore.RED
            else:
                trend_str = f"Trend: On {CONFIG['SMA_PERIOD']}-SMA ({sma_str})"
                trend_color = Fore.YELLOW
        else:
             trend_str = f"Trend: SMA ({CONFIG['SMA_PERIOD']}@{CONFIG['SMA_TIMEFRAME']}) unavailable"
             trend_color = Fore.YELLOW

        print_color(f"  Last Price: {price_color}{Style.BRIGHT}{current_price_str}{Style.RESET_ALL}", end='')
        print_color(f" | {trend_color}{trend_str}{Style.RESET_ALL}")

    else:
        print_color(f"  Last Price: {Fore.YELLOW}Unavailable{Style.RESET_ALL} | Trend: {Fore.YELLOW}Unavailable{Style.RESET_ALL}")

    # --- Position & PNL Info ---
    pnl_str = "Position: None"
    pnl_color = Fore.WHITE
    if position_info.get('has_position'):
        pos = position_info['position'] # Primary position (first found)
        side = pos.get('side', 'N/A').capitalize()
        size = decimal.Decimal(str(pos.get('contracts', '0')))
        entry_price = decimal.Decimal(str(pos.get('entryPrice', '0')))
        pnl = position_info.get('unrealizedPnl', decimal.Decimal('0'))
        size_str = format_decimal(size, amount_prec)
        entry_str = format_decimal(entry_price, price_prec)
        pnl_val_str = format_decimal(pnl, pnl_prec)

        side_color = Fore.GREEN if side.lower() == 'long' else Fore.RED if side.lower() == 'short' else Fore.WHITE
        pnl_color = Fore.GREEN if pnl > 0 else Fore.RED if pnl < 0 else Fore.WHITE

        pnl_str = (f"Position: {side_color}{side} {size_str}{Style.RESET_ALL}"
                   f" | Entry: {Fore.YELLOW}{entry_str}{Style.RESET_ALL}"
                   f" | uPNL: {pnl_color}{pnl_val_str} USDT{Style.RESET_ALL}") # Assuming USDT quote
    else:
         pnl_str = f"{Fore.LIGHTBLACK_EX}Position: None{Style.RESET_ALL}"

    print_color(f"  {pnl_str}")


    # --- Order Book Display (Condensed) ---
    print_color("-" * 75, color=Fore.CYAN)
    price_width = max(10, price_prec + 4)
    volume_width = max(12, amount_prec + 6)

    # Prepare display lines
    ask_lines = []
    bid_lines = []
    for ask in reversed(analyzed_orderbook['asks']):
        price_str = format_decimal(ask['price'], price_prec)
        ask_lines.append(f"{Style.NORMAL}{Fore.WHITE}{price_str:<{price_width}}{ask['style']}{ask['color']}{ask['volume_str']:<{volume_width}}{Style.RESET_ALL}")
    for bid in analyzed_orderbook['bids']:
        price_str = format_decimal(bid['price'], price_prec)
        bid_lines.append(f"{Style.NORMAL}{Fore.WHITE}{price_str:<{price_width}}{bid['style']}{bid['color']}{bid['volume_str']:<{volume_width}}{Style.RESET_ALL}")

    # Print side-by-side if possible, or stacked
    max_rows = max(len(ask_lines), len(bid_lines))
    print_color(f"{'Asks (Sell Walls)':^{price_width+volume_width}}{'Bids (Buy Walls)':^{price_width+volume_width}}", color=Fore.LIGHTBLACK_EX)
    print_color(f"{'-'*(price_width+volume_width-1):<{price_width+volume_width}} {'-'*(price_width+volume_width-1):<{price_width+volume_width}}", color=Fore.LIGHTBLACK_EX)

    for i in range(max_rows):
        ask_part = ask_lines[i] if i < len(ask_lines) else ' ' * (price_width + volume_width)
        bid_part = bid_lines[i] if i < len(bid_lines) else ' ' * (price_width + volume_width)
        print(f"{ask_part}  {bid_part}") # Add space between columns

    # --- Spread ---
    best_ask = analyzed_orderbook['asks'][-1]['price'] if analyzed_orderbook['asks'] else decimal.Decimal('0')
    best_bid = analyzed_orderbook['bids'][0]['price'] if analyzed_orderbook['bids'] else decimal.Decimal('0')
    spread = best_ask - best_bid if best_ask > 0 and best_bid > 0 else decimal.Decimal('0')
    print_color(f"\n--- Spread: {format_decimal(spread, price_prec)} ---", color=Fore.MAGENTA, style=Style.DIM)

    # --- Volume Analysis Footer ---
    print_color("\n--- Volume Analysis ---", color=Fore.BLUE)
    print_color(f"  Total Ask: {Fore.RED}{format_decimal(analyzed_orderbook['ask_total_volume'], amount_prec)}{Style.RESET_ALL}"
                f" | Total Bid: {Fore.GREEN}{format_decimal(analyzed_orderbook['bid_total_volume'], amount_prec)}{Style.RESET_ALL}")

    imbalance_ratio = analyzed_orderbook['volume_imbalance_ratio']
    imbalance_color = Fore.WHITE
    if imbalance_ratio.is_infinite(): imbalance_color = Fore.LIGHTGREEN_EX
    elif imbalance_ratio > decimal.Decimal('1.5'): imbalance_color = Fore.GREEN
    elif imbalance_ratio < decimal.Decimal('0.67') and not imbalance_ratio.is_zero(): imbalance_color = Fore.RED
    elif imbalance_ratio.is_zero() and analyzed_orderbook['ask_total_volume'] > 0: imbalance_color = Fore.LIGHTRED_EX

    imbalance_str = "inf" if imbalance_ratio.is_infinite() else format_decimal(imbalance_ratio, 2)
    print_color(f"  Imbalance (Bid/Ask): {imbalance_color}{imbalance_str}{Style.RESET_ALL}"
                f" | Ask VWAP: {Fore.YELLOW}{format_decimal(analyzed_orderbook['ask_weighted_price'], price_prec)}{Style.RESET_ALL}"
                f" | Bid VWAP: {Fore.YELLOW}{format_decimal(analyzed_orderbook['bid_weighted_price'], price_prec)}{Style.RESET_ALL}")

    # --- Pressure Reading ---
    print_color("--- Pressure Reading ---", color=Fore.BLUE)
    if imbalance_ratio.is_infinite(): print_color("  Extreme Bid Dominance", color=Fore.LIGHTYELLOW_EX)
    elif imbalance_ratio > decimal.Decimal('1.5'): print_color("  Strong Buy Pressure", color=Fore.GREEN, style=Style.BRIGHT)
    elif imbalance_ratio < decimal.Decimal('0.67') and not imbalance_ratio.is_zero(): print_color("  Strong Sell Pressure", color=Fore.RED, style=Style.BRIGHT)
    elif imbalance_ratio.is_zero() and analyzed_orderbook['ask_total_volume'] > 0: print_color("  Extreme Ask Dominance", color=Fore.LIGHTYELLOW_EX)
    else: print_color("  Volume Relatively Balanced", color=Fore.WHITE)
    print_color("=" * 75, color=Fore.CYAN)


def place_market_order(exchange, symbol, side, amount_str, market_info):
    """ Places a market order after validation and confirmation. """
    try:
        amount = decimal.Decimal(amount_str)
        min_amount = market_info.get('min_amount', decimal.Decimal('0'))
        amount_step = market_info.get('amount_step', decimal.Decimal('1'))

        if amount <= 0:
            print_color("Amount must be positive.", color=Fore.YELLOW); return
        if min_amount > 0 and amount < min_amount:
            print_color(f"Amount < minimum ({format_decimal(min_amount, market_info['amount_precision'])}).", color=Fore.YELLOW); return
        if amount_step > 0 and (amount % amount_step) != 0:
            rounded_amount = (amount // amount_step) * amount_step
            if min_amount > 0 and rounded_amount < min_amount:
                print_color(f"Amount step invalid, rounding down makes it < minimum.", color=Fore.YELLOW); return
            round_confirm = input(f"{Fore.YELLOW}Amount step invalid. Round down to {format_decimal(rounded_amount, market_info['amount_precision'])}? (yes/no): {Style.RESET_ALL}").strip().lower()
            if round_confirm == 'yes':
                amount = rounded_amount
                amount_str = format_decimal(amount, market_info['amount_precision'])
                print_color(f"Using rounded amount: {amount_str}", color=Fore.CYAN)
            else:
                print_color("Order cancelled.", color=Fore.YELLOW); return
    except (decimal.InvalidOperation, TypeError, ValueError) as e:
        print_color(f"Invalid amount format or market data error: {e}", color=Fore.YELLOW); return

    # --- Confirmation ---
    side_color = Fore.GREEN if side == 'buy' else Fore.RED
    prompt_text = (f"{Style.BRIGHT}Confirm MARKET {side_color}{side.upper()}{Style.RESET_ALL}"
                   f"{Style.BRIGHT} order for {Fore.YELLOW}{amount_str}{Style.RESET_ALL}"
                   f"{Style.BRIGHT} {Fore.MAGENTA}{symbol}{Style.RESET_ALL}"
                   f"{Style.BRIGHT} (yes/no): {Style.RESET_ALL}")
    confirmation = input(prompt_text).strip().lower()

    if confirmation == 'yes':
        print_color(f"{Fore.CYAN}# Transmitting order...", style=Style.DIM, end='\r')
        try:
            # !!! IMPORTANT: Bybit Hedge Mode requires positionIdx !!!
            # If you get error 10001, your account is likely in Hedge Mode.
            # You MUST specify positionIdx: 1 for Buy side, 2 for Sell side when OPENING.
            # For CLOSING orders, you might need to specify reduceOnly=True or the correct positionIdx.
            # This basic market order might only work correctly in One-Way mode.
            params = {}
            # --- Add Hedge Mode Params (EXAMPLE - uncomment/adjust if needed) ---
            # is_hedge_mode = True # Set this based on your account setting
            # if is_hedge_mode:
            #     if side == 'buy':
            #         params['positionIdx'] = 1 # For LONG position
            #     else: # sell
            #         params['positionIdx'] = 2 # For SHORT position
                # For closing, you might need 'reduceOnly': True instead or specific idx
                # params['reduceOnly'] = True # If closing an existing position

            formatted_amount = exchange.amount_to_precision(symbol, float(amount))
            order = exchange.create_market_order(symbol, side, formatted_amount, params=params)
            sys.stdout.write("\033[K")
            print_color(f"\n✅ Market order placed successfully! ID: {Fore.YELLOW}{order.get('id')}", color=Fore.GREEN, style=Style.BRIGHT)

        except ccxt.InsufficientFunds as e:
             sys.stdout.write("\033[K"); print_color(f"\n❌ Insufficient Funds: {e}", color=Fore.RED, style=Style.BRIGHT)
        except ccxt.ExchangeError as e:
            sys.stdout.write("\033[K")
            # Specifically check for the position mode error
            if "10001" in str(e) and "position idx not match position mode" in str(e):
                 print_color(f"\n❌ Exchange Error: {e}", color=Fore.RED, style=Style.BRIGHT)
                 print_color(f"{Fore.YELLOW}Suggestion: Your Bybit account might be in HEDGE MODE. Market orders need 'positionIdx' parameter (1=Buy/Long, 2=Sell/Short) or switch account to One-Way mode.", style=Style.BRIGHT)
            else:
                 print_color(f"\n❌ Exchange Error placing order: {e}", color=Fore.RED, style=Style.BRIGHT)
        except Exception as e:
            sys.stdout.write("\033[K"); print_color(f"\n❌ Unexpected error during order placement: {e}", color=Fore.RED, style=Style.BRIGHT)
    else:
        print_color("Order cancelled.", color=Fore.YELLOW)

# ==============================================================================
# Main Execution Block - The Ritual Begins
# ==============================================================================

def main():
    """Main execution function."""
    print_color("*" * 75, color=Fore.RED, style=Style.BRIGHT)
    print_color("   DISCLAIMER: ADVANCED TRADING SCRIPT. USE WITH EXTREME CAUTION.", color=Fore.RED, style=Style.BRIGHT)
    print_color("   MARKET ORDERS CAN SLIP. YOU ARE RESPONSIBLE FOR ALL TRADES.", color=Fore.RED, style=Style.BRIGHT)
    print_color("   Ensure sufficient funds and understand Bybit Futures risks.", color=Fore.RED, style=Style.BRIGHT)
    print_color("*" * 75, color=Fore.RED, style=Style.BRIGHT)
    print("\n")

    if not CONFIG["API_KEY"] or not CONFIG["API_SECRET"]:
        print_color("API Key/Secret missing in environment/.env", color=Fore.RED, style=Style.BRIGHT); return

    print_color(f"{Fore.CYAN}# Initializing Bybit connection ({CONFIG['DEFAULT_EXCHANGE_TYPE']})...", style=Style.DIM)
    try:
        exchange = ccxt.bybit({
            'apiKey': CONFIG["API_KEY"], 'secret': CONFIG["API_SECRET"],
            'options': {'defaultType': CONFIG["DEFAULT_EXCHANGE_TYPE"], 'adjustForTimeDifference': True,},
            'timeout': CONFIG["CONNECT_TIMEOUT"], 'enableRateLimit': True,
        })
        # Test connection implicitly via get_market_info later
        print_color("Connection object created. Market data loads on demand.", color=Fore.GREEN)
    except Exception as e:
        print_color(f"Failed to initialize Bybit connection: {e}", color=Fore.RED, style=Style.BRIGHT); return

    symbol = ""
    market_info = None
    while not market_info:
        try:
            symbol_input = input(f"{Style.BRIGHT}{Fore.BLUE}Enter Bybit Futures symbol (e.g., BTCUSDT): {Style.RESET_ALL}").strip().upper()
            if not symbol_input: continue
            market_info = get_market_info(exchange, symbol_input) # Validates symbol too
            if market_info: symbol = symbol_input # Keep symbol if info fetched
            elif market_info is None and 'Network Error' in locals().get('last_error_msg',''): # Check if failure was network related
                 print_color("Retrying market info fetch after network error...", color=Fore.YELLOW)
                 time.sleep(CONFIG["RETRY_DELAY_NETWORK_ERROR"]) # Wait before retrying input
            # Error messages handled within get_market_info

        except (EOFError, KeyboardInterrupt): print_color("\nExiting.", color=Fore.YELLOW); return
        except Exception as e: print_color(f"Error during symbol input: {e}", color=Fore.RED); time.sleep(2)


    print_color(f"\nStarting analysis for {Fore.MAGENTA}{symbol}{Fore.CYAN}. Press Ctrl+C to exit.", color=Fore.CYAN)
    last_error_msg = "" # Track last error to avoid repetition
    while True:
        additional_data_error = False
        orderbook_error = False
        try:
            # --- Fetch all data concurrently (or sequentially) ---
            print_color(f"{Fore.CYAN}# Fetching market data...", style=Style.DIM, end='\r')
            ticker_info, ohlcv_data, position_data, additional_data_error = fetch_additional_data(
                exchange, symbol, CONFIG["SMA_TIMEFRAME"], CONFIG["SMA_PERIOD"]
            )
            analyzed_orderbook, orderbook_error = analyze_orderbook_volume(exchange, symbol, market_info)
            sys.stdout.write("\033[K") # Clear fetch message

            # --- Process fetched data ---
            trend_info = {'sma': None}
            if ohlcv_data and len(ohlcv_data) >= CONFIG["SMA_PERIOD"]:
                # Extract close prices (index 4)
                close_prices = [decimal.Decimal(str(candle[4])) for candle in ohlcv_data]
                sma = calculate_sma(close_prices, CONFIG["SMA_PERIOD"])
                if sma is not None:
                    trend_info['sma'] = sma

            position_info = {'has_position': False, 'position': None, 'unrealizedPnl': decimal.Decimal('0')}
            if position_data: # Check if list is not empty
                # Find the first relevant position (ccxt might return multiple for hedge mode)
                # You might need more logic here for specific hedge mode handling if needed
                current_pos = position_data[0]
                position_info['has_position'] = True
                position_info['position'] = current_pos
                # Calculate PNL if possible (requires mark price or last price)
                if current_pos.get('unrealizedPnl') is not None:
                    position_info['unrealizedPnl'] = decimal.Decimal(str(current_pos['unrealizedPnl']))
                elif ticker_info and ticker_info.get('last') is not None and current_pos.get('entryPrice') is not None and current_pos.get('contracts') is not None:
                     # Manual PNL calculation as fallback
                     try:
                         last_price = decimal.Decimal(str(ticker_info['last']))
                         entry_price = decimal.Decimal(str(current_pos['entryPrice']))
                         size = decimal.Decimal(str(current_pos['contracts']))
                         side = current_pos.get('side', 'long') # Default to long if side missing
                         if side == 'long':
                             position_info['unrealizedPnl'] = (last_price - entry_price) * size
                         else: # short
                             position_info['unrealizedPnl'] = (entry_price - last_price) * size
                     except Exception as pnl_e:
                         print_color(f"Warning: Could not calculate PNL manually: {pnl_e}", color=Fore.YELLOW, style=Style.DIM)


            # --- Display ---
            if analyzed_orderbook:
                 display_combined_analysis(analyzed_orderbook, market_info, ticker_info, trend_info, position_info)
            elif not orderbook_error and not additional_data_error: # If OB just empty, still show ticker/pnl
                 print_color(f"\n{Fore.YELLOW}Order book data unavailable, showing other info:{Style.RESET_ALL}")
                 display_combined_analysis({'symbol': symbol, 'timestamp': exchange.iso8601(exchange.milliseconds()), 'asks':[], 'bids':[]}, # Fake OB
                                           market_info, ticker_info, trend_info, position_info)


            # --- Action Prompt ---
            if not additional_data_error and not orderbook_error: # Only prompt if fetches were okay
                action = input(
                    f"\n{Style.BRIGHT}{Fore.BLUE}Action "
                    f"({Fore.CYAN}analyze{Fore.BLUE}/"
                    f"{Fore.GREEN}buy{Fore.BLUE}/"
                    f"{Fore.RED}sell{Fore.BLUE}/"
                    f"{Fore.YELLOW}exit{Fore.BLUE}): {Style.RESET_ALL}"
                ).strip().lower()

                if action == 'buy':
                    amount_str = input(f"{Style.BRIGHT}{Fore.GREEN}Enter quantity to BUY: {Style.RESET_ALL}").strip()
                    place_market_order(exchange, symbol, 'buy', amount_str, market_info)
                elif action == 'sell':
                    amount_str = input(f"{Style.BRIGHT}{Fore.RED}Enter quantity to SELL: {Style.RESET_ALL}").strip()
                    place_market_order(exchange, symbol, 'sell', amount_str, market_info)
                elif action == 'analyze':
                    print_color("Refreshing analysis...", color=Fore.CYAN, style=Style.DIM)
                elif action == 'exit':
                    print_color("Exiting the arcane terminal.", color=Fore.YELLOW); break
                else:
                    print_color("Invalid action.", color=Fore.YELLOW)
            else:
                 print_color("Waiting due to data fetch errors...", color=Fore.YELLOW, style=Style.DIM)
                 # Wait longer if there were API errors
                 time.sleep(CONFIG["RETRY_DELAY_NETWORK_ERROR"])


            # --- Loop Delay ---
            time.sleep(CONFIG["REFRESH_INTERVAL"])

        except KeyboardInterrupt:
            print_color("\nOperation interrupted by user. Exiting.", color=Fore.YELLOW); break
        except Exception as e:
            current_error_msg = str(e)
            # Avoid spamming the same critical error repeatedly
            if current_error_msg != last_error_msg:
                print_color(f"\nAn critical error occurred in the main loop: {e}", color=Fore.RED, style=Style.BRIGHT)
                last_error_msg = current_error_msg
            else:
                 print_color(f".", color=Fore.RED, end='') # Print dots for repeated errors

            print_color(f"Waiting {CONFIG['REFRESH_INTERVAL'] * 2}s...", color=Fore.YELLOW, style=Style.DIM)
            time.sleep(CONFIG["REFRESH_INTERVAL"] * 2)
            last_error_msg = current_error_msg # Update last error


if __name__ == '__main__':
    main()
    print_color("\nWizard Pyrmethus bids you farewell. May your insights be sharp!", color=Fore.MAGENTA, style=Style.BRIGHT)
