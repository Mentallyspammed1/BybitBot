import ccxt
from rich.console import Console
from rich.text import Text
from rich.panel import Panel
from rich.table import Table
from rich.theme import Theme
import numpy as np
import time  # Time magic for our loops

# Define a more refined mystical theme for our console output
theme = Theme({
    "title": "bold bright_cyan",      # Main title
    "section": "bold neon_green",     # Section headers
    "metric": "dim cyan",           # Metric labels
    "value": "bright_magenta",        # Primary values
    "warning": "bold bright_yellow",   # Warnings
    "error": "bold bright_red",      # Errors
    "neutral": "bold bright_cyan",     # Neutral indicators
    "positive": "bright_green",      # Positive/bullish indicators
    "negative": "bright_red",       # Negative/bearish indicators
    "dim_text": "dim",              # Dimmed text for less important info
    "pivot_level": "bright_yellow",    # Pivot levels
    "reason": "italic dim cyan",      # Prediction reasons
    "data": "neon_cyan",            # General data values
    "volume": "neon_magenta",          # Volume data
    "atr": "neon_yellow",             # ATR value
    "prediction_up": "bold bright_green blink", # Strong bullish prediction
    "prediction_down": "bold bright_red blink",  # Strong bearish prediction
    "prediction_sideways": "bold bright_yellow", # Sideways/uncertain prediction
    "rsi_oversold": "bold green",      # RSI Oversold condition
    "rsi_overbought": "bold red",     # RSI Overbought condition
    "stoch_rsi_overbought": "bold red", # Stoch RSI Overbought
    "stoch_rsi_oversold": "bold green", # Stoch RSI Oversold
})

console = Console(theme=theme)

def calculate_fibonacci_pivots(high, low, close):
    """🔮 Calculates Fibonacci Pivot Points. Unveiling support and resistance levels through mystical ratios."""
    pp = (high + low + close) / 3
    r1 = pp + 0.382 * (high - low)
    s1 = pp - 0.382 * (high - low)
    r2 = pp + 0.618 * (high - low)
    s2 = pp - 0.618 * (high - low)
    r3 = pp + 1.000 * (high - low)
    s3 = pp - 1.000 * (high - low)
    return {"PP": pp, "R1": r1, "S1": s1, "R2": r2, "S2": s2, "R3": r3, "S3": s3}

def analyze_orderbook(orderbook, current_price):
    """🔍 Analyzes the orderbook to discern immediate liquidity and pressure at the current price."""
    best_bid = orderbook['bids'][0][0] if orderbook and orderbook['bids'] else None
    best_ask = orderbook['asks'][0][0] if orderbook and orderbook['asks'] else None
    spread = best_ask - best_bid if best_bid and best_ask else None

    bid_volume_near = 0
    ask_volume_near = 0
    near_price_range = current_price * 0.005  # 0.5% range - the immediate battleground

    if orderbook and orderbook['bids']:
        for price, volume in orderbook['bids']:
            if current_price - near_price_range <= price <= current_price + near_price_range:
                bid_volume_near += volume
    if orderbook and orderbook['asks']:
        for price, volume in orderbook['asks']:
            if current_price - near_price_range <= price <= current_price + near_price_range:
                ask_volume_near += volume

    return {
        'spread': spread,
        'bid_volume_near': bid_volume_near,
        'ask_volume_near': ask_volume_near
    }

def find_nearest_pivots(pivots, current_price, num_nearest=5):
    """📍 Finds the nearest Fibonacci Pivot levels, highlighting key interaction zones."""
    pivot_levels = list(pivots.items()) # Convert dict items to list for sorting
    pivot_levels.sort(key=lambda item: abs(item[1] - current_price)) # Sort by distance to current price
    return pivot_levels[:num_nearest]

def calculate_rsi(prices, period=14):
    """📈 Calculates Relative Strength Index (RSI), measuring momentum's mystical energy."""
    if len(prices) <= period:
        return np.nan

    price_diffs = np.diff(prices)
    gains = price_diffs[price_diffs > 0]
    losses = -price_diffs[price_diffs < 0]

    avg_gain = np.mean(gains) if gains.size > 0 else 0
    avg_loss = np.mean(losses) if losses.size > 0 else 0

    if avg_loss == 0:
        return 100 if avg_gain > 0 else 50  # Extreme case handling

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_stoch_rsi(prices, rsi_period=14, stoch_period=14):
    """📊 Calculates Stochastic RSI, a refined measure of momentum within momentum."""
    rsi_values = []
    for i in range(rsi_period, len(prices)):
        rsi_values.append(calculate_rsi(prices[i-rsi_period:i+1], rsi_period))

    if not rsi_values or len(rsi_values) < stoch_period:
        return np.nan, np.nan

    stoch_rsi_k_values = []
    for i in range(stoch_period - 1, len(rsi_values)):
        period_rsi = rsi_values[i - stoch_period + 1:i+1]
        highest_rsi = np.max(period_rsi)
        lowest_rsi = np.min(period_rsi)
        if highest_rsi == lowest_rsi:
            stoch_rsi_k = 50 # Avoid division by zero
        else:
            stoch_rsi_k = 100 * (rsi_values[i] - lowest_rsi) / (highest_rsi - lowest_rsi)
        stoch_rsi_k_values.append(stoch_rsi_k)

    if not stoch_rsi_k_values:
        return np.nan, np.nan

    stoch_rsi_d = np.mean(stoch_rsi_k_values[-3:]) if len(stoch_rsi_k_values) >= 3 else np.mean(stoch_rsi_k_values)
    return stoch_rsi_k_values[-1], stoch_rsi_d

def calculate_sma(prices, period=20):
    """📉 Calculates Simple Moving Average (SMA), smoothing price action through time."""
    if len(prices) < period:
        return np.nan
    return np.mean(prices[-period:])

def calculate_vma(volumes, period=20):
    """🌊 Calculates Volume Moving Average (VMA), gauging the tide of market participation."""
    if len(volumes) < period:
        return np.nan
    return np.mean(volumes[-period:])

def calculate_ma_rsi(prices, rsi_period=20, ma_period=10):
    """🔮 Calculates Moving Average of RSI, smoothing momentum's signals for clarity."""
    rsi_values = []
    for i in range(rsi_period, len(prices)):
        rsi_values.append(calculate_rsi(prices[i-rsi_period:i+1], rsi_period))

    if not rsi_values or len(rsi_values) < ma_period:
        return np.nan
    return calculate_sma(np.array(rsi_values), ma_period)

def calculate_atr(ohlcv, period=14):
    """🔥 Calculates Average True Range (ATR), measuring market volatility's fire."""
    if len(ohlcv) <= period:
        return np.nan

    high_prices = np.array([candle[2] for candle in ohlcv])
    low_prices = np.array([candle[3] for candle in ohlcv])
    close_prices = np.array([candle[4] for candle in ohlcv])

    true_range_values = []
    for i in range(1, len(ohlcv)):
        high_low = high_prices[i] - low_prices[i]
        high_close_prev = abs(high_prices[i] - close_prices[i-1])
        low_close_prev = abs(low_prices[i] - close_prices[i-1])
        true_range_values.append(max(high_low, high_close_prev, low_close_prev))

    atr = np.mean(true_range_values[:period]) # Initial ATR
    atr_values = [atr]

    for i in range(period, len(true_range_values)):
        atr = (atr * (period - 1) + true_range_values[i]) / period # Smoothed ATR
        atr_values.append(atr)

    return atr_values[-1]

def get_price_prediction(current_price, sma_short, sma_long, rsi_value, stoch_rsi_k, stoch_rsi_d, pivots):
    """🔮 Generates a price prediction based on SMA crossover, RSI, Stoch RSI, and Fibonacci Pivots."""
    prediction_parts = []
    prediction_reasons = []

    # SMA Crossover Predictions
    if not np.isnan(sma_short) and not np.isnan(sma_long):
        if sma_short > sma_long and current_price > sma_short:
            prediction_parts.append(f"[{theme.styles['prediction_up']}]Possible UPWARD Trend[/]")
            prediction_reasons.append(f"SMA({sma_short_period}) crossed above SMA({sma_long_period}) and price is above SMA({sma_short_period}), bullish signal.")
        elif sma_long > sma_short and current_price < sma_short:
            prediction_parts.append(f"[{theme.styles['prediction_down']}]Possible DOWNWARD Trend[/]")
            prediction_reasons.append(f"SMA({sma_long_period}) crossed above SMA({sma_short_period}) and price is below SMA({sma_short_period}), bearish signal.")
        else:
            prediction_parts.append(f"[{theme.styles['prediction_sideways']}]UNCERTAIN Trend[/]")
            prediction_reasons.append("SMA crossover is unclear or price is not confirming the signal, indicating uncertainty.")
    else:
        prediction_parts.append(f"[{theme.styles['dim_text']}]N/A[/] (SMA Data Insufficient)")
        prediction_reasons.append("Insufficient SMA data for trend prediction based on moving averages.")

    # RSI Predictions
    if not np.isnan(rsi_value):
        if rsi_value < 30:
            prediction_parts.append(f"[{theme.styles['rsi_oversold']}]OVERSOLD RSI[/]")
            prediction_reasons.append(f"RSI({rsi_period}) is oversold ({rsi_value:.2f} < 30), potential for price reversal upwards.")
        elif rsi_value > 70:
            prediction_parts.append(f"[{theme.styles['rsi_overbought']}]OVERBOUGHT RSI[/]")
            prediction_reasons.append(f"RSI({rsi_period}) is overbought ({rsi_value:.2f} > 70), potential for price reversal downwards.")

    # Stoch RSI Predictions
    if not np.isnan(stoch_rsi_k) and not np.isnan(stoch_rsi_d):
        if stoch_rsi_k < 20 and stoch_rsi_d < 20:
            prediction_parts.append(f"[{theme.styles['stoch_rsi_oversold']}]OVERSOLD Stoch RSI[/]")
            prediction_reasons.append(f"Stoch RSI (%K and %D) are both oversold (K={stoch_rsi_k:.2f}, D={stoch_rsi_d:.2f} < 20), potential for upward momentum.")
        elif stoch_rsi_k > 80 and stoch_rsi_d > 80:
            prediction_parts.append(f"[{theme.styles['stoch_rsi_overbought']}]OVERBOUGHT Stoch RSI[/]")
            prediction_reasons.append(f"Stoch RSI (%K and %D) are both overbought (K={stoch_rsi_k:.2f}, D={stoch_rsi_d:.2f} > 80), potential for downward momentum.")
        elif stoch_rsi_k > stoch_rsi_d and stoch_rsi_d < 20: # Bullish crossover in oversold zone
             prediction_parts.append(f"[{theme.styles['positive']}]Stoch RSI Bullish Crossover[/]")
             prediction_reasons.append(f"Stoch RSI %K crossed above %D in oversold territory, suggesting a potential buy signal.")
        elif stoch_rsi_k < stoch_rsi_d and stoch_rsi_d > 80: # Bearish crossover in overbought zone
             prediction_parts.append(f"[{theme.styles['negative']}]Stoch RSI Bearish Crossover[/]")
             prediction_reasons.append(f"Stoch RSI %K crossed below %D in overbought territory, suggesting a potential sell signal.")

    # Fibonacci Pivot Influence
    pivot_influence = False
    for level_name, level_price in pivots.items():
        distance_percent = abs(current_price - level_price) / current_price * 100
        if distance_percent < 0.2: # Near pivot if within 0.2%
            pivot_influence = True
            level_type_name = level_name  # Default level name
            if level_name == "PP": level_type_name = "Pivot Point (PP)"
            elif level_name.startswith("R"): level_type_name = "Resistance " + level_name
            elif level_name.startswith("S"): level_type_name = "Support " + level_name

            if current_price < level_price and level_name.startswith('R'):
                prediction_parts.append(f"[{theme.styles['negative']}]Near {level_type_name}[/]")
                prediction_reasons.append(f"Price near {level_type_name} ({level_price:.6f}), potential resistance.")
            elif current_price > level_price and level_name.startswith('S'):
                prediction_parts.append(f"[{theme.styles['positive']}]Near {level_type_name}[/]")
                prediction_reasons.append(f"Price near {level_type_name} ({level_price:.6f}), potential support.")
            elif level_name == 'PP':
                prediction_parts.append(f"[{theme.styles['pivot_level']}]Near {level_type_name}[/]")
                prediction_reasons.append(f"Price near Pivot Point ({level_price:.6f}), a key level for direction changes.")

    if not prediction_parts:
        return f"[{theme.styles['neutral']}]NEUTRAL[/]", []

    prediction_text = " ".join(prediction_parts)
    return prediction_text, prediction_reasons


def main():
    """Initiates the Neon Market Analysis, fetching data and displaying insights."""
    exchange_id = 'bybit'
    exchange_class = getattr(ccxt, exchange_id)
    exchange = exchange_class({'options': {'defaultType': 'spot'}})

    symbol = console.input(f"[{theme.styles['title']}]Enter trading symbol (e.g., BTC/USDT):[/] ").strip().upper()
    timeframe = console.input(f"[{theme.styles['title']}]Enter timeframe (e.g., 15m, 1h, 1d):[/] ").strip()

    # Indicator Periods - Customizable here if needed
    rsi_period = 14
    stoch_rsi_period = 14
    sma_short_period = 20
    sma_long_period = 50
    rsi_20_period = 20
    rsi_100_period = 100
    ma_rsi_20_period = 10
    vma_period = 20
    atr_period = 14

    pivots = None
    initial_ohlcv = None

    try:
        if not (exchange.has['watchTicker'] and exchange.has['watchOrderBook'] and exchange.has['fetchOHLCV']):
            console.print(f"[{theme.styles['warning']}]Warning:[/] Exchange {exchange_id} might not fully support Websockets, using REST for data.")

        # Initial OHLCV fetch for pivots and ATR (REST, outside loop)
        limit_initial_fetch = max(2, atr_period + 1)
        initial_ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit_initial_fetch)
        if not initial_ohlcv or len(initial_ohlcv) < limit_initial_fetch:
            console.print(f"[{theme.styles['error']}]Error:[/] Failed initial OHLCV fetch for pivots/ATR. Check symbol, timeframe, and Bybit IP block.")
            return
        prev_candle = initial_ohlcv[-2]
        pivots = calculate_fibonacci_pivots(prev_candle[2], prev_candle[3], prev_candle[4]) # High, Low, Close
        atr_value_initial = calculate_atr(initial_ohlcv, atr_period)

        while True: # Main Loop for continuous market analysis
            console.clear()

            # REST API Data Fetching (Ticker, Orderbook, OHLCV)
            ticker = exchange.fetch_ticker(symbol)
            orderbook = exchange.fetch_order_book(symbol)
            limit_ohlcv_update = max(rsi_period + stoch_rsi_period + 1, sma_long_period, rsi_100_period, rsi_20_period, ma_rsi_20_period + rsi_20_period, vma_period, atr_period, 1)
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit_ohlcv_update)

            # Error Handling for REST API fetches
            if not ticker: console.print(f"[{theme.styles['warning']}]Warning:[/] Failed to fetch ticker data. Retrying in 30s...")
            if not orderbook: console.print(f"[{theme.styles['warning']}]Warning:[/] Failed to fetch order book. Retrying in 30s...")
            if not ohlcv: console.print(f"[{theme.styles['warning']}]Warning:[/] Failed to fetch OHLCV data. Retrying with last known data if available...")

            if not ticker or not orderbook or not ohlcv: # If any fetch fails, retry after delay
                time.sleep(30)
                if initial_ohlcv is not None and ohlcv is None: # Fallback to initial OHLCV if REST fetch fails in loop
                    ohlcv = initial_ohlcv
                elif ohlcv is None:
                    console.print(f"[{theme.styles['error']}]Error:[/] No OHLCV data available, indicators cannot be calculated.")
                    continue # Skip iteration if no OHLCV data
                continue # Retry loop

            initial_ohlcv = ohlcv # Update OHLCV fallback

            closes = np.array([candle[4] for candle in ohlcv])
            volumes = np.array([candle[5] for candle in ohlcv])
            current_price = ticker['last']

            # Indicator Calculations
            rsi_value = calculate_rsi(closes, rsi_period)
            stoch_rsi_k, stoch_rsi_d = calculate_stoch_rsi(closes, rsi_period, stoch_rsi_period)
            sma_short = calculate_sma(closes, sma_short_period)
            sma_long = calculate_sma(closes, sma_long_period)
            rsi_20 = calculate_rsi(closes, rsi_20_period)
            rsi_100 = calculate_rsi(closes, rsi_100_period)
            ma_of_rsi_20 = calculate_ma_rsi(closes, rsi_20_period, ma_rsi_20_period)
            vma = calculate_vma(volumes, vma_period)
            atr_value = calculate_atr(ohlcv, atr_period)

            # Price Prediction Logic
            price_prediction_text, prediction_reasons = get_price_prediction(
                current_price, sma_short, sma_long, rsi_value, stoch_rsi_k, stoch_rsi_d, pivots
            )

            # --- Rich Output Display ---
            title = Text.from_markup(f"[{theme.styles['title']}]✨ Neon Market Analysis for {symbol} ({timeframe}) on {exchange_id.capitalize()} ✨ - Updated: {time.strftime('%H:%M:%S')}[/]")
            console.print(Panel(title, title_align="center", border_style="neon_cyan"))

            console.print(Panel(f"[{theme.styles['value']}]{current_price:.6f} [/]", title=f"[{theme.styles['section']}]Current Price[/]", border_style="neon_green"))

            # Order Book Table
            orderbook_table = Table(title=f"[{theme.styles['section']}]📊 Order Book Analysis 📊[/]", show_header=True, header_style="bold bright_magenta", border_style="bright_magenta")
            orderbook_table.add_column(f"[{theme.styles['metric']}]Metric[/]", style=f"{theme.styles['metric']}", width=18)
            orderbook_table.add_column(f"[{theme.styles['data']}]Value[/]", justify="right")
            order_analysis = analyze_orderbook(orderbook, current_price)
            if order_analysis['spread'] is not None:
                orderbook_table.add_row(f"[{theme.styles['pivot_level']}]Spread[/]", f"[{theme.styles['pivot_level']}] {order_analysis['spread']:.6f}[/]")
            orderbook_table.add_row(f"[{theme.styles['positive']}]Bid Volume (Near)[/]", f"[{theme.styles['positive']}] {order_analysis['bid_volume_near']:.2f}[/]")
            orderbook_table.add_row(f"[{theme.styles['negative']}]Ask Volume (Near)[/]", f"[{theme.styles['negative']}] {order_analysis['ask_volume_near']:.2f}[/]")
            console.print(orderbook_table)

            # Fibonacci Pivots Table
            pivots_table = Table(title=f"[{theme.styles['section']}]📈 Fibonacci Pivots 📉[/]", show_header=True, header_style="bold neon_cyan", border_style="neon_cyan")
            pivots_table.add_column(f"[{theme.styles['metric']}]Level[/]", style=f"{theme.styles['metric']}", width=10)
            pivots_table.add_column(f"[{theme.styles['pivot_level']}]Price[/]", justify="right")
            pivots_table.add_column(f"[{theme.styles['dim_text']} italic]Distance[/]", justify="right", style=f"{theme.styles['dim_text']} italic")
            nearest_pivots = find_nearest_pivots(pivots, current_price, num_nearest=5)
            for level_name, level_price in nearest_pivots:
                distance = abs(level_price - current_price)
                pivots_table.add_row(f"[{theme.styles['pivot_level']}]{level_name}[/]", f"[{theme.styles['value']}] {level_price:.6f}[/]", f"[{theme.styles['dim_text']}]{distance:.6f}[/]")
            console.print(pivots_table)

            # Trend & Indicator Analysis Table
            trend_table = Table(title=f"[{theme.styles['section']}]🔍 Trend & Indicator Analysis 🔍[/]", show_header=True, header_style="bold bright_yellow", border_style="bright_yellow")
            trend_table.add_column(f"[{theme.styles['metric']}]Indicator[/]", style=f"{theme.styles['metric']}", width=28) # Adjusted width
            trend_table.add_column(f"[{theme.styles['data']}]Value[/]", justify="right")

            def style_rsi(rsi): # Function to style RSI based on levels
                if np.isnan(rsi): return "[dim]N/A[/]"
                if rsi < 30: return f"[{theme.styles['rsi_oversold']}]{rsi:.2f}[/]"
                if rsi > 70: return f"[{theme.styles['rsi_overbought']}]{rsi:.2f}[/]"
                return f"[{theme.styles['data']}]{rsi:.2f}[/]"

            def style_stoch_rsi(k, d): # Function to style Stoch RSI
                if np.isnan(k) or np.isnan(d): return "[dim]N/A[/]"
                if k < 20 and d < 20: return f"[{theme.styles['stoch_rsi_oversold']}]{k:.2f} / {d:.2f}[/]"
                if k > 80 and d > 80: return f"[{theme.styles['stoch_rsi_overbought']}]{k:.2f} / {d:.2f}[/]"
                return f"[{theme.styles['data']}]{k:.2f} / {d:.2f}[/]"


            trend_table.add_row("RSI (14)", style_rsi(rsi_value))
            trend_table.add_row("Stoch RSI (%K / %D)", style_stoch_rsi(stoch_rsi_k, stoch_rsi_d))
            trend_table.add_row(f"SMA ({sma_short_period})", f"[{theme.styles['data']}]{sma_short:.2f}[/]" if not np.isnan(sma_short) else "[dim]N/A[/]")
            trend_table.add_row(f"SMA ({sma_long_period})", f"[{theme.styles['data']}]{sma_long:.2f}[/]" if not np.isnan(sma_long) else "[dim]N/A[/]")
            trend_table.add_row("RSI (20)", style_rsi(rsi_20))
            trend_table.add_row("MA RSI (20,10)", f"[{theme.styles['data']}]{ma_of_rsi_20:.2f}[/]" if not np.isnan(ma_of_rsi_20) else "[dim]N/A[/]")
            trend_table.add_row("RSI (100)", style_rsi(rsi_100))
            trend_table.add_row("Volume", f"[{theme.styles['volume']}]{volumes[-1]:.2f}[/]" if len(volumes) > 0 else "[dim]N/A[/]")
            trend_table.add_row(f"VMA ({vma_period})", f"[{theme.styles['volume']}]{vma:.2f}[/]" if not np.isnan(vma) else "[dim]N/A[/]")
            trend_table.add_row(f"ATR ({atr_period})", f"[{theme.styles['atr']}]{atr_value:.6f}[/]" if not np.isnan(atr_value) else "[dim]N/A[/]")
            trend_table.add_row("Prediction", price_prediction_text)
            if prediction_reasons:
                trend_table.add_row(f"[{theme.styles['reason']}]Prediction Reasons[/]", f"[{theme.styles['reason']}]" + ", ".join(prediction_reasons) + "[/]")

            console.print(trend_table)
            time.sleep(30)

    except ccxt.ExchangeError as e:
        console.print(f"[{theme.styles['error']}]CCXT Exchange Error:[/] {e}")
    except ccxt.NetworkError as e:
        console.print(f"[{theme.styles['error']}]CCXT Network Error:[/] {e}")
    except KeyboardInterrupt:
        console.print(f"[{theme.styles['warning']}]Exiting program...[/]")
    except Exception as e:
        console.print(f"[{theme.styles['error']}]An unexpected error occurred: [/] {e}")

if __name__ == "__main__":
    main()
