import requests
import os
from rich.console import Console
from rich.text import Text
from rich.rule import Rule
import pandas as pd
import pandas_ta as ta

console = Console()

def fetch_bybit_data(symbol="BTCUSDT", interval="15m",
limit=200):
    """
    Fetches OHLCV and orderbook data from the Bybit API.

    Args:
        symbol (str): The trading symbol (e.g., "BTCUSDT").
        interval (str): Timeframe interval (e.g., "1m", "5m",
"15m", "1h", "4h", "1d").
        limit (int): Number of historical candles to fetch.

    Returns:
        tuple: (ohlcv_df, orderbook_data) - Pandas DataFrame for
OHLCV, dict for orderbook, or (None, None) on error.
    """
    ohlcv_endpoint = f"https://api.bybit.com/v5/market/kline"
    orderbook_endpoint = f"https://api.bybit.com/v5/market/
orderbook"

    try:
        # Fetch OHLCV data
        ohlcv_params = {
            "category": "spot",
            "symbol": symbol,
            "interval": interval,
            "limit": limit
        }
        ohlcv_response = requests.get(ohlcv_endpoint,
params=ohlcv_params)
        ohlcv_response.raise_for_status()
        ohlcv_data = ohlcv_response.json()

        if ohlcv_data['retCode'] != 0:
            console.print(f"[bold red]Error fetching OHLCV data:
[/bold red] {ohlcv_data['retMsg']}")
            return None, None

        ohlcv_list = ohlcv_data['result']['list']
        ohlcv_df = pd.DataFrame(ohlcv_list,
columns=['timestamp', 'open', 'high', 'low', 'close', 'volume',
'turnover'])
        ohlcv_df['timestamp'] =
pd.to_datetime(ohlcv_df['timestamp'], unit='ms')
        for col in ['open', 'high', 'low', 'close', 'volume',
'turnover']:
            ohlcv_df[col] = pd.to_numeric(ohlcv_df[col])
        ohlcv_df = ohlcv_df.set_index('timestamp')
        ohlcv_df = ohlcv_df.sort_index()


        # Fetch Orderbook data
        orderbook_params = {
            "category": "spot",
            "symbol": symbol,
            "limit": 50  # Adjust limit as needed
        }
        orderbook_response = requests.get(orderbook_endpoint,
params=orderbook_params)
        orderbook_response.raise_for_status()
        orderbook_data = orderbook_response.json()

        if orderbook_data['retCode'] != 0:
            console.print(f"[bold red]Error fetching orderbook
data:[/bold red] {orderbook_data['retMsg']}")
            return ohlcv_df, None # Return OHLCV data even if
orderbook fails


        return ohlcv_df, orderbook_data['result']

    except requests.exceptions.RequestException as e:
        console.print_exception()
        console.print(f"[bold red]Error fetching data from Bybit
API:[/bold red] {e}")
        return None, None

def calculate_fibonacci_pivots(high, low, close, num_pivots=4):
    """
    Calculates Fibonacci pivot levels.

    Args:
        high (float): Current high price.
        low (float): Current low price.
        close (float): Current close price.
        num_pivots (int): Number of pivot levels above and below
to return.

    Returns:
        dict: Dictionary containing support and resistance
Fibonacci pivot levels.
    """
    pivot_point = (high + low + close) / 3.0

    diff_high = high - pivot_point
    diff_low = pivot_point - low

    if diff_high > diff_low:
        range_val = diff_high
    else:
        range_val = diff_low

    fib_levels = [0.236, 0.382, 0.5, 0.618, 0.786, 1.0, 1.618,
2.618, 4.236] # Common Fibonacci ratios                         
    resistances = {}
    supports = {}

    for i in range(1, num_pivots + 1):
        for level in fib_levels:                                            r_level = pivot_point + (range_val * level * i)
            s_level = pivot_point - (range_val * level * i)                 resistances[f"R{i}_Fib_{level}"] = r_level
            supports[f"S{i}_Fib_{level}"] = s_level

    # Return only the nearest levels, can be adjusted based on
needs
    nearest_resistances = {f"R{i+1}": resistances[f"R{i+1}
_Fib_0.382"] for i in range(num_pivots)} # Using 0.382 level for
nearest approximation
    nearest_supports = {f"S{i+1}": supports[f"S{i+1}_Fib_0.382"]
for i in range(num_pivots)} # Using 0.382 level for nearest     approximation


    return {"resistances": nearest_resistances, "supports":     nearest_supports}


def analyze_orderbook(orderbook_data):
    """
    Analyzes order book data (basic analysis).

    Args:
        orderbook_data (dict): Order book data from Bybit API.

    Returns:
        dict: Analysis results (basic - can be expanded).
    """
    if not orderbook_data:
        return {"bid_ask_spread": None, "bid_depth": None,
"ask_depth": None}                                              
    bids = orderbook_data.get('bids', [])                           asks = orderbook_data.get('asks', [])

    if not bids or not asks:
        return {"bid_ask_spread": None, "bid_depth": None,      "ask_depth": None}

    best_bid_price = float(bids[0][0]) if bids else None
    best_ask_price = float(asks[0][0]) if asks else None
    bid_ask_spread = best_ask_price - best_bid_price if
best_bid_price is not None and best_ask_price is not None else
None
                                                                    bid_depth = sum([float(bid[1]) for bid in bids[:5]]) # Top 5
bid depth
    ask_depth = sum([float(ask[1]) for ask in asks[:5]]) # Top 5ask depth

    return {
        "bid_ask_spread": bid_ask_spread,                               "bid_depth": bid_depth,
        "ask_depth": ask_depth
    }
                                                                
def generate_scalping_signals(ohlcv_df, orderbook_data):
    """
    Generates scalping signals based on volume, Stochastic RSI,
Fibonacci pivots, and order book analysis.                      
    Args:
        ohlcv_df (pd.DataFrame): OHLCV data from Bybit API.
        orderbook_data (dict): Order book data from Bybit API.

    Returns:
        list: A list of scalping signal dictionaries.
    """
    signals = []

    if ohlcv_df is None or ohlcv_df.empty:
        console.print("[bold yellow]Warning:[/bold yellow] No
OHLCV data received.")
        return signals

    if orderbook_data is None:
        console.print("[bold yellow]Warning:[/bold yellow] No
orderbook data received. Orderbook analysis will be limited.")

    # 1. Volume Analysis (Simple - can be enhanced)
    last_volume = ohlcv_df['volume'].iloc[-1]
    average_volume =
ohlcv_df['volume'].rolling(window=20).mean().iloc[-1] # 20      period avg volume

    volume_condition = last_volume > (average_volume * 1.5) #   Volume spike condition                                          

    # 2. Stochastic RSI                                             stoch_rsi = ta.stochrsi(ohlcv_df['close'], length=14,
rsi_length=14, k=3, d=3)
    if stoch_rsi.empty:                                                 console.print("[bold yellow]Warning:[/bold yellow] Could
not calculate Stochastic RSI.")                                         return signals

    stoch_rsi_k = stoch_rsi['STOCHRSIk_14_14_3_3'].iloc[-1]
    stoch_rsi_d = stoch_rsi['STOCHRSId_14_14_3_3'].iloc[-1]
                                                                    stoch_rsi_oversold = stoch_rsi_k < 20 and stoch_rsi_d < 20
    stoch_rsi_overbought = stoch_rsi_k > 80 and stoch_rsi_d > 80
    stoch_rsi_cross_up = stoch_rsi_k > stoch_rsi_d and
stoch_rsi_k < 50 and stoch_rsi_k[-2] <= stoch_rsi_d[-2] #K
crosses above D in oversold/neutral                                 stoch_rsi_cross_down = stoch_rsi_k < stoch_rsi_d and        stoch_rsi_k > 50 and stoch_rsi_k[-2] >= stoch_rsi_d[-2] #K
crosses below D in overbought/neutral                           
    # 3. Fibonacci Pivots                                           last_high = ohlcv_df['high'].iloc[-1]                           last_low = ohlcv_df['low'].iloc[-1]
    last_close = ohlcv_df['close'].iloc[-1]
    fib_pivots = calculate_fibonacci_pivots(last_high, last_low,
last_close)
    nearest_resistances = fib_pivots['resistances']
    nearest_supports = fib_pivots['supports']

    # 4. Order Book Analysis                                        orderbook_analysis = analyze_orderbook(orderbook_data)
    bid_ask_spread = orderbook_analysis['bid_ask_spread']
    bid_depth = orderbook_analysis['bid_depth']
    ask_depth = orderbook_analysis['ask_depth']

    # --- Signal Logic --- (Basic Example - Refine and expand
significantly)

    entry_price = last_close # Default entry at current close,
adjust based on signal

    # Long Signal Condition Example (Highly simplified)
    if stoch_rsi_cross_up and volume_condition and bid_depth >
ask_depth * 1.2 : #Stoch RSI cross up, volume spike, stronger
bid depth
        signal_details = {
            "Signal Type": "[bold bright_green]LONG[/bold
bright_green]",
            "Entry Price": f"[green]{entry_price:.2f}[/green]",
            "Exit Price": f"[yellow]
{nearest_resistances.get('R1', entry_price * 1.01):.2f}[/
yellow]", # Example exit at R1
            "TP": f"[blue]{nearest_resistances.get('R2',
entry_price * 1.02):.2f}[/blue]", # Example TP at R2
            "SL": f"[red]{nearest_supports.get('S1', entry_price
* 0.99):.2f}[/red]",   # Example SL at S1
            "Confidence Level": "[magenta]65%[/magenta]",
            "Commentary": f"[white]Stoch RSI K crossed above D,
Volume spike, Bid depth stronger. Potential Long at
{nearest_supports.get('S1', 'support level')}[/white]"
        }
        signals.append(signal_details)


    # Short Signal Condition Example (Highly simplified)
    if stoch_rsi_cross_down and volume_condition and ask_depth >
bid_depth * 1.2: #Stoch RSI cross down, volume spike, stronger
ask depth
        signal_details = {
            "Signal Type": "[bold bright_magenta]SHORT[/bold
bright_magenta]",
            "Entry Price": f"[magenta]{entry_price:.2f}[/
magenta]",
            "Exit Price": f"[yellow]{nearest_supports.get('S1',
entry_price * 0.99):.2f}[/yellow]", # Example exit at S1
            "TP": f"[blue]{nearest_supports.get('S2',
entry_price * 0.98):.2f}[/blue]", # Example TP at S2
            "SL": f"[red]{nearest_resistances.get('R1',
entry_price * 1.01):.2f}[/red]",   # Example SL at R1
            "Confidence Level": "[magenta]70%[/magenta]",
            "Commentary": f"[white]Stoch RSI K crossed below D,
Volume spike, Ask depth stronger. Potential Short at
{nearest_resistances.get('R1', 'resistance level')}[/white]"
        }
        signals.append(signal_details)


    return signals


if __name__ == "__main__":
    symbol_to_trade = "BTCUSDT"
    timeframe = "15m" # Adjust timeframe as needed
    ohlcv_data, orderbook_data =
fetch_bybit_data(symbol_to_trade, interval=timeframe)

    if ohlcv_data is not None:
        signals = generate_scalping_signals(ohlcv_data,
orderbook_data)

        console.print(Rule(f"[bold bright_cyan]Bybit Scalping
Signals for {symbol_to_trade} ({timeframe})[/bold bright_cyan]",
align="center"))

        if signals:
            for signal in signals:
                signal_type_text = Text.from_markup(f"Signal
Type: {signal['Signal Type']}")
                entry_text = Text.from_markup(f"[bold
green]Entry:[/bold green] {signal['Entry Price']}")
                exit_text = Text.from_markup(f"[bold
yellow]Exit:[/bold yellow] {signal['Exit Price']}")
                tp_text = Text.from_markup(f"[bold blue]TP:[/
bold blue] {signal['TP']}")
                sl_text = Text.from_markup(f"[bold red]SL:[/bold
red] {signal['SL']}")
                confidence_text = Text.from_markup(f"[bold
cyan]Confidence:[/bold cyan] {signal['Confidence Level']}")
                commentary_text = Text.from_markup(f"[bold
white]Reasoning:[/bold white] {signal['Commentary']}")

                console.print(signal_type_text)
                console.print(entry_text)
                console.print(exit_text)
                console.print(tp_text)
                console.print(sl_text)
                console.print(confidence_text)
                console.print(commentary_text)
                console.print("-" * 30, style="dim")
                console.print()
        else:
            console.print("[bold blue]No scalping signals
generated[/bold blue] based on current market data.")

        console.print(Rule("[bold bright_cyan]End of Signals[/
bold bright_cyan]", align="center"))
    else:
        console.print("[bold red]Failed to retrieve Bybit market
data. Cannot generate signals.[/bold red]")
