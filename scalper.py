import ccxt
import os
from rich.console import Console
from rich.text import Text
from rich.rule import Rule
import pandas as pd
import pandas_ta as ta

console = Console()

class BybitScalper:
    def __init__(self, api_key=None, api_secret=None):
        self.exchange = None
        self.initialize_exchange(api_key, api_secret)
        
    def initialize_exchange(self, api_key=None, api_secret=None):
        """Initialize CCXT exchange with credentials."""
        try:
            self.exchange = ccxt.bybit({
                'apiKey': api_key or os.getenv('BYBIT_API_KEY'),
                'secret': api_secret or os.getenv('BYBIT_SECRET'),
                'options': {'defaultType': 'spot'}
            })
            self.exchange.check_required_credentials()
            self.exchange.load_markets()
        except Exception as e:
            console.print(f"[bold red]Error initializing exchange:[/red] {e}")
            raise

    def fetch_data(self, symbol="BTCUSDT", timeframe="15m", limit=200):
        """Fetch OHLCV and orderbook data using CCXT."""
        try:
            ccxt_timeframe = self._convert_timeframe(timeframe)
            ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe=ccxt_timeframe, limit=limit)
            ohlcv_df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            ohlcv_df['timestamp'] = pd.to_datetime(ohlcv_df['timestamp'], unit='ms')
            
            orderbook = self.exchange.fetch_order_book(symbol)
            return ohlcv_df.set_index('timestamp'), orderbook
        except Exception as e:
            console.print(f"[bold red]Error fetching market data:[/red] {e}")
            return None, None

    def _convert_timeframe(self, timeframe):
        """Convert custom timeframe to CCXT format."""
        timeframe_map = {
            '15m': '15m',
            '1h': '1h',
            '4h': '4h',
            '1d': '1d'
        }
        return timeframe_map.get(timeframe, '1h')

    def calculate_fibonacci_pivots(self, high, low, close, num_pivots=4):
        """Calculate Fibonacci pivot levels."""
        pivot_point = (high + low + close) / 3.0
        
        diff_high = high - pivot_point
        diff_low = pivot_point - low
        
        if diff_high > diff_low:
            range_val = diff_high
        else:
            range_val = diff_low
            
        fib_levels = [0.236, 0.382, 0.5, 0.618, 0.786, 1.0, 1.618, 2.618, 4.236]
        
        resistances = {}
        supports = {}
        
        for i in range(1, num_pivots + 1):
            for level in fib_levels:
                r_level = pivot_point + (range_val * level * i)
                s_level = pivot_point - (range_val * level * i)
                resistances[f"R{i}_Fib_{level}"] = r_level
                supports[f"S{i}_Fib_{level}"] = s_level
        
        nearest_resistances = {f"R{i+1}": resistances[f"R{i+1}_Fib_0.382"] 
                             for i in range(num_pivots)}
        nearest_supports = {f"S{i+1}": supports[f"S{i+1}_Fib_0.382"] 
                          for i in range(num_pivots)}
        
        return {"resistances": nearest_resistances, "supports": nearest_supports}

    def analyze_orderbook(self, orderbook_data, depth_levels=5):
        """Analyze order book data."""
        bids = orderbook_data.get('bids', [])
        asks = orderbook_data.get('asks', [])

        if not bids or not asks:
            return {"bid_ask_spread": None, "bid_depth": None,
                    "ask_depth": None, "bid_ask_ratio": None}

        best_bid_price = float(bids[0][0])
        best_ask_price = float(asks[0][0])
        
        bid_ask_spread = best_ask_price - best_bid_price
        
        bid_depth = sum(float(bid[1]) for bid in bids[:depth_levels])
        ask_depth = sum(float(ask[1]) for ask in asks[:depth_levels])
        
        bid_ask_ratio = bid_depth / ask_depth if ask_depth != 0 else None
        
        return {
            "bid_ask_spread": bid_ask_spread,
            "bid_depth": bid_depth,
            "ask_depth": ask_depth,
            "bid_ask_ratio": bid_ask_ratio
        }

    def generate_scalping_signals(self, ohlcv_df, orderbook_data,
                                volume_multiplier=1.5,
                                stoch_rsi_oversold_level=30,
                                stoch_rsi_overbought_level=70,
                                bid_ask_depth_ratio_threshold=1.2):
        """Generate scalping signals based on multiple indicators."""
        signals = []
        
        if ohlcv_df is None or ohlcv_df.empty:
            console.print("[bold yellow]Warning:[/yellow] No OHLCV data received.")
            return signals

        if orderbook_data is None:
            console.print("[bold yellow]Warning:[/yellow] No orderbook data received.")

        last_volume = ohlcv_df['volume'].iloc[-1]
        avg_volume = ohlcv_df['volume'].rolling(window=20).mean().iloc[-1]
        volume_condition = last_volume > (avg_volume * volume_multiplier)

        ema_short = ta.ema(ohlcv_df['close'], length=9).iloc[-1]
        ema_long = ta.ema(ohlcv_df['close'], length=21).iloc[-1]
        uptrend = ema_short > ema_long
        downtrend = ema_short < ema_long

        stoch_rsi = ta.stochrsi(ohlcv_df['close'], length=14, rsi_length=14, k=3, d=3)
        if stoch_rsi.empty:
            console.print("[bold yellow]Warning:[/yellow] Could not calculate Stochastic RSI.")
            return signals

        stoch_rsi_k = stoch_rsi['STOCHRSIk_14_14_3_3'].iloc[-1]
        stoch_rsi_d = stoch_rsi['STOCHRSId_14_14_3_3'].iloc[-1]

        stoch_rsi_oversold = stoch_rsi_k < stoch_rsi_oversold_level and stoch_rsi_d < stoch_rsi_oversold_level
        stoch_rsi_overbought = stoch_rsi_k > stoch_rsi_overbought_level and stoch_rsi_d > stoch_rsi_overbought_level
        stoch_rsi_cross_up = stoch_rsi_k > stoch_rsi_d and stoch_rsi_k < 50 and stoch_rsi_k[-2] <= stoch_rsi_d[-2]
        stoch_rsi_cross_down = stoch_rsi_k < stoch_rsi_d and stoch_rsi_k > 50 and stoch_rsi_k[-2] >= stoch_rsi_d[-2]

        last_high = ohlcv_df['high'].iloc[-1]
        last_low = ohlcv_df['low'].iloc[-1]
        last_close = ohlcv_df['close'].iloc[-1]
        fib_pivots = self.calculate_fibonacci_pivots(last_high, last_low, last_close)
        
        orderbook_analysis = self.analyze_orderbook(orderbook_data)
        bid_ask_spread = orderbook_analysis['bid_ask_spread']
        bid_depth = orderbook_analysis['bid_depth']
        ask_depth = orderbook_analysis['ask_depth']
        bid_ask_ratio = orderbook_analysis['bid_ask_ratio']

        entry_price = last_close
        atr = ta.atr(ohlcv_df['high'], ohlcv_df['low'], ohlcv_df['close'], length=14).iloc[-1]
        atr_multiplier_tp = 2
        atr_multiplier_sl = 1.5

        if uptrend and stoch_rsi_cross_up and volume_condition and \
           bid_ask_ratio is not None and bid_ask_ratio > bid_ask_depth_ratio_threshold:
            sl_price = entry_price - (atr * atr_multiplier_sl)
            tp_price = entry_price + (atr * atr_multiplier_tp)

            signal_details = {
                "Signal Type": "[bold bright_green]LONG[/bold green]",
                "Entry Price": f"[green]{entry_price:.2f}[/green]",
                "Exit Price": f"[yellow]{nearest_resistances.get('R1', tp_price):.2f}[/yellow]",
                "TP": f"[blue]{tp_price:.2f}[/blue]",
                "SL": f"[red]{sl_price:.2f}[/red]",
                "Confidence Level": "[magenta]75%[/magenta]",
                "Commentary": f"[white]Uptrend (EMA), Stoch RSI K crossed up, Volume spike, Bid depth stronger.[/white]",
                "Stoch RSI K": f"{stoch_rsi_k:.2f}",
                "Stoch RSI D": f"{stoch_rsi_d:.2f}",
                "Volume": f"{last_volume:.2f}",
                "Avg Volume": f"{average_volume:.2f}",
                "Bid/Ask Ratio": f"{bid_ask_ratio:.2f}",
                "ATR": f"{atr:.4f}",
                "EMA9": f"{ema_short:.2f}",
                "EMA21": f"{ema_long:.2f}",
            }
            signals.append(signal_details)

        if downtrend and stoch_rsi_cross_down and volume_condition and \
           bid_ask_ratio is not None and bid_ask_ratio < (1/bid_ask_depth_ratio_threshold):
            sl_price = entry_price + (atr * atr_multiplier_sl)
            tp_price = entry_price - (atr * atr_multiplier_tp)

            signal_details = {
                "Signal Type": "[bold bright_magenta]SHORT[/bold magenta]",
                "Entry Price": f"[magenta]{entry_price:.2f}[magenta]",
                "Exit Price": f"[yellow]{nearest_supports.get('S1', tp_price):.2f}[/yellow]",
                "TP": f"[blue]{tp_price:.2f}[/blue]",
                "SL": f"[red]{sl_price:.2f}[/red]",
                "Confidence Level": "[magenta]70%[/magenta]",
                "Commentary": f"[white]Downtrend (EMA), Stoch RSI K crossed down, Volume spike, Ask depth stronger.[/white]",
                "Stoch RSI K": f"{stoch_rsi_k:.2f}",
                "Stoch RSI D": f"{stoch_rsi_d:.2f}",
                "Volume": f"{last_volume:.2f}",
                "Avg Volume": f"{average_volume:.2f}",
                "Bid/Ask Ratio": f"{bid_ask_ratio:.2f}",
                "ATR": f"{atr:.4f}",
                "EMA9": f"{ema_short:.2f}",
                "EMA21": f"{ema_long:.2f}",
            }
            signals.append(signal_details)

        return signals

def get_symbol_input():
    """Get and validate symbol input from user."""
    while True:
        symbol = input("\nEnter symbol (e.g., BTCUSDT, ETHUSDT) or 'q' to quit: ").upper()
        if symbol == 'Q':
            return None
        if symbol:
            return symbol
        console.print("[bold red]Please enter a valid symbol[/red]")

def get_timeframe_input():
    """Get and validate timeframe input from user."""
    valid_timeframes = ['15m', '1h', '4h', '1d']
    while True:
        timeframe = input("Enter timeframe (15m/1h/4h/1d) or 'q' to quit: ").lower()
        if timeframe == 'q':
            return None
        if timeframe in valid_timeframes:
            return timeframe
        console.print(f"[bold red]Please enter a valid timeframe ({', '.join(valid_timeframes)})[/red]")

def main():
    """Main function with user input loop."""
    console.print(Rule("Bybit Scalping Signal Generator", align="center"))
    
    scalper = BybitScalper()
    
    while True:
        symbol = get_symbol_input()
        if symbol is None:
            break
            
        timeframe = get_timeframe_input()
        if timeframe is None:
            break
            
        console.print(Rule(f"Analyzing {symbol} ({timeframe})", align="center"))
        
        ohlcv_data, orderbook_data = scalper.fetch_data(symbol, timeframe)
        
        if ohlcv_data is not None:
            signals = scalper.generate_scalping_signals(ohlcv_data, orderbook_data)
            
            if signals:
                for signal in signals:
                    console.print(Text.from_markup(f"Signal Type: {signal['Signal Type']}"))
                    console.print(Text.from_markup(f"[bold green]Entry:[/green] {signal['Entry Price']}"))
                    console.print(Text.from_markup(f"[bold yellow]Exit:[/yellow] {signal['Exit Price']}"))
                    console.print(Text.from_markup(f"[bold blue]TP:[/blue] {signal['TP']}"))
                    console.print(Text.from_markup(f"[bold red]SL:[/red] {signal['SL']}"))
                    console.print(Text.from_markup(f"[bold cyan]Confidence:[/cyan] {signal['Confidence Level']}"))
                    console.print(Text.from_markup(f"[bold white]Reasoning:[/white] {signal['Commentary']}"))
                    
                    console.print(f"  Stoch RSI (K, D): ({signal.get('Stoch RSI K', 'N/A')}, "
                                f"{signal.get('Stoch RSI D', 'N/A')})")
                    console.print(f"  Volume (Current, Avg): ({signal.get('Volume', 'N/A')}, "
                                f"{signal.get('Avg Volume', 'N/A')})")
                    console.print(f"  Bid/Ask Ratio: {signal.get('Bid/Ask Ratio', 'N/A')}")
                    console.print(f"  ATR: {signal.get('ATR', 'N/A')}")
                    console.print(f"  EMA (9, 21): ({signal.get('EMA9', 'N/A')}, "
                                f"{signal.get('EMA21', 'N/A')})")
                    
                    console.print("-" * 30, style="dim")
                    console.print()
            else:
                console.print("[bold blue]No scalping signals generated[/blue]")
        
        console.print(Rule("Press Enter to continue or 'q' to quit", align="center"))
        if input().lower() == 'q':
            break

    console.print(Rule("Thank you for using Bybit Scalping Signal Generator", align="center"))

if __name__ == "__main__":
    main()
