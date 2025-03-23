```python
#!/usr/bin/env python3

import os
import json
import time
import hmac
import hashlib
import requests
from dotenv import load_dotenv
from technicalindicators import SMA, StochasticRSI, ATR, MACD
from rich.console import Console
from rich.syntax import Syntax
from rich.theme import Theme
from rich.panel import Panel

load_dotenv()

# Mystical Theme for the Terminal
mystical_theme = Theme({
    "info": "cyan",
    "debug": "yellow",
    "error": "bold red",
    "signal_buy": "bold green",
    "signal_sell": "bold red",
    "indicator": "magenta",
    "value": "green",
    "reason": "italic cyan",
    "confidence": "bold white",
    "trade_positive": "green",
    "trade_negative": "red",
    "trade_neutral": "yellow",
    "pivot_level": "blue",
    "order_book_depth": "bold blue",
})

console = Console(theme=mystical_theme)

# Logger enchanted with Rich
class Logger:
    def info(self, msg):
        console.print(f"[info][[{time.strftime('%Y-%m-%d %H:%M:%S')}] [INFO]][/] {msg}")

    def debug(self, msg):
        if os.getenv('LOG_LEVEL') == 'DEBUG':
            console.print(f"[debug][[{time.strftime('%Y-%m-%d %H:%M:%S')}] [DEBUG]][/] {msg}")

    def error(self, msg):
        console.print(f"[error][[{time.strftime('%Y-%m-%d %H:%M:%S')}] [ERROR]][/] {msg}")

logger = Logger()

# Cache imbued with temporal magic (TTL)
class Cache:
    def __init__(self):
        self.data = {}

    def get(self, key):
        entry = self.data.get(key)
        if not entry or time.time() - entry['timestamp'] > entry['ttl']:
            return None
        return entry['value']

    def set(self, key, value, ttl):
        self.data[key] = {'value': value, 'timestamp': time.time(), 'ttl': ttl}

# Market Data Provider, weaver of market insights
class MarketDataProvider:
    def __init__(self, config):
        self.config = config
        self.logger = logger
        self.api_key = os.getenv('BYBIT_API_KEY')
        self.api_secret = os.getenv('BYBIT_API_SECRET')
        self.base_url = os.getenv('BYBIT_API_BASE_URL') or 'https://api.bybit.com'
        self.cache = Cache()

    async def fetch_ohlcv(self, symbol, timeframe, limit, retries=0):
        cache_key = f"{symbol}-{timeframe}-{limit}"
        cached_data = self.cache.get(cache_key)
        if cached_data:
            return cached_data

        params = {
            'category': 'linear',
            'symbol': symbol.replace('/', ''),
            'interval': timeframe,
            'limit': limit
        }
        try:
            response = requests.get(f"{self.base_url}/v5/market/kline", params=params)
            response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)
            data = response.json()['result']['list']
            if len(data) < limit * 0.9:
                raise Exception('Insufficient OHLCV data received')
            reversed_data = [[int(d[0]), float(d[1]), float(d[2]), float(d[3]), float(d[4]), float(d[5])] for d in reversed(data)]
            self.cache.set(cache_key, reversed_data, self.config['cache_ttl'])
            self.logger.debug(f"Fetched OHLCV data for {symbol}-{timeframe}")
            return reversed_data
        except requests.exceptions.RequestException as e:
            return await self._handle_retry(e, retries, f"OHLCV fetch for {symbol}", lambda: self.fetch_ohlcv(symbol, timeframe, limit, retries + 1))
        except Exception as e:
            return await self._handle_retry(e, retries, f"OHLCV fetch for {symbol}", lambda: self.fetch_ohlcv(symbol, timeframe, limit, retries + 1))

    async def fetch_order_book(self, symbol, retries=0):
        cache_key = symbol
        cached_order_book = self.cache.get(cache_key)
        if cached_order_book:
            return cached_order_book

        params = {'category': 'linear', 'symbol': symbol.replace('/', '')}
        try:
            response = requests.get(f"{self.base_url}/v5/market/orderbook", params=params)
            response.raise_for_status()
            data = response.json()['result']
            order_book = {
                'bids': [[float(price), float(volume)] for price, volume in data['b']],
                'asks': [[float(price), float(volume)] for price, volume in data['a']]
            }
            self.cache.set(cache_key, order_book, self.config['cache_ttl'])
            self.logger.debug(f"Fetched order book for {symbol}")
            return order_book
        except requests.exceptions.RequestException as e:
            return await self._handle_retry(e, retries, f"Order book fetch for {symbol}", lambda: self.fetch_order_book(symbol, retries + 1))

    async def fetch_balance(self, retries=0):
        timestamp = str(int(time.time() * 1000))
        params = {
            'accountType': 'UNIFIED',
            'timestamp': timestamp,
            'recvWindow': '5000',
            'apiKey': self.api_key
        }
        query_string = '&'.join([f"{k}={v}" for k, v in sorted(params.items())])
        sign = hmac.new(self.api_secret.encode('utf-8'), query_string.encode('utf-8'), hashlib.sha256).hexdigest()

        headers = {
            'X-BAPI-API-KEY': self.api_key,
            'X-BAPI-SIGN': sign,
            'X-BAPI-TIMESTAMP': timestamp,
            'X-BAPI-RECV-WINDOW': '5000'
        }
        try:
            response = requests.get(f"{self.base_url}/v5/account/wallet-balance", headers=headers, params=params)
            response.raise_for_status()
            response_data = response.json()
            if response_data['retCode'] != 0:
                raise Exception(f"API Error: {response_data['retMsg']}")
            total_equity = response_data['result']['list'][0]['totalEquity'] if response_data['result']['list'] else '0'
            self.logger.debug(f"Balance fetched: {total_equity} USDT")
            return {'USDT': float(total_equity)}
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Balance fetch attempt {retries + 1} failed: {e}")
            return await self._handle_retry(e, retries, 'Balance fetch', lambda: self.fetch_balance(retries + 1))
        except Exception as e:
            self.logger.error(f"Balance fetch attempt {retries + 1} failed: {e}")
            return await self._handle_retry(e, retries, 'Balance fetch', lambda: self.fetch_balance(retries + 1))

    async def _handle_retry(self, error, retries, action, retry_fn):
        if retries < self.config['max_retries']:
            delay = self.config['retry_delay'] * (2**retries) / 1000 # Convert ms to seconds for sleep
            self.logger.debug(f"Retrying {action} ({retries + 1}/{self.config['max_retries']}) after {delay*1000:.0f}ms")
            time.sleep(delay)
            return await retry_fn()
        else:
            raise Exception(f"Failed to {action} after {self.config['max_retries']} retries: {error}")

# Indicator Engine, alchemist of market signals
class IndicatorEngine:
    def __init__(self, config, market_data):
        self.config = config
        self.market_data = market_data
        self.logger = logger

    async def calculate_all(self, candles, order_book):
        if not candles or len(candles) < self.config['history_candles']:
            self.logger.error('Insufficient candle data to weave indicators.')
            return None

        closes = [candle[4] for candle in candles]
        highs = [candle[2] for candle in candles]
        lows = [candle[3] for candle in candles]
        volumes = [candle[5] for candle in candles]

        fib_pivots = await self.calculate_fibonacci_pivots() if self.config['fibonacci_pivots']['enabled'] else None
        pivot_order_book = self.analyze_order_book_for_pivots(order_book, fib_pivots) if fib_pivots else {}

        return {
            'price': closes[-1],
            'volume': volumes[-1],
            'average_volume': sum(volumes[-self.config['volume_confirmation']['lookback']:]) / self.config['volume_confirmation']['lookback'],
            'sma': {
                'fast': SMA(closes, period=self.config['indicators']['sma']['fast'])[-1],
                'slow': SMA(closes, period=self.config['indicators']['sma']['slow'])[-1],
            },
            'stoch_rsi': StochasticRSI(closes, **self.config['indicators']['stoch_rsi'])[-1],
            'atr': ATR(highs, lows, closes, period=self.config['indicators']['atr']['period'])[-1],
            'macd': MACD(closes, **self.config['indicators']['macd'])[-1],
            'fibonacci_pivots': fib_pivots,
            'pivot_order_book_analysis': pivot_order_book,
        }

    async def calculate_fibonacci_pivots(self):
        daily_candles = await self.market_data.fetch_ohlcv(self.config['symbol'], self.config['fibonacci_pivots']['period'], 2)
        if not daily_candles or len(daily_candles) < 2:
            return None

        _, prev_day = daily_candles
        high, low, close = prev_day[2], prev_day[3], prev_day[4]
        pivot = (high + low + close) / 3
        return {
            'pivot': pivot,
            'r1': 2 * pivot - low,
            's1': 2 * pivot - high,
            'r2': pivot + (high - low),
            's2': pivot - (high - low),
            'r3': high + 2 * (pivot - low),
            's3': low - 2 * (high - pivot),
        }

    def analyze_order_book_for_pivots(self, order_book, pivot_levels):
        if not order_book or not pivot_levels:
            return {}
        range_percentage = self.config['fibonacci_pivots']['order_book_range_percentage']
        depth_scores = {}

        for level in ['pivot', 's1', 's2', 'r1', 'r2']:
            pivot_level = pivot_levels.get(level)
            if not pivot_level:
                continue

            bid_depth = sum(vol for price, vol in order_book['bids'] if pivot_level * (1 - range_percentage) <= price <= pivot_level * (1 + range_percentage))
            ask_depth = sum(vol for price, vol in order_book['asks'] if pivot_level * (1 - range_percentage) <= price <= pivot_level * (1 + range_percentage))
            depth_scores[level] = {'bid_depth': bid_depth, 'ask_depth': ask_depth}
        return depth_scores

# Signal Generator, diviner of market actions
class SignalGenerator:
    def __init__(self, config):
        self.config = config
        self.logger = logger

    def generate(self, indicator_data):
        if not indicator_data:
            self.logger.debug('No indicator data received, signals remain hidden.')
            return []

        sma = indicator_data['sma']
        stoch_rsi = indicator_data['stoch_rsi']
        macd = indicator_data['macd']
        volume = indicator_data['volume']
        avg_volume = indicator_data['average_volume']
        price = indicator_data['price']
        fib_pivots = indicator_data['fibonacci_pivots']
        pivot_ob = indicator_data['pivot_order_book_analysis']

        trend_up = sma['fast'] > sma['slow']
        signals = []
        confidence_boost = 0
        reasons = []

        if self.config['volume_confirmation']['enabled'] and volume > avg_volume:
            confidence_boost += 5
            reasons.append('[reason]Volume Confirmed[/]')

        macd_bullish = macd['MACD'] > macd['signal'] and macd['histogram'] > 0
        macd_bearish = macd['MACD'] < macd['signal'] and macd['histogram'] < 0

        if self.config['fibonacci_pivots']['enabled'] and fib_pivots:
            pivot_boost, order_book_boost, pivot_reason = self.analyze_pivots(price, fib_pivots, pivot_ob)
            confidence_boost += pivot_boost + order_book_boost
            if pivot_reason:
                reasons.append(f'[reason]{pivot_reason}[/]')

        if stoch_rsi['k'] < self.config['thresholds']['oversold'] and stoch_rsi['d'] < self.config['thresholds']['oversold'] and trend_up:
            confidence = 75 + (10 if trend_up else 0) + confidence_boost + (10 if macd_bullish else 0)
            if macd_bullish:
                reasons.append('[reason]Bullish MACD[/]')
            confidence = max(50, confidence)
            signals.append({
                'type': 'BUY',
                'reason': f"Oversold Momentum with Uptrend & Confirmations, {', '.join(reasons)}",
                'confidence': confidence,
                'timestamp': int(time.time()),
            })

        if stoch_rsi['k'] > self.config['thresholds']['overbought'] and stoch_rsi['d'] > self.config['thresholds']['overbought'] and not trend_up:
            confidence = 75 + (10 if not trend_up else 0) + confidence_boost + (10 if macd_bearish else 0)
            if macd_bearish:
                reasons.append('[reason]Bearish MACD[/]')
            confidence = max(50, confidence)
            signals.append({
                'type': 'SELL',
                'reason': f"Overbought Momentum with Downtrend & Confirmations, {', '.join(reasons)}",
                'confidence': confidence,
                'timestamp': int(time.time()),
            })

        return [signal for signal in signals if signal['confidence'] >= self.config['thresholds']['min_confidence']]

    def analyze_pivots(self, price, fib_pivots, pivot_ob):
        proximity = self.config['fibonacci_pivots']['proximity_percentage']
        buy_levels = self.config['fibonacci_pivots']['levels_for_buy_entry']
        sell_levels = self.config['fibonacci_pivots']['levels_for_sell_entry']
        pivot_boost = 0
        order_book_boost = 0
        reason = ''

        for is_buy, levels in [(True, buy_levels), (False, sell_levels)]:
            for level in levels:
                pivot_level_val = fib_pivots.get(level.lower())
                if not pivot_level_val or abs(price - pivot_level_val) / price > proximity:
                    continue

                level_boost = self.config['fibonacci_pivots']['pivot_weight']
                ob_boost = 0
                if pivot_ob and pivot_ob.get(level.lower()):
                    ob_depth = pivot_ob[level.lower()]['bid_depth'] if is_buy else pivot_ob[level.lower()]['ask_depth']
                    ob_boost = (ob_depth * self.config['fibonacci_pivots']['order_book_weight'] / 1000)

                pivot_boost += level_boost
                order_book_boost += ob_boost
                reason = f'{level.upper()} Pivot Level Proximity'
                break # Prioritize the first level that triggers
            if reason: # Stop after finding a reason (prioritize levels in config order)
                break

        return pivot_boost, order_book_boost, reason

# Trading Bot, orchestrator of market maneuvers
class TradingBot:
    def __init__(self, config):
        self.config = config
        self.logger = logger
        self.market_data = MarketDataProvider(config)
        self.indicator_engine = IndicatorEngine(config, self.market_data)
        self.signal_generator = SignalGenerator(config)
        self.trade_context = {'current_position': None, 'entry_price': None, 'stop_loss': None, 'take_profit': None, 'position_size': None, 'entry_time': None}
        self.trade_history = []
        self.running = False
        self.listeners = {}

    def on(self, event, callback):
        self.listeners[event] = callback

    def emit(self, event, data):
        if event in self.listeners:
            self.listeners[event](data)

    async def initialize(self):
        try:
            balance = await self.market_data.fetch_balance()
            if not balance or 'USDT' not in balance or balance['USDT'] <= 0:
                self.logger.warn('No USDT balance available or API issue detected. Bot cannot initialize trading.')
                return False

            order_book = await self.market_data.fetch_order_book(self.config['symbol'])
            if not order_book or not order_book['bids']:
                self.logger.error('Could not fetch order book to determine position size. Initialization failed.')
                return False

            self.config['position_size'] = self.config['max_position_size_usdt'] / order_book['bids'][0][0]
            self.logger.info(f"Bot initialized with position size: [value]{self.config['position_size']:.4f}[/]")
            self.emit('initialized')
            return True
        except Exception as e:
            self.logger.error(f"Initialization incantation failed: {e}")
            return False

    async def start(self):
        self.logger.info(f"Starting bot for [indicator]{self.config['symbol']}[/] on [indicator]{self.config['timeframe']}[/]")
        if not await self.initialize():
            self.logger.error('Bot initialization failed, halting operations.')
            return
        self.running = True
        while self.running:
            await self.analysis_cycle()
            time.sleep(self.config['analysis_interval'] / 1000) # Convert ms to seconds

    async def analysis_cycle(self):
        try:
            candles = await self.market_data.fetch_ohlcv(self.config['symbol'], self.config['timeframe'], self.config['history_candles'])
            order_book = await self.market_data.fetch_order_book(self.config['symbol'])
            indicators = await self.indicator_engine.calculate_all(candles, order_book)
            if not indicators:
                return

            self.check_position(indicators['price'])
            signals = self.signal_generator.generate(indicators)
            self.emit('signals', signals)
            await self.execute_strategy(signals, indicators, order_book)

        except Exception as e:
            self.logger.error(f"Analysis cycle disrupted: {e}")

    def check_position(self, price):
        if not self.trade_context['current_position']:
            return
        if price <= self.trade_context['stop_loss']:
            self.close_position('Stop Loss Triggered')
        elif price >= self.trade_context['take_profit']:
            self.close_position('Take Profit Triggered')

    async def execute_strategy(self, signals, indicators, order_book):
        if not signals:
            return

        signal = signals[0] # Take the top signal, for now simple strategy
        signal_type_styled = f"[signal_{signal['type'].lower()}]{signal['type']}[/]"
        self.logger.info(f"Primary Signal: {signal_type_styled} ([reason]{signal['reason']}[/], [confidence]{signal['confidence']:.2f}%[/])")

        balance = await self.market_data.fetch_balance()
        usdt_available = balance.get('USDT', 0)
        entry_price = order_book['bids'][0][0] if order_book and order_book['bids'] else indicators['price'] # Fallback to current price
        position_size = min(self.config['position_size'], (usdt_available * self.config['risk_percentage']) / entry_price)

        if signal['type'] == 'BUY' and not self.trade_context['current_position']:
            position_size = min(position_size, (usdt_available * 0.9) / entry_price) # Conservative sizing
            await self.place_order('buy', position_size, entry_price, indicators['atr'], signal['reason'])
        elif signal['type'] == 'SELL' and self.trade_context['current_position']:
            await self.close_position('Sell Signal Triggered')

    async def place_order(self, side, quantity, price, atr, reason, retries=0):
        sl_tp = self.calculate_stop_loss_take_profit(side, price, atr)
        timestamp = str(int(time.time() * 1000))
        params = {
            'category': 'linear',
            'symbol': self.config['symbol'].replace('/', ''),
            'side': side.capitalize(),
            'orderType': 'Limit', # Limit order for entry
            'qty': str(quantity),
            'price': str(price),
            'apiKey': self.market_data.api_key,
            'timestamp': timestamp,
            'recvWindow': '5000'
        }
        query_string = '&'.join([f"{k}={v}" for k, v in sorted(params.items())])
        sign = hmac.new(self.market_data.api_secret.encode('utf-8'), query_string.encode('utf-8'), hashlib.sha256).hexdigest()

        headers = {
            'X-BAPI-API-KEY': self.market_data.api_key,
            'X-BAPI-SIGN': sign,
            'X-BAPI-TIMESTAMP': timestamp,
            'X-BAPI-RECV-WINDOW': '5000'
        }
        try:
            response = requests.post(f"{self.market_data.base_url}/v5/order/create", params=params, headers=headers)
            response.raise_for_status()
            response_data = response.json()
            if response_data['retCode'] != 0:
                raise Exception(f"API Order Error: {response_data['retMsg']}")

            self.logger.info(f"Order Placed: [signal_buy]{side.upper()}[/] [value]{quantity:.4f}[/] [indicator]{self.config['symbol']}[/] at [value]{price:.4f}[/], SL: [value]{sl_tp['stop_loss']:.4f}[/], TP: [value]{sl_tp['take_profit']:.4f}[/]")

            if side == 'buy':
                self.trade_context = {
                    'current_position': 'BUY',
                    'entry_price': price,
                    'stop_loss': sl_tp['stop_loss'],
                    'take_profit': sl_tp['take_profit'],
                    'position_size': quantity,
                    'entry_time': time.time(),
                }
                self.trade_history.append({**self.trade_context, 'status': 'OPEN', 'entry_reason': reason})

        except requests.exceptions.RequestException as e:
            self.logger.error(f"Order placement spell failed: {e}")
            await self._retry_order(side, quantity, price, atr, reason, retries, e)
        except Exception as e:
            self.logger.error(f"Order placement spell failed: {e}")
            await self._retry_order(side, quantity, price, atr, reason, retries, e)

    def calculate_stop_loss_take_profit(self, side, price, atr):
        offset = price * self.config['sl_tp_offset_percentage']
        if side == 'buy':
            return {
                'stop_loss': price - (atr * self.config['stop_loss_multiplier']) - offset,
                'take_profit': price + (atr * self.config['take_profit_multiplier']) + offset,
            }
        else: # sell
            return {
                'stop_loss': price + (atr * self.config['stop_loss_multiplier']) + offset,
                'take_profit': price - (atr * self.config['take_profit_multiplier']) - offset,
            }

    async def _retry_order(self, side, quantity, price, atr, reason, retries, original_error):
        if retries < self.config['max_retries']:
            delay = self.config['retry_delay'] * (2**retries) / 1000 # Convert ms to seconds
            self.logger.debug(f"Retrying order incantation ({retries + 1}/{self.config['max_retries']}) after {delay*1000:.0f}ms")
            time.sleep(delay)
            await self.place_order(side, quantity, price, atr, reason, retries + 1)
        else:
            raise Exception(f"Failed to place order after {self.config['max_retries']} retries: {original_error}")

    async def close_position(self, reason):
        if not self.trade_context['current_position']:
            return

        timestamp = str(int(time.time() * 1000))
        params = {
            'category': 'linear',
            'symbol': self.config['symbol'].replace('/', ''),
            'side': 'Sell', # Always sell to close
            'orderType': 'Market', # Market order for immediate execution
            'qty': str(self.trade_context['position_size']),
            'apiKey': self.market_data.api_key,
            'timestamp': timestamp,
            'recvWindow': '5000'
        }
        query_string = '&'.join([f"{k}={v}" for k, v in sorted(params.items())])
        sign = hmac.new(self.market_data.api_secret.encode('utf-8'), query_string.encode('utf-8'), hashlib.sha256).hexdigest()

        headers = {
            'X-BAPI-API-KEY': self.market_data.api_key,
            'X-BAPI-SIGN': sign,
            'X-BAPI-TIMESTAMP': timestamp,
            'X-BAPI-RECV-WINDOW': '5000'
        }

        try:
            response = requests.post(f"{self.market_data.base_url}/v5/order/create", params=params, headers=headers)
            response.raise_for_status()
            response_data = response.json()
            if response_data['retCode'] != 0:
                raise Exception(f"API Close Position Error: {response_data['retMsg']}")

            exit_price = float(response_data['result'].get('price', self.trade_context['entry_price'])) # Fallback to entry price if not in response
            self.logger.info(f"Position closed ([reason]{reason}[/]). Exit Price: [value]{exit_price:.4f}[/]")

            trade = next((t for t in self.trade_history if t['status'] == 'OPEN' and t['entry_time'] == self.trade_context['entry_time']), None)
            if trade:
                trade['status'] = 'CLOSED'
                trade['exit_time'] = time.time()
                trade['exit_price'] = exit_price
                trade['exit_reason'] = reason
                trade['profit'] = (exit_price - trade['entry_price']) * trade['position_size']

            self.trade_context = {'current_position': None, 'entry_price': None, 'stop_loss': None, 'take_profit': None, 'position_size': None, 'entry_time': None}

        except requests.exceptions.RequestException as e:
            self.logger.error(f"Position closing spell failed ([reason]{reason}[/]): {e}")
        except Exception as e:
            self.logger.error(f"Position closing spell failed ([reason]{reason}[/]): {e}")


    def stop(self):
        self.running = False
        self.logger.info('Bot operations ceased.')
        self.report_trade_history()

    def report_trade_history(self):
        console.rule("[bold magenta]Trade History & Performance Report[/]")
        if not self.trade_history:
            console.print("[info]No trades have been executed in this cycle.[/]")
            return

        wins = 0
        losses = 0
        total_profit = 0

        for trade in self.trade_history:
            profit_color = "trade_neutral"
            if trade['profit'] > 0:
                profit_color = "trade_positive"
                wins += 1
            elif trade['profit'] < 0:
                profit_color = "trade_negative"
                losses += 1
            total_profit += trade['profit'] if trade['profit'] else 0

            console.print(
                f"Symbol: [indicator]{self.config['symbol']}[/], Side: [indicator]{trade['current_position']}[/], "
                f"Entry: [value]{trade['entry_price']:.4f}[/], Exit: [value]{trade.get('exit_price', 'N/A')}[/], "
                f"Profit: [[{profit_color}]]{trade.get('profit', 'N/A'):.4f} USDT[/], Status: [indicator]{trade['status']}[/]"
            )

        win_rate = (wins / len(self.trade_history)) * 100 if self.trade_history else 0
        console.print(Panel(f"""
[bold]Summary[/]:
Total Trades: [value]{len(self.trade_history)}[/]
Wins: [value]{wins}[/]
Losses: [value]{losses}[/]
Win Rate: [value]{win_rate:.2f}%[/]
Total Profit: [value]{total_profit:.2f} USDT[/]
        """, title="[bold blue]Performance Metrics[/]", border_style="blue"))


# Main Execution, summoning the bot to life
async def run_bot():
    try:
        with open('config.json', 'r') as f:
            config = json.load(f)
    except FileNotFoundError:
        console.error("Configuration file 'config.json' not found. Ensure it exists in the same directory.")
        return
    except json.JSONDecodeError:
        console.error("Error decoding 'config.json'. Please check for JSON syntax errors.")
        return

    bot = TradingBot(config)

    bot.on('initialized', bot.start)
    bot.on('signals', lambda signals: [
        console.print(f"[signal_{s['type'].lower()}]{s['type']}[/] - [reason]{s['reason']}[/] ([confidence]{s['confidence']:.2f}%[/])")
        for s in signals
    ])

    import signal
    async def handle_shutdown(sig, frame):
        logger.info('Initiating graceful shutdown...')
        await bot.close_position('Shutdown signal received')
        bot.stop()
        exit(0)

    signal.signal(signal.SIGINT, handle_shutdown)
    signal.signal(signal.SIGTERM, handle_shutdown)


    await bot.initialize() # Initial initialization attempt

if __name__ == '__main__':
    import asyncio
    asyncio.run(run_bot())
```
