import asyncio
import json
import os
import sys
import signal
from datetime import datetime
import ccxt
import dotenv
from rich.console import Console
from rich.syntax import Syntax
from rich.theme import Theme

# Mystical Color Theme
PYRMETHUS_THEME = Theme({
    "info": "cyan", "success": "green", "warn": "yellow", "error": "bold red",
    "critical": "bold underline red", "signal_buy": "bold green", "signal_sell": "bold red",
    "neutral": "dim white", "heartbeat": "magenta", "profit": "bold green", "loss": "bold red",
    "debug": "dim blue", "silly": "dim purple", "code": "bold white", "config": "italic yellow",
    "timestamp": "dim white", "level": "bold blue", "signal_reason": "italic green",
    "confidence": "bold yellow", "indicator": "cyan", "value": "magenta", "order_type": "bold magenta",
    "order_side": "bold green", "order_quantity": "bold cyan", "order_price": "bold blue",
    "balance": "bold green", "change_profit": "bold green", "change_loss": "bold red",
})

console = Console(theme=PYRMETHUS_THEME)
dotenv.load_dotenv()

DEFAULT_LOG_LEVEL = 'info'
CONFIG_FILE_PATH = 'config.json'
LOG_FILE_PATH = 'bot.log'

CACHE = {
    'markets': None, 'ohlcv': {}, 'orderBook': {}, 'lastTradeTime': None,
    'cooldownActiveUntil': 0, 'initialBalance': None, 'positionHistory': []
}

DEFAULT_CONFIG = {
    'exchange': {
        'name': 'bybit', 'apiKey': os.getenv('BYBIT_API_KEY'), 'secret': os.getenv('BYBIT_API_SECRET'),
        'rateLimit': 500, 'testMode': False
    },
    'symbol': 'BTC/USDT', 'timeframe': '1h', 'historyCandles': 300, 'analysisInterval': 60000,
    'logLevel': DEFAULT_LOG_LEVEL, 'heartbeatInterval': 3600000, 'maxRetries': 7, 'retryDelay': 3000,
    'riskPercentage': 0.15, 'maxPositionSizeUSDT': 150, 'estimatedSLPercentage': 0.01,
    'stopLossMultiplier': 2.5, 'takeProfitMultiplier': 4.5, 'slTpOffsetPercentage': 0.0015,
    'cacheTTL': {'ohlcv': 450000, 'orderBook': 75000}, 'cooldownPeriod': 3600000,
    'indicators': {
        'sma': {'fast': 12, 'slow': 35}, 'rsi': {'period': 14}, 'bollinger': {'period': 20, 'stdDev': 2.0},
        'macd': {'fastPeriod': 12, 'slowPeriod': 26, 'signalPeriod': 9}, 'atr': {'period': 14},
        'ema': {'period': 20}  # Added EMA for trend confirmation
    },
    'thresholds': {'oversold': 30, 'overbought': 70, 'minConfidence': 65, 'volSpikeMultiplier': 1.5},
    'volumeConfirmation': {'enabled': True, 'lookback': 25, 'multiplier': 1.2},
    'maxConsecutiveLosses': 3  # New parameter for risk control
}

class Logger:
    def __init__(self, log_file_path=LOG_FILE_PATH, default_level=DEFAULT_LOG_LEVEL):
        self.log_file_path = log_file_path
        self.log_levels = {
            'error': 0, 'warn': 1, 'info': 2, 'success': 3, 'signal_buy': 4, 'signal_sell': 5,
            'neutral': 6, 'heartbeat': 7, 'profit': 8, 'loss': 9, 'debug': 10, 'silly': 11, 'critical': 12
        }
        self.current_level = self.log_levels.get(default_level.lower(), 2)

    def set_level(self, level_name):
        level_name = level_name.lower()
        if level_name in self.log_levels:
            self.current_level = self.log_levels[level_name]
            self.info(f"Log level set to: [level]{level_name.upper()}[/level]")
        else:
            self.warn(f"Invalid log level: [warn]{level_name}[/warn]")

    def _log(self, level_name, message):
        if self.log_levels[level_name] <= self.current_level:
            timestamp_str = datetime.now().isoformat()
            log_message = f"{timestamp_str} {level_name.upper()} {message}"
            console.print(f"[timestamp]{timestamp_str}[/timestamp] [[level]{level_name.upper()}[/level]] {message}", style=level_name)
            with open(self.log_file_path, 'a') as logfile:
                logfile.write(log_message + '\n')

    def error(self, message): self._log('error', message)
    def warn(self, message): self._log('warn', message)
    def info(self, message): self._log('info', message)
    def success(self, message): self._log('success', message)
    def signal_buy(self, message): self._log('signal_buy', message)
    def signal_sell(self, message): self._log('signal_sell', message)
    def neutral(self, message): self._log('neutral', message)
    def heartbeat(self, message): self._log('heartbeat', message)
    def profit(self, message): self._log('profit', message)
    def loss(self, message): self._log('loss', message)
    def critical(self, message): self._log('critical', message)
    def debug(self, message): self._log('debug', message)
    def silly(self, message): self._log('silly', message)

logger = Logger()

class ConfigManager:
    def __init__(self):
        self.config = {}

    async def load(self, user_config):
        file_config = {}
        if os.path.exists(CONFIG_FILE_PATH):
            try:
                with open(CONFIG_FILE_PATH, 'r') as f:
                    file_config = json.load(f)
                logger.info(f"Loaded config from [config]{CONFIG_FILE_PATH}[/config]")
            except Exception as e:
                logger.error(f"Error loading config: {e}")
        self.config = {**DEFAULT_CONFIG, **file_config, **user_config}
        self._validate_config()
        logger.set_level(self.config['logLevel'])
        return self.config

    def _validate_config(self):
        required = ['symbol', 'timeframe', 'exchange', 'riskPercentage', 'maxPositionSizeUSDT', 'indicators']
        for key in required:
            if not self._nested_get(self.config, key):
                raise ValueError(f"Missing required config: {key}")
        if not all(k in self.config['exchange'] for k in ['name', 'apiKey', 'secret']):
            raise ValueError("Invalid 'exchange' config")
        if not all(k in self.config['indicators'] for k in ['sma', 'rsi', 'bollinger', 'macd', 'atr', 'ema']):
            raise ValueError("Invalid 'indicators' config")
        if not 0 < self.config['riskPercentage'] <= 100:
            raise ValueError("'riskPercentage' must be 0-100")
        if self.config['maxPositionSizeUSDT'] <= 0:
            raise ValueError("'maxPositionSizeUSDT' must be positive")

    def _nested_get(self, obj, key):
        parts = key.split('.')
        current = obj
        for part in parts:
            if isinstance(current, dict) and part in current:
                current = current[part]
            else:
                return None
        return current

    def display_config(self):
        logger.info("Current Configuration:")
        console.print(Syntax(json.dumps(self.config, indent=4), "json", theme="monokai"))

class MarketDataProvider:
    def __init__(self, exchange_config, config):
        exchange_class = getattr(ccxt, exchange_config['name'])
        self.exchange = exchange_class({
            'apiKey': exchange_config['apiKey'], 'secret': exchange_config['secret'],
            'enableRateLimit': True, 'rateLimit': exchange_config['rateLimit'],
            'options': {'defaultType': 'spot'}, 'testMode': exchange_config['testMode']
        })
        self.config = config

    async def load_markets(self):
        if not CACHE['markets']:
            try:
                CACHE['markets'] = await self.exchange.load_markets()
                logger.info("Markets loaded")
            except ccxt.NetworkError as e:
                logger.critical(f"Network error loading markets: {e}")
                raise
        return CACHE['markets']

    async def fetch_ohlcv(self, symbol, timeframe, limit, retries=0):
        cache_key = f"{symbol}-{timeframe}-{limit}"
        cached = CACHE['ohlcv'].get(cache_key)
        now = datetime.now().timestamp() * 1000
        if cached and (now - cached['timestamp'] < self.config['cacheTTL']['ohlcv']):
            return cached['data']
        try:
            data = await self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            CACHE['ohlcv'][cache_key] = {'data': data, 'timestamp': now}
            return data
        except ccxt.NetworkError as e:
            if retries < self.config['maxRetries']:
                logger.warn(f"Retrying OHLCV fetch ({retries + 1}/{self.config['maxRetries']})")
                await asyncio.sleep(self.config['retryDelay'] / 1000 * (2 ** retries))
                return await self.fetch_ohlcv(symbol, timeframe, limit, retries + 1)
            logger.error(f"OHLCV fetch failed: {e}")
            raise

    async def fetch_ticker(self, symbol):
        try:
            return await self.exchange.fetch_ticker(symbol)
        except ccxt.NetworkError as e:
            logger.error(f"Ticker fetch failed: {e}")
            raise

    def get_exchange_instance(self):
        return self.exchange

class IndicatorEngine:
    def __init__(self, config):
        self.config = config

    async def calculate_indicators(self, candles):
        if not candles:
            logger.warn("No candles for indicators")
            return None
        closes = [c[4] for c in candles]
        highs = [c[2] for c in candles]
        lows = [c[3] for c in candles]
        volumes = [c[5] for c in candles]
        return {
            'sma': self._calculate_sma(closes, self.config['indicators']['sma']),
            'rsi': self._calculate_rsi(closes, self.config['indicators']['rsi']['period']),
            'bollingerBands': self._calculate_bollinger_bands(closes, self.config['indicators']['bollinger']),
            'macd': self._calculate_macd(closes, self.config['indicators']['macd']),
            'atr': self._calculate_atr(highs, lows, closes, self.config['indicators']['atr']['period']),
            'ema': self._calculate_ema(closes, self.config['indicators']['ema']['period']),
            'volumeSMA': self._calculate_sma(volumes, {'fast': self.config['volumeConfirmation']['lookback'], 'slow': None})['fast'] if self.config['volumeConfirmation']['enabled'] else None
        }

    def _calculate_sma(self, values, sma_config):
        return {'fast': self._sma(values, sma_config['fast']), 'slow': self._sma(values, sma_config['slow'])}

    def _sma(self, values, period):
        if len(values) < period:
            return None
        return sum(values[-period:]) / period

    def _calculate_rsi(self, closes, period):
        if len(closes) < period + 1:
            return None
        gains, losses = [], []
        for i in range(1, len(closes)):
            diff = closes[i] - closes[i - 1]
            gains.append(diff if diff > 0 else 0)
            losses.append(-diff if diff < 0 else 0)
        avg_gain = self._ema(gains[:period], period)
        avg_loss = self._ema(losses[:period], period)
        for i in range(period, len(gains)):
            avg_gain = (avg_gain * (period - 1) + gains[i]) / period
            avg_loss = (avg_loss * (period - 1) + losses[i]) / period
        if avg_loss == 0:
            return 100
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))

    def _calculate_bollinger_bands(self, closes, bollinger_config):
        period, std_dev = bollinger_config['period'], bollinger_config['stdDev']
        if len(closes) < period:
            return None
        sma = self._sma(closes, period)
        if sma is None:
            return None
        variance = sum((c - sma) ** 2 for c in closes[-period:]) / period
        std = variance ** 0.5
        return {'upper': sma + std * std_dev, 'middle': sma, 'lower': sma - std * std_dev}

    def _calculate_macd(self, closes, macd_config):
        fast, slow, signal = macd_config['fastPeriod'], macd_config['slowPeriod'], macd_config['signalPeriod']
        if len(closes) < slow:
            return None
        ema_fast = self._ema(closes[-fast:], fast)
        ema_slow = self._ema(closes[-slow:], slow)
        macd_line = ema_fast - ema_slow
        macd_values = [self._ema(closes[i:i + fast], fast) - self._ema(closes[i:i + slow], slow) 
                       for i in range(len(closes) - slow + 1)]
        signal_line = self._ema(macd_values[-signal:], signal)
        return {'macd': macd_line, 'signal': signal_line}

    def _calculate_atr(self, highs, lows, closes, period):
        if len(highs) < period + 1:
            return None
        trs = []
        for i in range(1, len(highs)):
            tr = max(highs[i] - lows[i], abs(highs[i] - closes[i - 1]), abs(lows[i] - closes[i - 1]))
            trs.append(tr)
        atr = sum(trs[:period]) / period
        for i in range(period, len(trs)):
            atr = (atr * (period - 1) + trs[i]) / period
        return atr

    def _calculate_ema(self, values, period):
        if len(values) < period:
            return None
        ema = sum(values[:period]) / period
        multiplier = 2 / (period + 1)
        for value in values[period:]:
            ema = (value - ema) * multiplier + ema
        return ema

class SignalGenerator:
    def __init__(self, config):
        self.config = config

    def generate_signals(self, indicator_data, current_price, last_candle):
        if not indicator_data:
            return []
        signals = []
        sma = indicator_data.get('sma')
        rsi = indicator_data.get('rsi')
        bb = indicator_data.get('bollingerBands')
        macd = indicator_data.get('macd')
        atr = indicator_data.get('atr')
        ema = indicator_data.get('ema')
        vol_sma = indicator_data.get('volumeSMA')

        # Trend confirmation with EMA
        trend_direction = 'up' if current_price > ema else 'down'
        vol_spike = last_candle[5] > vol_sma * self.config['thresholds']['volSpikeMultiplier'] if vol_sma else False
        volatility_high = atr > self.config['indicators']['atr']['period'] * 0.015 if atr else False

        if sma and sma['fast'] and sma['slow']:
            if sma['fast'] > sma['slow'] and trend_direction == 'up':
                conf = 75 + (10 if vol_spike else 0) - (5 if volatility_high else 0)
                signals.append({'type': 'BUY', 'reason': 'SMA Crossover', 'confidence': conf})
            elif sma['fast'] < sma['slow'] and trend_direction == 'down':
                conf = 75 + (10 if vol_spike else 0) - (5 if volatility_high else 0)
                signals.append({'type': 'SELL', 'reason': 'SMA Crossover', 'confidence': conf})

        if rsi:
            if rsi < self.config['thresholds']['oversold'] and trend_direction == 'up':
                conf = 80 + (10 if vol_spike else 0) - (5 if volatility_high else 0)
                signals.append({'type': 'BUY', 'reason': 'RSI Oversold', 'confidence': conf})
            elif rsi > self.config['thresholds']['overbought'] and trend_direction == 'down':
                conf = 80 + (10 if vol_spike else 0) - (5 if volatility_high else 0)
                signals.append({'type': 'SELL', 'reason': 'RSI Overbought', 'confidence': conf})

        if bb and last_candle:
            if last_candle[4] <= bb['lower'] and trend_direction == 'up':
                conf = 70 + (10 if vol_spike else 0) - (5 if volatility_high else 0)
                signals.append({'type': 'BUY', 'reason': 'BB Lower Touch', 'confidence': conf})
            elif last_candle[4] >= bb['upper'] and trend_direction == 'down':
                conf = 70 + (10 if vol_spike else 0) - (5 if volatility_high else 0)
                signals.append({'type': 'SELL', 'reason': 'BB Upper Touch', 'confidence': conf})

        if macd:
            if macd['macd'] > macd['signal'] and trend_direction == 'up':
                conf = 75 + (10 if vol_spike else 0) - (5 if volatility_high else 0)
                signals.append({'type': 'BUY', 'reason': 'MACD Crossover', 'confidence': conf})
            elif macd['macd'] < macd['signal'] and trend_direction == 'down':
                conf = 75 + (10 if vol_spike else 0) - (5 if volatility_high else 0)
                signals.append({'type': 'SELL', 'reason': 'MACD Crossover', 'confidence': conf})

        # Risk management: Check consecutive losses
        recent_losses = sum(1 for p in CACHE['positionHistory'][-self.config['maxConsecutiveLosses']:] if p['profit'] < 0)
        if recent_losses >= self.config['maxConsecutiveLosses']:
            logger.warn("Max consecutive losses reached, reducing signal confidence")
            for s in signals:
                s['confidence'] = max(s['confidence'] - 20, 0)

        return [s for s in signals if s['confidence'] >= self.config['thresholds']['minConfidence']]

class TradingBot:
    def __init__(self, config):
        self.config = config
        self.market_data = MarketDataProvider(self.config['exchange'], self.config)
        self.indicator_engine = IndicatorEngine(self.config)
        self.signal_generator = SignalGenerator(self.config)
        self.trade_context = {'current_position': None, 'last_order': None, 'balance': None}
        self.intervals = []
        self.running = True

    async def start(self):
        await self.market_data.load_markets()
        await self.fetch_initial_balance()
        logger.success(f"Bot started for [success]{self.config['symbol']}[/success] on [success]{self.config['timeframe']}[/success]")
        self.intervals.extend([
            asyncio.create_task(self.analysis_cycle_loop()),
            asyncio.create_task(self.heartbeat_loop())
        ])

    async def fetch_initial_balance(self):
        try:
            balance = await self.market_data.get_exchange_instance().fetch_balance()
            self.trade_context['balance'] = balance.get('USDT', {}).get('free', 0)
            CACHE['initialBalance'] = self.trade_context['balance']
            logger.info(f"Initial balance: [balance]USDT {self.trade_context['balance']}[/balance]")
        except Exception as e:
            logger.error(f"Balance fetch failed: {e}")

    async def analysis_cycle_loop(self):
        while self.running:
            await self.analysis_cycle()
            await asyncio.sleep(self.config['analysisInterval'] / 1000)

    async def heartbeat_loop(self):
        while self.running:
            logger.heartbeat("Bot heartbeat")
            await asyncio.sleep(self.config['heartbeatInterval'] / 1000)

    async def analysis_cycle(self):
        try:
            now_ms = datetime.now().timestamp() * 1000
            if now_ms < CACHE['cooldownActiveUntil']:
                logger.info("Cooldown active, skipping analysis")
                return
            candles = await self.market_data.fetch_ohlcv(self.config['symbol'], self.config['timeframe'], self.config['historyCandles'])
            if not candles:
                logger.warn("No candle data")
                return
            ticker = await self.market_data.fetch_ticker(self.config['symbol'])
            current_price = ticker['last']
            indicators = await self.indicator_engine.calculate_indicators(candles)
            if not indicators:
                logger.warn("No indicators calculated")
                return
            signals = self.signal_generator.generate_signals(indicators, current_price, candles[-1])
            if signals:
                logger.debug(f"Signals: {', '.join(f'{s['type']} ({s['reason']}, {s['confidence']}%)' for s in signals)}")
            for signal in signals:
                if signal['type'] == 'BUY' and not self.trade_context['current_position']:
                    await self.execute_buy_order(current_price, signal, indicators['atr'])
                elif signal['type'] == 'SELL' and self.trade_context['current_position'] == 'BUY':
                    await self.execute_sell_order(current_price, signal, indicators['atr'])
        except ccxt.NetworkError as e:
            logger.error(f"Network error in analysis: {e}")
        except Exception as e:
            logger.critical(f"Analysis error: {e}")

    async def calculate_order_quantity(self, price, atr):
        if not self.trade_context['balance']:
            logger.warn("No balance for quantity calc")
            return 0
        risk_amount = self.trade_context['balance'] * (self.config['riskPercentage'] / 100)
        sl_distance = atr * self.config['stopLossMultiplier']
        quantity = risk_amount / (sl_distance * price)
        max_quantity = self.config['maxPositionSizeUSDT'] / price
        quantity = min(quantity, max_quantity)
        market = CACHE['markets'][self.config['symbol']]
        qty_prec = self.market_data.get_exchange_instance().amount_to_precision(self.config['symbol'], quantity)
        min_qty = market['limits']['amount']['min']
        if float(qty_prec) < min_qty:
            qty_prec = str(min_qty)
            logger.warn(f"Quantity {qty_prec} below min {min_qty}, adjusted")
        return float(qty_prec)

    async def execute_buy_order(self, price, signal, atr):
        qty = await self.calculate_order_quantity(price, atr)
        if qty <= 0:
            logger.warn("Invalid buy quantity")
            return
        try:
            params = self._create_order_params(price, 'buy', qty, atr)
            order = await self.market_data.get_exchange_instance().create_order(
                self.config['symbol'], 'limit', 'buy', qty, price, params=params)
            self.trade_context['current_position'] = 'BUY'
            self.trade_context['last_order'] = order
            CACHE['cooldownActiveUntil'] = datetime.now().timestamp() * 1000 + self.config['cooldownPeriod']
            logger.signal_buy(f"BUY: qty=[value]{qty}[/value], price=[value]{price}[/value], TP=[value]{params['takeProfit']}[/value], SL=[value]{params['stopLoss']}[/value] - {signal['reason']}")
        except Exception as e:
            logger.error(f"Buy order failed: {e}")

    async def execute_sell_order(self, price, signal, atr):
        if not self.trade_context['current_position']:
            logger.warn("No position to sell")
            return
        qty = self.trade_context['last_order']['amount']
        try:
            params = self._create_order_params(price, 'sell', qty, atr)
            order = await self.market_data.get_exchange_instance().create_order(
                self.config['symbol'], 'market', 'sell', qty, None, params=params)
            profit = (price - self.trade_context['last_order']['price']) * qty
            CACHE['positionHistory'].append({'profit': profit, 'time': datetime.now().timestamp()})
            self.trade_context['current_position'] = None
            self.trade_context['last_order'] = order
            CACHE['cooldownActiveUntil'] = datetime.now().timestamp() * 1000 + self.config['cooldownPeriod']
            logger.signal_sell(f"SELL: qty=[value]{qty}[/value], price=[value]{price}[/value] - {signal['reason']}")
            await self.update_balance(profit)
        except Exception as e:
            logger.error(f"Sell order failed: {e}")

    async def update_balance(self, profit):
        balance = await self.market_data.get_exchange_instance().fetch_balance()
        new_balance = balance.get('USDT', {}).get('free', 0)
        change = new_balance - CACHE['initialBalance']
        if change > 0:
            logger.profit(f"Profit: USDT [change_profit]{change:.2f}[/change_profit] (Trade: {profit:.2f})")
        elif change < 0:
            logger.loss(f"Loss: USDT [change_loss]{change:.2f}[/change_loss] (Trade: {profit:.2f})")
        else:
            logger.neutral("Breakeven")
        self.trade_context['balance'] = new_balance
        CACHE['initialBalance'] = new_balance

    def _create_order_params(self, price, side, qty, atr):
        sl_price = price - atr * self.config['stopLossMultiplier'] if side == 'buy' else price + atr * self.config['stopLossMultiplier']
        tp_price = price + atr * self.config['takeProfitMultiplier'] if side == 'buy' else price - atr * self.config['takeProfitMultiplier']
        exchange = self.market_data.get_exchange_instance()
        return {
            'stopLoss': exchange.price_to_precision(self.config['symbol'], sl_price),
            'takeProfit': exchange.price_to_precision(self.config['symbol'], tp_price)
        }

    def stop(self):
        self.running = False
        for task in self.intervals:
            task.cancel()
        logger.warn("Bot stopped")

def get_user_config():
    console.print("[bold cyan]Configure Pyrmethus Trading Bot[/bold cyan]")
    while True:
        symbol = input("Enter symbol (e.g., BTC/USDT): ").upper()
        if '/' in symbol:
            break
        print("Invalid format. Use 'BASE/QUOTE'.")
    while True:
        timeframe = input("Enter timeframe (e.g., 1h): ").lower()
        if timeframe in ['1m', '5m', '15m', '30m', '1h', '4h', '1d']:
            break
        print("Valid options: 1m, 5m, 15m, 30m, 1h, 4h, 1d")
    while True:
        log_level = input("Enter log level (info/debug/warn/error): ").lower()
        if log_level in ['info', 'debug', 'warn', 'error']:
            break
        print("Valid options: info, debug, warn, error")
    while True:
        test_mode = input("Run in test mode? (yes/no): ").lower()
        if test_mode in ['yes', 'no']:
            test_mode = test_mode == 'yes'
            break
        print("Enter 'yes' or 'no'")
    while True:
        try:
            sma_fast = int(input("Enter SMA fast period (e.g., 12): "))
            sma_slow = int(input("Enter SMA slow period (e.g., 35): "))
            if sma_fast > 0 and sma_slow > sma_fast:
                break
            print("Periods must be positive and slow > fast")
        except ValueError:
            print("Enter valid integers")
    while True:
        try:
            rsi_period = int(input("Enter RSI period (e.g., 14): "))
            if rsi_period > 0:
                break
            print("Period must be positive")
        except ValueError:
            print("Enter a valid integer")
    while True:
        try:
            bollinger_period = int(input("Enter Bollinger Bands period (e.g., 20): "))
            bollinger_std_dev = float(input("Enter Bollinger Bands std dev (e.g., 2.0): "))
            if bollinger_period > 0 and bollinger_std_dev > 0:
                break
            print("Values must be positive")
        except ValueError:
            print("Enter valid numbers")
    while True:
        try:
            risk_percentage = float(input("Enter risk % per trade (0-100): "))
            if 0 < risk_percentage <= 100:
                break
            print("Must be between 0 and 100")
        except ValueError:
            print("Enter a valid number")
    while True:
        try:
            max_position_size_usdt = float(input("Enter max position size in USDT (e.g., 150): "))
            if max_position_size_usdt > 0:
                break
            print("Must be positive")
        except ValueError:
            print("Enter a valid number")
    return {
        'symbol': symbol, 'timeframe': timeframe, 'logLevel': log_level, 'exchange': {'testMode': test_mode},
        'indicators': {
            'sma': {'fast': sma_fast, 'slow': sma_slow}, 'rsi': {'period': rsi_period},
            'bollinger': {'period': bollinger_period, 'stdDev': bollinger_std_dev},
            'macd': DEFAULT_CONFIG['indicators']['macd'], 'atr': DEFAULT_CONFIG['indicators']['atr'],
            'ema': DEFAULT_CONFIG['indicators']['ema']
        },
        'riskPercentage': risk_percentage, 'maxPositionSizeUSDT': max_position_size_usdt
    }

async def run_bot():
    config_manager = ConfigManager()
    user_config = get_user_config()
    config = await config_manager.load(user_config)
    config_manager.display_config()
    bot = TradingBot(config)
    await bot.start()
    stop_event = asyncio.Event()

    def stop_handler():
        bot.stop()
        stop_event.set()

    loop = asyncio.get_event_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, stop_handler)
    console.print("[bold green]Bot running. Press Ctrl+C to stop.[/bold green]")
    await stop_event.wait()

if __name__ == "__main__":
    asyncio.run(run_bot())
