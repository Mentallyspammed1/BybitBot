import asyncio
import json
import os
import sys
from datetime import datetime
import ccxt
import dotenv
from rich.console import Console
from rich.syntax import Syntax
from rich.theme import Theme

# Mystical Color Theme for the Coding Wizard
PYRMETHUS_THEME = Theme({
    "info": "cyan",
    "success": "green",
    "warn": "yellow",
    "error": "bold red",
    "critical": "bold underline red",
    "signal_buy": "bold green",
    "signal_sell": "bold red",
    "neutral": "dim white",
    "heartbeat": "magenta",
    "profit": "bold green",
    "loss": "bold red",
    "debug": "dim blue",
    "silly": "dim purple",
    "code": "bold white",
    "config": "italic yellow",
    "timestamp": "dim white",
    "level": "bold blue",
    "signal_reason": "italic green",
    "confidence": "bold yellow",
    "indicator": "cyan",
    "value": "magenta",
    "order_type": "bold magenta",
    "order_side": "bold green",
    "order_quantity": "bold cyan",
    "order_price": "bold blue",
    "balance": "bold green",
    "change_profit": "bold green",
    "change_loss": "bold red",
})


console = Console(theme=PYRMETHUS_THEME)

dotenv.load_dotenv()

DEFAULT_LOG_LEVEL = 'info'
CONFIG_FILE_PATH = 'config.json'
LOG_FILE_PATH = 'bot.log'


class Logger:
    """
    A mystical conduit for logging messages with varying levels of importance,
    illuminating the bot's path through the digital darkness.
    """
    def __init__(self, log_file_path=LOG_FILE_PATH, default_level=DEFAULT_LOG_LEVEL):
        self.log_file_path = log_file_path
        self.log_levels = {
            'error': 0, 'warn': 1, 'info': 2, 'success': 3,
            'signal_buy': 4, 'signal_sell': 5, 'neutral': 6,
            'heartbeat': 7, 'profit': 8, 'loss': 9, 'debug': 10,
            'silly': 11, 'critical': 12
        }
        self.current_level = self.log_levels[default_level]

    def set_level(self, level_name):
        """Adjusts the mystical veil to reveal logs of a certain importance."""
        if level_name.lower() in self.log_levels:
            self.current_level = self.log_levels[level_name.lower()]
            self.info(f"Log level attuned to: [level]{level_name.upper()}[/level]")
        else:
            self.warn(f"Invalid log level: [warn]{level_name}[/warn], maintaining default: [warn]{DEFAULT_LOG_LEVEL.upper()}[/warn]")

    def _log(self, level_name, message):
        """Whispers the message into the log file and displays it on the console, if deemed important enough."""
        if self.log_levels[level_name] <= self.current_level:
            timestamp_str = datetime.now().isoformat()
            log_message = f"{timestamp_str} {level_name.upper()} {message}"
            console.print(f"[timestamp]{timestamp_str}[/timestamp] [[level]{level_name.upper()}[/level]] {message}", style=level_name)
            with open(self.log_file_path, 'a') as logfile:
                logfile.write(log_message + '\n')

    def error(self, message):
        self._log('error', message)

    def warn(self, message):
        self._log('warn', message)

    def info(self, message):
        self._log('info', message)

    def success(self, message):
        self._log('success', message)

    def signal_buy(self, message):
        self._log('signal_buy', message)

    def signal_sell(self, message):
        self._log('signal_sell', message)

    def neutral(self, message):
        self._log('neutral', message)

    def heartbeat(self, message):
        self._log('heartbeat', message)

    def profit(self, message):
        self._log('profit', message)

    def loss(self, message):
        self._log('loss', message)

    def critical(self, message):
        self._log('critical', message)

    def debug(self, message):
        self._log('debug', message)

    def silly(self, message):
        self._log('silly', message)


logger = Logger(LOG_FILE_PATH)


DEFAULT_CONFIG = {
    'exchange': {
        'name': 'bybit',
        'apiKey': os.getenv('BYBIT_API_KEY'),
        'secret': os.getenv('BYBIT_API_SECRET'),
        'rateLimit': 500,
        'testMode': False
    },
    'symbol': 'BTC/USDT',
    'timeframe': '1h',
    'historyCandles': 300,
    'analysisInterval': 60000,
    'logLevel': DEFAULT_LOG_LEVEL,
    'heartbeatInterval': 3600000,
    'maxRetries': 7,
    'retryDelay': 3000,
    'riskPercentage': 0.15,
    'maxPositionSizeUSDT': 150,
    'stopLossMultiplier': 2.5,
    'takeProfitMultiplier': 4.5,
    'slTpOffsetPercentage': 0.0015,
    'cacheTTL': {'ohlcv': 450000, 'orderBook': 75000},
    'indicators': {
        'sma': {'fast': 12, 'slow': 35},
        'stochRsi': {'rsiPeriod': 16, 'stochasticPeriod': 16, 'kPeriod': 4, 'dPeriod': 4},
        'atr': {'period': 16},
        'macd': {'fastPeriod': 12, 'slowPeriod': 26, 'signalPeriod': 9},
        'rsi': {'period': 14},
        'bollinger': {'period': 20, 'stdDev': 2}
    },
    'thresholds': {'oversold': 30, 'overbought': 70, 'minConfidence': 65},
    'volumeConfirmation': {'enabled': True, 'lookback': 25, 'multiplier': 1.2},
    'fibonacciPivots': {
        'enabled': False,
        'period': '1d',
        'proximityPercentage': 0.006,
        'orderBookRangePercentage': 0.0025,
        'pivotWeight': 20,
        'orderBookWeight': 12,
        'levelsForBuyEntry': ['S1', 'S2', 'S3'],
        'levelsForSellEntry': ['R1', 'R2', 'R3']
    }
}

CACHE = {
    'markets': None,
    'ohlcv': {},
    'orderBook': {},
    'lastTradeTime': None,
    'cooldownActiveUntil': 0,
    'initialBalance': None
}


class ConfigManager:
    """
    Manages the configuration, loading defaults, file configurations, and CLI overrides.
    Ensures the bot is properly configured before casting its spells.
    """
    def __init__(self):
        self.config = {}

    async def load(self, cli_config):
        """Loads configuration from file and CLI, merging with defaults."""
        file_config = {}
        try:
            if os.path.exists(CONFIG_FILE_PATH):
                with open(CONFIG_FILE_PATH, 'r') as f:
                    file_config = json.load(f)
                logger.info(f"Loaded configuration from [config]{CONFIG_FILE_PATH}[/config]")
            else:
                logger.info(f"[config]{CONFIG_FILE_PATH}[/config] not found, using default configuration.")
        except Exception as e:
            logger.error(f"Error loading [config]{CONFIG_FILE_PATH}[/config]: {e}")

        self.config = {**DEFAULT_CONFIG, **file_config, **cli_config}
        self._validate_config()
        self._apply_log_level()
        return self.config

    def _apply_log_level(self):
        """Sets the log level based on the configuration."""
        logger.set_level(self.config['logLevel'])

    def _validate_config(self):
        """Ensures essential configuration parameters are present."""
        required_keys = ['symbol', 'timeframe', 'exchange', 'exchange.apiKey', 'exchange.secret']
        for key in required_keys:
            if not self._nested_get(self.config, key):
                raise ValueError(f"Missing required configuration: {key}")
        if not isinstance(self.config['exchange'], dict) or not all(k in self.config['exchange'] for k in ['name', 'apiKey', 'secret']):
            raise ValueError("Invalid 'exchange' configuration format.")


    def _nested_get(self, obj, key):
        """Helper function to get nested keys safely."""
        parts = key.split('.')
        current = obj
        for part in parts:
            if isinstance(current, dict) and part in current:
                current = current[part]
            else:
                return None
        return current

    def display_config(self):
        """Presents the current configuration in a structured, readable format."""
        logger.info("Current Configuration:")
        config_syntax = Syntax(json.dumps(self.config, indent=4), "json", theme="monokai", line_numbers=False)
        console.print(config_syntax)


class MarketDataProvider:
    """
    Fetches market data from the exchange, caching results to reduce API calls.
    A mystical diviner of market secrets, providing the bot with necessary insights.
    """
    def __init__(self, exchange_config, config):
        exchange_class = getattr(ccxt, exchange_config['name'])
        self.exchange = exchange_class({
            'apiKey': exchange_config['apiKey'],
            'secret': exchange_config['secret'],
            'enableRateLimit': True,
            'rateLimit': exchange_config['rateLimit'],
            'options': {
                'defaultType': 'spot',
            },
            'testMode': exchange_config['testMode']
        })
        self.config = config

    async def load_markets(self):
        """Loads markets from the exchange, caching for future use."""
        if not CACHE['markets']:
            try:
                CACHE['markets'] = await self.exchange.load_markets()
                logger.info("Markets loaded from exchange.")
            except Exception as e:
                logger.critical(f"Failed to load markets: {e}")
                raise e
        return CACHE['markets']

    async def fetch_ohlcv(self, symbol, timeframe, limit, retries=0):
        """Fetches OHLCV data, utilizing cache and retry mechanisms for resilience."""
        cache_key = f"{symbol}-{timeframe}-{limit}"
        cached_data = CACHE['ohlcv'].get(cache_key)
        if cached_data and (datetime.now().timestamp() * 1000 - cached_data['timestamp'] < self.config['cacheTTL']['ohlcv']):
            return cached_data['data']

        try:
            data = await self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            CACHE['ohlcv'][cache_key] = {'data': data, 'timestamp': datetime.now().timestamp() * 1000}
            return data
        except Exception as e:
            if retries < self.config['maxRetries']:
                logger.warn(f"Network error fetching OHLCV, retrying ({retries + 1}/{self.config['maxRetries']})...")
                await asyncio.sleep(self.config['retryDelay'] * (2**retries) / 1000) # Convert to seconds for asyncio.sleep
                return await self.fetch_ohlcv(symbol, timeframe, limit, retries + 1)
            logger.error(f"Failed to fetch OHLCV after multiple retries: {e}")
            raise e

    async def fetch_order_book(self, symbol, retries=0):
        """Fetches order book data, employing cache and retries for robustness."""
        cache_key = symbol
        cached_data = CACHE['orderBook'].get(cache_key)
        if cached_data and (datetime.now().timestamp() * 1000 - cached_data['timestamp'] < self.config['cacheTTL']['orderBook']):
            return cached_data['data']

        try:
            data = await self.exchange.fetch_order_book(symbol)
            CACHE['orderBook'][cache_key] = {'data': data, 'timestamp': datetime.now().timestamp() * 1000}
            return data
        except Exception as e:
            if retries < self.config['maxRetries']:
                logger.warn(f"Network error fetching order book, retrying ({retries + 1}/{self.config['maxRetries']})...")
                await asyncio.sleep(self.config['retryDelay'] * (2**retries) / 1000) # Convert to seconds for asyncio.sleep
                return await self.fetch_order_book(symbol, retries + 1)
            logger.error(f"Failed to fetch order book after multiple retries: {e}")
            raise e

    async def fetch_ticker(self, symbol, retries=0):
        """Fetches ticker data, with retry logic for network hiccups."""
        try:
            return await self.exchange.fetch_ticker(symbol)
        except Exception as e:
            if retries < self.config['maxRetries']:
                logger.warn(f"Network error fetching ticker, retrying ({retries + 1}/{self.config['maxRetries']})...")
                await asyncio.sleep(self.config['retryDelay'] * (2**retries) / 1000) # Convert to seconds for asyncio.sleep
                return await self.fetch_ticker(symbol, retries + 1)
            logger.error(f"Failed to fetch ticker after multiple retries: {e}")
            raise e

    def get_exchange_instance(self):
        """Returns the CCXT exchange instance."""
        return self.exchange


class IndicatorEngine:
    """
    Calculates technical indicators from market data.
    A mystical forge where raw data is transmuted into actionable insights.
    """
    def __init__(self, config):
        self.config = config

    async def calculate_indicators(self, candles):
        """Calculates all configured indicators."""
        if not candles or len(candles) == 0:
            logger.warn("No candles data provided for indicator calculation.")
            return None

        closes = [candle[4] for candle in candles]
        volumes = [candle[5] for candle in candles]

        return {
            'sma': self._calculate_sma(closes, self.config['indicators']['sma']),
            'rsi': self._calculate_rsi(closes, self.config['indicators']['rsi']['period']),
            'bollingerBands': self._calculate_bollinger_bands(closes, self.config['indicators']['bollinger']),
            'volumeSMA': self._calculate_sma(volumes, {'fast': self.config['volumeConfirmation']['lookback'], 'slow': None})['fast'] if self.config['volumeConfirmation']['enabled'] else None
        }

    def _calculate_sma(self, values, sma_config):
        """Calculates Simple Moving Averages (SMA)."""
        fast_sma = self._sma(values, sma_config['fast']) if sma_config['fast'] else None
        slow_sma = self._sma(values, sma_config['slow']) if sma_config['slow'] else None
        return {'fast': fast_sma, 'slow': slow_sma}

    def _sma(self, values, period):
        """Helper function to calculate SMA for a given period."""
        if not period or len(values) < period:
            return None
        return sum(values[-period:]) / period

    def _calculate_rsi(self, closes, period):
        """Calculates Relative Strength Index (RSI)."""
        if len(closes) < period + 1:
            return None

        diffs = [closes[i] - closes[i-1] for i in range(1, len(closes))]
        gains = [d if d > 0 else 0 for d in diffs[-period-1:]] # Adjusted slicing
        losses = [-d if d < 0 else 0 for d in diffs[-period-1:]] # Adjusted slicing

        if not gains or not losses: # Handle edge case where gains or losses are empty
            return 50.0 # Neutral RSI if no changes in price

        avg_gain = sum(gains[1:]) / period # Start from index 1 to align with diffs after period
        avg_loss = sum(losses[1:]) / period # Start from index 1 to align with diffs after period


        for i in range(period + 1, len(diffs) + 1): # Iterate from period+1 to length of diffs
            gain = gains[i] if i < len(gains) else 0 # Ensure index is within bounds
            loss = losses[i] if i < len(losses) else 0 # Ensure index is within bounds
            avg_gain = (avg_gain * (period - 1) + gain) / period
            avg_loss = (avg_loss * (period - 1) + loss) / period

        if avg_loss == 0:
            return 100.0
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))


    def _calculate_bollinger_bands(self, closes, bollinger_config):
        """Calculates Bollinger Bands."""
        period = bollinger_config['period']
        std_dev_mult = bollinger_config['stdDev']
        if len(closes) < period:
            return None

        sma = self._sma(closes, period)
        if sma is None: # Handle case where SMA is None
            return None

        squared_differences = [(close - sma) ** 2 for close in closes[-period:]]
        variance = sum(squared_differences) / period
        std_deviation = variance ** 0.5

        upper_band = sma + (std_dev_mult * std_deviation)
        lower_band = sma - (std_dev_mult * std_deviation)
        return {'upper': upper_band, 'middle': sma, 'lower': lower_band}


class SignalGenerator:
    """
    Generates trading signals based on indicator data and configured thresholds.
    The oracle of the bot, interpreting market signs and suggesting actions.
    """
    def __init__(self, config):
        self.config = config

    def generate_signals(self, indicator_data, current_price, last_candle):
        """Analyzes indicators and generates buy/sell signals."""
        if not indicator_data:
            return []

        signals = []
        sma = indicator_data.get('sma')
        rsi = indicator_data.get('rsi')
        bollinger_bands = indicator_data.get('bollingerBands')
        volume_sma = indicator_data.get('volumeSMA')

        if sma:
            if sma['fast'] and sma['slow'] and sma['fast'] > sma['slow']:
                signals.append({'type': 'BUY', 'reason': 'SMA Crossover', 'confidence': 70})
            elif sma['fast'] and sma['slow'] and sma['fast'] < sma['slow']:
                signals.append({'type': 'SELL', 'reason': 'SMA Crossover', 'confidence': 70})

        if rsi:
            if rsi < self.config['thresholds']['oversold']:
                signals.append({'type': 'BUY', 'reason': 'RSI Oversold', 'confidence': 75})
            elif rsi > self.config['thresholds']['overbought']:
                signals.append({'type': 'SELL', 'reason': 'RSI Overbought', 'confidence': 75})

        if bollinger_bands and last_candle:
            if last_candle[4] <= bollinger_bands['lower']:
                signals.append({'type': 'BUY', 'reason': 'Bollinger Bands Lower Band Touch', 'confidence': 65})
            elif last_candle[4] >= bollinger_bands['upper']:
                signals.append({'type': 'SELL', 'reason': 'Bollinger Bands Upper Band Touch', 'confidence': 65})

        if self.config['volumeConfirmation']['enabled'] and volume_sma and last_candle:
            if last_candle[5] > volume_sma * self.config['volumeConfirmation']['multiplier']:
                for signal in signals:
                    signal['confidence'] = min(signal['confidence'] + 10, 95)

        return [signal for signal in signals if signal['confidence'] >= self.config['thresholds']['minConfidence']]


class TradingBot:
    """
    The heart of the trading algorithm, orchestrating market data, signals, and order execution.
    Pyrmethus, the coding wizard, brought to life to navigate the volatile seas of crypto trading.
    """
    def __init__(self, config):
        self.config = config
        self.market_data = MarketDataProvider(self.config['exchange'], self.config)
        self.indicator_engine = IndicatorEngine(self.config)
        self.signal_generator = SignalGenerator(self.config)
        self.trade_context = {'current_position': None, 'last_order': None, 'balance': None}
        self.intervals = []
        self.initial_balance_fetched = False

    async def start(self):
        """Initiates the bot, loading markets and starting analysis cycles."""
        await self.market_data.load_markets()
        logger.success(f"Bot awakened, trading [success]{self.config['symbol']}[/success] on [success]{self.config['timeframe']}[/success]")
        await self.fetch_initial_balance()
        self.intervals.append(asyncio.ensure_future(self.analysis_cycle_loop()))
        self.intervals.append(asyncio.ensure_future(self.heartbeat_loop()))


    async def heartbeat_loop(self):
        """Sends heartbeat messages at intervals to ensure the bot's spirit is alive."""
        while True:
            logger.heartbeat("Bot heartbeat...")
            await asyncio.sleep(self.config['heartbeatInterval'] / 1000) # Convert to seconds for asyncio.sleep


    async def analysis_cycle_loop(self):
        """Loops the analysis cycle at configured intervals."""
        while True:
            await self.analysis_cycle()
            await asyncio.sleep(self.config['analysisInterval'] / 1000) # Convert to seconds for asyncio.sleep


    async def fetch_initial_balance(self):
        """Fetches the initial balance from the exchange."""
        try:
            balance = await self.market_data.get_exchange_instance().fetch_balance()
            self.trade_context['balance'] = balance['USDT']['free'] if 'USDT' in balance and 'free' in balance['USDT'] else 0
            CACHE['initialBalance'] = self.trade_context['balance']
            logger.info(f"Initial balance divined: [balance]USDT {self.trade_context['balance']}[/balance]")
            self.initial_balance_fetched = True
        except Exception as error:
            logger.error(f"Error fetching initial balance: {error}")


    async def analysis_cycle(self):
        """Performs one analysis cycle: fetches data, calculates indicators, generates signals, and executes orders."""
        try:
            candles = await self.market_data.fetch_ohlcv(self.config['symbol'], self.config['timeframe'], self.config['historyCandles'])
            if not candles or len(candles) == 0:
                logger.warn("No candle data received. Skipping analysis cycle.")
                return

            current_price_ticker = await self.market_data.fetch_ticker(self.config['symbol'])
            current_price = current_price_ticker['last']
            indicators = await self.indicator_engine.calculate_indicators(candles)

            if not indicators:
                logger.warn("Indicators could not be calculated. Skipping signal generation.")
                return

            signals = self.signal_generator.generate_signals(indicators, current_price, candles[-1])

            if signals:
                signal_messages = ", ".join([f"[signal_{sig['type'].lower()}]{sig['type']}[/signal_{sig['type'].lower()}] ([signal_reason]{sig['reason']}[/signal_reason], [confidence]{sig['confidence']}%[/confidence])" for sig in signals])
                logger.debug(f"Generated signals: {signal_messages}")

            for signal in signals:
                log_level = f"signal_{signal['type'].lower()}"
                logger._log(log_level, f"[signal_{signal['type'].lower()}]{signal['type']}[/signal_{signal['type'].lower()}] Signal - [signal_reason]{signal['reason']}[/signal_reason] ([confidence]{signal['confidence']}%[/confidence])")

                if signal['type'] == 'BUY' and not self.trade_context['current_position']:
                    await self.execute_buy_order(current_price, signal)
                elif signal['type'] == 'SELL' and self.trade_context['current_position'] == 'BUY':
                    await self.execute_sell_order(current_price, signal)
                else:
                    logger.neutral(f"Neutral signal or no action needed: [neutral]{signal['type']}[/neutral] - [neutral]{signal['reason']}[/neutral]")

        except Exception as e:
            logger.error(f"Analysis cycle error: {e}")


    async def calculate_order_quantity(self, price):
        """Calculates the order quantity based on risk percentage and current balance."""
        if not self.trade_context['balance']:
            logger.warn("Balance not available, cannot calculate order quantity.")
            return 0

        risk_amount = self.trade_context['balance'] * (self.config['riskPercentage'] / 100)
        quantity = risk_amount / price
        market = CACHE['markets'][self.config['symbol']]
        quantity_to_precision = self.market_data.get_exchange_instance().amount_to_precision(self.config['symbol'], quantity)

        if market and market['limits'] and market['limits']['amount'] and market['limits']['amount']['min']:
            if float(quantity_to_precision) < market['limits']['amount']['min']:
                quantity_to_precision = str(market['limits']['amount']['min'])
                logger.warn(f"Calculated quantity [warn]{quantity_to_precision}[/warn] is less than minimum [warn]{market['limits']['amount']['min']}[/warn], using minimum.")

        return float(quantity_to_precision)


    async def execute_buy_order(self, current_price, signal):
        """Executes a buy order based on the generated signal."""
        quantity = await self.calculate_order_quantity(current_price)
        if quantity <= 0:
            logger.warn("Calculated order quantity is zero or less, skipping buy order.")
            return

        try:
            order_params = self._create_order_params(current_price, 'buy', quantity)
            order = await self.market_data.get_exchange_instance().create_order(
                self.config['symbol'], 'limit', 'buy', quantity, current_price, params=order_params)
            self.trade_context['current_position'] = 'BUY'
            self.trade_context['last_order'] = order

            log_message = (f"[signal_buy]BUY[/signal_buy] order placed: [order_quantity]quantity[/order_quantity]=[value]{quantity}[/value], "
                           f"[order_price]price[/order_price]=[value]{current_price}[/value], "
                           f"[indicator]TP[/indicator]=[value]{order_params['takeProfit']}[/value], [indicator]SL[/indicator]=[value]{order_params['stopLoss']}[/value] - [signal_reason]{signal['reason']}[/signal_reason]")
            logger.signal_buy(log_message)


        except Exception as e:
            logger.error(f"Error placing BUY order: {e}")


    async def execute_sell_order(self, current_price, signal):
        """Executes a sell order to close a position."""
        if not self.trade_context['current_position']:
            logger.warn("No open position to sell.")
            return

        position_quantity = self.trade_context['last_order']['amount'] if self.trade_context['last_order'] else 0.001 # Fallback quantity

        try:
            order_params = self._create_order_params(current_price, 'sell', position_quantity)
            order = await self.market_data.get_exchange_instance().create_order(
                self.config['symbol'], 'market', 'sell', position_quantity, None, params=order_params) # Market order to close position
            self.trade_context['current_position'] = None
            self.trade_context['last_order'] = order

            log_message = (f"[signal_sell]SELL[/signal_sell] order placed to close position: [order_quantity]quantity[/order_quantity]=[value]{position_quantity}[/value], "
                           f"[order_price]price[/order_price]=[value]{current_price}[/value] - [signal_reason]{signal['reason']}[/signal_reason]")
            logger.signal_sell(log_message)


            if CACHE['initialBalance'] is not None and self.trade_context['balance'] is not None:
                current_balance = await self.market_data.get_exchange_instance().fetch_balance()
                balance_change = current_balance['USDT']['free'] - CACHE['initialBalance'] if 'USDT' in current_balance and 'free' in current_balance['USDT'] else 0

                if balance_change > 0:
                    logger.profit(f"Position closed in [profit]PROFIT[/profit]: Change USDT [change_profit]{balance_change:.2f}[/change_profit]")
                elif balance_change < 0:
                    logger.loss(f"Position closed in [loss]LOSS[/loss]: Change USDT [change_loss]{balance_change:.2f}[/change_loss]")
                else:
                    logger.neutral("Position closed at breakeven.")

                self.trade_context['balance'] = current_balance['USDT']['free'] if 'USDT' in current_balance and 'free' in current_balance['USDT'] else 0
                CACHE['initialBalance'] = self.trade_context['balance']


        except Exception as e:
            logger.error(f"Error placing SELL order: {e}")


    def _create_order_params(self, current_price, side, quantity):
        """Creates order parameters including stop loss and take profit."""
        sl_price = current_price * (1 - self.config['stopLossMultiplier'] * self.config['slTpOffsetPercentage']) if side == 'buy' else current_price * (1 + self.config['stopLossMultiplier'] * self.config['slTpOffsetPercentage'])
        tp_price = current_price * (1 + self.config['takeProfitMultiplier'] * self.config['slTpOffsetPercentage']) if side == 'buy' else current_price * (1 - self.config['takeProfitMultiplier'] * self.config['slTpOffsetPercentage'])

        return {
            'stopLoss': self.market_data.get_exchange_instance().price_to_precision(self.config['symbol'], sl_price),
            'takeProfit': self.market_data.get_exchange_instance().price_to_precision(self.config['symbol'], tp_price),
        }


    def stop(self):
        """Stops the bot by clearing all intervals."""
        for interval_task in self.intervals:
            interval_task.cancel() # Cancel asyncio tasks
        logger.warn("Bot deactivated.")


async def run_bot():
    """Initiates the bot and handles command line interaction."""
    config_manager = ConfigManager()

    symbol = 'BTC/USDT'
    timeframe = '1h'
    log_level = DEFAULT_LOG_LEVEL
    test_mode = False

    if not sys.stdin.isatty(): # Check if running interactively
        console.print("Running in non-interactive mode, using default configurations or environment variables.")
    else:
        console.print("[bold cyan]Initializing Trading Bot Setup...[/bold cyan]")
        exchange_test = ccxt.bybit() # No API keys needed for market data
        await exchange_test.load_markets()
        symbols = list(exchange_test.markets.keys())
        timeframes = list(exchange_test.timeframes.keys()) if exchange_test.timeframes else ['1m', '5m', '15m', '30m', '1h', '4h', '1d']


        def prompt(text, default_value):
            return console.input(f"[bold magenta]{text} (default: {default_value}):[/bold magenta] ") or default_value


        symbol = prompt("Enter symbol (e.g., BTC/USDT)", 'BTC/USDT').upper()
        timeframe = prompt("Enter timeframe (e.g., 1h)", '1h').lower()
        log_level = prompt(f"Enter log level (e.g., info, debug, warn, error) default: {DEFAULT_LOG_LEVEL}", DEFAULT_LOG_LEVEL).lower()
        test_mode_input = prompt("Run in test mode? (yes/no)", 'no').lower()
        test_mode = test_mode_input == 'yes'


    cli_config = {'symbol': symbol, 'timeframe': timeframe, 'logLevel': log_level, 'exchange': {'testMode': test_mode}}
    config = await config_manager.load(cli_config)
    config_manager.display_config()


    bot = TradingBot(config)
    await bot.start()

    async def shutdown_handler():
        """Handles shutdown signals gracefully."""
        console.print("[bold red]Initiating bot shutdown...[/bold red]")
        bot.stop()
        pending = asyncio.Task.all_tasks()
        console.print("[bold yellow]Waiting for pending tasks to complete...[/bold yellow]")
        await asyncio.gather(*pending, return_exceptions=True) # Wait for all tasks to finish, even if cancelled
        console.print("[bold green]Bot shutdown complete.[/bold green]")


    loop = asyncio.get_event_loop()
    for signal in [signal.SIGINT, signal.SIGTERM]:
        loop.add_signal_handler(signal, lambda: asyncio.create_task(shutdown_handler())) # Handle signals for graceful shutdown


    console.print("[bold green]Bot is now running. Press Ctrl+C to stop.[/bold green]")
    try:
        await asyncio.Future()  # Keep the script running indefinitely until Ctrl+C
    except asyncio.CancelledError:
        pass # Expected on shutdown


if __name__ == "__main__":
    import signal
    asyncio.run(run_bot())
