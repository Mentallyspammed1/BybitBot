# advanced_trading_bot.py
import logging
import os
import time
import numpy as np
import pandas as pd
import yaml
import ccxt
from colorama import Fore, Style, init as colorama_init
from dotenv import load_dotenv
from sklearn.linear_model import LinearRegression  # Optional ML integration

# Initialize colorama and environment
colorama_init(autoreset=True)
load_dotenv()

# Configure logging
logger = logging.getLogger("AdvancedTradingBot")
logger.setLevel(logging.DEBUG)

formatter = logging.Formatter(
    f"{Fore.CYAN}%(asctime)s - {Fore.YELLOW}%(levelname)s - {Fore.GREEN}%(message)s{Style.RESET_ALL}"
)

console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)
file_handler = logging.FileHandler("advanced_trading_bot.log", mode='a')
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

# Retry decorator for API calls
def retry_api_call(max_retries=3, initial_delay=1):
    def decorator(func):
        def wrapper(*args, **kwargs):
            retries = 0
            delay = initial_delay
            while retries <= max_retries:
                try:
                    return func(*args, **kwargs)
                except ccxt.RateLimitExceeded as e:
                    logger.warning(f"{Fore.YELLOW}Rate limit exceeded, retrying in {delay}s... ({retries+1}/{max_retries})")
                    time.sleep(delay)
                    delay *= 2
                    retries += 1
                except Exception as e:
                    logger.error(f"{Fore.RED}Error: {e} - Retrying in {delay}s ({retries+1}/{max_retries})")
                    time.sleep(delay)
                    delay *= 2
                    retries += 1
            logger.error(f"{Fore.RED}Max retries exceeded for {func.__name__}")
            return None
        return wrapper
    return decorator

class AdvancedTradingBot:
    def __init__(self):
        self.load_config()
        self.symbol = input("Enter trading symbol (e.g., BTC/USDT): ").upper()
        self.exchange = self.initialize_exchange()
        self.position = None
        self.entry_price = None
        self.confidence_level = 0
        self.open_positions = []
        self.trade_history = []
        self.risk_per_trade = self.config['risk_management']['risk_per_trade']
        self.leverage = self.config['risk_management']['leverage']

    def load_config(self):
        try:
            with open('config.yaml') as f:
                self.config = yaml.safe_load(f)
            logger.info(f"{Fore.GREEN}Loaded configuration successfully")
        except Exception as e:
            logger.error(f"{Fore.RED}Config error: {e}")
            raise

    def initialize_exchange(self):
        try:
            exchange = ccxt.bybit({
                'apiKey': os.getenv('BYBIT_API_KEY'),
                'secret': os.getenv('BYBIT_API_SECRET'),
                'options': {'defaultType': 'future'},
                'enableRateLimit': True
            })
            exchange.load_markets()
            logger.info(f"{Fore.GREEN}Connected to Bybit successfully")
            return exchange
        except Exception as e:
            logger.error(f"{Fore.RED}Exchange init failed: {e}")
            raise

    @retry_api_call()
    def get_market_data(self):
        """Fetch comprehensive market data across multiple timeframes"""
        timeframes = ['1m', '5m', '15m']
        data = {}
        for tf in timeframes:
            data[tf] = {
                'ohlcv': self.exchange.fetch_ohlcv(self.symbol, tf, limit=100),
                'orderbook': self.exchange.fetch_order_book(self.symbol, limit=20)
            }
        return data

    def analyze_market(self, data):
        """Generate multi-timeframe, multi-factor market analysis"""
        analysis = {
            'price': self.exchange.fetch_ticker(self.symbol)['last'],
            'timeframes': {},
            'signals': [],
            'confidence': 0
        }

        for tf, tf_data in data.items():
            closes = [c[4] for c in tf_data['ohlcv']]
            analysis['timeframes'][tf] = {
                'ema': self.calculate_ema(closes),
                'rsi': self.calculate_rsi(closes),
                'macd': self.calculate_macd(closes),
                'order_book': self.analyze_order_book(tf_data['orderbook'])
            }

        # Generate signals
        for tf, tf_analysis in analysis['timeframes'].items():
            if tf_analysis['price'] > tf_analysis['ema']:
                analysis['signals'].append((f"{tf} EMA Bullish", self.config['weights']['ema']))
            if tf_analysis['rsi'] < 30:
                analysis['signals'].append((f"{tf} RSI Oversold", self.config['weights']['rsi']))
            if tf_analysis['order_book']['imbalance'] > 1.2:
                analysis['signals'].append((f"{tf} Bid Imbalance", self.config['weights']['imbalance']))

        # Calculate confidence level
        analysis['confidence'] = sum(w for _, w in analysis['signals'])
        self.confidence_level = analysis['confidence']

        return analysis

    def calculate_ema(self, closes, period=20):
        return np.mean(closes[-period:])

    def calculate_rsi(self, closes, period=14):
        deltas = np.diff(closes)
        gains = deltas.clip(min=0)
        losses = -deltas.clip(max=0)
        avg_gain = np.mean(gains[-period:])
        avg_loss = np.mean(losses[-period:])
        return 100 - (100 / (1 + (avg_gain / avg_loss))) if avg_loss != 0 else 100

    def calculate_macd(self, closes):
        ema12 = np.mean(closes[-12:])
        ema26 = np.mean(closes[-26:])
        return {'macd_line': ema12 - ema26, 'signal': np.mean(closes[-9:])}

    def analyze_order_book(self, orderbook):
        bids = orderbook['bids']
        asks = orderbook['asks']
        bid_vol = sum(b[1] for b in bids)
        ask_vol = sum(a[1] for a in asks)
        return {
            'imbalance': bid_vol / ask_vol if ask_vol > 0 else 1,
            'spread': asks[0][0] - bids[0][0],
            'top_bid': bids[0][0],
            'top_ask': asks[0][0]
        }

    def execute_strategy(self, analysis):
        """Determine trading actions based on analysis"""
        action = 'hold'
        reason = []
        confidence = 'low'

        if self.confidence_level >= 3.0:
            confidence = 'high'
            if not self.position:
                action = 'long' if analysis['price'] > analysis['timeframes']['15m']['ema'] else 'short'
                reason = [s[0] for s in analysis['signals']]
        elif self.confidence_level <= -3.0:
            confidence = 'high'
            if self.position:
                action = 'close'

        # Trailing stop logic
        if self.position:
            self.manage_trailing_stop(analysis['price'])

        return action, reason, confidence

    def manage_trailing_stop(self, current_price):
        if self.position == 'long':
            trail_price = current_price * (1 - self.config['trailing_stop']['callback'])
            if trail_price > self.entry_price * (1 - self.config['stop_loss']):
                self.stop_loss = trail_price
        elif self.position == 'short':
            trail_price = current_price * (1 + self.config['trailing_stop']['callback'])
            if trail_price < self.entry_price * (1 + self.config['stop_loss']):
                self.stop_loss = trail_price

    def run(self):
        logger.info(f"{Fore.MAGENTA}Starting trading bot for {self.symbol}")
        try:
            while True:
                data = self.get_market_data()
                analysis = self.analyze_market(data)
                action, reasons, confidence = self.execute_strategy(analysis)

                logger.info(f"\n{Fore.CYAN}=== Market Analysis ===")
                logger.info(f"Price: {analysis['price']:.2f}")
                for tf, tf_analysis in analysis['timeframes'].items():
                    logger.info(f"\n{Fore.YELLOW}=== {tf} Analysis ===")
                    logger.info(f"EMA: {tf_analysis['ema']:.2f}")
                    logger.info(f"RSI: {tf_analysis['rsi']:.2f}")
                    logger.info(f"Order Book Imbalance: {tf_analysis['order_book']['imbalance']:.2f}x")
                logger.info(f"Signals: {', '.join([s[0] for s in analysis['signals']])}")
                logger.info(f"Confidence: {confidence.upper()} ({self.confidence_level:.2f})")

                if action != 'hold':
                    self.execute_trade(action, reasons, analysis['price'])

                time.sleep(60)

        except KeyboardInterrupt:
            logger.info(f"{Fore.YELLOW}Shutting down bot...")

    def execute_trade(self, action, reasons, price):
        logger.info(f"{Fore.GREEN}Executing {action.upper()} trade because: {', '.join(reasons)}")
        # Add actual order execution logic here
        self.position = action if action != 'close' else None
        self.entry_price = price if action != 'close' else None

if __name__ == "__main__":
    bot = AdvancedTradingBot()
    bot.run()
