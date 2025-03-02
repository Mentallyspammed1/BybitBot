import logging
import os
import time
import numpy as np
import pandas as pd
import yaml
import ccxt
from colorama import Fore, Style, init as colorama_init
from dotenv import load_dotenv
from typing import Dict, List, Tuple, Optional, Any

# Initialize colorama and load environment variables
colorama_init(autoreset=True)
load_dotenv()

# Configure logging with colored output
logger = logging.getLogger("ImprovedTradingBot")
logger.setLevel(logging.DEBUG)

formatter = logging.Formatter(
    f"{Fore.CYAN}%(asctime)s - {Fore.YELLOW}%(levelname)s - {Fore.GREEN}%(message)s{Style.RESET_ALL}"
)

console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)
file_handler = logging.FileHandler("trading_bot.log", mode='a')
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

def retry_api_call(max_retries: int = 3, initial_delay: int = 1):
    """
    Decorator for handling API call retries with exponential backoff.
    """
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

class ImprovedTradingBot:
    def __init__(self):
        """
        Initialize the trading bot with enhanced configuration and features.
        This version uses no external ML libraries and relies entirely on technical analysis.
        """
        self.load_config()
        self.symbol = input("Enter trading symbol (e.g., BTC/USDT): ").upper()
        self.exchange = self.initialize_exchange()
        self.position = None
        self.entry_price = None
        self.confidence_level = 0
        self.open_positions = []
        self.trade_history = []
        
        # Risk management parameters
        self.risk_per_trade = self.config['risk_management']['risk_per_trade']
        self.leverage = self.config['risk_management']['leverage']
        self.stop_loss_pct = float(os.getenv('STOP_LOSS_PERCENTAGE', '0.015'))
        self.take_profit_pct = float(os.getenv('TAKE_PROFIT_PERCENTAGE', '0.03'))
        
        # Technical analysis parameters
        self.timeframes = ['1m', '5m', '15m']
        self.ema_period = int(os.getenv('EMA_PERIOD', '10'))
        self.rsi_period = int(os.getenv('RSI_PERIOD', '14'))
        self.macd_fast = 12
        self.macd_slow = 26
        self.macd_signal = 9
        self.stoch_period = 14
        self.bollinger_period = 20
        self.bollinger_std = 2
        
        # Order book parameters
        self.order_book_depth = int(os.getenv('ORDER_BOOK_DEPTH', '20'))
        self.imbalance_threshold = float(os.getenv('IMBALANCE_THRESHOLD', '1.5'))
        
        # Performance tracking
        self.iteration = 0
        self.daily_pnl = 0.0
        self.simulation_mode = os.getenv('SIMULATION_MODE', 'True').lower() in ('true', '1', 'yes')

    def load_config(self) -> None:
        """Load configuration from YAML file with error handling."""
        try:
            with open('config.yaml') as f:
                self.config = yaml.safe_load(f)
            logger.info(f"{Fore.GREEN}Loaded configuration successfully")
        except Exception as e:
            logger.error(f"{Fore.RED}Config error: {e}")
            raise

    @retry_api_call()
    def initialize_exchange(self) -> ccxt.Exchange:
        """Initialize exchange connection with enhanced error handling."""
        try:
            exchange = ccxt.bybit({
                'apiKey': os.getenv('BYBIT_API_KEY'),
                'secret': os.getenv('BYBIT_API_SECRET'),
                'options': {'defaultType': 'future'},
                'enableRateLimit': True,
                'recvWindow': 60000
            })
            exchange.load_markets()
            logger.info(f"{Fore.GREEN}Connected to Bybit successfully")
            return exchange
        except Exception as e:
            logger.error(f"{Fore.RED}Exchange initialization failed: {e}")
            raise

    @retry_api_call()
    def get_market_data(self) -> Dict[str, Dict]:
        """Fetch comprehensive market data across multiple timeframes."""
        data = {}
        for tf in self.timeframes:
            data[tf] = {
                'ohlcv': self.exchange.fetch_ohlcv(self.symbol, tf, limit=100),
                'orderbook': self.exchange.fetch_order_book(self.symbol, limit=self.order_book_depth)
            }
        return data

    def analyze_market(self, data: Dict[str, Dict]) -> Dict[str, Any]:
        """
        Generate multi-timeframe, multi-factor market analysis.
        Calculates technical indicators and aggregates signals.
        """
        analysis = {
            'price': self.exchange.fetch_ticker(self.symbol)['last'],
            'timeframes': {},
            'signals': [],
            'confidence': 0
        }

        for tf, tf_data in data.items():
            closes = np.array([c[4] for c in tf_data['ohlcv']])
            highs = np.array([c[2] for c in tf_data['ohlcv']])
            lows = np.array([c[3] for c in tf_data['ohlcv']])
            
            analysis['timeframes'][tf] = {
                'ema': self.calculate_ema(closes),
                'rsi': self.calculate_rsi(closes),
                'macd': self.calculate_macd(closes),
                'stoch_rsi': self.calculate_stoch_rsi(closes),
                'bollinger': self.calculate_bollinger_bands(closes),
                'atr': self.calculate_atr(highs, lows, closes),
                'order_book': self.analyze_order_book(tf_data['orderbook'])
            }

            tf_analysis = analysis['timeframes'][tf]
            
            # Price vs EMA signal
            if closes[-1] > tf_analysis['ema']:
                analysis['signals'].append((f"{tf} EMA Bullish", self.config['weights']['ema']))
            elif closes[-1] < tf_analysis['ema']:
                analysis['signals'].append((f"{tf} EMA Bearish", -self.config['weights']['ema']))

            # RSI signal
            if tf_analysis['rsi'] < 30:
                analysis['signals'].append((f"{tf} RSI Oversold", self.config['weights']['rsi']))
            elif tf_analysis['rsi'] > 70:
                analysis['signals'].append((f"{tf} RSI Overbought", -self.config['weights']['rsi']))

            # MACD signal
            if tf_analysis['macd']['macd'] > tf_analysis['macd']['signal']:
                analysis['signals'].append((f"{tf} MACD Bullish", self.config['weights']['macd']))
            elif tf_analysis['macd']['macd'] < tf_analysis['macd']['signal']:
                analysis['signals'].append((f"{tf} MACD Bearish", -self.config['weights']['macd']))

            # Bollinger Bands signal
            if closes[-1] < tf_analysis['bollinger']['lower']:
                analysis['signals'].append((f"{tf} BB Oversold", self.config['weights']['bollinger']))
            elif closes[-1] > tf_analysis['bollinger']['upper']:
                analysis['signals'].append((f"{tf} BB Overbought", -self.config['weights']['bollinger']))

            # Order book imbalance signal
            ob_analysis = tf_analysis['order_book']
            if ob_analysis['imbalance'] > self.imbalance_threshold:
                analysis['signals'].append((f"{tf} Strong Bid Pressure", self.config['weights']['imbalance']))
            elif ob_analysis['imbalance'] < 1 / self.imbalance_threshold:
                analysis['signals'].append((f"{tf} Strong Ask Pressure", -self.config['weights']['imbalance']))

        # Aggregate confidence level
        analysis['confidence'] = sum(weight for _, weight in analysis['signals'])
        self.confidence_level = analysis['confidence']

        return analysis

    def calculate_ema(self, data: np.ndarray, period: int = None) -> float:
        """Calculate Exponential Moving Average"""
        if period is None:
            period = self.ema_period
        weights = np.exp(np.linspace(-1., 0., period))
        weights /= weights.sum()
        return np.convolve(data, weights, mode='valid')[-1]

    def calculate_rsi(self, data: np.ndarray) -> float:
        """Calculate Relative Strength Index"""
        deltas = np.diff(data)
        gains = np.maximum(deltas, 0)
        losses = -np.minimum(deltas, 0)
        avg_gain = np.mean(gains[-self.rsi_period:])
        avg_loss = np.mean(losses[-self.rsi_period:])
        return 100 - (100 / (1 + avg_gain / avg_loss)) if avg_loss != 0 else 100

    def calculate_macd(self, data: np.ndarray) -> Dict[str, float]:
        """Calculate MACD indicators"""
        ema_fast = self.calculate_ema(data, self.macd_fast)
        ema_slow = self.calculate_ema(data, self.macd_slow)
        macd_line = ema_fast - ema_slow
        signal_line = self.calculate_ema(np.array([macd_line]), self.macd_signal)
        return {
            'macd': macd_line,
            'signal': signal_line,
            'histogram': macd_line - signal_line
        }

    def calculate_bollinger_bands(self, data: np.ndarray) -> Dict[str, float]:
        """Calculate Bollinger Bands"""
        sma = np.mean(data[-self.bollinger_period:])
        std = np.std(data[-self.bollinger_period:])
        return {
            'upper': sma + self.bollinger_std * std,
            'middle': sma,
            'lower': sma - self.bollinger_std * std
        }

    def calculate_atr(self, highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, period: int = 14) -> float:
        """Calculate Average True Range"""
        tr_list = []
        for i in range(1, len(closes)):
            tr = max(
                highs[i] - lows[i],
                abs(highs[i] - closes[i - 1]),
                abs(lows[i] - closes[i - 1])
            )
            tr_list.append(tr)
        return np.mean(tr_list[-period:])

    def calculate_stoch_rsi(self, data: np.ndarray) -> Tuple[float, float]:
        """Calculate Stochastic RSI"""
        rsi_values = np.array([self.calculate_rsi(data[i:]) for i in range(len(data) - self.stoch_period + 1)])
        min_val = np.min(rsi_values)
        max_val = np.max(rsi_values)
        k = 100 * (rsi_values[-1] - min_val) / (max_val - min_val) if max_val != min_val else 50
        d = np.mean(rsi_values[-3:])
        return k, d

    def analyze_order_book(self, orderbook: Dict) -> Dict[str, float]:
        """Analyze order book for market microstructure insights."""
        bids = orderbook['bids']
        asks = orderbook['asks']
        bid_vol = sum(b[1] for b in bids)
        ask_vol = sum(a[1] for a in asks)
        return {
            'imbalance': bid_vol / ask_vol if ask_vol > 0 else float('inf'),
            'spread': asks[0][0] - bids[0][0],
            'bid_wall': max(b[1] for b in bids),
            'ask_wall': max(a[1] for a in asks)
        }

    def calculate_position_size(self, price: float) -> float:
        """Calculate optimal position size based on risk management rules."""
        balance = self.exchange.fetch_balance().get('USDT', {}).get('free', 0)
        volatility = self.calculate_volatility()
        base_size = balance * self.risk_per_trade
        if volatility:
            adjusted_size = base_size * (1 + volatility * self.config['risk_management']['volatility_multiplier'])
            return min(adjusted_size, balance * 0.05)  # Cap at 5% of balance
        return base_size

    def calculate_volatility(self) -> Optional[float]:
        """Calculate market volatility based on recent 1m candles."""
        try:
            ohlcv = self.exchange.fetch_ohlcv(self.symbol, '1m', limit=30)
            prices = np.array([candle[4] for candle in ohlcv])
            returns = np.diff(prices) / prices[:-1]
            return np.std(returns)
        except Exception as e:
            logger.error(f"Error calculating volatility: {e}")
            return None

    @retry_api_call()
    def execute_trade(self, side: str, size: float, price: float) -> Dict[str, Any]:
        """
        Execute a market order while handling errors.
        In simulation mode, trade details are logged without executing real orders.
        """
        try:
            if self.simulation_mode:
                trade_details = {
                    "status": "simulated",
                    "side": side,
                    "size": size,
                    "price": price,
                    "timestamp": time.time()
                }
                logger.info(f"{Fore.CYAN}[SIMULATION] {side.upper()} order of size {size:.4f} at {price:.2f}")
                return trade_details
            
            order = self.exchange.create_market_order(
                symbol=self.symbol,
                side=side,
                amount=size,
                params={'leverage': self.leverage}
            )
            
            logger.info(f"{Fore.GREEN}Executed {side.upper()} order: Size={size:.4f}, Price={price:.2f}")
            return order
        except Exception as e:
            logger.error(f"{Fore.RED}Trade execution error: {e}")
            return None

    def run(self) -> None:
        """
        Main trading loop:
         - Fetches market data.
         - Analyzes multiple timeframes using technical indicators.
         - Checks entry/exit conditions based on aggregated confidence.
         - Manages positions with stop-loss and take-profit logic.
        """
        logger.info(f"{Fore.MAGENTA}Starting improved trading bot for {self.symbol}")
        try:
            while True:
                self.iteration += 1
                logger.info(f"\n=== Iteration {self.iteration} ===")
                
                # Fetch and analyze market data
                data = self.get_market_data()
                analysis = self.analyze_market(data)
                
                # Log current market conditions
                logger.info(f"Current price: {analysis['price']:.2f}")
                logger.info(f"Confidence level: {analysis['confidence']:.2f}")
                logger.info(f"Active position: {self.position}")
                
                # Check for entry signals
                if analysis['confidence'] > self.config['confidence_threshold']:
                    if self.position is None:
                        # Enter a new long position
                        size = self.calculate_position_size(analysis['price'])
                        self.position = 'buy'
                        self.entry_price = analysis['price']
                        trade = self.execute_trade(self.position, size, self.entry_price)
                        if trade:
                            self.open_positions.append(trade)
                            logger.info(f"{Fore.GREEN}Opened long position at {self.entry_price:.2f}")
                    elif self.position == 'sell':
                        # Close short position and open long position
                        size = self.calculate_position_size(analysis['price'])
                        close_trade = self.execute_trade('buy', size, analysis['price'])
                        if close_trade:
                            logger.info(f"{Fore.YELLOW}Closed short position at {analysis['price']:.2f}")
                            self.position = 'buy'
                            self.entry_price = analysis['price']
                            trade = self.execute_trade(self.position, size, self.entry_price)
                            if trade:
                                self.open_positions.append(trade)
                                logger.info(f"{Fore.GREEN}Opened long position at {self.entry_price:.2f}")
                
                # Check for exit signals / short entry signals
                elif analysis['confidence'] < -self.config['confidence_threshold']:
                    if self.position is None:
                        # Enter a new short position
                        size = self.calculate_position_size(analysis['price'])
                        self.position = 'sell'
                        self.entry_price = analysis['price']
                        trade = self.execute_trade(self.position, size, self.entry_price)
                        if trade:
                            self.open_positions.append(trade)
                            logger.info(f"{Fore.RED}Opened short position at {self.entry_price:.2f}")
                    elif self.position == 'buy':
                        # Close long position and open short position
                        size = self.calculate_position_size(analysis['price'])
                        close_trade = self.execute_trade('sell', size, analysis['price'])
                        if close_trade:
                            logger.info(f"{Fore.YELLOW}Closed long position at {analysis['price']:.2f}")
                            self.position = 'sell'
                            self.entry_price = analysis['price']
                            trade = self.execute_trade(self.position, size, self.entry_price)
                            if trade:
                                self.open_positions.append(trade)
                                logger.info(f"{Fore.RED}Opened short position at {self.entry_price:.2f}")
                
                # Manage open position with stop-loss and take-profit logic
                if self.position is not None:
                    current_price = analysis['price']
                    if self.position == 'buy':
                        stop_loss_price = self.entry_price * (1 - self.stop_loss_pct)
                        take_profit_price = self.entry_price * (1 + self.take_profit_pct)
                        trailing_stop = current_price * (1 - self.stop_loss_pct)
                        if trailing_stop > stop_loss_price:
                            stop_loss_price = trailing_stop
                    else:
                        stop_loss_price = self.entry_price * (1 + self.stop_loss_pct)
                        take_profit_price = self.entry_price * (1 - self.take_profit_pct)
                        trailing_stop = current_price * (1 + self.stop_loss_pct)
                        if trailing_stop < stop_loss_price:
                            stop_loss_price = trailing_stop

                    logger.info(f"Position: {self.position.upper()}")
                    logger.info(f"Entry Price: {self.entry_price:.2f}")
                    logger.info(f"Current Price: {current_price:.2f}")
                    logger.info(f"Stop Loss: {stop_loss_price:.2f}")
                    logger.info(f"Take Profit: {take_profit_price:.2f}")
                    
                    # Check if stop-loss or take-profit levels are hit
                    if (self.position == 'buy' and (current_price <= stop_loss_price or current_price >= take_profit_price)) or \
                       (self.position == 'sell' and (current_price >= stop_loss_price or current_price <= take_profit_price)):
                        
                        exit_reason = "Stop Loss" if (
                            (self.position == 'buy' and current_price <= stop_loss_price) or \
                            (self.position == 'sell' and current_price >= stop_loss_price)
                        ) else "Take Profit"
                        
                        size = self.calculate_position_size(current_price)
                        close_trade = self.execute_trade('sell' if self.position == 'buy' else 'buy', size, current_price)
                        if close_trade:
                            logger.info(f"{Fore.YELLOW}Position closed ({exit_reason}) at {current_price:.2f}")
                            if self.position == 'buy':
                                pnl = (current_price - self.entry_price) / self.entry_price * 100
                            else:
                                pnl = (self.entry_price - current_price) / self.entry_price * 100
                            self.daily_pnl += pnl
                            logger.info(f"{Fore.CYAN}Trade PnL: {pnl:.2f}% | Daily PnL: {self.daily_pnl:.2f}%")
                            self.position = None
                            self.entry_price = None

                logger.info("=== Performance Metrics ===")
                logger.info(f"Daily PnL: {self.daily_pnl:.2f}%")
                logger.info(f"Open Positions: {len(self.open_positions)}")
                logger.info("========================")
                
                sleep_time = self.config.get('trading_loop_interval', 60)
                logger.info(f"Sleeping for {sleep_time} seconds...")
                time.sleep(sleep_time)

        except KeyboardInterrupt:
            logger.info(f"{Fore.MAGENTA}Trading bot stopped by user{Style.RESET_ALL}")
            if self.position is not None:
                logger.warning(f"{Fore.YELLOW}Warning: Bot stopped with open position: {self.position}{Style.RESET_ALL}")
        except Exception as e:
            logger.error(f"{Fore.RED}Unexpected error: {e}{Style.RESET_ALL}")
            raise
        finally:
            logger.info(f"{Fore.CYAN}Final Daily PnL: {self.daily_pnl:.2f}%{Style.RESET_ALL}")
            logger.info(f"{Fore.CYAN}Trading session ended at {time.strftime('%Y-%m-%d %H:%M:%S')}{Style.RESET_ALL}")

if __name__ == "__main__":
    try:
        logger.info(f"{Fore.CYAN}Starting trading bot at {time.strftime('%Y-%m-%d %H:%M:%S')}{Style.RESET_ALL}")
        bot = ImprovedTradingBot()
        bot.run()
    except Exception as e:
        logger.error(f"{Fore.RED}Fatal error: {e}{Style.RESET_ALL}")
