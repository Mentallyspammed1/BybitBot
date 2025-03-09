import ccxt                                                import time                                                import os
import numpy as np                                         import pandas as pd
import logging                                             from dotenv import load_dotenv
from colorama import Fore, Style, init as colorama_init                                                               colorama_init(autoreset=True)
                                                           NEON_GREEN = Fore.GREEN
NEON_CYAN = Fore.CYAN                                      NEON_YELLOW = Fore.YELLOW
NEON_MAGENTA = Fore.MAGENTA                                NEON_RED = Fore.RED                                        RESET_COLOR = Style.RESET_ALL
                                                           logger = logging.getLogger("EnhancedTradingBot")
logger.setLevel(logging.DEBUG)                             formatter = logging.Formatter(
    f"{NEON_CYAN}%(asctime)s - {NEON_YELLOW}%(levelname)s -{NEON_GREEN}%(message)s{RESET_COLOR}")                     
console_handler = logging.StreamHandler()                  console_handler.setFormatter(formatter)
logger.addHandler(console_handler)                         
file_handler =                                             logging.FileHandler("enhanced_trading_bot.log")            file_handler.setFormatter(formatter)
logger.addHandler(file_handler)                            
load_dotenv()                                              
                                                           class EnhancedTradingBot:                                      """
    An enhanced cryptocurrency trading bot that integrates technical indicators,
    order book analysis, and risk management strategies to automate trading decisions.
    """                                                                                                                   def __init__(self, symbol):                                    """                                                        Initializes the EnhancedTradingBot with exchange   configurations, trading parameters,                                and technical indicator settings.                                                                                     Args:                                                          symbol (str): The trading symbol (e.g., 'BTC/
USDT').                                                            """
        logger.info("Initializing EnhancedTradingBot...")  
        self.exchange_id = os.getenv('EXCHANGE_ID',        'bybit')                                                           self.api_key = os.getenv('BYBIT_API_KEY')
        self.api_secret = os.getenv('BYBIT_API_SECRET')            self.simulation_mode = os.getenv(
            'SIMULATION_MODE', 'True').lower() in ('true', '1', 'yes')
                                                                   if not symbol:                                                 logger.error("Symbol must be provided.")
            raise ValueError("Symbol cannot be empty.")            self.symbol = symbol.upper()
                                                                   # --- Order Size Configuration ---
        try:                                                           self.base_order_size_percentage = float(                       os.getenv('ORDER_SIZE_PERCENTAGE', 0.01))
            if not 0 < self.base_order_size_percentage <=  1:                                                                         raise ValueError(                                              "ORDER_SIZE_PERCENTAGE must be between
0 and 1.")                                                         except ValueError as e:                                        logger.error(f"Invalid ORDER_SIZE_PERCENTAGE:
{e}")                                                                  raise
                                                                   # --- Risk Management Parameters ---
        try:                                                           self.take_profit_pct = float(                                  os.getenv('TAKE_PROFIT_PERCENTAGE', 0.03))
            if self.take_profit_pct < 0:                                   raise ValueError("TAKE_PROFIT_PERCENTAGE
cannot be negative.")                                              except ValueError as e:
            logger.error(f"Invalid TAKE_PROFIT_PERCENTAGE: {e}")                                                                  raise
                                                                   try:
            self.stop_loss_pct = float(                                    os.getenv('STOP_LOSS_PERCENTAGE', 0.015))
            if self.stop_loss_pct < 0:                                     raise ValueError("STOP_LOSS_PERCENTAGE     cannot be negative.")
        except ValueError as e:                                        logger.error(f"Invalid STOP_LOSS_PERCENTAGE:
{e}")                                                                  raise
                                                                   self.trailing_stop_loss_active = os.getenv(                    'TRAILING_STOP_ACTIVE', 'False').lower() in
('true', '1', 'yes')                                               try:
            self.trailing_stop_callback = float(                           os.getenv('TRAILING_STOP_CALLBACK', 0.02))
            if self.trailing_stop_callback < 0:                            raise ValueError("TRAILING_STOP_CALLBACK   cannot be negative.")
        except ValueError as e:                                        logger.error(f"Invalid TRAILING_STOP_CALLBACK:
{e}")                                                                  raise
        self.high_since_entry = -np.inf                            self.low_since_entry = np.inf                      
        # --- Technical Indicator Periods ---                      try:
            self.ema_period = int(os.getenv('EMA_PERIOD',  10))
            if self.ema_period <= 0:                                       raise ValueError("EMA_PERIOD must be a     positive integer.")
        except ValueError as e:                                        logger.error(f"Invalid EMA_PERIOD: {e}")
            raise                                          
        try:                                                           self.rsi_period = int(os.getenv('RSI_PERIOD',  14))
            if self.rsi_period <= 0:                                       raise ValueError("RSI_PERIOD must be a
positive integer.")                                                except ValueError as e:
            logger.error(f"Invalid RSI_PERIOD: {e}")                   raise                                          
        self.macd_short_period = 12                                self.macd_long_period = 26                                 self.macd_signal_period = 9                                                                                           try:                                                           self.stoch_rsi_period =                        int(os.getenv('STOCH_RSI_PERIOD', 14))
            if self.stoch_rsi_period <= 0:                                 raise ValueError(
                    "STOCH_RSI_PERIOD must be a positive   integer.")
        except ValueError as e:                                        logger.error(f"Invalid STOCH_RSI_PERIOD: {e}")             raise
                                                                   try:
            self.stoch_rsi_k_period =                      int(os.getenv('STOCH_RSI_K_PERIOD', 3))
            if self.stoch_rsi_k_period <= 0:                               raise ValueError(                                              "STOCH_RSI_K_PERIOD must be a positive
integer.")                                                         except ValueError as e:
            logger.error(f"Invalid STOCH_RSI_K_PERIOD:     {e}")
            raise                                                                                                             try:
            self.stoch_rsi_d_period =                      int(os.getenv('STOCH_RSI_D_PERIOD', 3))
            if self.stoch_rsi_d_period <= 0:                               raise ValueError(                                              "STOCH_RSI_D_PERIOD must be a positive integer.")                                                         except ValueError as e:
            logger.error(f"Invalid STOCH_RSI_D_PERIOD:     {e}")
            raise                                          
        try:                                                           self.vwap_period = int(os.getenv('VWAP_PERIOD',20))
            if self.vwap_period <= 0:                                      raise ValueError("VWAP_PERIOD must be a
positive integer.")                                                except ValueError as e:
            logger.error(f"Invalid VWAP_PERIOD: {e}")                  self.vwap_period = 20                          
        try:                                                           self.bb_period = int(os.getenv('BB_PERIOD',
20))                                                                   if self.bb_period <= 0:
                raise ValueError("BB_PERIOD must be a      positive integer.")                                                except ValueError as e:                                        logger.error(f"Invalid BB_PERIOD: {e}")                    self.bb_period = 20
        try:                                                           self.bb_std_dev = float(os.getenv('BB_STD_DEV',
2.0))                                                              except ValueError as e:                                        logger.error(f"Invalid BB_STD_DEV: {e}")
            self.bb_std_dev = 2.0                          
        try:                                                           self.atr_period = int(os.getenv('ATR_PERIOD',
14))                                                                   if self.atr_period <= 0:                                       raise ValueError("ATR_PERIOD must be a
positive integer.")                                                except ValueError as e:
            logger.error(f"Invalid ATR_PERIOD: {e}")                   self.atr_period = 14
        try:                                                           self.atr_order_size_multiplier = float(                        os.getenv('ATR_ORDER_SIZE_MULTIPLIER',
2.0))                                                              except ValueError as e:
            logger.error(f"Invalid                         ATR_ORDER_SIZE_MULTIPLIER: {e}")
            self.atr_order_size_multiplier = 2.0                                                                      
        # --- Order Book Analysis Parameters ---                   try:
            self.order_book_depth =                        int(os.getenv('ORDER_BOOK_DEPTH', 10))
            if not 1 <= self.order_book_depth <= 50:                       raise ValueError("ORDER_BOOK_DEPTH must be between 1 and 50.")
        except ValueError as e:                                        logger.error(f"Invalid ORDER_BOOK_DEPTH: {e}")
            raise                                          
        try:                                                           self.imbalance_threshold = float(                              os.getenv('IMBALANCE_THRESHOLD', 1.5))
            if self.imbalance_threshold < 0:                               raise ValueError("IMBALANCE_THRESHOLD
cannot be negative.")                                              except ValueError as e:                                        logger.error(f"Invalid IMBALANCE_THRESHOLD:    {e}")                                                                  raise
                                                                   try:
            self.volume_cluster_threshold = float(                         os.getenv('VOLUME_CLUSTER_THRESHOLD',
10000))                                                                if self.volume_cluster_threshold < 0:                          raise ValueError(
                    "VOLUME_CLUSTER_THRESHOLD cannot be    negative.")
        except ValueError as e:                                        logger.error(f"Invalid
VOLUME_CLUSTER_THRESHOLD: {e}")                                        raise                                          
        try:                                                           self.ob_delta_lookback =
int(os.getenv('OB_DELTA_LOOKBACK', 5))                                 if self.ob_delta_lookback <= 0:
                raise ValueError(                                              "OB_DELTA_LOOKBACK must be a positive  integer.")
        except ValueError as e:                                        logger.error(f"Invalid OB_DELTA_LOOKBACK: {e}")
            raise                                          
        try:                                                           self.cluster_proximity_threshold_pct = float(  
os.getenv('CLUSTER_PROXIMITY_THRESHOLD_PCT', 0.005))                   if not 0 <=
self.cluster_proximity_threshold_pct <= 1:                                 raise ValueError(
                    "CLUSTER_PROXIMITY_THRESHOLD_PCT must  be between 0 and 1.")                                              except ValueError as e:
            logger.error(f"Invalid                         CLUSTER_PROXIMITY_THRESHOLD_PCT: {e}")
            raise                                          
                                                                   # --- Signal Weights ---                                   try:
            self.ema_weight = float(os.getenv('EMA_WEIGHT',1.0))
        except ValueError as e:                                        logger.error(f"Invalid EMA_WEIGHT: {e}")                   self.ema_weight = 1.0                                                                                             try:
            self.rsi_weight = float(os.getenv('RSI_WEIGHT',0.8))
        except ValueError as e:                                        logger.error(f"Invalid RSI_WEIGHT: {e}")
            self.rsi_weight = 0.8                                                                                             try:
            self.macd_weight =                             float(os.getenv('MACD_WEIGHT', 1.2))
        except ValueError as e:                                        logger.error(f"Invalid MACD_WEIGHT: {e}")
            self.macd_weight = 1.2                                                                                            try:
            self.stoch_rsi_weight =                        float(os.getenv('STOCH_RSI_WEIGHT', 0.7))
        except ValueError as e:                                        logger.error(f"Invalid STOCH_RSI_WEIGHT: {e}")
            self.stoch_rsi_weight = 0.7                                                                                       try:                                                           self.imbalance_weight =                        float(os.getenv('IMBALANCE_WEIGHT', 1.5))
        except ValueError as e:                                        logger.error(f"Invalid IMBALANCE_WEIGHT: {e}")             self.imbalance_weight = 1.5                                                                                       try:
            self.ob_delta_change_weight = float(                           os.getenv('OB_DELTA_CHANGE_WEIGHT', 0.6))
        except ValueError as e:                                        logger.error(f"Invalid OB_DELTA_CHANGE_WEIGHT: {e}")                                                                  self.ob_delta_change_weight = 0.6              
        try:                                                           self.spread_weight =
float(os.getenv('SPREAD_WEIGHT', -0.3))                            except ValueError as e:
            logger.error(f"Invalid SPREAD_WEIGHT: {e}")                self.spread_weight = -0.3                      
        try:                                                           self.cluster_proximity_weight = float(
                os.getenv('CLUSTER_PROXIMITY_WEIGHT', 0.4))        except ValueError as e:                                        logger.error(f"Invalid                         CLUSTER_PROXIMITY_WEIGHT: {e}")                                        self.cluster_proximity_weight = 0.4
                                                                   try:
            self.vwap_weight =                             float(os.getenv('VWAP_WEIGHT', 0.9))                               except ValueError as e:                                        logger.error(f"Invalid VWAP_WEIGHT: {e}")                  self.vwap_weight = 0.9                                                                                            try:                                                           self.bb_weight = float(os.getenv('BB_WEIGHT',  0.8))
        except ValueError as e:                                        logger.error(f"Invalid BB_WEIGHT: {e}")                    self.bb_weight = 0.8
                                                                                                                              # --- Trading State Variables ---                          self.position = None  # Current position: LONG,    SHORT, or None/FLAT                                                self.entry_price = None                                    self.order_amount = None
        self.trade_count = 0                                       self.last_ob_delta = None
        self.last_spread = None                                    self.bot_running_flag = True                                                                                          if not self.api_key or not self.api_secret:                    logger.error(                                                  "API key and secret must be set in         environment variables.")                                               raise ValueError("API credentials not found.") 
        self.exchange = self._initialize_exchange()                if self.exchange is None:                                      logger.critical(
                "Exchange initialization failed. Bot cannotstart.")                                                               raise Exception("Exchange initialization       failed.")                                                                                                                     logger.info(                                                   f"EnhancedTradingBot initialized for symbol:   {self.symbol}")                                                    logger.info("EnhancedTradingBot initialization
complete.")                                                
    def _initialize_exchange(self):                                """                                                        Initializes the exchange connection using ccxt     library.                                                                                                                      Returns:                                                       ccxt.Exchange: Exchange object if connection issuccessful, None otherwise.                                        """                                                        logger.info(f"Initializing exchange:
{self.exchange_id.upper()}...")                                    try:                                                           exchange_class = getattr(ccxt,                 self.exchange_id)
            exchange = exchange_class({                                    'apiKey': self.api_key,                                    'secret': self.api_secret,
                'enableRateLimit': True,                                   'options': {'defaultType': 'future'}                   })                                                         exchange.load_markets()                                    logger.info(                                                   f"{Fore.GREEN}Connected to                 {self.exchange_id.upper()} successfully.{Style.RESET_ALL}")            return exchange                                        except AttributeError:
            logger.error(                                                  f"Exchange ID '{self.exchange_id}' is not
valid or supported by ccxt.")                                          return None                                            except ccxt.ExchangeError as e:                                logger.error(f"CCXT Exchange error during      initialization: {e}")
            return None                                            except ccxt.NetworkError as e:
            logger.error(f"CCXT Network error during       initialization: {e}")                                                  return None
        except Exception as e:                                         logger.error(
                f"Unexpected error during exchange         initialization: {e}")
            return None                                                                                                   def fetch_market_price(self):                                  """                                                        Fetches the current market price for the trading   symbol.                                                    
        Returns:                                                       float: Current market price, None if unable to fetch.
        """                                                        try:
            ticker =                                       self.exchange.fetch_ticker(self.symbol)
            if ticker and 'last' in ticker and             ticker['last'] is not None:                                                price = ticker['last']                                     logger.debug(f"Fetched market price:       {price:.2f}")                                                              return price                                           else:                                                          logger.warning(                                                f"Could not fetch valid ticker data for{self.symbol}.")                                                           return None                                        except ccxt.NetworkError as e:                                 logger.error(                                                  f"Network error fetching market price for  {self.symbol}: {e}")                                                   return None                                            except ccxt.ExchangeError as e:
            logger.error(                                                  f"Exchange error fetching market price for
{self.symbol}: {e}")                                                   return None
        except Exception as e:                                         logger.error(                                                  f"Unexpected error fetching market price
for {self.symbol}: {e}")                                               return None
                                                               def fetch_order_book(self):
        """                                                        Fetches and analyzes the order book for volume     clusters and imbalance.
                                                                   Returns:
            tuple: Order book data, imbalance ratio, order book delta, delta change, spread, spread change, bid
clusters, ask clusters.                                                       Returns None for order book data and    other values as None or empty lists if fetching fails.             """                                                        try:                                                           order_book = self.exchange.fetch_order_book(                   self.symbol, limit=self.order_book_depth)              bid_clusters, ask_clusters =                   self.detect_volume_clusters(                                               order_book)                                                                                                       bids = order_book.get('bids', [])                          asks = order_book.get('asks', [])                                                                                     if bids and asks:                                              bid_volume = sum(bid[1] for bid in bids)                   ask_volume = sum(ask[1] for ask in asks)
                imbalance_ratio = ask_volume / \                               bid_volume if bid_volume > 0 else
float('inf')                                                               ob_delta = bid_volume - ask_volume                         ob_delta_change = None                                     if self.last_ob_delta is not None:                             ob_delta_change = ob_delta -
self.last_ob_delta                                                         self.last_ob_delta = ob_delta
                                                                           spread = asks[0][0] - bids[0][0] if bids   and asks else None                                                         spread_change = None                                       if self.last_spread is not None and spread
is not None:                                                                   spread_change = spread -
self.last_spread                                                           self.last_spread = spread                                                                                             logger.info(                                                   f"Order Book: Bid Vol =
{bid_volume:.2f}, Ask Vol = {ask_volume:.2f}, Imbalance    Ratio = {imbalance_ratio:.2f}, OB Delta = {ob_delta:.2f},
OB Delta Change = {ob_delta_change}, Spread = {spread:.2f},Spread Change = {spread_change}")                                          return order_book, imbalance_ratio,        ob_delta, ob_delta_change, spread, spread_change,          bid_clusters, ask_clusters
            else:                                                          logger.warning("Order book data unavailable
or empty.")                                                                return order_book, None, None, None, None, None, [], []                                                                                                                  except ccxt.NetworkError as e:
            logger.error(                                                  f"Network error fetching order book for
{self.symbol}: {e}")                                                   return None, None, None, None, None, None, [], []                                                                 except ccxt.ExchangeError as e:                                logger.error(                                                  f"Exchange error fetching order book for   {self.symbol}: {e}")                                                   return None, None, None, None, None, None, [], []                                                                 except Exception as e:                                         logger.error(f"Error fetching order book for   {self.symbol}: {e}")
            return None, None, None, None, None, None, [], []
                                                               def detect_volume_clusters(self, order_book):                  """                                                        Detects significant volume clusters in the order   book.
                                                                   Args:
            order_book (dict): Order book data from        exchange.                                                                                                                     Returns:                                                       tuple: Lists of bid and ask cluster prices.
        """                                                        bid_cluster_prices = []
        ask_cluster_prices = []                                    bids = np.array(order_book.get('bids', []))                asks = np.array(order_book.get('asks', []))                                                                           if bids.size:
            bid_clusters = bids[bids[:, 1] >               self.volume_cluster_threshold]
            if bid_clusters.size:                                          logger.info(                                                   f"Significant bid clusters detected:   {bid_clusters}")                                                           bid_cluster_prices = bid_clusters[:,
0].tolist()                                                        if asks.size:
            ask_clusters = asks[asks[:, 1] >               self.volume_cluster_threshold]                                         if ask_clusters.size:                                          logger.info(                                                   f"Significant ask clusters detected:
{ask_clusters}")                                                           ask_cluster_prices = ask_clusters[:,
0].tolist()                                                        return bid_cluster_prices, ask_cluster_prices      
    def get_cluster_proximity_signal(self, current_price,  bid_clusters, ask_clusters):                                       """                                                        Generates a trading signal based on proximity to
volume clusters.                                                                                                              Args:                                                          current_price (float): Current market price.               bid_clusters (list): List of bid cluster
prices.                                                                ask_clusters (list): List of ask cluster
prices.                                                                                                                       Returns:
            float: Signal value based on cluster proximity.Positive for bid cluster proximity, negative for ask       cluster.                                                           """
        signal = 0                                                 proximity_threshold = current_price *              self.cluster_proximity_threshold_pct                                                                                          if not bid_clusters:
            bid_clusters = []                                      if not ask_clusters:
            ask_clusters = []                                                                                                 for bid_price in bid_clusters:                                 if 0 < current_price - bid_price <=            proximity_threshold:
                signal += 0.5
                logger.info(                                                   f"Price is close to bid cluster at     {bid_price:.2f} (potential support).")
                                                                   for ask_price in ask_clusters:
            if 0 < ask_price - current_price <=            proximity_threshold:
                signal -= 0.5                                              logger.info(                                                   f"Price is close to ask cluster at
{ask_price:.2f} (potential resistance).")                          return signal
                                                               def fetch_historical_prices(self, timeframe='1m',
limit=100):                                                        """                                                        Fetches historical price data for the symbol.
                                                                   Args:                                                          timeframe (str): Timeframe for historical data
(e.g., '1m', '5m', '1h'). Default '1m'.                                limit (int): Number of historical data points  to fetch. Default 100.
                                                                   Returns:
            pandas.DataFrame: DataFrame with historical    price data, None if fetching fails.
        """                                                        try:                                                           ohlcv = self.exchange.fetch_ohlcv(
                self.symbol, timeframe=timeframe,          limit=limit)
            if ohlcv and len(ohlcv) > 0:                                   df = pd.DataFrame(
                    ohlcv, columns=['timestamp', 'open',   'high', 'low', 'close', 'volume'])                                         df['timestamp'] =
pd.to_datetime(df['timestamp'], unit='ms')                                 logger.debug(
                    f"Fetched {len(df)} historical price   entries for {self.symbol} ({timeframe}).")
                return df                                              else:                                                          logger.warning(
                    f"No historical data fetched for       {self.symbol} ({timeframe}).")
                return None                                        except ccxt.NetworkError as e:
            logger.error(                                                  f"Network error fetching historical prices for {self.symbol} ({timeframe}): {e}")                                 return None
        except ccxt.ExchangeError as e:                                logger.error(                                                  f"Exchange error fetching historical pricesfor {self.symbol} ({timeframe}): {e}")                                 return None
        except Exception as e:                                         logger.error(
                f"Error fetching historical prices for     {self.symbol} ({timeframe}): {e}")                                     return None                                                                                                   def calculate_ema(self, prices, period=None):                  """                                                        Calculates Exponential Moving Average (EMA).                                                                          Args:
            prices (list): List of prices.                             period (int, optional): EMA period. Defaults toself.ema_period.
                                                                   Returns:                                                       float: EMA value, None if calculation fails.
        """                                                        if period is None:                                             period = self.ema_period                               if prices is None or len(prices) < period:                     logger.warning(
                "Not enough price data for EMA calculation or prices are None.")                                                  return None                                            try:                                                           prices_np = np.array(prices)
            weights = np.exp(np.linspace(-1., 0., period))             weights /= weights.sum()                                   ema = np.convolve(prices_np, weights,
mode='valid')[-1]                                                      logger.debug(f"Calculated EMA: {ema:.2f}")                 return ema                                             except Exception as e:                                         logger.error(f"Error calculating EMA: {e}")
            return None                                    
    def calculate_rsi(self, prices, period=None):                  """                                                        Calculates Relative Strength Index (RSI).                                                                             Args:
            prices (list): List of prices.                             period (int, optional): RSI period. Defaults to
self.rsi_period.                                                                                                              Returns:                                                       float: RSI value, None if calculation fails.           """                                                        if period is None:                                             period = self.rsi_period                               if prices is None or len(prices) < period + 1:                 logger.warning(                                                "Not enough price data for RSI calculation
or prices are None.")                                                  return None
        try:                                                           prices_np = np.array(prices)
            deltas = np.diff(prices_np)                                gains = np.maximum(deltas, 0)
            losses = -np.minimum(deltas, 0)                            avg_gain = np.mean(gains[-period:]) if
gains.size > 0 else 0                                                  avg_loss = np.mean(losses[-period:]) if
losses.size > 0 else 0                                                                                                            if avg_loss == 0:
                rsi = 100                                              else:
                rs = avg_gain / avg_loss                                   rsi = 100 - (100 / (1 + rs))
            logger.debug(f"Calculated RSI: {rsi:.2f}")                 return rsi                                             except Exception as e:
            logger.error(f"Error calculating RSI: {e}")                return None                                                                                                   def calculate_macd(self, prices):
        """                                                        Calculates Moving Average Convergence Divergence   (MACD).
                                                                   Args:
            prices (list): List of prices.                 
        Returns:                                                       tuple: MACD, Signal, Histogram values, None if calculation fails.
        """                                                        if prices is None or len(prices) <
self.macd_long_period:                                                 logger.warning(                                                "Not enough price data for MACD calculationor prices are None.")                                                  return None, None, None
        try:                                                           short_ema = self.calculate_ema(
                prices[-self.macd_short_period:],          period=self.macd_short_period)
            long_ema = self.calculate_ema(                                 prices[-self.macd_long_period:],           period=self.macd_long_period)
            if short_ema is None or long_ema is None:                      logger.warning("Could not calculate EMAs
for MACD.")                                                                return None, None, None
            macd = short_ema - long_ema                                signal = self.calculate_ema([macd],            period=self.macd_signal_period)
            if signal is None:                                             logger.warning("Could not calculate MACD
signal line.")                                                             return macd, None, None
            hist = macd - signal                                       logger.debug(                                                  f"MACD: {macd:.2f}, Signal: {signal:.2f},  Histogram: {hist:.2f}")                                                return macd, signal, hist
        except Exception as e:
            logger.error(f"Error calculating MACD: {e}")               return None, None, None                                                                                       def calculate_stoch_rsi(self, df):
        """                                                        Calculates Stochastic RSI.
                                                                   Args:                                                          df (pandas.DataFrame): DataFrame with 'close'
prices.
                                                                   Returns:
            tuple: Stochastic RSI K and D values, None if  calculation fails.                                                 """                                                        period = self.stoch_rsi_period                             smooth_k = self.stoch_rsi_k_period                         smooth_d = self.stoch_rsi_d_period                 
        if df is None or len(df) < period:                             logger.warning(                                                "Not enough DataFrame data for Stochastic  RSI calculation or DataFrame is None.")                                return None, None
        try:                                                           delta = df['close'].diff(1)                                gain =                                         delta.clip(lower=0).rolling(window=period).mean()                      loss =
(-delta.clip(upper=0)).rolling(window=period).mean()       
            rs = gain / loss                                           rsi = 100 - (100 / (1 + rs))
            stoch_rsi = (rsi -                             rsi.rolling(window=period).min()) / \                      
(rsi.rolling(window=period).max() -                                                  rsi.rolling(window=period).min())
            k = stoch_rsi.rolling(window=smooth_k).mean()              d = k.rolling(window=smooth_d).mean()
            k_last, d_last = k.iloc[-1], d.iloc[-1]                    logger.debug(f"Stoch RSI: K = {k_last:.2f}, D ={d_last:.2f}")                                                         return k_last, d_last                                  except Exception as e:
            logger.error(f"Error calculating Stochastic    RSI: {e}")
            return None, None                                                                                             def calculate_vwap(self, df):
        """                                                        Calculates Volume Weighted Average Price (VWAP).
                                                                   Args:                                                          df (pandas.DataFrame): DataFrame with 'high',  'low', 'close', 'volume' columns.                          
        Returns:                                                       float: VWAP value, None if calculation fails.
        """                                                        if df is None or len(df) < self.vwap_period:
            logger.warning("Not enough DataFrame data for  VWAP calculation.")                                                    return None                                            try:                                                           typical_price = (df['high'] + df['low'] +      df['close']) / 3                                                       df['typical_price'] = typical_price
            df['volume_typical_price'] = df['volume'] *    df['typical_price']                                                    cumulative_vtp =
df['volume_typical_price'].rolling(                                        window=self.vwap_period,
min_periods=self.vwap_period).sum()                                    cumulative_volume = df['volume'].rolling(
                window=self.vwap_period,                   min_periods=self.vwap_period).sum()                                    vwap = cumulative_vtp / cumulative_volume
            vwap_last = vwap.iloc[-1]                                  logger.debug(f"Calculated VWAP:
{vwap_last:.2f}")                                                      return vwap_last
        except Exception as e:                                         logger.error(f"Error calculating VWAP: {e}")               return None
                                                               def calculate_bollinger_bands(self, prices):
        """                                                        Calculates Bollinger Bands.
                                                                   Args:                                                          prices (list): List of prices.                                                                                    Returns:
            tuple: Upper, Middle, Lower Bollinger Band     values, None if calculation fails.
        """                                                        if prices is None or len(prices) < self.bb_period:             logger.warning(
                "Not enough price data for Bollinger Bands calculation.")
            return None, None, None                                try:
            prices_series = pd.Series(prices)                          rolling_mean =                                 prices_series.rolling(window=self.bb_period).mean()
            rolling_std =                                  prices_series.rolling(window=self.bb_period).std()
            upper_band = rolling_mean + (rolling_std *     self.bb_std_dev)
            lower_band = rolling_mean - (rolling_std *     self.bb_std_dev)                                                       middle_band = rolling_mean  # Middle band is
just the moving average                                                upper_band_last, middle_band_last,             lower_band_last = upper_band.iloc[                                         -1], middle_band.iloc[-1],
lower_band.iloc[-1]                                                    logger.debug(                                                  f"Bollinger Bands: Upper =
{upper_band_last:.2f}, Middle = {middle_band_last:.2f},    Lower = {lower_band_last:.2f}")
            return upper_band_last, middle_band_last,      lower_band_last
        except Exception as e:                                         logger.error(f"Error calculating Bollinger     Bands: {e}")
            return None, None, None                        
    def calculate_atr(self, df):                                   """
        Calculates Average True Range (ATR).                                                                                  Args:
            df (pandas.DataFrame): DataFrame with 'high',  'low', 'close' columns.

        Returns:                                                       float: ATR value, None if calculation fails.           """                                                        if df is None or len(df) < self.atr_period:                    logger.warning("Not enough DataFrame data for
ATR calculation.")                                                     return None
        try:                                                           high_low = df['high'] - df['low']                          high_close_prev = np.abs(df['high'] -
df['close'].shift(1))                                                  low_close_prev = np.abs(df['low'] -
df['close'].shift(1))                                                  ranges = pd.concat(
                [high_low, high_close_prev,                low_close_prev], axis=1).max(axis=1)                                   atr =
ranges.rolling(window=self.atr_period).mean()                          atr_last = atr.iloc[-1]
            logger.debug(f"Calculated ATR: {atr_last:.4f}")            return atr_last                                        except Exception as e:                                         logger.error(f"Error calculating ATR: {e}")                return None
                                                               def calculate_dynamic_order_size(self, atr):
        """
        Calculates dynamic order size based on ATR.                                                                           Args:                                                          atr (float): Average True Range value.         
        Returns:                                                       float: Order amount, base order size if ATR is
None or calculation fails.                                         """                                                        if atr is None:
            logger.warning("ATR is None, using base order  size percentage.")
            return self.calculate_order_size()                     try:
            atr_factor = atr / self.fetch_market_price() ifself.fetch_market_price() else 0.01                                    dynamic_order_size_percentage = min(
                self.base_order_size_percentage *
self.atr_order_size_multiplier / atr_factor, 0.1)                      dynamic_order_size_percentage = max(
                dynamic_order_size_percentage,             self.base_order_size_percentage * 0.1)                     
            balance =                                      self.exchange.fetch_balance().get('USDT', {}).get('free',
0)                                                                     if balance is None:                                            logger.warning(                                                "Could not retrieve free balance for   dynamic order size calculation.")
                return 0                                   
            order_size_usd = balance *                     dynamic_order_size_percentage
            last_price = self.fetch_market_price()                     if last_price is None:
                logger.warning(                                                "Could not fetch market price for
dynamic order size calculation.")                                          return 0
                                                                       if last_price <= 0:                                            logger.warning(
                    f"Invalid last price: {last_price}.    Cannot calculate dynamic order amount.")
                return 0                                   
            order_amount = order_size_usd / last_price                 logger.debug(                                                  f"Calculated dynamic order amount:
{order_amount:.4f} {self.symbol.replace('/USDT', '')},     using ATR factor.")
            return order_amount                            
        except Exception as e:                                         logger.error(f"Error calculating dynamic order size: {e}")
            return self.calculate_order_size()  # Fallback to base order size
                                                               def calculate_order_size(self):
        """                                                        Calculates order size based on base order size     percentage of balance.
                                                                   Returns:                                                       float: Order amount.                                   """
        try:                                                           balance =                                      self.exchange.fetch_balance().get('USDT', {}).get('free',
0)                                                                     if balance is None:
                logger.warning("Could not retrieve free    balance from account.")
                return 0                                           except ccxt.NetworkError as e:                                 logger.error(f"Network error fetching balance:
{e}")                                                                  return 0
        except ccxt.ExchangeError as e:                                logger.error(f"Exchange error fetching balance:
{e}")                                                                  return 0                                               except Exception as e:
            logger.error(f"Error fetching balance: {e}")               return 0

        order_size_usd = balance *                         self.base_order_size_percentage                                    last_price = self.fetch_market_price()                     if last_price is None:                                         logger.warning(                                                "Could not fetch market price to calculate order amount.")                                                        return 0                                                                                                          if last_price <= 0:
            logger.warning(                                                f"Invalid last price: {last_price}. Cannot
calculate order amount.")                                              return 0
                                                                   order_amount = order_size_usd / last_price                 logger.debug(
            f"Calculated order amount: {order_amount:.4f}  {self.symbol.replace('/USDT', '')} based on balance and
price.")                                                           return order_amount
                                                               def make_decision(self, df, current_price, order_book, imbalance_ratio, ob_delta, ob_delta_change, spread,
spread_change, bid_clusters, ask_clusters, ema, rsi, macd,
signal_macd, hist_macd, stoch_k, stoch_d, vwap, bb_upper,  bb_lower):
        """                                                        Analyzes market conditions and technical indicatorsto make a trading decision.
                                                                   Args:                                                          df (pandas.DataFrame): DataFrame of historical prices.
            current_price (float): Current market price.               order_book (dict): Current order book data.                imbalance_ratio (float): Order book imbalance
ratio.                                                                 ob_delta (float): Order book delta.
            ob_delta_change (float): Change in order book  delta.                                                                 spread (float): Current spread.                            spread_change (float): Change in spread.                   bid_clusters (list): List of bid volume clusterprices.
            ask_clusters (list): List of ask volume clusterprices.                                                                ema (float): EMA value.                                    rsi (float): RSI value.                                    macd (float): MACD value.
            signal_macd (float): MACD signal line value.               hist_macd (float): MACD histogram value.                   stoch_k (float): Stochastic RSI K value.
            stoch_d (float): Stochastic RSI D value.                   vwap (float): VWAP value.                                  bb_upper (float): Upper Bollinger Band value.
            bb_lower (float): Lower Bollinger Band value.                                                                     Returns:                                                       str: 'BUY', 'SELL', or 'WAIT' based on         analysis.                                                          """                                                        if df is None:
            logger.warning("No historical data to make     decision.")                                                            return 'WAIT'                                  
        if any(indicator is None for indicator in [ema,    rsi, macd, signal_macd, stoch_k, stoch_d, vwap, bb_upper,  bb_lower]):
            logger.warning(                                                "One or more technical indicators could notbe calculated. Waiting for reliable data.")                            return 'WAIT'
                                                                   weighted_signal = 0                                
        # --- EMA Signal ---                                       if current_price > ema:                                        weighted_signal += self.ema_weight * 0.5  #    Bullish signal                                                         logger.debug(f"EMA bullish signal, weight:     {self.ema_weight * 0.5}")                                          elif current_price < ema:
            weighted_signal -= self.ema_weight * 0.5  #    Bearish signal
            logger.debug(f"EMA bearish signal, weight:     {self.ema_weight * 0.5}")
                                                                   # --- RSI Signal ---                                       if rsi < 30:                                                   weighted_signal += self.rsi_weight * 0.6  #    Oversold                                                               logger.debug(f"RSI oversold signal, weight:    {self.rsi_weight * 0.6}")                                          elif rsi > 70:                                                 weighted_signal -= self.rsi_weight * 0.6  #    Overbought                                                             logger.debug(f"RSI overbought signal, weight:  {self.rsi_weight * 0.6}")                                                                                                     # --- MACD Signal ---                                      if macd > signal_macd and hist_macd > 0:                       weighted_signal += self.macd_weight * 0.7  #   MACD bullish                                                           logger.debug(f"MACD bullish signal, weight:
{self.macd_weight * 0.7}")                                         elif macd < signal_macd and hist_macd < 0:
            weighted_signal -= self.macd_weight * 0.7  #   MACD bearish                                                           logger.debug(f"MACD bearish signal, weight:    {self.macd_weight * 0.7}")                                                                                                    # --- Stochastic RSI Signal ---
        if stoch_k < 20 and stoch_d < 20 and stoch_k <     stoch_d:                                                               weighted_signal += self.stoch_rsi_weight * 0.5
# Stoch RSI oversold and crossing up
            logger.debug(f"Stoch RSI oversold bullish      signal, weight: {self.stoch_rsi_weight * 0.5}")
        elif stoch_k > 80 and stoch_d > 80 and stoch_k >   stoch_d:                                                               weighted_signal -= self.stoch_rsi_weight * 0.5
# Stoch RSI overbought and crossing down                               logger.debug(f"Stoch RSI overbought bearish    signal, weight: {self.stoch_rsi_weight * 0.5}")                                                                               # --- Order Book Imbalance Signal ---                      if imbalance_ratio > self.imbalance_threshold:                 weighted_signal -= self.imbalance_weight * 0.4 # Bearish imbalance                                                    logger.debug(f"Order book imbalance bearish
signal, weight: {self.imbalance_weight * 0.4}")                    elif imbalance_ratio < (1 /
self.imbalance_threshold) and imbalance_ratio !=           float('inf') and imbalance_ratio != 0:                                 weighted_signal += self.imbalance_weight * 0.4
# Bullish imbalance                                                    logger.debug(f"Order book imbalance bullish
signal, weight: {self.imbalance_weight * 0.4}")            
        # --- Order Book Delta Change Signal ---                   if ob_delta_change is not None and ob_delta_change > 0:                                                                   weighted_signal += self.ob_delta_change_weight * 0.3  # Increasing buying pressure
            logger.debug(f"OB Delta increasing bullish     signal, weight: {self.ob_delta_change_weight * 0.3}")
        elif ob_delta_change is not None and               ob_delta_change < 0:                                                   weighted_signal -= self.ob_delta_change_weight * 0.3 # Increasing selling pressure                                    logger.debug(f"OB Delta increasing bearish
signal, weight: {self.ob_delta_change_weight * 0.3}")      
        # --- Spread Change Signal ---                             if spread is not None and spread > self.last_spreadif self.last_spread is not None else False:  # Spread      widening is bearish                                                    weighted_signal -= self.spread_weight * 0.2 #
Bearish spread widening                                                logger.debug(f"Spread widening bearish signal, weight: {self.spread_weight * 0.2}")                               elif spread is not None and spread <               self.last_spread if self.last_spread is not None else
False: # Spread tightening is bullish                                  weighted_signal += self.spread_weight * 0.2 #
Bullish spread tightening                                              logger.debug(f"Spread tightening bullish
signal, weight: {self.spread_weight * 0.2}")
                                                                   # --- Cluster Proximity Signal ---
        weighted_signal += self.cluster_proximity_weight *
cluster_proximity_signal                                           logger.debug(f"Cluster proximity signal, weight:
{self.cluster_proximity_weight * cluster_proximity_signal},
signal value: {cluster_proximity_signal}")

        # --- VWAP Signal ---
        if current_price > vwap:
            weighted_signal += self.vwap_weight * 0.3  #
Price above VWAP bullish
            logger.debug(f"Price above VWAP bullish signal,weight: {self.vwap_weight * 0.3}")
        elif current_price < vwap:
            weighted_signal -= self.vwap_weight * 0.3 #
Price below VWAP bearish
            logger.debug(f"Price below VWAP bearish signal,
weight: {self.vwap_weight * 0.3}")

        # --- Bollinger Bands Signal ---
        if current_price > bb_upper:
            weighted_signal -= self.bb_weight * 0.4 # Price
above upper BB overbought
            logger.debug(f"Price above upper Bollinger Band
bearish signal, weight: {self.bb_weight * 0.4}")
        elif current_price < bb_lower:
            weighted_signal += self.bb_weight * 0.4 # Pricebelow lower BB oversold
            logger.debug(f"Price below lower Bollinger Band
bullish signal, weight: {self.bb_weight * 0.4}")

        logger.info(f"Composite weighted signal:
{weighted_signal:.2f}")

        if weighted_signal > 1.5:  # Increased threshold
for BUY signal
            return 'BUY'
        elif weighted_signal < -1.5: # Increased threshold
for SELL signal
            return 'SELL'
        else:
            return 'WAIT'
                                                               def execute_trade(self, action, price):                        """                                                        Executes a trade based on the trading action and   current price.
        Handles both simulation and real trading modes.

        Args:
            action (str): 'BUY', 'SELL', or 'WAIT'.
            price (float): Current market price.
        """
        if self.simulation_mode:                                       logger.info(
                f"{NEON_YELLOW}[SIMULATION MODE] - {action}
signal at price: {price:.2f}{RESET_COLOR}")
            if action == 'BUY' and self.position != 'LONG':
                amount =
self.calculate_dynamic_order_size(self.calculate_atr(self.f
etch_historical_prices()))
                if amount > 0:
                    logger.info(
                        f"{NEON_YELLOW}[SIMULATION MODE] -
Placing {action} order for {amount:.4f}
{self.symbol.replace('/USDT', '')}{RESET_COLOR}")
                    self.position = 'LONG'
                    self.entry_price = price
                    self.order_amount = amount                                 self.high_since_entry = price  # Reset
high for trailing stop
                    self.low_since_entry = price  # Reset
low for trailing stop                                                          self.trade_count += 1
                    logger.info(
                        f"{NEON_YELLOW}[SIMULATION MODE] -
Position updated to {self.position}, Entry Price:
{self.entry_price:.2f}, Amount: {self.order_amount:.4f},
Trade Count: {self.trade_count}{RESET_COLOR}")
                else:
                    logger.warning(
                        f"Calculated order amount is zero
or negative, skipping {action} order.")
            elif action == 'SELL' and self.position ==
'LONG':  # Closing long position.
                amount = self.order_amount if
self.order_amount else
self.calculate_dynamic_order_size(self.calculate_atr(self.f
etch_historical_prices()))                                                 if amount > 0:                                                 logger.info(
                        f"{NEON_YELLOW}[SIMULATION MODE] - Placing {action} order to close LONG position for
{amount:.4f} {self.symbol.replace('/USDT', '')}            {RESET_COLOR}")
                    self.position = 'FLAT'  # or None                          self.entry_price = None                                    self.order_amount = None
                    profit_loss_pct = (price -             self.entry_price) / self.entry_price if self.entry_price
else 0                                                                         logger.info(
                        f"{NEON_YELLOW}[SIMULATION MODE] - Closed LONG position with Profit/Loss:                     {profit_loss_pct:.2%}{RESET_COLOR}")
                else:                                                          logger.warning(
                        f"Calculated close order amount is zero or negative, skipping {action} order.")
                                                                       elif action == 'WAIT':                                         logger.info(
                    f"{NEON_CYAN}[SIMULATION MODE] -       Waiting for a trading signal...{RESET_COLOR}")
        else:  # Real trading logic (place actual orders)              if action == 'BUY' and self.position != 'LONG':
                amount =                                   self.calculate_dynamic_order_size(self.calculate_atr(self.fetch_historical_prices()))
                if amount > 0:                                                 try:
                        order =                            self.exchange.create_market_order(self.symbol, 'buy',
amount)                                                                            logger.info(                                                   f"{NEON_GREEN}BUY order placed
successfully: {order}{RESET_COLOR}")                                               self.position = 'LONG'
                        self.entry_price = price                                   self.order_amount = amount
                        self.high_since_entry = price  #   Reset high for trailing stop                                                       self.low_since_entry = price  #
Reset low for trailing stop                                                        self.trade_count += 1
                        logger.info(                                                   f"{NEON_GREEN}Position updated
to {self.position}, Entry Price: {self.entry_price:.2f},   Amount: {self.order_amount:.4f}, Trade Count:              {self.trade_count}{RESET_COLOR}")
                    except Exception as e:                                         logger.error(f"{NEON_RED}Error
placing BUY order: {e}{RESET_COLOR}")                                      else:
                    logger.warning(f"Calculated order      amount is zero or negative, skipping BUY order.")          
            elif action == 'SELL' and self.position ==     'LONG':  # Closing long position
                amount = self.order_amount if              self.order_amount else
self.calculate_dynamic_order_size(self.calculate_atr(self.fetch_historical_prices()))                                                 if amount > 0:
                    try:                                                           order =
self.exchange.create_market_order(self.symbol, 'sell',     amount)
                        logger.info(                                                   f"{NEON_GREEN}SELL order placedto close LONG position successfully: {order}{RESET_COLOR}")
                        self.position = 'FLAT'  # or None                          self.entry_price = None
                        self.order_amount = None                                   profit_loss_pct = (price -
self.entry_price) / self.entry_price if self.entry_price   else 0                                                                             logger.info(
                            f"{NEON_GREEN}Closed LONG      position with Profit/Loss: {profit_loss_pct:.2%}
{RESET_COLOR}")                                                                except Exception as e:
                        logger.error(                                                  f"{NEON_RED}Error placing SELL order to close LONG position: {e}{RESET_COLOR}")
                else:                                                          logger.warning(
                        f"Calculated close order amount is zero or negative, skipping SELL order.")
                                                                       elif action == 'WAIT':                                         logger.info(
                    f"{NEON_CYAN}Waiting for a trading     signal...{RESET_COLOR}")
                                                               def check_open_orders(self):
        """                                                        Checks open positions for take profit, stop loss,  and trailing stop loss conditions.
                                                                   Returns:
            str: 'SELL' if TP/SL is hit, 'WAIT' otherwise.         """
        try:                                                           if self.position == 'LONG' and self.entry_priceis not None:
                current_price = self.fetch_market_price()                  if current_price is None:
                    logger.warning("Could not fetch currentprice to check open orders.")
                    return 'WAIT'                                                                                                     # Take Profit Check
                take_profit_price = self.entry_price * (1 +self.take_profit_pct)
                if current_price >= take_profit_price:                         logger.info(
                        f"Take Profit condition met.       Current price: {current_price:.2f}, Take Profit Price:     {take_profit_price:.2f}")
                    return 'SELL'  # Signal to close       position
                                                                           # Stop Loss Check
                stop_loss_price = self.entry_price * (1 -  self.stop_loss_pct)                                                        if current_price <= stop_loss_price:
                    logger.info(                                                   f"Stop Loss condition met. Current
price: {current_price:.2f}, Stop Loss Price:               {stop_loss_price:.2f}")
                    return 'SELL'  # Signal to close       position                                                                                                                              # Trailing Stop Loss (if active)                           if self.trailing_stop_loss_active:
                    self.high_since_entry =                max(self.high_since_entry, current_price)  # Update high
                    trailing_stop_loss_price =             self.high_since_entry * (1 - self.trailing_stop_callback)                      if current_price <=
trailing_stop_loss_price:                                                          logger.info(
                            f"Trailing Stop Loss triggered.Current price: {current_price:.2f}, Trailing Stop Price:
{trailing_stop_loss_price:.2f}, High since entry:          {self.high_since_entry:.2f}")                                                      return 'SELL'  # Signal to close
position                                                   
                return 'WAIT'  # No action needed based on TP/SL
        except Exception as e:                                         logger.error(f"Error checking open orders and  TP/SL: {e}")
            return 'WAIT'  # Default to wait in case of    errors
                                                               def run_bot(self):
        """                                                        Main loop for running the trading bot. Fetches     data, makes decisions, and executes trades.
        """                                                        logger.info(f"{NEON_MAGENTA}------- Starting
Trading Bot for {self.symbol} -------{RESET_COLOR}")               while self.bot_running_flag:
            try:                                                           current_price = self.fetch_market_price()                  if current_price is None:
                    logger.warning("Could not fetch currentprice. Retrying in 1 minute...")
                    time.sleep(60)  # Wait for 60 seconds  before retrying
                    continue                                                                                                          df = self.fetch_historical_prices()
                order_book, imbalance_ratio, ob_delta,     ob_delta_change, spread, spread_change, bid_clusters,
ask_clusters = self.fetch_order_book()                     
                if order_book is None or imbalance_ratio isNone:                                                                          logger.warning("Could not fetch order
book data. Retrying in 1 minute...")                                           time.sleep(60)
                    continue                               
                # Calculate Indicators - Fetch data and    calculate indicators once per loop                                         prices = df['close'].tolist() if df is not
None and not df.empty else None # Ensure prices is not Noneif df is empty
                ema = self.calculate_ema(prices) if prices else None
                rsi = self.calculate_rsi(prices) if prices else None                                                                  macd, signal_macd, hist_macd =
self.calculate_macd(prices) if prices else (None, None,    None)
                stoch_k, stoch_d =                         self.calculate_stoch_rsi(df) if df is not None and not
df.empty else (None, None)                                                 vwap = self.calculate_vwap(df) if df is notNone and not df.empty else None
                bb_upper, bb_middle, bb_lower =            self.calculate_bollinger_bands(prices) if prices else
(None, None, None)                                                         atr = self.calculate_atr(df) if df is not
None and not df.empty else None                                                                                                       cluster_proximity_signal =
self.get_cluster_proximity_signal(                                             current_price, bid_clusters,
ask_clusters)                                              
                if self.position == 'LONG':                                    tp_sl_signal = self.check_open_orders()                    if tp_sl_signal == 'SELL':
                        action = 'SELL'                                        else:
                        action = self.make_decision(df,    current_price, order_book, imbalance_ratio, ob_delta,      ob_delta_change, spread, spread_change, bid_clusters,      ask_clusters, ema, rsi, macd, signal_macd, hist_macd,      stoch_k, stoch_d, vwap, bb_upper, bb_lower)                                else:  # position is None or 'FLAT' (not ina trade)                                                                       action = self.make_decision(df,        current_price, order_book, imbalance_ratio, ob_delta,      ob_delta_change, spread, spread_change, bid_clusters,      ask_clusters, ema, rsi, macd, signal_macd, hist_macd,      stoch_k, stoch_d, vwap, bb_upper, bb_lower)
                                                                           self.execute_trade(action, current_price)
                                                                           time.sleep(30)  # Check every 30 seconds
            except ccxt.NetworkError as e:                                 logger.error(                                                  f"Network error during bot operation:
{e}. Retrying in 60 seconds...")                                           time.sleep(60)
            except ccxt.ExchangeError as e:                                logger.error(
                    f"Exchange error during bot operation: {e}. Retrying in 60 seconds...")                                           time.sleep(60)
            except Exception as e:                                         logger.critical(
                    f"Unexpected error during bot          operation: {e}. Bot will continue to run, but check logs.")
                logger.exception(e)  # Log detailed        exception info                                                             time.sleep(60)  # Wait and retry
                                                           
if __name__ == '__main__':                                     symbol = os.getenv('TRADING_SYMBOL')
    if not symbol:                                                 logger.error("TRADING_SYMBOL environment variable  is not set.")
        exit(1)                                                try:
        bot = EnhancedTradingBot(symbol)                           bot.run_bot()
    except ValueError as ve:                                       logger.error(f"Configuration error: {ve}")                 exit(1)
    except Exception as e:                                         logger.critical(f"Bot failed to start: {e}")
        exit(1)
