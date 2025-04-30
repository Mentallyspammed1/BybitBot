#!/usr/bin/env node

const dotenv = require('dotenv');
dotenv.config();

const axios = require('axios');
const { SMA, StochasticRSI, ATR, MACD } = require('technicalindicators');
const chalk = require('chalk');
const crypto = require('crypto');

// Logger with chalk
const logger = {
  info: (msg) => console.log(chalk.cyan(`[${new Date().toISOString()}] [INFO] ${msg}`)),
  debug: (msg) => {
    if (process.env.LOG_LEVEL === 'DEBUG') {
      console.log(chalk.yellow(`[${new Date().toISOString()}] [DEBUG] ${msg}`));
    }
  },
  error: (msg) => console.error(chalk.red(`[${new Date().toISOString()}] [ERROR] ${msg}`)),
};

// Cache with TTL
class Cache {
  constructor() {
    this.data = new Map();
  }

  get(key) {
    const entry = this.data.get(key);
    if (!entry || Date.now() - entry.timestamp > entry.ttl) return null;
    return entry.value;
  }

  set(key, value, ttl) {
    this.data.set(key, { value, timestamp: Date.now(), ttl });
  }
}

// Market Data Provider
class MarketDataProvider {
  constructor(config) {
    this.config = config;
    this.logger = logger;
    this.apiKey = process.env.BYBIT_API_KEY;
    this.apiSecret = process.env.BYBIT_API_SECRET;
    this.baseUrl = process.env.BYBIT_API_BASE_URL || 'https://api.bybit.com';
    this.cache = new Cache();
  }

  async fetchOHLCV(symbol, timeframe, limit, retries = 0) {
    const cacheKey = `${symbol}-${timeframe}-${limit}`;
    const cached = this.cache.get(cacheKey);
    if (cached) return cached;

    const params = {
      category: 'linear',
      symbol: symbol.replace('/', ''),
      interval: timeframe,
      limit,
    };
    try {
      const response = await axios.get(`${this.baseUrl}/v5/market/kline`, { params });
      const data = response.data.result.list;
      if (data.length < limit * 0.9) throw new Error('Insufficient OHLCV data');
      const reversed = data.reverse().map(d => [parseInt(d[0]), parseFloat(d[1]), parseFloat(d[2]), parseFloat(d[3]), parseFloat(d[4]), parseFloat(d[5])]);
      this.cache.set(cacheKey, reversed, this.config.cache_ttl);
      this.logger.debug(`Fetched OHLCV for ${symbol}-${timeframe}`);
      return reversed;
    } catch (error) {
      return this.handleRetry(error, retries, `OHLCV fetch for ${symbol}`, () => this.fetchOHLCV(symbol, timeframe, limit, retries + 1));
    }
  }

  async fetchOrderBook(symbol, retries = 0) {
    const cached = this.cache.get(symbol);
    if (cached) return cached;

    const params = { category: 'linear', symbol: symbol.replace('/', '') };
    try {
      const response = await axios.get(`${this.baseUrl}/v5/market/orderbook`, { params });
      const data = response.data.result;
      const orderBook = {
        bids: data.b.map(([price, volume]) => [parseFloat(price), parseFloat(volume)]),
        asks: data.a.map(([price, volume]) => [parseFloat(price), parseFloat(volume)]),
      };
      this.cache.set(symbol, orderBook, this.config.cache_ttl);
      this.logger.debug(`Fetched order book for ${symbol}`);
      return orderBook;
    } catch (error) {
      return this.handleRetry(error, retries, `Order book fetch for ${symbol}`, () => this.fetchOrderBook(symbol, retries + 1));
    }
  }

  async fetchBalance(retries = 0) {
    const params = {
      api_key: this.apiKey,
      timestamp: Date.now(),
      recv_window: 5000,
    };
    params.sign = this.generateSignature(params);
    try {
      const response = await axios.get(`${this.baseUrl}/v5/account/wallet-balance`, {
        headers: { 'X-BAPI-SIGN': params.sign, 'X-BAPI-API-KEY': this.apiKey, 'X-BAPI-TIMESTAMP': params.timestamp, 'X-BAPI-RECV-WINDOW': params.recv_window },
        params: { accountType: 'UNIFIED' },
      });
      const total = response.data.result.list[0]?.totalEquity || 0;
      return { USDT: parseFloat(total) };
    } catch (error) {
      return this.handleRetry(error, retries, 'Balance fetch', () => this.fetchBalance(retries + 1));
    }
  }

  generateSignature(params) {
    const queryString = Object.keys(params).sort().map(k => `${k}=${params[k]}`).join('&');
    return crypto.createHmac('sha256', this.apiSecret).update(queryString).digest('hex');
  }

  async handleRetry(error, retries, action, retryFn) {
    if (retries < this.config.max_retries) {
      const delay = this.config.retry_delay * (2 ** retries);
      this.logger.debug(`Retrying ${action} (${retries + 1}/${this.config.max_retries}) after ${delay}ms`);
      await new Promise(resolve => setTimeout(resolve, delay));
      return retryFn();
    } else {
      throw new Error(`Failed to ${action}: ${error.message}`);
    }
  }
}

// Indicator Engine
class IndicatorEngine {
  constructor(config, marketData) {
    this.config = config;
    this.marketData = marketData;
    this.logger = logger;
  }

  async calculateAll(candles, orderBook) {
    if (!candles || candles.length < this.config.history_candles) {
      this.logger.error('Insufficient candle data');
      return null;
    }

    const closes = candles.map(c => c[4]);
    const highs = candles.map(c => c[2]);
    const lows = candles.map(c => c[3]);
    const volumes = candles.map(c => c[5]);

    const fibPivots = this.config.fibonacci_pivots.enabled ? await this.calculateFibonacciPivots() : null;
    const pivotOrderBook = fibPivots ? this.analyzeOrderBookForPivots(orderBook, fibPivots) : {};

    return {
      price: closes[closes.length - 1],
      volume: volumes[volumes.length - 1],
      average_volume: volumes.slice(-this.config.volume_confirmation.lookback).reduce((a, b) => a + b, 0) / this.config.volume_confirmation.lookback,
      sma: {
        fast: SMA.calculate({ period: this.config.indicators.sma.fast, values: closes }).slice(-1)[0],
        slow: SMA.calculate({ period: this.config.indicators.sma.slow, values: closes }).slice(-1)[0],
      },
      stoch_rsi: StochasticRSI.calculate(this.config.indicators.stoch_rsi.merge({ values: closes })).slice(-1)[0],
      atr: ATR.calculate({ high: highs, low: lows, close: closes, period: this.config.indicators.atr.period }).slice(-1)[0],
      macd: MACD.calculate(this.config.indicators.macd.merge({ values: closes })).slice(-1)[0],
      fibonacci_pivots: fibPivots,
      pivot_order_book_analysis: pivotOrderBook,
    };
  }

  async calculateFibonacciPivots() {
    const dailyCandles = await this.marketData.fetchOHLCV(this.config.symbol, this.config.fibonacci_pivots.period, 2);
    if (!dailyCandles || dailyCandles.length < 2) return null;

    const [_, prevDay] = dailyCandles;
    const high = prevDay[2], low = prevDay[3], close = prevDay[4];
    const pivot = (high + low + close) / 3;
    return {
      pivot,
      r1: 2 * pivot - low,
      s1: 2 * pivot - high,
      r2: pivot + (high - low),
      s2: pivot - (high - low),
      r3: high + 2 * (pivot - low),
      s3: low - 2 * (high - pivot),
    };
  }

  analyzeOrderBookForPivots(orderBook, pivotLevels) {
    if (!orderBook || !pivotLevels) return {};
    const rangePercentage = this.config.fibonacci_pivots.order_book_range_percentage;
    const depthScores = {};

    ['pivot', 's1', 's2', 'r1', 'r2'].forEach(level => {
      const pivotLevel = pivotLevels[level];
      if (!pivotLevel) return;

      const bidDepth = orderBook.bids.reduce((sum, [price, vol]) => {
        return price >= pivotLevel * (1 - rangePercentage) && price <= pivotLevel * (1 + rangePercentage) ? sum + vol : sum;
      }, 0);
      const askDepth = orderBook.asks.reduce((sum, [price, vol]) => {
        return price >= pivotLevel * (1 - rangePercentage) && price <= pivotLevel * (1 + rangePercentage) ? sum + vol : sum;
      }, 0);
      depthScores[level] = { bid_depth: bidDepth, ask_depth: askDepth };
    });
    return depthScores;
  }
}

// Signal Generator
class SignalGenerator {
  constructor(config) {
    this.config = config;
    this.logger = logger;
  }

  generate(indicatorData) {
    if (!indicatorData) {
      this.logger.debug('No indicator data provided, returning empty signals');
      return [];
    }

    const { sma, stoch_rsi, macd, volume, average_volume: avgVol, price, fibonacci_pivots: fibPivots, pivot_order_book_analysis: pivotOb } = indicatorData;

    const trendUp = sma.fast > sma.slow;
    const signals = [];
    let confidenceBoost = 0;
    const reasons = [];

    if (this.config.volume_confirmation.enabled && volume > avgVol) {
      confidenceBoost += 5;
      reasons.push('Volume Confirmed');
    }

    const macdBullish = macd.MACD > macd.signal && macd.histogram > 0;
    const macdBearish = macd.MACD < macd.signal && macd.histogram < 0;

    if (this.config.fibonacci_pivots.enabled && fibPivots) {
      const [pivotBoost, orderBookBoost, pivotReason] = this.analyzePivots(price, fibPivots, pivotOb);
      confidenceBoost += pivotBoost + orderBookBoost;
      if (pivotReason) reasons.push(pivotReason);
    }

    if (stoch_rsi.k < this.config.thresholds.oversold && stoch_rsi.d < this.config.thresholds.oversold && trendUp) {
      let confidence = 75 + (trendUp ? 10 : 0) + confidenceBoost + (macdBullish ? 10 : 0);
      if (macdBullish) reasons.push('Bullish MACD');
      confidence = Math.max(50, confidence);
      signals.push({
        type: 'BUY',
        reason: `Oversold Momentum with Uptrend & Confirmations, ${reasons.join(', ')}`,
        confidence,
        timestamp: Math.floor(Date.now() / 1000),
      });
    }

    if (stoch_rsi.k > this.config.thresholds.overbought && stoch_rsi.d > this.config.thresholds.overbought && !trendUp) {
      let confidence = 75 + (!trendUp ? 10 : 0) + confidenceBoost + (macdBearish ? 10 : 0);
      if (macdBearish) reasons.push('Bearish MACD');
      confidence = Math.max(50, confidence);
      signals.push({
        type: 'SELL',
        reason: `Overbought Momentum with Downtrend & Confirmations, ${reasons.join(', ')}`,
        confidence,
        timestamp: Math.floor(Date.now() / 1000),
      });
    }

    return signals.filter(signal => signal.confidence >= this.config.thresholds.min_confidence);
  }

  analyzePivots(price, fibPivots, pivotOb) {
    const proximity = this.config.fibonacci_pivots.proximity_percentage;
    const buyLevels = this.config.fibonacci_pivots.levels_for_buy_entry;
    const sellLevels = this.config.fibonacci_pivots.levels_for_sell_entry;
    let pivotBoost = 0;
    let orderBookBoost = 0;
    let reason = '';

    for (const isBuy of [true, false]) {
      const levels = isBuy ? buyLevels : sellLevels;
      for (const level of levels) {
        const pivotLevel = fibPivots[level.toLowerCase()];
        if (!pivotLevel || Math.abs(price - pivotLevel) / price > proximity) continue;

        const levelBoost = this.config.fibonacci_pivots.pivot_weight;
        const obBoost = pivotOb?.[level.toLowerCase()] ? 
          ((isBuy ? pivotOb[level.toLowerCase()].bid_depth : pivotOb[level.toLowerCase()].ask_depth) * 
            this.config.fibonacci_pivots.order_book_weight / 1000) : 0;

        pivotBoost += levelBoost;
        orderBookBoost += obBoost;
        reason = `${level.toUpperCase()} Pivot Level Proximity`;
        break;
      }
      if (reason) break;
    }

    return [pivotBoost, orderBookBoost, reason];
  }
}

// Trading Bot
class TradingBot {
  constructor(config) {
    this.config = config;
    this.logger = logger;
    this.marketData = new MarketDataProvider(config);
    this.indicatorEngine = new IndicatorEngine(config, this.marketData);
    this.signalGenerator = new SignalGenerator(config);
    this.tradeContext = { current_position: null, entry_price: null, stop_loss: null, take_profit: null, position_size: null, entry_time: null };
    this.tradeHistory = [];
    this.running = false;
    this.listeners = new Map();
  }

  on(event, callback) {
    this.listeners.set(event, callback);
  }

  emit(event, data) {
    const callback = this.listeners.get(event);
    if (callback) callback(data);
  }

  async initialize() {
    const balance = await this.marketData.fetchBalance();
    if (!balance.USDT) throw new Error('API health check failed');
    this.config.position_size = this.config.max_position_size_usdt / (await this.marketData.fetchOrderBook(this.config.symbol)).bids[0][0];
    this.logger.info(`Bot initialized with position size: ${this.config.position_size}`);
    this.emit('initialized');
  }

  async start() {
    this.logger.info(`Starting bot for ${this.config.symbol} on ${this.config.timeframe}`);
    this.running = true;
    await this.initialize();
    while (this.running) {
      await this.analysisCycle();
      await new Promise(resolve => setTimeout(resolve, this.config.analysis_interval));
    }
  }

  async analysisCycle() {
    try {
      const candles = await this.marketData.fetchOHLCV(this.config.symbol, this.config.timeframe, this.config.history_candles);
      const orderBook = await this.marketData.fetchOrderBook(this.config.symbol);
      const indicators = await this.indicatorEngine.calculateAll(candles, orderBook);
      if (!indicators) return;

      this.checkPosition(indicators.price);
      const signals = this.signalGenerator.generate(indicators);
      this.emit('signals', signals);
      await this.executeStrategy(signals, indicators, orderBook);
    } catch (error) {
      this.logger.error(`Analysis cycle error: ${error.message}`);
    }
  }

  checkPosition(price) {
    if (!this.tradeContext.current_position) return;
    if (price <= this.tradeContext.stop_loss) {
      this.closePosition('Stop Loss Triggered');
    } else if (price >= this.tradeContext.take_profit) {
      this.closePosition('Take Profit Triggered');
    }
  }

  async executeStrategy(signals, indicators, orderBook) {
    if (!signals.length) return;
    const signal = signals[0];
    this.logger.info(`Primary Signal: ${signal.type} (${signal.reason}, ${signal.confidence}%)`);

    const balance = await this.marketData.fetchBalance();
    const usdtAvailable = balance.USDT || 0;
    const entryPrice = orderBook.bids[0][0] || indicators.price;
    let positionSize = Math.min(this.config.position_size, (usdtAvailable * this.config.risk_percentage) / entryPrice);

    if (signal.type === 'BUY' && !this.tradeContext.current_position) {
      positionSize = Math.min(positionSize, (usdtAvailable * 0.9) / entryPrice);
      await this.placeOrder('buy', positionSize, entryPrice, indicators.atr, signal.reason);
    } else if (signal.type === 'SELL' && this.tradeContext.current_position) {
      await this.closePosition('Sell Signal Triggered');
    }
  }

  async placeOrder(side, quantity, price, atr, reason, retries = 0) {
    const slTp = this.calculateStopLossTakeProfit(side, price, atr);
    const params = {
      category: 'linear',
      symbol: this.config.symbol.replace('/', ''),
      side: side.charAt(0).toUpperCase() + side.slice(1),
      orderType: 'Limit',
      qty: quantity.toString(),
      price: price.toString(),
      api_key: this.marketData.apiKey,
      timestamp: Date.now(),
      recv_window: 5000,
    };
    params.sign = this.marketData.generateSignature(params);

    try {
      const response = await axios.post(`${this.marketData.baseUrl}/v5/order/create`, params, {
        headers: { 'X-BAPI-SIGN': params.sign, 'X-BAPI-API-KEY': this.marketData.apiKey, 'X-BAPI-TIMESTAMP': params.timestamp, 'X-BAPI-RECV-WINDOW': params.recv_window },
      });
      this.logger.info(`Order placed: ${side.toUpperCase()} ${quantity} ${this.config.symbol} at ${price}, SL: ${slTp.stop_loss}, TP: ${slTp.take_profit}`);

      if (side === 'buy') {
        this.tradeContext = {
          current_position: 'BUY',
          entry_price: price,
          stop_loss: slTp.stop_loss,
          take_profit: slTp.take_profit,
          position_size: quantity,
          entry_time: Date.now(),
        };
        this.tradeHistory.push({ ...this.tradeContext, status: 'OPEN', entry_reason: reason });
      }
    } catch (error) {
      this.logger.error(`Order placement failed: ${error.message}`);
      await this.retryOrder(side, quantity, price, atr, reason, retries, error);
    }
  }

  calculateStopLossTakeProfit(side, price, atr) {
    const offset = price * this.config.sl_tp_offset_percentage;
    return side === 'buy' ? {
      stop_loss: price - (atr * this.config.stop_loss_multiplier) - offset,
      take_profit: price + (atr * this.config.take_profit_multiplier) + offset,
    } : {
      stop_loss: price + (atr * this.config.stop_loss_multiplier) + offset,
      take_profit: price - (atr * this.config.take_profit_multiplier) - offset,
    };
  }

  async retryOrder(side, quantity, price, atr, reason, retries, error) {
    if (retries < this.config.max_retries) {
      const delay = this.config.retry_delay * (2 ** retries);
      this.logger.debug(`Retrying order (${retries + 1}/${this.config.max_retries}) after ${delay}ms`);
      await new Promise(resolve => setTimeout(resolve, delay));
      await this.placeOrder(side, quantity, price, atr, reason, retries + 1);
    } else {
      throw error;
    }
  }

  async closePosition(reason) {
    if (!this.tradeContext.current_position) return;
    const params = {
      category: 'linear',
      symbol: this.config.symbol.replace('/', ''),
      side: 'Sell',
      orderType: 'Market',
      qty: this.tradeContext.position_size.toString(),
      api_key: this.marketData.apiKey,
      timestamp: Date.now(),
      recv_window: 5000,
    };
    params.sign = this.marketData.generateSignature(params);

    try {
      const response = await axios.post(`${this.marketData.baseUrl}/v5/order/create`, params, {
        headers: { 'X-BAPI-SIGN': params.sign, 'X-BAPI-API-KEY': this.marketData.apiKey, 'X-BAPI-TIMESTAMP': params.timestamp, 'X-BAPI-RECV-WINDOW': params.recv_window },
      });
      const exitPrice = parseFloat(response.data.result.price || this.tradeContext.entry_price);
      this.logger.info(`Position closed (${reason}). Exit Price: ${exitPrice}`);

      const trade = this.tradeHistory.find(t => t.status === 'OPEN' && t.entry_time === this.tradeContext.entry_time);
      if (trade) {
        trade.status = 'CLOSED';
        trade.exit_time = Date.now();
        trade.exit_price = exitPrice;
        trade.exit_reason = reason;
        trade.profit = (exitPrice - trade.entry_price) * trade.position_size;
      }
      this.tradeContext = { current_position: null, entry_price: null, stop_loss: null, take_profit: null, position_size: null, entry_time: null };
    } catch (error) {
      this.logger.error(`Position close failed (${reason}): ${error.message}`);
    }
  }

  stop() {
    this.running = false;
    this.logger.info('Bot stopped');
    this.reportTradeHistory();
  }

  reportTradeHistory() {
    this.logger.info('=== Trade History & Performance ===');
    if (!this.tradeHistory.length) {
      this.logger.info('No trades executed');
      return;
    }

    let wins = 0, losses = 0;
    this.tradeHistory.forEach(trade => {
      this.logger.info(`Symbol: ${this.config.symbol}, Side: ${trade.current_position}, Entry: ${trade.entry_price}, Exit: ${trade.exit_price || 'N/A'}, Profit: ${trade.profit || 'N/A'}, Status: ${trade.status}`);
      if (trade.profit > 0) wins++;
      else if (trade.profit < 0) losses++;
    });

    const totalProfit = this.tradeHistory.reduce((sum, t) => sum + (t.profit || 0), 0);
    const winRate = (wins / this.tradeHistory.length) * 100;
    this.logger.info(`Total Trades: ${this.tradeHistory.length}, Wins: ${wins}, Losses: ${losses}, Win Rate: ${winRate.toFixed(2)}%, Total Profit: ${totalProfit.toFixed(2)} USDT`);
  }
}

// Main Execution
async function runBot() {
  const config = require('./config.json');
  const bot = new TradingBot(config);

  bot.on('initialized', () => bot.start());
  bot.on('signals', (signals) => {
    signals.forEach(s => logger.info(`${s.type} - ${s.reason} (${s.confidence}%)`));
  });

  process.on('SIGINT', async () => {
    logger.info('Shutting down bot...');
    await bot.closePosition('Shutdown signal (SIGINT)');
    bot.stop();
    process.exit(0);
  });

  await bot.initialize();
}

if (require.main === module) {
  runBot();
}
