require('dotenv').config();
const ccxt = require('ccxt');
const { StochasticRSI, SMA, EMA, ATR } = require('technicalindicators');
const EventEmitter = require('events');
const readline = require('readline');
const fs = require('fs');

// Neon Color Scheme
const COLORS = {
    RESET: "\x1b[0m",
    NEON_GREEN: "\x1b[38;5;46m",
    NEON_PINK: "\x1b[38;5;201m",
    NEON_CYAN: "\x1b[38;5;51m",
    NEON_YELLOW: "\x1b[38;5;226m",
    NEON_RED: "\x1b[38;5;196m"
};

// Logging Utility
const log = (level, color, message, ...args) => {
    const levels = { ERROR: 0, INFO: 1, DEBUG: 2 };
    const logLevel = process.env.LOG_LEVEL || 'INFO';
    if (levels[level] <= levels[logLevel]) {
        const logMessage = `${color}[${new Date().toISOString()}] [${level}] ${message}${COLORS.RESET}`;
        console.log(logMessage, ...args);
        if (process.env.LOG_FILE) {
            fs.appendFileSync(process.env.LOG_FILE, `${logMessage} ${JSON.stringify(args)}\n`);
        }
    }
};

// Default Configuration
const DEFAULT_CONFIG = {
    exchange: { name: 'bybit', apiKey: process.env.BYBIT_API_KEY, secret: process.env.BYBIT_API_SECRET, rateLimit: 500 },
    symbol: 'BTC/USDT',
    timeframe: '1h',
    historyCandles: 200,
    analysisInterval: 60000,
    maxRetries: 5,
    retryDelay: 2000,
    riskPercentage: 0.1,
    stopLossOffset: 0,
    stopLossMultiplier: 2,
    takeProfitOffset: 0,
    takeProfitMultiplier: 4,
    cacheTTL: 300000,
    circuitBreaker: { maxFailures: 5, resetTime: 300000 },
    indicators: {
        sma: { fast: 10, slow: 30, ribbon: [20, 50, 200] },
        ema: { fast: 12, slow: 26 },
        stochRsi: { rsiPeriod: 14, stochasticPeriod: 14, kPeriod: 3, dPeriod: 3 },
        atr: { period: 14 },
        momentum: { period: 14, mamFast: 5, mamSlow: 14 }
    },
    thresholds: { oversold: 20, overbought: 80, minConfidence: 60 }
};

// Global Cache and State
const CACHE = { markets: null, ohlcv: new Map(), orderBook: new Map(), balance: { data: {}, timestamp: 0 } };
let botInstance = null;

// Config Manager
class ConfigManager {
    constructor() { this.config = null; }
    async load(config) {
        this.config = { ...DEFAULT_CONFIG, ...config };
        this.validateConfig();
        return this.config;
    }
    validateConfig() {
        const required = ['symbol', 'timeframe', 'exchange.apiKey', 'exchange.secret'];
        for (const key of required) {
            if (!key.split('.').reduce((obj, k) => obj?.[k], this.config)) {
                throw new Error(`Missing required config: ${key}`);
            }
        }
    }
}

// Fibonacci Analyzer
class FibonacciAnalyzer {
    constructor(marketData, config) {
        this.marketData = marketData;
        this.config = config;
    }

    async getPivotRange(symbol, timeframe) {
        if (['1h', '4h', '1d'].includes(timeframe)) {
            const dailyCandles = await this.marketData.fetchOHLCV(symbol, '1d', 2);
            const yesterday = dailyCandles[0];
            return { high: yesterday[2], low: yesterday[3], close: yesterday[4] };
        } else {
            const candles = await this.marketData.fetchOHLCV(symbol, timeframe, 24);
            const highs = candles.map(c => c[2]);
            const lows = candles.map(c => c[3]);
            const closes = candles.map(c => c[4]);
            return { high: Math.max(...highs), low: Math.min(...lows), close: closes[closes.length - 1] };
        }
    }

    calculateFibonacciLevels(high, low) {
        const range = high - low;
        return [
            { level: 0, price: low },
            { level: 23.6, price: low + 0.236 * range },
            { level: 38.2, price: low + 0.382 * range },
            { level: 50, price: low + 0.50 * range },
            { level: 61.8, price: low + 0.618 * range },
            { level: 78.6, price: low + 0.786 * range },
            { level: 100, price: high }
        ];
    }

    findNearestLevels(levels, currentPrice) {
        return levels
            .map(level => ({ ...level, distance: Math.abs(level.price - currentPrice) }))
            .sort((a, b) => a.distance - b.distance)
            .slice(0, 4);
    }

    async getNearestFibonacciLevels(symbol, currentPrice) {
        const { high, low, close } = await this.getPivotRange(symbol, this.config.timeframe);
        const fibLevels = this.calculateFibonacciLevels(high, low);
        const nearestLevels = this.findNearestLevels(fibLevels, currentPrice);
        log('DEBUG', COLORS.NEON_CYAN, `Nearest Fibonacci levels: ${nearestLevels.map(l => `${l.level}%: ${l.price}`).join(', ')}`);
        return nearestLevels;
    }
}

// Market Data Provider
class MarketDataProvider {
    constructor(exchangeConfig, config) {
        this.exchange = new ccxt.bybit({ apiKey: exchangeConfig.apiKey, secret: exchangeConfig.secret, enableRateLimit: true, rateLimit: exchangeConfig.rateLimit });
        this.config = config;
        this.failures = 0;
    }

    async loadMarkets() {
        if (!CACHE.markets) {
            try {
                CACHE.markets = await this.exchange.loadMarkets();
                this.failures = 0;
            } catch (error) {
                log('ERROR', COLORS.NEON_RED, 'Failed to load markets:', error.message);
                CACHE.markets = {};
                throw error;
            }
        }
        return CACHE.markets;
    }

    async getMinOrderSize(symbol) {
        const markets = await this.loadMarkets();
        const minSize = markets[symbol]?.limits?.amount?.min;
        if (!minSize || isNaN(minSize)) {
            log('DEBUG', COLORS.NEON_YELLOW, `No min order size for ${symbol}, defaulting to 0.01`);
            return 0.01;
        }
        return minSize;
    }

    async getBalance(forceRefresh = false) {
        if (!forceRefresh && CACHE.balance.timestamp && Date.now() - CACHE.balance.timestamp < this.config.cacheTTL) {
            return CACHE.balance.data;
        }
        try {
            const balance = await this.exchange.fetchBalance();
            CACHE.balance = { data: balance.total, timestamp: Date.now() };
            this.failures = 0;
            return balance.total;
        } catch (error) {
            log('ERROR', COLORS.NEON_RED, 'Failed to fetch balance:', error.message);
            throw error;
        }
    }

    async apiHealthCheck() {
        try {
            const balance = await this.getBalance(true);
            log('INFO', COLORS.NEON_GREEN, 'API health check passed. Balance:', balance);
            return true;
        } catch (error) {
            log('ERROR', COLORS.NEON_RED, 'API health check failed:', error.message);
            return false;
        }
    }

    async fetchOHLCV(symbol, timeframe, limit, retries = 0) {
        const cacheKey = `${symbol}-${timeframe}-${limit}`;
        const cached = CACHE.ohlcv.get(cacheKey);
        if (cached && Date.now() - cached.timestamp < this.config.cacheTTL) return cached.data;

        try {
            const data = await this.exchange.fetchOHLCV(symbol, timeframe, undefined, limit);
            if (data.length < limit * 0.9) throw new Error('Insufficient OHLCV data');
            CACHE.ohlcv.set(cacheKey, { data, timestamp: Date.now() });
            this.failures = 0;
            return data; // Oldest to newest order
        } catch (error) {
            if (retries < this.config.maxRetries && !error.message.includes('429')) {
                const delay = this.config.retryDelay * Math.pow(2, retries);
                log('DEBUG', COLORS.NEON_YELLOW, `Retrying OHLCV fetch (${retries + 1}/${this.config.maxRetries}) after ${delay}ms`);
                await new Promise(resolve => setTimeout(resolve, delay));
                return this.fetchOHLCV(symbol, timeframe, limit, retries + 1);
            }
            if (error.message.includes('429')) {
                const delay = this.config.retryDelay * Math.pow(2, Math.min(retries, 5));
                log('INFO', COLORS.NEON_YELLOW, `Rate limit hit, backing off for ${delay}ms`);
                await new Promise(resolve => setTimeout(resolve, delay));
                return this.fetchOHLCV(symbol, timeframe, limit, retries + 1);
            }
            throw new Error(`Failed to fetch OHLCV: ${error.message}`);
        }
    }

    async fetchOrderBook(symbol, retries = 0) {
        const cacheKey = symbol;
        const cached = CACHE.orderBook.get(cacheKey);
        if (cached && Date.now() - cached.timestamp < this.config.cacheTTL) return cached.data;

        try {
            const data = await this.exchange.fetchOrderBook(symbol);
            CACHE.orderBook.set(cacheKey, { data, timestamp: Date.now() });
            this.failures = 0;
            return data;
        } catch (error) {
            if (retries < this.config.maxRetries) {
                const delay = this.config.retryDelay * Math.pow(2, retries);
                log('DEBUG', COLORS.NEON_YELLOW, `Retrying order book fetch (${retries + 1}/${this.config.maxRetries}) after ${delay}ms`);
                await new Promise(resolve => setTimeout(resolve, delay));
                return this.fetchOrderBook(symbol, retries + 1);
            }
            throw new Error(`Failed to fetch order book: ${error.message}`);
        }
    }

    async hasSufficientLiquidity(symbol, side, quantity, price) {
        const orderBook = await this.fetchOrderBook(symbol);
        let cumulativeVolume = 0;
        const levels = side === 'buy' ? orderBook.asks : orderBook.bids;
        for (const [levelPrice, levelVolume] of levels) {
            if (side === 'buy' && levelPrice <= price) cumulativeVolume += levelVolume;
            if (side === 'sell' && levelPrice >= price) cumulativeVolume += levelVolume;
            if (cumulativeVolume >= quantity) return true;
        }
        log('DEBUG', COLORS.NEON_YELLOW, `Insufficient ${side} liquidity at ${price}: ${cumulativeVolume}/${quantity}`);
        return false;
    }

    async findOptimalEntryPrice(symbol, side, quantity) {
        const orderBook = await this.fetchOrderBook(symbol);
        let cumulativeVolume = 0;
        const levels = side === 'buy' ? orderBook.asks : orderBook.bids;
        for (const [price, volume] of levels) {
            cumulativeVolume += volume;
            if (cumulativeVolume >= quantity) {
                log('DEBUG', COLORS.NEON_CYAN, `Optimal ${side} price for ${quantity}: ${price}`);
                return price;
            }
        }
        return side === 'buy' ? levels[levels.length - 1][0] : levels[0][0];
    }
}

// Indicator Engine
class IndicatorEngine {
    constructor(config) { this.config = config; }
    getAdaptivePeriods(atr, meanATR) {
        const volatilityFactor = atr / meanATR;
        if (volatilityFactor > 1.5) return { smaFast: 5, smaSlow: 20, emaFast: 6, emaSlow: 13 };
        return { smaFast: this.config.indicators.sma.fast, smaSlow: this.config.indicators.sma.slow, emaFast: this.config.indicators.ema.fast, emaSlow: this.config.indicators.ema.slow };
    }

    calculateAll(candles) {
        if (!candles || candles.length < this.config.historyCandles) return null;
        const closes = candles.map(c => c[4]);
        const highs = candles.map(c => c[2]);
        const lows = candles.map(c => c[3]);
        const atrValues = ATR.calculate({ high: highs, low: lows, close: closes, period: this.config.indicators.atr.period });
        const meanATR = atrValues.slice(-50).reduce((sum, val) => sum + val, 0) / Math.min(50, atrValues.length);
        const { smaFast, smaSlow, emaFast, emaSlow } = this.getAdaptivePeriods(atrValues[atrValues.length - 1], meanATR);

        const smaFastValues = SMA.calculate({ period: smaFast, values: closes });
        const smaSlowValues = SMA.calculate({ period: smaSlow, values: closes });
        const emaFastValues = EMA.calculate({ period: emaFast, values: closes });
        const emaSlowValues = EMA.calculate({ period: emaSlow, values: closes });
        const ribbonValues = this.config.indicators.sma.ribbon.map(period => SMA.calculate({ period, values: closes }));
        const stochRSIValues = StochasticRSI.calculate({ values: closes, ...this.config.indicators.stochRsi });

        const momentumValues = closes.map((close, i) => i < this.config.indicators.momentum.period ? 0 : close - closes[i - this.config.indicators.momentum.period]);
        const mamFastValues = SMA.calculate({ period: this.config.indicators.momentum.mamFast, values: momentumValues });
        const mamSlowValues = SMA.calculate({ period: this.config.indicators.momentum.mamSlow, values: momentumValues });

        return {
            lastValues: {
                price: closes[closes.length - 1],
                sma: {
                    fast: smaFastValues[smaFastValues.length - 1],
                    slow: smaSlowValues[smaSlowValues.length - 1],
                    fastSlope: (smaFastValues[smaFastValues.length - 1] - smaFastValues[smaFastValues.length - 2]) / smaFastValues[smaFastValues.length - 2],
                    slowSlope: (smaSlowValues[smaSlowValues.length - 1] - smaSlowValues[smaSlowValues.length - 2]) / smaSlowValues[smaSlowValues.length - 2],
                    ribbon: ribbonValues.map(r => r[r.length - 1])
                },
                ema: {
                    fast: emaFastValues[emaFastValues.length - 1],
                    slow: emaSlowValues[emaSlowValues.length - 1],
                    fastSlope: (emaFastValues[emaFastValues.length - 1] - emaFastValues[emaFastValues.length - 2]) / emaFastValues[emaFastValues.length - 2],
                    slowSlope: (emaSlowValues[emaSlowValues.length - 1] - emaSlowValues[emaSlowValues.length - 2]) / emaSlowValues[emaSlowValues.length - 2]
                },
                stochRSI: stochRSIValues[stochRSIValues.length - 1],
                atr: atrValues[atrValues.length - 1],
                momentum: {
                    value: momentumValues[momentumValues.length - 1],
                    mamFast: mamFastValues[mamFastValues.length - 1],
                    mamSlow: mamSlowValues[mamSlowValues.length - 1],
                    mamFastSlope: (mamFastValues[mamFastValues.length - 1] - mamFastValues[mamFastValues.length - 2]) / Math.abs(mamFastValues[mamFastValues.length - 2] || 1),
                    mamSlowSlope: (mamSlowValues[mamSlowValues.length - 1] - mamSlowValues[mamSlowValues.length - 2]) / Math.abs(mamSlowValues[mamSlowValues.length - 2] || 1)
                },
                meanATR
            },
            fullArrays: {
                closes,
                highs,
                lows,
                smaFast: smaFastValues,
                smaSlow: smaSlowValues,
                emaFast: emaFastValues,
                emaSlow: emaSlowValues,
                stochRSI: stochRSIValues,
                atr: atrValues,
                momentum: momentumValues,
                mamFast: mamFastValues,
                mamSlow: mamSlowValues
            }
        };
    }
}

// Signal Generator
class SignalGenerator {
    constructor(config) { this.config = config; }
    findSwingLows(candles) {
        const lows = candles.map(c => c[3]);
        let swingLows = [];
        for (let i = 1; i < lows.length - 1; i++) {
            if (lows[i] < lows[i - 1] && lows[i] < lows[i + 1]) {
                swingLows.push({ index: i, price: lows[i] });
            }
        }
        return swingLows;
    }

    findSwingHighs(candles) {
        const highs = candles.map(c => c[2]);
        let swingHighs = [];
        for (let i = 1; i < highs.length - 1; i++) {
            if (highs[i] > highs[i - 1] && highs[i] > highs[i + 1]) {
                swingHighs.push({ index: i, price: highs[i] });
            }
        }
        return swingHighs;
    }

    async generate(indicatorData, orderBook, symbol) {
        if (!indicatorData || !orderBook) return [];
        const { lastValues, fullArrays } = indicatorData;
        const { sma, ema, stochRSI, atr, momentum } = lastValues;
        const spread = orderBook.asks[0][0] - orderBook.bids[0][0];
        const midPrice = (orderBook.bids[0][0] + orderBook.asks[0][0]) / 2;
        const signals = [];

        // SMA and EMA Crossovers
        const smaCrossoverUp = sma.fast > sma.slow && sma.fastSlope > 0;
        const smaCrossoverDown = sma.fast < sma.slow && sma.fastSlope < 0;
        const emaCrossoverUp = ema.fast > ema.slow && ema.fastSlope > 0;
        const emaCrossoverDown = ema.fast < ema.slow && ema.fastSlope < 0;

        // Ribbon Convergence
        const ribbonConverging = sma.ribbon.every((val, i, arr) => i === 0 || Math.abs(val - arr[i - 1]) < atr * 0.5);

        // MAM Cross
        const mamCrossUp = momentum.mamFast > momentum.mamSlow && momentum.mamFastSlope > 0;
        const mamCrossDown = momentum.mamFast < momentum.mamSlow && momentum.mamFastSlope < 0;
        const strongMomentum = Math.abs(momentum.value) > midPrice * 0.01;

        // Base Signals
        if (mamCrossUp && smaCrossoverUp && emaCrossoverUp && stochRSI.k < this.config.thresholds.oversold && stochRSI.d < this.config.thresholds.oversold) {
            let confidence = 75 + (sma.fastSlope + ema.fastSlope + momentum.mamFastSlope) * 100 - spread / atr * 10;
            confidence += ribbonConverging ? 5 : 0;
            confidence += strongMomentum ? 10 : 0;
            signals.push({ type: 'BUY', reason: 'MAM Cross Up with SMA/EMA and Oversold RSI', confidence: Math.min(95, confidence), timestamp: Date.now(), price: midPrice });
        }
        if (mamCrossDown && smaCrossoverDown && emaCrossoverDown && stochRSI.k > this.config.thresholds.overbought && stochRSI.d > this.config.thresholds.overbought) {
            let confidence = 75 - (sma.fastSlope + ema.fastSlope + momentum.mamFastSlope) * 100 - spread / atr * 10;
            confidence += ribbonConverging ? 5 : 0;
            confidence += strongMomentum ? 10 : 0;
            signals.push({ type: 'SELL', reason: 'MAM Cross Down with SMA/EMA and Overbought RSI', confidence: Math.min(95, confidence), timestamp: Date.now(), price: midPrice });
        }

        // Divergence Detection - Last 50 candles
        const last50Candles = fullArrays.closes.slice(-50).map((_, i) => fullArrays.closes[fullArrays.closes.length - 50 + i]);
        const last50RSI = fullArrays.stochRSI.slice(-50).map(rsi => rsi.k);
        const swingLows = this.findSwingLows(last50Candles.map((c, i) => [0, 0, fullArrays.highs[fullArrays.highs.length - 50 + i], fullArrays.lows[fullArrays.lows.length - 50 + i], c]));
        const swingHighs = this.findSwingHighs(last50Candles.map((c, i) => [0, 0, fullArrays.highs[fullArrays.highs.length - 50 + i], fullArrays.lows[fullArrays.lows.length - 50 + i], c]));

        // Bullish Divergence
        if (swingLows.length >= 2) {
            const recentLow = swingLows[swingLows.length - 1];
            const previousLow = swingLows[swingLows.length - 2];
            const recentRSI = last50RSI[recentLow.index];
            const previousRSI = last50RSI[previousLow.index];

            if (recentLow.price < previousLow.price && recentRSI > previousRSI) {
                log('INFO', COLORS.NEON_GREEN, 'Bullish Divergence Detected');
                const buySignals = signals.filter(s => s.type === 'BUY');
                if (buySignals.length > 0) {
                    buySignals.forEach(s => s.confidence = Math.min(95, s.confidence + 10));
                } else {
                    signals.push({ type: 'BUY', reason: 'Bullish Divergence', confidence: 80, timestamp: Date.now(), price: midPrice });
                }
            }
        }

        // Bearish Divergence
        if (swingHighs.length >= 2) {
            const recentHigh = swingHighs[swingHighs.length - 1];
            const previousHigh = swingHighs[swingHighs.length - 2];
            const recentRSI = last50RSI[recentHigh.index];
            const previousRSI = last50RSI[previousHigh.index];

            if (recentHigh.price > previousHigh.price && recentRSI < previousRSI) {
                log('INFO', COLORS.NEON_GREEN, 'Bearish Divergence Detected');
                const sellSignals = signals.filter(s => s.type === 'SELL');
                if (sellSignals.length > 0) {
                    sellSignals.forEach(s => s.confidence = Math.min(95, s.confidence + 10));
                } else {
                    signals.push({ type: 'SELL', reason: 'Bearish Divergence', confidence: 80, timestamp: Date.now(), price: midPrice });
                }
            }
        }

        return signals.filter(s => s.confidence >= this.config.thresholds.minConfidence);
    }
}

// Trading Bot
class TradingBot extends EventEmitter {
    constructor(config) {
        super();
        this.config = config;
        this.marketData = new MarketDataProvider(this.config.exchange, this.config);
        this.fibAnalyzer = new FibonacciAnalyzer(this.marketData, this.config);
        this.indicatorEngine = new IndicatorEngine(this.config);
        this.signalGenerator = new SignalGenerator(this.config);
        this.tradeContext = { currentPosition: null, entryPrice: null, stopLoss: null, takeProfit: null, positionSize: null, entryTime: null, balance: null, fibLevels: [] };
        this.circuitBreaker = { failures: 0, pausedUntil: 0 };
    }

    async initialize() {
        try {
            if (!await this.marketData.apiHealthCheck()) throw new Error('API health check failed');
            const minSize = await this.marketData.getMinOrderSize(this.config.symbol);
            this.tradeContext.balance = await this.marketData.getBalance();
            const usdtAvailable = this.tradeContext.balance.USDT || 0;
            this.config.positionSize = Math.max(this.config.positionSize || 0.01, minSize * 1.1);
            log('DEBUG', COLORS.NEON_CYAN, `Calculated positionSize: ${this.config.positionSize} (minSize: ${minSize})`);
            if (usdtAvailable < this.config.positionSize * 0.1) {
                log('INFO', COLORS.NEON_YELLOW, `Low USDT balance (${usdtAvailable}), position size may be adjusted`);
            }
            log('INFO', COLORS.NEON_GREEN, `Bot initialized for ${this.config.symbol} on Bybit. Base positionSize: ${this.config.positionSize}`);
            this.emit('initialized');
        } catch (error) {
            log('ERROR', COLORS.NEON_RED, 'Initialization failed:', error.message);
            throw error;
        }
    }

    async start() {
        try {
            log('INFO', COLORS.NEON_CYAN, `Trading bot started for ${this.config.symbol} on ${this.config.timeframe}`);
            await this.analysisCycle();
            this.interval = setInterval(() => this.analysisCycle(), this.config.analysisInterval);
        } catch (error) {
            log('ERROR', COLORS.NEON_RED, 'Start failed:', error.message);
        }
    }

    async analysisCycle() {
        if (this.circuitBreaker.pausedUntil > Date.now()) {
            log('INFO', COLORS.NEON_YELLOW, `Circuit breaker active until ${new Date(this.circuitBreaker.pausedUntil).toISOString()}`);
            return;
        }

        try {
            const candles = await this.marketData.fetchOHLCV(this.config.symbol, this.config.timeframe, this.config.historyCandles);
            const indicators = this.indicatorEngine.calculateAll(candles);
            const orderBook = await this.marketData.fetchOrderBook(this.config.symbol);
            if (!indicators) return;

            const currentPrice = orderBook.bids[0][0];
            this.tradeContext.fibLevels = await this.fibAnalyzer.getNearestFibonacciLevels(this.config.symbol, currentPrice);

            if (this.tradeContext.currentPosition) {
                const { price } = indicators.lastValues;
                if (price <= this.tradeContext.stopLoss) {
                    await this.closePosition('Stop Loss');
                } else if (price >= this.tradeContext.takeProfit) {
                    await this.closePosition('Take Profit');
                }
            }

            const signals = await this.signalGenerator.generate(indicators, orderBook, this.config.symbol);
            this.emit('signals', signals);
            await this.executeStrategy(signals, indicators, orderBook);
            this.circuitBreaker.failures = 0;
        } catch (error) {
            log('ERROR', COLORS.NEON_RED, 'Analysis error:', error.message);
            this.circuitBreaker.failures++;
            if (this.circuitBreaker.failures >= this.config.circuitBreaker.maxFailures) {
                this.circuitBreaker.pausedUntil = Date.now() + this.config.circuitBreaker.resetTime;
                log('INFO', COLORS.NEON_RED, `Circuit breaker tripped, pausing until ${new Date(this.circuitBreaker.pausedUntil).toISOString()}`);
            }
        }
    }

    async executeStrategy(signals, indicators, orderBook) {
        if (!signals.length) return;
        const primarySignal = signals[0];
        log('INFO', COLORS.NEON_PINK, `Primary Signal: ${primarySignal.type} (${primarySignal.reason}, ${primarySignal.confidence.toFixed(2)}%)`);

        const usdtAvailable = this.tradeContext.balance.USDT || 0;
        const entryPrice = primarySignal.type === 'BUY' ? orderBook.asks[0][0] : orderBook.bids[0][0];

        // Volatility-based risk adjustment
        const currentATR = indicators.lastValues.atr;
        const meanATR = indicators.lastValues.meanATR;
        const volatilityFactor = currentATR / meanATR;
        let riskAdjustment = 1; // Default: no adjustment
        if (volatilityFactor > 1.5) riskAdjustment = 0.5; // High volatility: halve risk
        else if (volatilityFactor < 0.5) riskAdjustment = 2; // Low volatility: double risk
        const adjustedRiskPercentage = this.config.riskPercentage * riskAdjustment;

        log('DEBUG', COLORS.NEON_CYAN, `Volatility Metrics - Current ATR: ${currentATR.toFixed(4)}, Mean ATR: ${meanATR.toFixed(4)}, Volatility Factor: ${volatilityFactor.toFixed(2)}, Risk Adjustment: ${riskAdjustment}, Adjusted Risk: ${adjustedRiskPercentage.toFixed(4)}`);

        let positionSize = Math.min(this.config.positionSize, (usdtAvailable * adjustedRiskPercentage) / entryPrice) * (primarySignal.confidence / 100);
        positionSize = Math.max(positionSize, (await this.marketData.getMinOrderSize(this.config.symbol)) * 1.1);

        const fibSupport = this.tradeContext.fibLevels.filter(l => l.price < primarySignal.price);
        const fibResistance = this.tradeContext.fibLevels.filter(l => l.price > primarySignal.price);

        if (primarySignal.type === 'BUY' && !this.tradeContext.currentPosition) {
            const nearestSupport = fibSupport.sort((a, b) => b.price - a.price)[0];
            if (!nearestSupport || Math.abs(primarySignal.price - nearestSupport.price) / primarySignal.price > 0.001) {
                log('INFO', COLORS.NEON_YELLOW, 'No nearby Fibonacci support for BUY, skipping');
                return;
            }
            const optimalPrice = await this.marketData.findOptimalEntryPrice(this.config.symbol, 'buy', positionSize);
            const hasLiquidity = await this.marketData.hasSufficientLiquidity(this.config.symbol, 'buy', positionSize, optimalPrice);
            if (!hasLiquidity) {
                log('INFO', COLORS.NEON_YELLOW, 'Insufficient liquidity for BUY, skipping');
                return;
            }
            const cost = optimalPrice * positionSize;
            if (cost > usdtAvailable) {
                positionSize = (usdtAvailable * 0.9) / optimalPrice;
                log('DEBUG', COLORS.NEON_YELLOW, `Adjusted positionSize to ${positionSize} due to insufficient balance (${usdtAvailable} USDT)`);
            }
            await this.placeOrder('buy', positionSize, optimalPrice, indicators.lastValues.atr);
        } else if (primarySignal.type === 'SELL' && this.tradeContext.currentPosition) {
            const nearestResistance = fibResistance.sort((a, b) => a.price - b.price)[0];
            if (!nearestResistance || Math.abs(primarySignal.price - nearestResistance.price) / primarySignal.price > 0.001) {
                log('INFO', COLORS.NEON_YELLOW, 'No nearby Fibonacci resistance for SELL, skipping');
                return;
            }
            const optimalPrice = await this.marketData.findOptimalEntryPrice(this.config.symbol, 'sell', this.tradeContext.positionSize);
            const hasLiquidity = await this.marketData.hasSufficientLiquidity(this.config.symbol, 'sell', this.tradeContext.positionSize, optimalPrice);
            if (!hasLiquidity) {
                log('INFO', COLORS.NEON_YELLOW, 'Insufficient liquidity for SELL, skipping');
                return;
            }
            await this.closePosition('Sell Signal');
        }
    }

    async placeOrder(side, quantity, price, atr, retries = 0) {
        try {
            log('INFO', COLORS.NEON_GREEN, `Placing ${side.toUpperCase()} order at ${price.toFixed(2)} with size ${quantity}`);
            const order = await this.marketData.exchange.createOrder(this.config.symbol, 'limit', side, quantity, price);
            log('INFO', COLORS.NEON_GREEN, `Order placed: ${order.id}`);
            if (side === 'buy') {
                const fibSupport = this.tradeContext.fibLevels.filter(l => l.price < price).sort((a, b) => b.price - a.price)[0];
                const fibResistance = this.tradeContext.fibLevels.filter(l => l.price > price).sort((a, b) => a.price - b.price)[0];
                this.tradeContext = {
                    currentPosition: 'BUY',
                    entryPrice: price,
                    stopLoss: fibSupport ? fibSupport.price : price - (atr * this.config.stopLossMultiplier + this.config.stopLossOffset),
                    takeProfit: fibResistance ? fibResistance.price : price + (atr * this.config.takeProfitMultiplier + this.config.takeProfitOffset),
                    positionSize: quantity,
                    entryTime: Date.now(),
                    balance: this.tradeContext.balance,
                    fibLevels: this.tradeContext.fibLevels
                };
                log('DEBUG', COLORS.NEON_CYAN, `Set SL: ${this.tradeContext.stopLoss}, TP: ${this.tradeContext.takeProfit}`);
            }
            return order;
        } catch (error) {
            log('ERROR', COLORS.NEON_RED, 'Order failed:', error.message);
            if (error.message.includes('170136') && retries < this.config.maxRetries) {
                const minSize = await this.marketData.getMinOrderSize(this.config.symbol);
                quantity = Math.max(quantity, minSize * 1.1);
                const delay = this.config.retryDelay * Math.pow(2, retries);
                log('DEBUG', COLORS.NEON_YELLOW, `Retrying with size ${quantity} after ${delay}ms (${retries + 1}/${this.config.maxRetries})`);
                await new Promise(resolve => setTimeout(resolve, delay));
                return this.placeOrder(side, quantity, price, atr, retries + 1);
            }
            throw error;
        }
    }

    async closePosition(reason) {
        if (!this.tradeContext.currentPosition) return;
        try {
            const order = await this.marketData.exchange.createOrder(this.config.symbol, 'market', 'sell', this.tradeContext.positionSize);
            log('INFO', COLORS.NEON_GREEN, `Position closed (${reason}): ${order.id}`);
            this.tradeContext = { currentPosition: null, entryPrice: null, stopLoss: null, takeProfit: null, positionSize: null, entryTime: null, balance: this.tradeContext.balance, fibLevels: this.tradeContext.fibLevels };
            this.tradeContext.balance = await this.marketData.getBalance(true);
        } catch (error) {
            log('ERROR', COLORS.NEON_RED, 'Close failed:', error.message);
        }
    }

    stop() {
        clearInterval(this.interval);
        log('INFO', COLORS.NEON_RED, 'Bot stopped');
    }
}

// Prompt Helper
async function promptForInput(promptText, defaultValue, validator) {
    const rl = readline.createInterface({ input: process.stdin, output: process.stdout });
    while (true) {
        const answer = await new Promise(resolve => rl.question(promptText, resolve));
        if (!answer.trim()) {
            rl.close();
            return defaultValue;
        } else if (validator(answer)) {
            rl.close();
            return answer;
        }
        log('INFO', COLORS.NEON_YELLOW, `Invalid input: ${answer}. Please try again.`);
    }
}

// Main Execution
async function runBot() {
    try {
        if (!StochasticRSI || !SMA || !EMA || !ATR) throw new Error('Missing technicalindicators dependency');
        const tempExchange = new ccxt.bybit();
        await tempExchange.loadMarkets();
        const symbols = tempExchange.symbols;
        const timeframes = Object.keys(tempExchange.timeframes);

        const symbol = await promptForInput(
            'Enter symbol (e.g., BTC/USDT, IP/USDT, default: BTC/USDT): ',
            'BTC/USDT',
            input => symbols.includes(input)
        );
        const timeframe = await promptForInput(
            'Enter timeframe (e.g., 1m, 15m, 1h, default: 1h): ',
            '1h',
            input => timeframes.includes(input)
        );

        const config = { symbol, timeframe };
        const configManager = new ConfigManager();
        await configManager.load(config);

        botInstance = new TradingBot(configManager.config);
        botInstance.on('initialized', () => botInstance.start());
        botInstance.on('signals', signals => {
            if (signals.length) {
                log('INFO', COLORS.NEON_CYAN, '\n=== Signals ===');
                signals.forEach(s => log('INFO', COLORS.NEON_GREEN, `${s.type} - ${s.reason} (${s.confidence.toFixed(2)}%)`));
            }
        });

        await botInstance.initialize();
    } catch (error) {
        log('ERROR', COLORS.NEON_RED, 'Fatal error:', error.message);
        process.exit(1);
    }

    process.on('SIGINT', async () => {
        log('INFO', COLORS.NEON_YELLOW, 'Shutting down...');
        if (botInstance) await botInstance.closePosition('Shutdown');
        if (botInstance) botInstance.stop();
        process.exit(0);
    });
}

runBot().catch(error => {
    log('ERROR', COLORS.NEON_RED, 'Startup error:', error.message);
    process.exit(1);
});
