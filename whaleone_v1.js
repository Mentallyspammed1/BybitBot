require('dotenv').config();
const ccxt = require('ccxt');
const { StochasticRSI, SMA, ATR } = require('technicalindicators');
const EventEmitter = require('events');

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
    const levels = { INFO: 1, DEBUG: 2 };
    if (levels[level] <= levels[process.env.LOG_LEVEL || 'INFO']) {
        console.log(`${color}[${new Date().toISOString()}] [${level}] ${message}${COLORS.RESET}`, ...args);
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
    stopLossMultiplier: 2,
    takeProfitMultiplier: 4,
    cacheTTL: 300000,
    indicators: {
        sma: { fast: 10, slow: 30 },
        stochRsi: { rsiPeriod: 14, stochasticPeriod: 14, kPeriod: 3, dPeriod: 3 },
        atr: { period: 14 }
    },
    thresholds: { oversold: 20, overbought: 80, minConfidence: 60 }
};

// Global Cache
const CACHE = { markets: null, ohlcv: new Map(), orderBook: new Map() };

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

// Market Data Provider
class MarketDataProvider {
    constructor(exchangeConfig, config) {
        this.exchange = new ccxt.bybit({ apiKey: exchangeConfig.apiKey, secret: exchangeConfig.secret, enableRateLimit: true, rateLimit: exchangeConfig.rateLimit });
        this.config = config;
    }

    async loadMarkets() {
        if (!CACHE.markets) {
            try {
                CACHE.markets = await this.exchange.loadMarkets();
            } catch (error) {
                log('INFO', COLORS.NEON_RED, 'Failed to load markets:', error.message);
                CACHE.markets = {};
            }
        }
        return CACHE.markets;
    }

    async getMinOrderSize(symbol) {
        const markets = await this.loadMarkets();
        const minSize = markets[symbol]?.limits?.amount?.min;
        if (!minSize || isNaN(minSize)) {
            log('DEBUG', COLORS.NEON_YELLOW, `No min order size for ${symbol}, defaulting to 0.01`);
            return 0.01; // Fallback
        }
        return minSize;
    }

    async getBalance() {
        try {
            const balance = await this.exchange.fetchBalance();
            return balance.total;
        } catch (error) {
            log('INFO', COLORS.NEON_RED, 'Failed to fetch balance:', error.message);
            return {};
        }
    }

    async apiHealthCheck() {
        try {
            const balance = await this.getBalance();
            log('INFO', COLORS.NEON_GREEN, 'API health check passed. Balance:', balance);
            return true;
        } catch (error) {
            log('INFO', COLORS.NEON_RED, 'API health check failed:', error.message);
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
            const reversed = data.reverse();
            CACHE.ohlcv.set(cacheKey, { data: reversed, timestamp: Date.now() });
            return reversed;
        } catch (error) {
            if (retries < this.config.maxRetries) {
                const delay = this.config.retryDelay * Math.pow(2, retries);
                log('DEBUG', COLORS.NEON_YELLOW, `Retrying OHLCV fetch (${retries + 1}/${this.config.maxRetries}) after ${delay}ms`);
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
}

// Indicator Engine
class IndicatorEngine {
    constructor(config) { this.config = config; }
    calculateAll(candles) {
        if (!candles || candles.length < this.config.historyCandles) return null;
        const closes = candles.map(c => c[4]);
        const highs = candles.map(c => c[2]);
        const lows = candles.map(c => c[3]);
        return {
            price: closes[closes.length - 1],
            sma: {
                fast: SMA.calculate({ period: this.config.indicators.sma.fast, values: closes }).slice(-1)[0],
                slow: SMA.calculate({ period: this.config.indicators.sma.slow, values: closes }).slice(-1)[0]
            },
            stochRSI: StochasticRSI.calculate({ values: closes, ...this.config.indicators.stochRsi }).slice(-1)[0],
            atr: ATR.calculate({ high: highs, low: lows, close: closes, period: this.config.indicators.atr.period }).slice(-1)[0]
        };
    }
}

// Signal Generator
class SignalGenerator {
    constructor(config) { this.config = config; }
    generate(indicatorData) {
        if (!indicatorData) return [];
        const { sma, stochRSI } = indicatorData;
        const trendUp = sma.fast > sma.slow;
        const signals = [];

        if (stochRSI.k < this.config.thresholds.oversold && stochRSI.d < this.config.thresholds.oversold && trendUp) {
            signals.push({ type: 'BUY', reason: 'Oversold Momentum with Uptrend', confidence: 75 + (trendUp ? 10 : 0), timestamp: Date.now() });
        }
        if (stochRSI.k > this.config.thresholds.overbought && stochRSI.d > this.config.thresholds.overbought && !trendUp) {
            signals.push({ type: 'SELL', reason: 'Overbought Momentum with Downtrend', confidence: 75 + (!trendUp ? 10 : 0), timestamp: Date.now() });
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
        this.indicatorEngine = new IndicatorEngine(this.config);
        this.signalGenerator = new SignalGenerator(this.config);
        this.tradeContext = { currentPosition: null, entryPrice: null, stopLoss: null, takeProfit: null, positionSize: null, entryTime: null };
    }

    async initialize() {
        try {
            if (!await this.marketData.apiHealthCheck()) throw new Error('API health check failed');
            const minSize = await this.marketData.getMinOrderSize(this.config.symbol);
            const balance = await this.marketData.getBalance();
            const usdtAvailable = balance.USDT || 0;
            this.config.positionSize = Math.max(this.config.positionSize || 0.01, minSize * 1.1);
            log('DEBUG', COLORS.NEON_CYAN, `Calculated positionSize: ${this.config.positionSize} (minSize: ${minSize})`);
            if (usdtAvailable < this.config.positionSize * 0.1) {
                log('INFO', COLORS.NEON_YELLOW, `Low USDT balance (${usdtAvailable}), position size may be adjusted`);
            }
            log('INFO', COLORS.NEON_GREEN, `Bot initialized for ${this.config.symbol} on Bybit. Base positionSize: ${this.config.positionSize}`);
            this.emit('initialized');
        } catch (error) {
            log('INFO', COLORS.NEON_RED, 'Initialization failed:', error.message);
            throw error;
        }
    }

    async start() {
        try {
            log('INFO', COLORS.NEON_CYAN, `Trading bot started for ${this.config.symbol} on ${this.config.timeframe}`);
            await this.analysisCycle();
            this.interval = setInterval(() => this.analysisCycle(), this.config.analysisInterval);
        } catch (error) {
            log('INFO', COLORS.NEON_RED, 'Start failed:', error.message);
        }
    }

    async analysisCycle() {
        try {
            const candles = await this.marketData.fetchOHLCV(this.config.symbol, this.config.timeframe, this.config.historyCandles);
            const indicators = this.indicatorEngine.calculateAll(candles);
            const orderBook = await this.marketData.fetchOrderBook(this.config.symbol);
            if (!indicators) return;

            if (this.tradeContext.currentPosition) {
                const { price } = indicators;
                if (price <= this.tradeContext.stopLoss) {
                    await this.closePosition('Stop Loss');
                } else if (price >= this.tradeContext.takeProfit) {
                    await this.closePosition('Take Profit');
                }
            }

            const signals = this.signalGenerator.generate(indicators);
            this.emit('signals', signals);
            await this.executeStrategy(signals, indicators, orderBook);
        } catch (error) {
            log('INFO', COLORS.NEON_RED, 'Analysis error:', error.message);
        }
    }

    async executeStrategy(signals, indicators, orderBook) {
        if (!signals.length) return;
        const primarySignal = signals[0];
        log('INFO', COLORS.NEON_PINK, `Primary Signal: ${primarySignal.type} (${primarySignal.reason}, ${primarySignal.confidence.toFixed(2)}%)`);

        const balance = await this.marketData.getBalance();
        const usdtAvailable = balance.USDT || 0;
        const entryPrice = orderBook.bids[0]?.[0] || indicators.price;
        let positionSize = Math.min(this.config.positionSize, (usdtAvailable * this.config.riskPercentage) / entryPrice);

        if (primarySignal.type === 'BUY' && !this.tradeContext.currentPosition) {
            const minSize = await this.marketData.getMinOrderSize(this.config.symbol);
            positionSize = Math.max(positionSize, minSize * 1.1);
            const cost = entryPrice * positionSize;
            if (cost > usdtAvailable) {
                positionSize = (usdtAvailable * 0.9) / entryPrice;
                log('DEBUG', COLORS.NEON_YELLOW, `Adjusted positionSize to ${positionSize} due to insufficient balance (${usdtAvailable} USDT)`);
            }
            await this.placeOrder('buy', positionSize, entryPrice, indicators.atr);
        } else if (primarySignal.type === 'SELL' && this.tradeContext.currentPosition) {
            await this.closePosition('Sell Signal');
        }
    }

    async placeOrder(side, quantity, price, atr, retries = 0) {
        try {
            log('INFO', COLORS.NEON_GREEN, `Placing ${side.toUpperCase()} order at ${price.toFixed(2)} with size ${quantity}`);
            const order = await this.marketData.exchange.createOrder(this.config.symbol, 'limit', side, quantity, price);
            log('INFO', COLORS.NEON_GREEN, `Order placed: ${order.id}`);
            if (side === 'buy') {
                this.tradeContext = {
                    currentPosition: 'BUY',
                    entryPrice: price,
                    stopLoss: price - (atr * this.config.stopLossMultiplier),
                    takeProfit: price + (atr * this.config.takeProfitMultiplier),
                    positionSize: quantity,
                    entryTime: Date.now()
                };
                log('DEBUG', COLORS.NEON_CYAN, `Set SL: ${this.tradeContext.stopLoss}, TP: ${this.tradeContext.takeProfit}`);
            }
            return order;
        } catch (error) {
            log('INFO', COLORS.NEON_RED, 'Order failed:', error.message);
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
            this.tradeContext = { currentPosition: null, entryPrice: null, stopLoss: null, takeProfit: null, positionSize: null, entryTime: null };
        } catch (error) {
            log('INFO', COLORS.NEON_RED, 'Close failed:', error.message);
        }
    }

    stop() {
        clearInterval(this.interval);
        log('INFO', COLORS.NEON_RED, 'Bot stopped');
    }
}

// Prompt Helper
async function promptForInput(promptText, defaultValue, validator) {
    const rl = require('readline').createInterface({ input: process.stdin, output: stdout });
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

    const bot = new TradingBot(configManager.config);
    bot.on('initialized', () => bot.start());
    bot.on('signals', signals => {
        if (signals.length) {
            log('INFO', COLORS.NEON_CYAN, '\n=== Signals ===');
            signals.forEach(s => log('INFO', COLORS.NEON_GREEN, `${s.type} - ${s.reason} (${s.confidence.toFixed(2)}%)`));
        }
    });

    try {
        await bot.initialize();
    } catch (error) {
        log('INFO', COLORS.NEON_RED, 'Fatal error:', error.message);
        process.exit(1);
    }

    process.on('SIGINT', async () => {
        log('INFO', COLORS.NEON_YELLOW, 'Shutting down...');
        await bot.closePosition('Shutdown');
        bot.stop();
        process.exit(0);
    });
}

runBot().catch(error => {
    log('INFO', COLORS.NEON_RED, 'Startup error:', error.message);
    process.exit(1);
});
