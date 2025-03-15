#!/usr/bin/env node

const fs = require('fs').promises;
const { exec } = require('child_process');
const util = require('util');
const execPromise = util.promisify(exec);
const semver = require('semver'); // Added for version checking

// ANSI color codes (used until chalk is installed)
const COLORS = {
  reset: '\x1b[0m',
  cyan: '\x1b[36m',
  yellow: '\x1b[33m',
  red: '\x1b[31m',
};

// Simple logger
const logger = {
  info: (msg) => console.log(`${COLORS.cyan}[${new Date().toISOString()}] [INFO] ${msg}${COLORS.reset}`),
  warn: (msg) => console.log(`${COLORS.yellow}[${new Date().toISOString()}] [WARN] ${msg}${COLORS.reset}`),
  error: (msg) => console.error(`${COLORS.red}[${new Date().toISOString()}] [ERROR] ${msg}${COLORS.reset}`),
};

// List of required npm packages
const PACKAGES = [
  'dotenv',
  'axios',
  'technicalindicators',
  'chalk',
  'semver', // Added for version checking
];

// Default environment variables for .env file
const DEFAULT_ENV = `
BYBIT_API_KEY=your_bybit_api_key_here
BYBIT_API_SECRET=your_bybit_api_secret_here
LOG_LEVEL=INFO
BYBIT_API_BASE_URL=https://api-testnet.bybit.com
`.trim();

// Default configuration for config.json
const DEFAULT_CONFIG = {
  symbol: 'BTC/USDT',
  timeframe: '1h',
  history_candles: 300,
  analysis_interval: 60000,
  max_retries: 5,
  retry_delay: 2000,
  risk_percentage: 0.1,
  max_position_size_usdt: 100,
  stop_loss_multiplier: 2,
  take_profit_multiplier: 4,
  sl_tp_offset_percentage: 0.001,
  cache_ttl: 300000,
  indicators: {
    sma: { fast: 10, slow: 30 },
    stoch_rsi: { rsi_period: 14, stochastic_period: 14, k_period: 3, d_period: 3 },
    atr: { period: 14 },
    macd: { fast_period: 12, slow_period: 26, signal_period: 9 },
  },
  thresholds: { oversold: 20, overbought: 80, min_confidence: 60 },
  volume_confirmation: { enabled: true, lookback: 20 },
  fibonacci_pivots: {
    enabled: true,
    period: '1d',
    proximity_percentage: 0.005,
    order_book_range_percentage: 0.002,
    pivot_weight: 15,
    order_book_weight: 10,
    levels_for_buy_entry: ['S1', 'S2'],
    levels_for_sell_entry: ['R1', 'R2'],
  },
};

async function checkNodeVersion() {
  const requiredVersion = '14.0.0';
  const { stdout } = await execPromise('node -v');
  const currentVersion = stdout.trim().replace('v', '');

  if (semver.lt(currentVersion, requiredVersion)) {
    logger.error(`Node.js version ${currentVersion} detected. Please upgrade to ${requiredVersion} or higher.`);
    process.exit(1);
  } else {
    logger.info(`Node.js version check passed: ${currentVersion}`);
  }
}

async function installPackages() {
  logger.info('Installing required npm packages...');
  for (const pkg of PACKAGES) {
    try {
      await execPromise(`npm list ${pkg} --depth=0`);
      logger.info(`${pkg} is already installed`);
    } catch {
      logger.info(`Installing package: ${pkg}`);
      try {
        const { stderr } = await execPromise(`npm install ${pkg}`);
        if (stderr && !stderr.includes('up to date')) {
          logger.warn(`Warnings during ${pkg} installation: ${stderr}`);
        }
        logger.info(`Successfully installed ${pkg}`);
      } catch (error) {
        logger.error(`Failed to install ${pkg}: ${error.message}`);
        process.exit(1);
      }
    }
  }
}

async function createConfigFile() {
  const configPath = 'config.json';
  try {
    await fs.access(configPath);
    logger.info(`Configuration file '${configPath}' already exists`);
  } catch {
    await fs.writeFile(configPath, JSON.stringify(DEFAULT_CONFIG, null, 2));
    logger.info(`Created default configuration file at '${configPath}'`);
  }
}

async function createEnvFile() {
  const envPath = '.env';
  try {
    await fs.access(envPath);
    logger.info(`'.env' file already exists`);
  } catch {
    await fs.writeFile(envPath, DEFAULT_ENV);
    logger.info(`Created '.env' file with default environment variables`);
  }
}

async function verifySetup() {
  logger.info('Verifying setup...');
  const missingPackages = [];
  for (const pkg of PACKAGES) {
    try {
      await execPromise(`npm list ${pkg} --depth=0`);
    } catch {
      missingPackages.push(pkg);
    }
  }
  if (missingPackages.length > 0) {
    logger.error(`Missing packages: ${missingPackages.join(', ')}`);
    process.exit(1);
  }

  try {
    await fs.access('.env');
  } catch {
    logger.error(`'.env' file not found`);
    process.exit(1);
  }

  try {
    await fs.access('config.json');
  } catch {
    logger.error(`'config.json' file not found`);
    process.exit(1);
  }

  const dotenv = require('dotenv');
  dotenv.config();
  if (!process.env.BYBIT_API_KEY || !process.env.BYBIT_API_SECRET) {
    logger.warn(`API credentials not set in '.env'. Please add them before running the bot.`);
  } else {
    logger.info(`API credentials found in '.env'`);
  }

  logger.info('Setup verification passed');
}

async function runSetup() {
  try {
    logger.info('Starting setup for Trading Bot...');
    await checkNodeVersion();
    await installPackages();
    await createConfigFile();
    await createEnvFile();
    await verifySetup();
    logger.info('Setup completed successfully! You can now run the trading bot.');
    logger.info('Next steps:\n1. Edit \'.env\' with your Bybit API credentials.\n2. Customize \'config.json\' if needed.\n3. Run the bot with \'node trading_bot.js\'');
  } catch (error) {
    logger.error(`Setup failed: ${error.message}`);
    console.error(error.stack);
    process.exit(1);
  }
}

if (require.main === module) {
  runSetup();
}
