#!/usr/bin/env ruby

require 'fileutils'
require 'open3'
require 'logger' # Added this line to fix the NameError

# Setup script for the trading bot
class TradingBotSetup
  # List of required Ruby gems
  GEMS = %w[dotenv ccxt technicalindicators colorize async hashie].freeze

  # Default environment variables for .env file
  DEFAULT_ENV = <<~ENV
    BYBIT_API_KEY='your_bybit_api_key_here'
    BYBIT_API_SECRET='your_bybit_api_secret_here'
    LOG_LEVEL='INFO'
  ENV

  def initialize
    @logger = Logger.new(STDOUT)
    @logger.formatter = proc { |severity, datetime, _, msg| "[#{datetime.iso8601}] [#{severity}] #{msg}\n" }
  end

  def run
    @logger.info('Starting setup for Trading Bot...')
    check_ruby_version
    install_gems
    create_config_file
    create_env_file
    verify_setup
    @logger.info('Setup completed successfully! You can now run the trading bot.')
    @logger.info("Next steps:\n1. Edit '.env' with your Bybit API credentials.\n2. Customize 'config.json' if needed.\n3. Run the bot with 'ruby trading_bot.rb'")
  rescue StandardError => e
    @logger.error("Setup failed: #{e.message}")
    @logger.error(e.backtrace.join("\n"))
    exit 1
  end

  private

  def check_ruby_version
    required_version = '3.0.0'
    current_version = RUBY_VERSION
    if Gem::Version.new(current_version) < Gem::Version.new(required_version)
      @logger.error("Ruby version #{current_version} detected. Please upgrade to #{required_version} or higher.")
      exit 1
    else
      @logger.info("Ruby version check passed: #{current_version}")
    end
  end

  def install_gems
    @logger.info('Installing required Ruby gems...')
    GEMS.each do |gem|
      installed = system("gem list -i #{gem} > /dev/null 2>&1")
      unless installed
        @logger.info("Installing gem: #{gem}")
        stdout, stderr, status = Open3.capture3("gem install #{gem}")
        unless status.success?
          @logger.error("Failed to install #{gem}: #{stderr}")
          exit 1
        end
        @logger.info("Successfully installed #{gem}")
      else
        @logger.info("#{gem} is already installed")
      end
    end
  end

  def create_config_file
    config_path = 'config.json'
    unless File.exist?(config_path)
      default_config = {
        symbol: 'BTC/USDT',
        timeframe: '1h',
        history_candles: 300,
        analysis_interval: 60_000,
        max_retries: 5,
        retry_delay: 2000,
        risk_percentage: 0.1,
        max_position_size_usdt: 100,
        stop_loss_multiplier: 2,
        take_profit_multiplier: 4,
        sl_tp_offset_percentage: 0.001,
        cache_ttl: 300_000,
        indicators: {
          sma: { fast: 10, slow: 30 },
          stoch_rsi: { rsi_period: 14, stochastic_period: 14, k_period: 3, d_period: 3 },
          atr: { period: 14 },
          macd: { fast_period: 12, slow_period: 26, signal_period: 9 }
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
          levels_for_buy_entry: %w[S1 S2],
          levels_for_sell_entry: %w[R1 R2]
        }
      }
      File.write(config_path, JSON.pretty_generate(default_config))
      @logger.info("Created default configuration file at '#{config_path}'")
    else
      @logger.info("Configuration file '#{config_path}' already exists")
    end
  end

  def create_env_file
    env_path = '.env'
    unless File.exist?(env_path)
      File.write(env_path, DEFAULT_ENV)
      @logger.info("Created '.env' file with default environment variables")
    else
      @logger.info("'.env' file already exists")
    end
  end

  def verify_setup
    @logger.info('Verifying setup...')
    missing_gems = GEMS.reject { |gem| system("gem list -i #{gem} > /dev/null 2>&1") }
    unless missing_gems.empty?
      @logger.error("Missing gems: #{missing_gems.join(', ')}")
      exit 1
    end

    unless File.exist?('.env')
      @logger.error("'.env' file not found")
      exit 1
    end

    unless File.exist?('config.json')
      @logger.error("'config.json' file not found")
      exit 1
    end

    require 'dotenv'
    Dotenv.load
    unless ENV['BYBIT_API_KEY'] && ENV['BYBIT_API_SECRET']
      @logger.warn("API credentials not set in '.env'. Please add them before running the bot.")
    else
      @logger.info("API credentials found in '.env'")
    end

    @logger.info('Setup verification passed')
  end
end

# Run the setup
if $PROGRAM_NAME == __FILE__
  TradingBotSetup.new.run
end
