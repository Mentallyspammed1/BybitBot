from flask import Flask, render_template, request, jsonify
import threading
import time
import os
import logging
import signal
import sys
import argparse
from colorama import Fore, Style, init as colorama_init

colorama_init(autoreset=True)

# Neon color definitions
NEON_GREEN = Fore.GREEN
NEON_CYAN = Fore.CYAN
NEON_YELLOW = Fore.YELLOW
NEON_MAGENTA = Fore.MAGENTA
NEON_RED = Fore.RED
RESET_COLOR = Style.RESET_ALL

from trading_bot import EnhancedTradingBot

app = Flask(__name__)
bot_instance = None
bot_thread = None
bot_running = False
log_file_path = "enhanced_trading_bot.log"

# Logging configuration
flask_logger = logging.getLogger('flask.app')
flask_logger.setLevel(logging.DEBUG)
flask_formatter = logging.Formatter(f"{NEON_CYAN}%(asctime)s - {NEON_YELLOW}%(levelname)s - {NEON_GREEN}%(message)s{RESET_COLOR}")
flask_console_handler = logging.StreamHandler()
flask_console_handler.setFormatter(flask_formatter)
flask_logger.addHandler(flask_console_handler)
flask_file_handler = logging.FileHandler("flask_app.log")
flask_file_handler.setFormatter(flask_formatter)
flask_logger.addHandler(flask_file_handler)

def signal_handler(sig, frame):
    flask_logger.info("Signal received, shutting down Flask app...")
    global bot_running
    if bot_running:
        flask_logger.info("Stopping trading bot...")
        bot_running = False
        if bot_instance:
            bot_instance.stop_bot_loop()
        if bot_thread and bot_thread.is_alive():
            bot_thread.join(timeout=10)
            if bot_thread.is_alive():
                flask_logger.warning("Bot thread did not stop gracefully after 10 seconds. Forcefully exiting.")
            else:
                flask_logger.info("Trading bot stopped gracefully.")
    flask_logger.info("Flask app shutdown complete.")
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

@app.route('/')
def index():
    return render_template('index.html', bot_running=bot_running)

@app.route('/start_bot', methods=['POST'])
def start_bot():
    global bot_instance, bot_thread, bot_running
    if not bot_running:
        symbol = request.form.get('symbol')
        if not symbol:
            flask_logger.warning("No trading symbol provided.")
            return jsonify({'status': 'error', 'message': 'Trading symbol is required.'}), 400

        flask_logger.info(f"Starting trading bot with symbol: {symbol}")
        bot_instance = EnhancedTradingBot(symbol=symbol)
        if bot_instance.exchange:
            bot_running = True
            bot_thread = threading.Thread(target=bot_instance.run_bot)
            bot_thread.daemon = True
            bot_thread.start()
            flask_logger.info("Trading bot started in a new thread.")
            return jsonify({'status': 'success', 'message': f'Trading bot started with symbol {symbol}.'})
        else:
            bot_running = False
            flask_logger.error("Failed to start bot due to exchange initialization error.")
            return jsonify({'status': 'error', 'message': 'Failed to start bot. Check logs for errors.'}), 500
    else:
        flask_logger.warning("Bot is already running.")
        return jsonify({'status': 'warning', 'message': 'Bot is already running.'}), 400

@app.route('/stop_bot', methods=['POST'])
def stop_bot():
    global bot_running, bot_instance
    if bot_running:
        flask_logger.info("Stopping trading bot...")
        bot_running = False
        if bot_instance:
            bot_instance.stop_bot_loop()
        flask_logger.info("Trading bot signaled to stop. It might take a moment to fully stop.")
        return jsonify({'status': 'success', 'message': 'Trading bot stopped.'})
    else:
        flask_logger.warning("Bot is not running.")
        return jsonify({'status': 'warning', 'message': 'Bot is not running.'}), 400

@app.route('/account_data')
def get_account_data():
    """Combines balance, open orders, and PnL into a single endpoint."""
    if bot_instance:
        try:
            balance = bot_instance.fetch_account_balance()
            orders = bot_instance.fetch_open_orders()
            pnl = bot_instance.fetch_position_pnl()

            return jsonify({
                'status': 'success',
                'balance': balance if balance is not None else 'Error',
                'orders': orders,
                'pnl': pnl if pnl is not None else 'Error'
            })
        except Exception as e:
            flask_logger.error(f"Error fetching account data: {e}")
            return jsonify({'status': 'error', 'message': 'Failed to fetch account data. Check logs.'}), 500
    else:
        return jsonify({'status': 'warning', 'message': 'Bot instance not running.'}), 400

@app.route('/logs')
def get_logs():
    try:
        with open(log_file_path, 'r') as f:
            logs = f.readlines()
        return render_template('logs.html', logs=logs)
    except FileNotFoundError:
        return "Log file not found.", 404
    except Exception as e:
        flask_logger.error(f"Error reading log file: {e}")
        return "Error reading log file.", 500

@app.route('/status')
def get_status():
    return jsonify({'bot_running': bot_running})

if __name__ == '__main__':
    # --- Port Configuration using argparse ---
    parser = argparse.ArgumentParser(description="Flask app for Enhanced Trading Bot")
    parser.add_argument('--port', type=int, default=5000, help='Port to run the Flask app on (default: 5000)')
    args = parser.parse_args()
    port = args.port

    flask_logger.info("Starting Flask app...")
    app.run(debug=True, host='0.0.0.0', port=port)
