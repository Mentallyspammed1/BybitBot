from flask import Flask, render_template, jsonify
import os
import threading
import time

# Import your trading bot class (assumes your bot code is in trading_bv2.py)
from trading_bv2 import EnhancedTradingBot

app = Flask(__name__)

# Instantiate the trading bot and start its main loop in a background thread.
bot = EnhancedTradingBot('BTC/USDT')
bot_thread = threading.Thread(target=bot.run, daemon=True)
bot_thread.start()

def compute_total_pnl():
    """Compute total PnL from open positions using current market price."""
    current_price = bot.fetch_market_price() or 0
    pnl_total = 0
    for pos in bot.open_positions:
        entry_price = pos.get('entry_price', current_price)
        amount = pos.get('amount', 0)
        if pos['side'] == 'BUY':
            pnl = (current_price - entry_price) * amount
        else:  # For SELL positions
            pnl = (entry_price - current_price) * amount
        pnl_total += pnl
    return pnl_total

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/status')
def api_status():
    balance = bot.get_account_balance() or 0
    open_orders = bot.open_positions
    current_price = bot.fetch_market_price() or 0
    pnl = compute_total_pnl()
    orders = []
    for pos in open_orders:
        entry_price = pos.get('entry_price', current_price)
        if pos['side'] == 'BUY':
            pos_pnl = (current_price - entry_price) * pos.get('amount', 0)
        else:
            pos_pnl = (entry_price - current_price) * pos.get('amount', 0)
        orders.append({
            'symbol': pos.get('symbol', ''),
            'side': pos.get('side', ''),
            'amount': pos.get('amount', 0),
            'entry_price': entry_price,
            'current_price': current_price,
            'pnl': pos_pnl
        })
    return jsonify({
        'balance': balance,
        'open_orders': orders,
        'total_pnl': pnl
    })

@app.route('/api/logs')
def api_logs():
    log_file = 'enhanced_trading_bot.log'
    if os.path.exists(log_file):
        with open(log_file, 'r') as f:
            lines = f.readlines()
        # Return the last 50 lines
        return jsonify({'logs': lines[-50:]})
    else:
        return jsonify({'logs': []})

if __name__ == '__main__':
    # Runs on 0.0.0.0:5000 so you can access it from your device.
    app.run(host='0.0.0.0', port=5000)
