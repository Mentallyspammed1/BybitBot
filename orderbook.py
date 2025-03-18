import ccxt
exchange = ccxt.binance()
symbol = 'APT/USDT'
orderbook = exchange.fetch_order_book(symbol)
bid = orderbook['bids'][0][0] if orderbook['bids'] else None
ask = orderbook['asks'][0][0] if orderbook['asks'] else None
print(f"Bid: {bid}")
print(f"Ask: {ask}")
EOF && python orderbook.py