curl -s "https://api.bybit.com/v5/market/recent-trade?category=spot&symbol=BTCUSDT" | jq '.result.list'
