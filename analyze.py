#!/usr/bin/env python3
import pandas as pd
import numpy as np
from datetime import datetime
import sys

def load_data(file_path):
    """Load and validate the data from the file."""
    try:
        data = pd.read_csv(file_path)
        data['Date'] = pd.to_datetime(data['Date'])
        data.set_index('Date', inplace=True)
        return data
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def calculate_indicators(data):
    """Calculate technical indicators for scalping."""
    if data is None:
        raise ValueError("Data not loaded")
        
    indicators = {}
    
    # Calculate RSI (14-period)
    delta = data['Close'].diff()
    up, down = delta.copy(), delta.copy()
    up[up < 0] = 0
    down[down > 0] = 0
    roll_up = up.rolling(window=14).mean()
    roll_down = down.rolling(window=14).mean().abs()
    RS = roll_up / roll_down
    indicators['RSI'] = 100.0 - (100.0 / (1.0 + RS))
    
    # Calculate MACD (12,26,9)
    ema12 = data['Close'].ewm(span=12, adjust=False).mean()
    ema26 = data['Close'].ewm(span=26, adjust=False).mean()
    indicators['MACD'] = ema12 - ema26
    indicators['Signal'] = indicators['MACD'].ewm(span=9, adjust=False).mean()
    
    # Calculate Stochastic Oscillator (14,3,3)
    low14 = data['Low'].rolling(window=14).min()
    high14 = data['High'].rolling(window=14).max()
    indicators['Stoch'] = ((data['Close'] - low14) / 
                          (high14 - low14) * 100)
    indicators['Stoch_Signal'] = indicators['Stoch'].rolling(window=3).mean()
    
    return indicators

def generate_signals(indicators):
    """Generate scalping signals based on indicators."""
    signals = pd.DataFrame(index=indicators['RSI'].index)
    signals['Buy'] = False
    signals['Sell'] = False
    
    # Generate signals based on multiple indicators
    signals['Buy'] = (
        (indicators['RSI'] < 30) &  # Oversold condition
        (indicators['MACD'] > indicators['Signal']) &  # MACD crossover
        (indicators['Stoch'] < 20) &  # Stochastic oversold
        (indicators['Stoch'] > indicators['Stoch_Signal'])  # Stochastic crossover
    )
    
    signals['Sell'] = (
        (indicators['RSI'] > 70) &  # Overbought condition
        (indicators['MACD'] < indicators['Signal']) &  # MACD crossover
        (indicators['Stoch'] > 80) &  # Stochastic overbought
        (indicators['Stoch'] < indicators['Stoch_Signal'])  # Stochastic crossover
    )
    
    return signals

def analyze_file(file_path):
    """Complete analysis of the file and signal generation."""
    try:
        # Load data
        data = load_data(file_path)
        if data is None:
            return "Error: Could not load data"
            
        # Calculate indicators
        indicators = calculate_indicators(data)
        
        # Generate signals
        signals = generate_signals(indicators)
        
        # Find latest signals
        latest_buy = signals[signals['Buy'] == True].tail(1)
        latest_sell = signals[signals['Sell'] == True].tail(1)
        
        # Generate output
        output = []
        if not latest_buy.empty:
            output.append(f"Buy signal at {latest_buy.index[0]}")
        if not latest_sell.empty:
            output.append(f"Sell signal at {latest_sell.index[0]}")
            
        return "\n".join(output)
        
    except Exception as e:
        return f"Error: {str(e)}"

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: aichat -e analyze <file_path>")
        sys.exit(1)
    
    result = analyze_file(sys.argv[1])
    print(result)
