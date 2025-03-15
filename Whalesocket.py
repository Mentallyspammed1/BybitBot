import os
import logging
import requests
import pandas as pd
import numpy as np
from datetime import datetime
import hmac
import hashlib
import time
from dotenv import load_dotenv
from typing import Dict, Tuple, List, Union, Optional
from zoneinfo import ZoneInfo
from decimal import Decimal, getcontext
import json
from logging.handlers import RotatingFileHandler
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import ccxt
from rich.console import Console
from rich.table import Table
from rich.syntax import Syntax
from rich.logging import RichHandler
from rich.panel import Panel
from rich.theme import Theme

getcontext().prec = 10
load_dotenv()

# Rich Theme Definition
custom_theme = Theme({
    "repr.number": "bold bright_white",
    "level.support": "bold green",
    "level.resistance": "bold red",
    "level.pivot": "bold cyan",
    "indicator.bullish": "bold green",
    "indicator.bearish": "bold red",
    "indicator.neutral": "bold yellow",
    "signal.long": "bold green",
    "signal.short": "bold red",
    "signal.neutral": "bold yellow",
})

console = Console(theme=custom_theme)

# Configuration and Setup
class TradingConfig:
    def __init__(self, config_file: str = "config.json"):
        self.config_file = config_file
        self.config = self._load_config()
        self._validate_config()

    def _load_config(self) -> dict:
        default_config = {
            "interval": "15",
            "analysis_interval": 30,
            "retry_delay": 5,
            "momentum_period": 10,
            "momentum_ma_short": 12,
            "momentum_ma_long": 26,
            "volume_ma_period": 20,
            "atr_period": 14,
            "trend_strength_threshold": 0.4,
            "sideways_atr_multiplier": 1.5,
            "indicators": {
                "ema_alignment": True,
                "momentum": True,
                "volume_confirmation": True,
                "divergence": True,
                "stoch_rsi": True,
                "rsi": True,
                "macd": False,
                "bollinger_bands": True
            },
            "weight_sets": {
                "low_volatility": {
                    "ema_alignment": 0.3,
                    "momentum": 0.3,
                    "volume_confirmation": 0.2,
                    "divergence": 0.1,
                    "stoch_rsi": 0.5,
                    "rsi": 0.1,
                    "macd": 0.1,
                    "bollinger_bands": 0.1
                }
            },
            "rsi_period": 14,
            "bollinger_bands_period": 40,
            "bollinger_bands_std_dev": 2,
            "orderbook_limit": 200,
            "orderbook_cluster_threshold": 1000,
            "signal_config": {
                "significant_bid_volume_ratio": 1.5,
                "significant_ask_volume_ratio": 1.5,
                "oversold_rsi_threshold": 33,
                "overbought_rsi_threshold": 66,
                "oversold_stoch_rsi_threshold": 0.2,
                "overbought_stoch_rsi_threshold": 0.8,
                "stoch_rsi_crossover_lookback": 2,
                "stop_loss_atr_multiplier": 2,
                "take_profit_risk_reward_ratio": 2,
                "fib_level_proximity_threshold_long": 0.005,
                "fib_level_proximity_threshold_short": 0.005,
                "pivot_level_proximity_threshold_long": 0.005,
                "pivot_level_proximity_threshold_short": 0.005,
                "bollinger_band_breakout_lookback": 1,
                "trend_confirmation_lookback": 3,
                "min_bollinger_band_squeeze_ratio": 0.1
            }
        }

        if not os.path.exists(self.config_file):
            with open(self.config_file, 'w') as f:
                json.dump(default_config, f, indent=2)
                console.print(
                    Panel(f"[bold yellow]Created new config file with defaults at '{self.config_file}'[/bold yellow]",
                          title="[bold cyan]Configuration Genesis[/bold cyan]")
                )
                return default_config

        try:
            with open(self.config_file, "r", encoding="utf-8") as f:
                config = json.load(f)
                console.print(
                    Panel(f"[bold green]Loaded configuration from '{self.config_file}'[/bold green]",
                          title="[bold cyan]Configuration Loaded[/bold cyan]")
                )
                return config
        except FileNotFoundError:
            console.print(
                Panel("[bold yellow]Config file not found. Loading defaults.[/bold yellow]",
                      title="[bold cyan]Configuration Warning[/bold cyan]")
            )
            return default_config
        except json.JSONDecodeError:
            console.print(
                Panel("[bold yellow]Could not parse config file. Loading defaults.[/bold yellow]",
                      title="[bold red]Configuration Error[/bold red]")
            )
            return default_config

    def _validate_config(self) -> None:
        """Validates configuration values."""
        config = self.config

        # Validate numeric values
        numeric_configs = {
            "analysis_interval": (1, 3600),
            "retry_delay": (1, 300),
            "momentum_period": (1, 100),
            "momentum_ma_short": (1, 100),
            "momentum_ma_long": (1, 100),
            "volume_ma_period": (1, 100),
            "atr_period": (1, 100),
            "trend_strength_threshold": (0, 1),
            "sideways_atr_multiplier": (0.1, 5),
            "orderbook_limit": (1, 1000),
            "orderbook_cluster_threshold": (100, 1000000),
            "stop_loss_atr_multiplier": (0.5, 5),
            "take_profit_risk_reward_ratio": (0.5, 5)
        }

        for key, (min_val, max_val) in numeric_configs.items():
            if key in config:
                value = config[key]
                if not isinstance(value, (int, float)):
                    raise ValueError(f"Configuration value '{key}' must be numeric")
                if value < min_val or value > max_val:
                    raise ValueError(f"Configuration value '{key}' must be between {min_val} and {max_val}")

        # Validate weight sets
        for strategy, weights in config.get("weight_sets", {}).items():
            total = sum(weights.values())
            if abs(total - 1.0) > 0.001:
                raise ValueError(f"Weight set '{strategy}' weights must sum to 1.0")

# API Client
class BybitAPIClient:
    def __init__(self, api_key: str, api_secret: str, base_url: str = "https://api.bybit.com"):
        self.api_key = api_key
        self.api_secret = api_secret
        self.base_url = base_url
        self.session = self._create_session()

    def _create_session(self) -> requests.Session:
        """Creates a requests session with retry mechanism."""
        session = requests.Session()
        retries = Retry(
            total=3,
            backoff_factor=0.5,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET", "POST"]
        )
        session.mount('https://', HTTPAdapter(max_retries=retries))
        return session

    def _sign_request(self, method: str, endpoint: str, params: Optional[dict] = None) -> dict:
        """Signs API requests with HMAC signature."""
        timestamp = str(int(datetime.now().timestamp() * 1000))
        params = params or {}
        signature_params = params.copy()
        signature_params['timestamp'] = timestamp
        param_str = "&".join(f"{key}={value}" for key, value in sorted(signature_params.items()))
        signature = hmac.new(self.api_secret.encode(), param_str.encode(), hashlib.sha256).hexdigest()

        return {
            "X-BAPI-API-KEY": self.api_key,
            "X-BAPI-TIMESTAMP": timestamp,
            "X-BAPI-SIGN": signature,
            "Content-Type": "application/json"
        }

    def request(self, method: str, endpoint: str, params: Optional[dict] = None,
                logger: Optional[logging.Logger] = None) -> Optional[dict]:
        """Makes a signed request to Bybit API with enhanced error handling."""
        try:
            headers = self._sign_request(method, endpoint, params)
            url = f"{self.base_url}{endpoint}"
            request_kwargs = {
                'method': method,
                'url': url,
                'headers': headers,
                'timeout': 10
            }

            if method == "GET":
                request_kwargs['params'] = params
            elif method == "POST":
                request_kwargs['json'] = params

            response = self.session.request(**request_kwargs)
            response.raise_for_status()
            json_response = response.json()

            if json_response and json_response.get("retCode") == 0:
                return json_response
            else:
                ret_code = json_response.get('retCode')
                ret_msg = json_response.get('retMsg')
                if logger:
                    logger.error(f"Bybit API error: [bold red]{ret_code}[/bold red] - [yellow]{ret_msg}[/yellow]")
                return None

        except requests.exceptions.HTTPError as http_err:
            if logger:
                logger.error(f"[bold red]HTTP error occurred:[/bold red] {http_err}")
            return None
        except requests.exceptions.RequestException as req_err:
            if logger:
                logger.error(f"[bold red]API request failed:[/bold red] {req_err}")
            return None
        except json.JSONDecodeError as json_err:
            if logger:
                logger.error(f"[bold red]JSON decode error:[/bold red] {json_err}. Response text: {response.text if 'response' in locals() else 'No response'}")
            return None
        except Exception as e:
            if logger:
                logger.exception(f"[bold red]Unexpected error in API request:[/bold red] {e}")
            return None

# Trading Analyzer
class TradingAnalyzer:
    def __init__(self, df: pd.DataFrame, logger: logging.Logger, config: dict, symbol: str, interval: str):
        self.df = df
        self.logger = logger
        self.levels = {}
        self.fib_levels = {}
        self.config = config
        self.signal = None
        self.weight_sets = config["weight_sets"]
        self.user_defined_weights = self.weight_sets["low_volatility"]
        self.symbol = symbol
        self.interval = interval
        self.current_price = None
        self.indicator_values = {}

    def analyze_orderbook_levels(self, orderbook: dict, current_price: Decimal) -> str:
        """Analyzes orderbook data for significant bid/ask volumes near support/resistance levels."""
        if not orderbook:
            return "[yellow]Orderbook data not available.[/yellow]"

        try:
            bids = pd.DataFrame(orderbook['bids'], columns=['price', 'size'], dtype=float)
            asks = pd.DataFrame(orderbook['asks'], columns=['price', 'size'], dtype=float)

            analysis_output = ""
            threshold = self.config["orderbook_cluster_threshold"]

            def check_cluster_at_level(level_name: str, level_price: float) -> Optional[str]:
                """Checks for significant orderbook clusters at a given price level."""
                bid_volume_at_level = bids[
                    (bids['price'] <= level_price + (level_price * 0.0005)) &
                    (bids['price'] >= level_price - (level_price * 0.0005))
                    ]['size'].sum()

                ask_volume_at_level = asks[
                    (asks['price'] <= level_price + (level_price * 0.0005)) &
                    (asks['price'] >= level_price - (level_price * 0.0005))
                    ]['size'].sum()

                if bid_volume_at_level > threshold:
                    return f"[level.support]Significant bid volume[/level.support] ([repr.number]{bid_volume_at_level:.0f}[/repr.number]) near [level.support]{level_name}[/level.support] [repr.number]${level_price:.2f}[/repr.number]."
                if ask_volume_at_level > threshold:
                    return f"[level.resistance]Significant ask volume[/level.resistance] ([repr.number]{ask_volume_at_level:.0f}[/repr.number]) near [level.resistance]{level_name}[/level.resistance] [repr.number]${level_price:.2f}[/repr.number]."
                return None

            for level_type, levels in self.levels.get("Support", {}).items():
                cluster_analysis = check_cluster_at_level(f"Support {level_type}", levels)
                if cluster_analysis:
                    analysis_output += f"  {cluster_analysis}\n"

            for level_type, levels in self.levels.get("Resistance", {}).items():
                cluster_analysis = check_cluster_at_level(f"Resistance {level_type}", levels)
                if cluster_analysis:
                    analysis_output += f"  {cluster_analysis}\n"

            for level_name, level_price in self.levels.items():
                if isinstance(level_price, (float, int)):
                    cluster_analysis = check_cluster_at_level(f"Pivot {level_name}", level_price)
                    if cluster_analysis:
                        analysis_output += f"  {cluster_analysis}\n"

            if not analysis_output:
                return "  No significant orderbook clusters detected near Fibonacci/Pivot levels."

            return analysis_output.strip()
        except Exception as e:
            self.logger.error(f"Error analyzing orderbook levels: {e}")
            return "[yellow]Orderbook analysis failed.[/yellow]"

    def calculate_indicators(self) -> Dict[str, List[float]]:
        """Calculates all technical indicators."""
        try:
            indicators = {
                "obv": self.calculate_obv().tail(3).tolist(),
                "rsi_20": self.calculate_rsi(window=20).tail(3).tolist(),
                "rsi_100": self.calculate_rsi(window=100).tail(3).tolist(),
                "mfi": self.calculate_mfi().tail(3).tolist(),
                "cci": self.calculate_cci().tail(3).tolist(),
                "wr": self.calculate_williams_r().tail(3).tolist(),
                "adx": [self.calculate_adx()] * 3,
                "adi": self.calculate_adi().tail(3).tolist(),
                "mom": [self.determine_trend_momentum()] * 3,
                "sma": [self.calculate_sma(10).iloc[-1]] if not self.calculate_sma(10).empty else [np.nan],
                "psar": self.calculate_psar().tail(3).tolist(),
                "fve": self.calculate_fve().tail(3).tolist(),
                "macd": self.calculate_macd().tail(3).values.tolist() if not self.calculate_macd().empty else [],
                "bollinger_bands": self.calculate_bollinger_bands(
                    window=self.config["bollinger_bands_period"],
                    num_std_dev=self.config["bollinger_bands_std_dev"]
                ).tail(3).values.tolist() if not self.calculate_bollinger_bands().empty else [],
                "stoch_rsi_k": self.calculate_stoch_rsi()["k"].tail(3).tolist() if not self.calculate_stoch_rsi().empty else [np.nan] * 3,
                "stoch_rsi_d": self.calculate_stoch_rsi()["d"].tail(3).tolist() if not self.calculate_stoch_rsi().empty else [np.nan] * 3,
                "vwap": self.calculate_vwap().tail(3).tolist() if not self.calculate_vwap().empty else [np.nan] * 3,
                "ema_20": self.calculate_ema(window=20).tail(3).tolist() if not self.calculate_ema(window=20).empty else [np.nan] * 3
            }
            return indicators
        except Exception as e:
            self.logger.error(f"Error calculating indicators: {e}")
            return {}

    def analyze(self, current_price: Decimal, timestamp: str) -> str:
        """Performs market analysis and returns formatted output."""
        self.current_price = float(current_price)
        analysis_start_time = time.time()

        high = self.df["high"].max()
        low = self.df["low"].min()
        close = self.df["close"].iloc[-1]

        self._calculate_levels(high, low, close)
        nearest_supports, nearest_resistances = self.find_nearest_levels(self.current_price)

        self.indicator_values = self.calculate_indicators()
        orderbook_data = fetch_orderbook(self.symbol, self.config["orderbook_limit"], self.logger)
        orderbook_analysis_str = self.analyze_orderbook_levels(orderbook_data, current_price)

        analysis_output_str = self._format_analysis_output(timestamp, current_price, nearest_supports, nearest_resistances, orderbook_analysis_str)

        analysis_end_time = time.time()
        analysis_duration = analysis_end_time - analysis_start_time
        self.logger.info(f"Analysis completed in {analysis_duration:.4f} seconds.")

        console.print(Panel.fit(analysis_output_str, title=f"[bold cyan]Mystical Market Analysis for {self.symbol}[/bold cyan]", border_style="blue"))
        self.logger.info(analysis_output_str.replace("[/]", "").replace("[bold", "").replace("[cyan]", "").replace("[yellow]", "").replace("[green]", "").replace("[red]", "").replace("[blue]", "").replace("[magenta]", "").replace("[repr.number]", "").replace("[level.support]", "").replace("[level.resistance]", "").replace("[level.pivot]", "").replace("[indicator.bullish]", "").replace("[indicator.bearish]", "").replace("[indicator.neutral]", "").replace("[signal.long]", "").replace("[signal.short]", "").replace("[signal.neutral]", ""))

        return analysis_output_str

    def _calculate_levels(self, high: float, low: float, close: float) -> None:
        """Calculates Fibonacci retracement and pivot point levels."""
        self.calculate_fibonacci_retracement(high, low, self.current_price)
        self.calculate_pivot_points(high, low, close)

    def _format_analysis_output(self, timestamp: str, current_price: float,
                              nearest_supports: List[Tuple[str, float]],
                              nearest_resistances: List[Tuple[str, float]],
                              orderbook_analysis_str: str) -> str:
        """Formats the analysis output string using Rich markup."""
        analysis_output = f"""
[cyan]Exchange:[/cyan] Bybit
[cyan]Symbol:[/cyan] {self.symbol}
[cyan]Interval:[/cyan] {self.interval}
[cyan]Timestamp:[/cyan] {timestamp}
[cyan]Price:[/cyan]   [repr.number]{self.df['close'].iloc[-3]:.2f}[/repr.number] | [repr.number]{self.df['close'].iloc[-2]:.2f}[/repr.number] | [repr.number]{self.df['close'].iloc[-1]:.2f}[/repr.number]
[cyan]Vol:[/cyan]   [repr.number]{self.df['volume'].iloc[-3]:,}[/repr.number] | [repr.number]{self.df['volume'].iloc[-2]:,}[/repr.number] | [repr.number]{self.df['volume'].iloc[-1]:,}[/repr.number]
[cyan]Current Price:[/cyan] [repr.number]{current_price:.2f}[/repr.number]
"""

        # Add indicator interpretations
        for indicator_name, values in self.indicator_values.items():
            interp = self.interpret_indicator(indicator_name, values)
            if interp:
                analysis_output += interp + "\n"

        # Add levels and orderbook analysis
        levels_output = f"""
[cyan]Support and Resistance Levels:[/cyan]
"""
        for s in nearest_supports:
            levels_output += f"S: [level.support]{s[0]}[/level.support] [repr.number]${s[1]:.3f}[/repr.number]\n"
        for r in nearest_resistances:
            levels_output += f"R: [level.resistance]{r[0]}[/level.resistance] [repr.number]${r[1]:.3f}[/repr.number]\n"

        orderbook_output = f"""
[cyan]Orderbook Analysis near Fibonacci/Pivot Levels:[/cyan]
{orderbook_analysis_str}
"""

        return analysis_output + levels_output + orderbook_output

    def interpret_indicator(self, indicator_name: str, values: List[float]) -> Optional[str]:
        """Interprets indicator values and returns a Rich-formatted string."""
        if values is None or not values or all(np.isnan(v) if isinstance(v, (float, int)) else False for v in values):
            return f"[yellow]{indicator_name.upper()}: No data or calculation error.[/yellow]"

        try:
            if indicator_name == "rsi_20":
                return self._interpret_rsi(values[-1], "RSI 20")
            elif indicator_name == "rsi_100":
                return self._interpret_rsi(values[-1], "RSI 100")
            elif indicator_name == "mfi":
                return self._interpret_mfi(values[-1])
            elif indicator_name == "cci":
                return self._interpret_cci(values[-1])
            elif indicator_name == "wr":
                return self._interpret_williams_r(values[-1])
            elif indicator_name == "adx":
                return self._interpret_adx(values[0])
            elif indicator_name == "obv":
                return self._interpret_obv(values)
            elif indicator_name == "adi":
                return self._interpret_adi(values)
            elif indicator_name == "mom":
                return self._interpret_momentum(values[0])
            elif indicator_name == "sma":
                return self._interpret_sma(values[0])
            elif indicator_name == "psar":
                return self._interpret_psar(values[-1])
            elif indicator_name == "fve":
                return self._interpret_fve(values[-1])
            elif indicator_name == "macd":
                return self._interpret_macd(values[-1])
            elif indicator_name == "bollinger_bands":
                return self._interpret_bollinger_bands(values[-1])
            elif indicator_name == "stoch_rsi_k":
                return self._interpret_stoch_rsi_k(values, self.indicator_values.get("stoch_rsi_d", []))
            elif indicator_name == "vwap":
                return self._interpret_vwap()
            elif indicator_name == "ema_20":
                return self._interpret_ema_20()
            else:
                return None
        except (TypeError, IndexError) as e:
            self.logger.error(f"Error interpreting {indicator_name}: {e}")
            return f"[yellow]{indicator_name.upper()}: Interpretation error.[/yellow]"
        except Exception as e:
            self.logger.error(f"Unexpected error interpreting {indicator_name}: {e}")
            return f"[bold red]{indicator_name.upper()}: Unexpected error.[/bold red]"

    def _interpret_rsi(self, rsi_value: float, rsi_name: str) -> str:
        """Interprets RSI values."""
        if rsi_value > 70:
            return f"[red]{rsi_name}:[/red] Overbought ([repr.number]{rsi_value:.2f}[/repr.number])"
        elif rsi_value < 30:
            return f"[green]{rsi_name}:[/green] Oversold ([repr.number]{rsi_value:.2f}[/repr.number])"
        else:
            return f"[yellow]{rsi_name}:[/yellow] Neutral ([repr.number]{rsi_value:.2f}[/repr.number])"

    def _interpret_mfi(self, mfi_value: float) -> str:
        """Interprets MFI values."""
        if mfi_value > 80:
            return f"[red]MFI:[/red] Overbought ([repr.number]{mfi_value:.2f}[/repr.number])"
        elif mfi_value < 20:
            return f"[green]MFI:[/green] Oversold ([repr.number]{mfi_value:.2f}[/repr.number])"
        else:
            return f"[yellow]MFI:[/yellow] Neutral ([repr.number]{mfi_value:.2f}[/repr.number])"

    def _interpret_cci(self, cci_value: float) -> str:
        """Interprets CCI values."""
        if cci_value > 100:
            return f"[red]CCI:[/red] Overbought ([repr.number]{cci_value:.2f}[/repr.number])"
        elif cci_value < -100:
            return f"[green]CCI:[/green] Oversold ([repr.number]{cci_value:.2f}[/repr.number])"
        else:
            return f"[yellow]CCI:[/yellow] Neutral ([repr.number]{cci_value:.2f}[/repr.number])"

    def _interpret_williams_r(self, wr_value: float) -> str:
        """Interprets Williams %R values."""
        if wr_value < -80:
            return f"[green]Williams %R:[/green] Oversold ([repr.number]{wr_value:.2f}[/repr.number])"
        elif wr_value > -20:
            return f"[red]Williams %R:[/red] Overbought ([repr.number]{wr_value:.2f}[/repr.number])"
        else:
            return f"[yellow]Williams %R:[/yellow] Neutral ([repr.number]{wr_value:.2f}[/repr.number])"

    def _interpret_adx(self, adx_value: float) -> str:
        """Interprets ADX values."""
        if adx_value > 25:
            return f"[indicator.bullish]ADX:[/indicator.bullish] Trending ([repr.number]{adx_value:.2f}[/repr.number])"
        else:
            return f"[indicator.neutral]ADX:[/indicator.neutral] Ranging ([repr.number]{adx_value:.2f}[/repr.number])"

    def _interpret_obv(self, obv_values: List[float]) -> str:
        """Interprets OBV values."""
        return f"[blue]OBV:[/blue] {'[indicator.bullish]Bullish[/indicator.bullish]' if obv_values[-1] > obv_values[-2] else '[indicator.bearish]Bearish[/indicator.bearish]' if obv_values[-1] < obv_values[-2] else '[indicator.neutral]Neutral[/indicator.neutral]'}"

    def _interpret_adi(self, adi_values: List[float]) -> str:
        """Interprets ADI values."""
        return f"[blue]ADI:[/blue] {'[indicator.bullish]Accumulation[/indicator.bullish]' if adi_values[-1] > adi_values[-2] else '[indicator.bearish]Distribution[/indicator.bearish]' if adi_values[-1] < adi_values[-2] else '[indicator.neutral]Neutral[/indicator.neutral]'}"

    def _interpret_momentum(self, mom_data: dict) -> str:
        """Interprets Momentum data."""
        return f"[magenta]Momentum:[/magenta] {mom_data.get('trend', 'Unknown')} (Strength: [repr.number]{mom_data.get('strength', 0):.2f}[/repr.number])"

    def _interpret_sma(self, sma_value: float) -> str:
        """Interprets SMA values."""
        return f"[yellow]SMA (10):[/yellow] [repr.number]{sma_value:.2f}[/repr.number]"

    def _interpret_psar(self, psar_value: float) -> str:
        """Interprets PSAR values."""
        return f"[blue]PSAR:[/blue] [repr.number]{psar_value:.4f}[/repr.number] (Last Value)"

    def _interpret_fve(self, fve_value: float) -> str:
        """Interprets FVE values."""
        return f"[blue]FVE:[/blue] [repr.number]{fve_value:.0f}[/repr.number] (Last Value)"

    def _interpret_macd(self, macd_values: List[float]) -> str:
        """Interprets MACD values."""
        if len(macd_values) == 3:
            macd_line, signal_line, histogram = macd_values[0]
            return f"[green]MACD:[/green] MACD=[repr.number]{macd_line:.2f}[/repr.number], Signal=[repr.number]{signal_line:.2f}[/repr.number], Histogram=[repr.number]{histogram:.2f}[/repr.number]"
        else:
            return f"[red]MACD:[/red] Calculation issue."

    def _interpret_bollinger_bands(self, bb_values: List[float]) -> str:
        """Interprets Bollinger Bands status."""
        if len(bb_values) == 3:
            upper_band, middle_band, lower_band = bb_values[0]
            if self.current_price > upper_band:
                return f"[red]Bollinger Bands:[/red] Price above Upper Band ([repr.number]{upper_band:.2f}[/repr.number])"
            elif self.current_price < lower_band:
                return f"[green]Bollinger Bands:[/green] Price below Lower Band ([repr.number]{lower_band:.2f}[/repr.number])"
            else:
                return f"[yellow]Bollinger Bands:[/yellow] Price within Bands (Upper=[repr.number]{upper_band:.2f}[/repr.number], Middle=[repr.number]{middle_band:.2f}[/repr.number], Lower=[repr.number]{lower_band:.2f}[/repr.number])"
        else:
            return f"[red]Bollinger Bands:[/red] Calculation issue."

    def _interpret_stoch_rsi_k(self, k_values: List[float], d_values: List[float]) -> str:
        """Interprets Stochastic RSI K values and crossovers."""
        if len(k_values) >= 2 and len(d_values) >= 2:
            if k_values[-1] > d_values[-1] and k_values[-2] <= d_values[-2]:
                return f"[green]Stochastic RSI:[/green] Bullish crossover (K=[repr.number]{k_values[-1]:.2f}[/repr.number], D=[repr.number]{d_values[-1]:.2f}[/repr.number])"
            elif k_values[-1] < d_values[-1] and k_values[-2] >= d_values[-2]:
                return f"[red]Stochastic RSI:[/red] Bearish crossover (K=[repr.number]{k_values[-1]:.2f}[/repr.number], D=[repr.number]{d_values[-1]:.2f}[/repr.number])"

        if k_values[-1] > 0.8:
            return f"[red]Stochastic RSI K:[/red] Overbought ([repr.number]{k_values[-1]:.2f}[/repr.number])"
        elif k_values[-1] < 0.2:
            return f"[green]Stochastic RSI K:[/green] Oversold ([repr.number]{k_values[-1]:.2f}[/repr.number])"
        else:
            return f"[yellow]Stochastic RSI K:[/yellow] Neutral ([repr.number]{k_values[-1]:.2f}[/repr.number])"

    def _interpret_vwap(self) -> str:
        """Interprets VWAP status relative to current price."""
        vwap_value = self.indicator_values["vwap"][-1]
        if self.current_price > vwap_value:
            return f"[green]VWAP:[/green] Price above VWAP ([repr.number]{self.current_price:.2f}[/repr.number] > [repr.number]{vwap_value:.2f}[/repr.number])"
        elif self.current_price < vwap_value:
            return f"[red]VWAP:[/red] Price below VWAP ([repr.number]{self.current_price:.2f}[/repr.number] < [repr.number]{vwap_value:.2f}[/repr.number])"
        else:
            return f"[yellow]VWAP:[/yellow] Price at VWAP ([repr.number]{self.current_price:.2f}[/repr.number] == [repr.number]{vwap_value:.2f}[/repr.number])"

    def _interpret_ema_20(self) -> str:
        """Interprets EMA 20 status relative to current price."""
        ema_value = self.indicator_values["ema_20"][-1]
        if self.current_price > ema_value:
            return f"[green]EMA 20:[/green] Price above EMA ([repr.number]{self.current_price:.2f}[/repr.number] > [repr.number]{ema_value:.2f}[/repr.number])"
        elif self.current_price < ema_value:
            return f"[red]EMA 20:[/red] Price below EMA ([repr.number]{self.current_price:.2f}[/repr.number] < [repr.number]{ema_value:.2f}[/repr.number])"
        else:
            return f"[yellow]EMA 20:[/yellow] Price at EMA ([repr.number]{self.current_price:.2f}[/repr.number] == [repr.number]{ema_value:.2f}[/repr.number])"

def analyze_market_data_signals(indicators: Dict[str, List[float]],
                              support_resistance: Dict[str, float],
                              orderbook_analysis: str,
                              config: dict,
                              df: pd.DataFrame,
                              indicators_df: Dict[str, List[float]]) -> Optional[dict]:
    """Analyzes market data indicators and generates trading signals based on combined strategies."""
    signal_config = config.get("signal_config", {})
    significant_bid_volume_ratio = signal_config.get("significant_bid_volume_ratio", 1.5)
    significant_ask_volume_ratio = signal_config.get("significant_ask_volume_ratio", 1.5)
    oversold_rsi_threshold = signal_config.get("oversold_rsi_threshold", 30)
    overbought_rsi_threshold = config.get("overbought_rsi_threshold", 70)
    oversold_stoch_rsi_threshold = signal_config.get("oversold_stoch_rsi_threshold", 0.2)
    overbought_stoch_rsi_threshold = signal_config.get("overbought_stoch_rsi_threshold", 0.8)
    stoch_rsi_crossover_lookback = signal_config.get("stoch_rsi_crossover_lookback", 2)
    stop_loss_atr_multiplier = signal_config.get("stop_loss_atr_multiplier", 2)
    take_profit_risk_reward_ratio = signal_config.get("take_profit_risk_reward_ratio", 2)
    fib_level_proximity_threshold_long = signal_config.get("fib_level_proximity_threshold_long", 0.005)
    fib_level_proximity_threshold_short = signal_config.get("fib_level_proximity_threshold_short", 0.005)
    pivot_level_proximity_threshold_long = signal_config.get("pivot_level_proximity_threshold_long", 0.005)
    pivot_level_proximity_threshold_short = signal_config.get("pivot_level_proximity_threshold_short", 0.005)
    bollinger_band_breakout_lookback = signal_config.get("bollinger_band_breakout_lookback", 1)
    trend_confirmation_lookback = signal_config.get("trend_confirmation_lookback", 3)

    signal = None
    signal_type = None
    entry_price = None
    stop_loss = None
    take_profit = None
    confidence = "Low"
    rationale_parts = []

    current_price = indicators.get("Current Price", None)
    atr_value = indicators.get("ATR", None)
    trend_status = indicators.get("Trend", "Neutral")
    obv_status = indicators.get("OBV", "Neutral")
    ema_20_status = indicators.get("EMA 20 Status", None)
    sma_10_value = indicators.get("SMA (10)", None)
    stoch_rsi_k_status = indicators.get("Stochastic RSI K Status", None)
    stoch_rsi_status = indicators.get("Stochastic RSI Status", None)
    williams_r_status = indicators.get("Williams %R Status", None)
    rsi_20_100_status = indicators.get("RSI 20/100", None)
    bb_upper = indicators.get("Bollinger Bands Upper", None)
    bb_lower = indicators.get("Bollinger Bands Lower", None)
    bb_middle = indicators.get("Bollinger Bands Middle", None)
    bb_status = indicators.get("Bollinger Bands Status", None)

    # Long Signal Logic
    if "Significant bid volume" in orderbook_analysis:
        is_oversold_stoch_rsi = stoch_rsi_k_status == "Oversold"
        is_williams_r_oversold = williams_r_status == "Oversold"
        is_ema_above_sma = ema_20_status == "Price above EMA"

        long_entry_level_options = ['Support (Fib 50.0%)', 'Support (Fib 61.8%)', 'pivot', 's1', 's2']
        entry_level_name = None

        for level_name in long_entry_level_options:
            level_data = support_resistance.get(level_name)
            if level_data and current_price:
                proximity_threshold = fib_level_proximity_threshold_long if "Fib" in level_name else pivot_level_proximity_threshold_long
                if abs(current_price - level_data['price']) / current_price <= proximity_threshold:
                    entry_level_name = level_name
                    entry_price = level_data['price']
                    break

        is_stoch_rsi_bullish_crossover = stoch_rsi_status == "Bullish crossover"

        if entry_level_name and (is_oversold_stoch_rsi or is_stoch_rsi_bullish_crossover) and is_ema_above_sma:
            signal_type = "Long"
            stop_loss = round(current_price - stop_loss_atr_multiplier * atr_value, 4) if atr_value and current_price else None
            take_profit = round(current_price + (take_profit_risk_reward_ratio * (current_price - stop_loss)), 4) if stop_loss and current_price else None
            confidence = "Medium" if is_oversold_stoch_rsi and is_ema_above_sma else "Low"
            rationale_parts.extend([
                f"Detected [bold signal.long]Significant bid volume[/bold signal.long] near [level.support]{entry_level_name}[/level.support] at [repr.number]${entry_price:.2f}[/repr.number].",
                f"Stochastic RSI K is [indicator.neutral]{stoch_rsi_k_status}[/indicator.neutral] and showing [indicator.bullish]{stoch_rsi_status}[/indicator.bullish] signal.",
                f"Williams %R is [indicator.neutral]{williams_r_status}[/indicator.neutral].",
                f"EMA 20 is above SMA 10 indicating short term bullish momentum.",
                f"Overall trend is [indicator.bearish]{trend_status}[/indicator.bearish] and OBV is [indicator.bearish]{obv_status}[/indicator.bearish], consider counter-trend trade with caution.",
                f"Stop-loss at [repr.number]${stop_loss:.4f}[/repr.number], Take-profit at [repr.number]${take_profit:.4f}[/repr.number] (Risk:Reward [repr.number]{take_profit_risk_reward_ratio}:1[/repr.number])."
            ])
            signal = {
                "signal_type": signal_type,
                "entry_price": entry_price,
                "stop_loss": stop_loss,
                "take_profit": take_profit,
                "confidence": confidence,
                "rationale": " ".join(rationale_parts)
            }

    # Short Signal Logic
    elif "Significant ask volume" in orderbook_analysis:
        is_overbought_stoch_rsi = stoch_rsi_k_status == "Overbought"
        is_williams_r_overbought = williams_r_status == "Overbought"
        is_ema_below_sma = ema_20_status == "Price below EMA"

        short_entry_level_options = ['Resistance (Fib 38.2%)', 'Resistance (Fib 23.6%)', 'r1', 'r2', 'pivot']
        entry_level_name = None

        for level_name in short_entry_level_options:
            level_data = support_resistance.get(level_name)
            if level_data and current_price:
                proximity_threshold = fib_level_proximity_threshold_short if "Fib" in level_name else pivot_level_proximity_threshold_short
                if abs(current_price - level_data['price']) / current_price <= proximity_threshold:
                    entry_level_name = level_name
                    entry_price = level_data['price']
                    break

        is_stoch_rsi_bearish_crossover = stoch_rsi_status == "Bearish crossover"

        if entry_level_name and (is_overbought_stoch_rsi or is_stoch_rsi_bearish_crossover) and is_ema_below_sma:
            signal_type = "Short"
            stop_loss = round(current_price + stop_loss_atr_multiplier * atr_value, 4) if atr_value and current_price else None
            take_profit = round(current_price - (take_profit_risk_reward_ratio * (stop_loss - current_price)), 4) if stop_loss and current_price else None
            confidence = "Medium" if is_overbought_stoch_rsi and is_ema_above_sma else "Low"
            rationale_parts.extend([
                f"Detected [bold signal.short]Significant ask volume[/bold signal.short] near [level.resistance]{entry_level_name}[/level.resistance] at [repr.number]${entry_price:.2f}[/repr.number].",
                f"Stochastic RSI K is [indicator.neutral]{stoch_rsi_k_status}[/indicator.neutral] and showing [indicator.bearish]{stoch_rsi_status}[/indicator.bearish] signal.",
                f"Williams %R is [indicator.neutral]{williams_r_overbought}[/indicator.neutral].",
                f"EMA 20 is below SMA 10 indicating short term bearish momentum.",
                f"Overall trend is [indicator.neutral]{trend_status}[/indicator.neutral] and OBV is [indicator.neutral]{obv_status}[/indicator.neutral].",
                f"Stop-loss at [repr.number]${stop_loss:.4f}[/repr.number], Take-profit at [repr.number]${take_profit:.4f}[/repr.number] (Risk:Reward [repr.number]{take_profit_risk_reward_ratio}:1[/repr.number])."
            ])
            signal = {
                "signal_type": signal_type,
                "entry_price": entry_price,
                "stop_loss": stop_loss,
                "take_profit": take_profit,
                "confidence": confidence,
                "rationale": " ".join(rationale_parts)
            }

    # Bollinger Band Breakout Long Signal
    elif bb_status == "Price above Upper Band":
        is_oversold_stoch_rsi = stoch_rsi_k_status == "Oversold"
        is_stoch_rsi_bullish_crossover = stoch_rsi_status == "Bullish crossover"
        is_uptrend = trend_status == "Uptrend" or trend_status == "Neutral"
        is_breaking_resistance = False

        if (is_oversold_stoch_rsi or is_stoch_rsi_bullish_crossover) and is_breaking_resistance and is_uptrend:
            signal_type = "Long"
            entry_price = current_price
            stop_loss = bb_middle if bb_middle else round(current_price - stop_loss_atr_multiplier * atr_value, 4)
            take_profit = round(current_price + (take_profit_risk_reward_ratio * (current_price - stop_loss)), 4) if stop_loss and current_price else None
            confidence = "High"
            rationale_parts.extend([
                f"Price breaking above [bold signal.long]Bollinger Upper Band[/bold signal.long] at [repr.number]${bb_upper:.2f}[/repr.number], confirming a bullish breakout.",
                f"Stochastic RSI K is [indicator.neutral]{stoch_rsi_k_status}[/indicator.neutral] and showing [indicator.bullish]{stoch_rsi_status}[/indicator.bullish] signal, supporting breakout momentum.",
                f"Trend is [indicator.neutral]{trend_status}[/indicator.neutral], aligning with breakout direction.",
                f"Breaking above resistance level (confirmation needed).",
                f"Stop-loss set at Bollinger Middle Band ([repr.number]${bb_middle:.2f}[/repr.number]) for dynamic support.",
                f"Take-profit target is set at [repr.number]${take_profit:.4f}[/repr.number] (Risk:Reward [repr.number]{take_profit_risk_reward_ratio}:1[/repr.number])."
            ])
            signal = {
                "signal_type": signal_type,
                "entry_price": entry_price,
                "stop_loss": stop_loss,
                "take_profit": take_profit,
                "confidence": confidence,
                "rationale": " ".join(rationale_parts)
            }

    # Bollinger Band Breakout Short Signal
    elif bb_status == "Price below Lower Band":
        is_overbought_stoch_rsi = stoch_rsi_k_status == "Overbought"
        is_stoch_rsi_bearish_crossover = stoch_rsi_status == "Bearish crossover"
        is_downtrend = trend_status == "Downtrend" or trend_status == "Neutral"
        is_breaking_support = False

        if (is_overbought_stoch_rsi or is_stoch_rsi_bearish_crossover) and is_breaking_support and is_downtrend:
            signal_type = "Short"
            entry_price = current_price
            stop_loss = bb_middle if bb_middle else round(current_price + stop_loss_atr_multiplier * atr_value, 4)
            take_profit = round(current_price - (take_profit_risk_reward_ratio * (stop_loss - current_price)), 4) if stop_loss and current_price else None
            confidence = "High"
            rationale_parts.extend([
                f"Price breaking below [bold signal.short]Bollinger Lower Band[/bold signal.short] at [repr.number]${bb_lower:.2f}[/repr.number], signaling a bearish breakout.",
                f"Stochastic RSI K is [indicator.neutral]{stoch_rsi_k_status}[/indicator.neutral] and showing [indicator.bearish]{stoch_rsi_status}[/indicator.bearish] signal, reinforcing breakout momentum.",
                f"Trend is [indicator.neutral]{trend_status}[/indicator.neutral], in line with breakout direction.",
                f"Breaking below support level (confirmation needed).",
                f"Stop-loss set at Bollinger Middle Band ([repr.number]${bb_middle:.2f}[/repr.number]) for dynamic resistance.",
                f"Take-profit target is set at [repr.number]${take_profit:.4f}[/repr.number] (Risk:Reward [repr.number]{take_profit_risk_reward_ratio}:1[/repr.number])."
            ])
            signal = {
                "signal_type": signal_type,
                "entry_price": entry_price,
                "stop_loss": stop_loss,
                "take_profit": take_profit,
                "confidence": confidence,
                "rationale": " ".join(rationale_parts)
            }

    return signal

def format_signal_output(signal: Optional[dict], indicators: dict) -> None:
    """Formats the trading signal output using Rich library."""
    if signal:
        table_title = f"[bold magenta]{signal['signal_type']} Signal for {indicators.get('Symbol', 'Unknown Symbol')} on {indicators.get('Exchange', 'Unknown Exchange')} ({indicators.get('Interval', 'Unknown Interval')} Interval)[/bold magenta]"
        table = Table(title=table_title, title_justify="center")
        table.add_column("Entry Price", style="magenta", justify="center")
        table.add_column("Stop-Loss", style="red", justify="center")
        table.add_column("Take-Profit", style="green", justify="center")
        table.add_column("Confidence", style="cyan", justify="center")
        table.add_column("Rationale", style="green", justify="left")

        table.add_row(
            f"[bold]{signal['entry_price']:.4f}[/bold]" if signal.get('entry_price') is not None else "N/A",
            f"[bold]{signal['stop_loss']:.4f}[/bold]" if signal.get('stop_loss') is not None else "N/A",
            f"[bold]{signal['take_profit']:.4f}[/bold]" if signal.get('take_profit') is not None else "N/A",
            f"[bold]{signal['confidence']}[/bold]",
            signal['rationale']
        )

        console.print(Panel.fit(table, padding=(1, 2), title="[bold cyan]Trading Signal[/bold cyan]", border_style="cyan"))
    else:
        console.print(Panel("[bold yellow]No scalping signal generated based on current market data.[/bold yellow]", title="[bold cyan]Signal Status[/bold cyan]", border_style="yellow"))

def setup_logger(symbol: str) -> logging.Logger:
    """Sets up logging for the trading bot."""
    log_level = os.getenv("LOG_LEVEL", "INFO").upper()
    numeric_level = getattr(logging, log_level, None)
    if not isinstance(numeric_level, int):
        console.print(Panel(f"[bold red]Invalid log level:[/bold red] [yellow]{log_level}[/yellow]. Defaulting to [cyan]INFO[/cyan].", title="[bold red]Logging Configuration Error[/bold red]"))
        numeric_level = logging.INFO

    log_dir = "logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    log_file = os.path.join(log_dir, f"trading_bot_{symbol}_{datetime.now().strftime('%Y%m%d')}.log")

    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            RotatingFileHandler(log_file, maxBytes=1024*1024*5, backupCount=7), # 5MB log files, 7 backups
            RichHandler(console=console)
        ]
    )
    logger = logging.getLogger(__name__)
    logger.info(f"Starting trading bot with log level: {logging.getLevelName(logger.getEffectiveLevel())}")
    return logger

def fetch_klines(symbol: str, interval: str, logger: logging.Logger) -> pd.DataFrame:
    """Fetches kline data from Bybit API."""
    bybit_api_url = "https://api.bybit.com"
    endpoint = "/v5/market/kline"
    params = {
        "category": "spot",
        "symbol": symbol,
        "interval": interval,
        "limit": 200
    }

    client = BybitAPIClient(os.getenv("BYBIT_API_KEY"), os.getenv("BYBIT_API_SECRET"))
    response_data = client.request("GET", endpoint, params, logger)

    if response_data and response_data.get('retCode') == 0 and response_data.get('result'):
        klines_data = response_data['result']['list']
        return pd.DataFrame(klines_data)
    else:
        logger.error(f"Failed to fetch klines for {symbol}. API response: {response_data}")
        return pd.DataFrame()

def fetch_orderbook(symbol: str, limit: int, logger: logging.Logger) -> Optional[dict]:
    """Fetches orderbook data from Bybit API."""
    bybit_api_url = "https://api.bybit.com"
    endpoint = "/v5/market/orderbook"
    params = {
        "category": "spot",
        "symbol": symbol,
        "limit": limit
    }

    client = BybitAPIClient(os.getenv("BYBIT_API_KEY"), os.getenv("BYBIT_API_SECRET"))
    response_data = client.request("GET", endpoint, params, logger)

    if response_data and response_data.get('retCode') == 0 and response_data.get('result'):
        return response_data['result']
    else:
        logger.error(f"Failed to fetch orderbook for {symbol}. API response: {response_data}")
        return None

def fetch_current_price(symbol: str, logger: logging.Logger) -> Optional[Decimal]:
    """Fetches the current price for a symbol from Bybit."""
    bybit_api_url = "https://api.bybit.com"
    endpoint = "/v5/market/tickers"
    params = {
        "category": "spot",
        "symbol": symbol
    }
    client = BybitAPIClient(os.getenv("BYBIT_API_KEY"), os.getenv("BYBIT_API_SECRET"))
    response_data = client.request("GET", endpoint, params, logger)

    if response_data and response_data.get('retCode') == 0 and response_data.get('result') and response_data['result']['list']:
        ticker_info = response_data['result']['list'][0]
        return Decimal(ticker_info['lastPrice'])
    else:
        logger.error(f"Failed to fetch current price for {symbol}. API response: {response_data}")
        return None

def main():
    """Main execution function for the trading analyzer."""
    console.print(Panel("Initiating the Mystical Market Scanner...", border_style="cyan", title="[bold cyan]Pyrmethus Scans the Market[/bold cyan]"))

    # Load Configuration
    config = TradingConfig()
    CONFIG = config.config

    VALID_INTERVALS = ["1", "3", "5", "15", "30", "60", "120", "240", "360", "720", "D", "W", "M"]

    # Get user input for symbol and interval
    symbol = ""
    while True:
        symbol = console.input("[cyan]Enter trading symbol (e.g., BTCUSDT):[/cyan] ").upper().strip()
        if symbol:
            break
        console.print("[bold red]Symbol cannot be empty.[/bold red]")

    interval = ""
    while True:
        interval = console.input(f"[cyan]Enter timeframe (e.g., {', '.join(VALID_INTERVALS)}):[/cyan] ").strip()
        if not interval:
            interval = CONFIG["interval"]
            console.print(f"[bold yellow]No interval provided. Using default of {interval}[/bold yellow]")
            break
        if interval in VALID_INTERVALS:
            break
        console.print(f"[bold red]Invalid interval: {interval}[/bold red]")

    # Initialize components
    logger = setup_logger(symbol)
    analysis_interval = CONFIG["analysis_interval"]
    retry_delay = CONFIG["retry_delay"]

    try:
        # Main analysis loop
        while True:
            try:
                current_price = fetch_current_price(symbol, logger)
                if current_price is None:
                    logger.error(f"[bold red]Failed to fetch current price. Retrying in {retry_delay} seconds...[/bold red]")
                    time.sleep(retry_delay)
                    continue

                df = fetch_klines(symbol, interval, logger=logger)
                if df.empty:
                    logger.error(f"[bold red]Failed to fetch kline data. Retrying in {retry_delay} seconds...[/bold red]")
                    time.sleep(retry_delay)
                    continue

                analyzer = TradingAnalyzer(df, logger, CONFIG, symbol, interval)
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                analyzer.analyze(current_price, timestamp)

                signal = analyze_market_data_signals(
                    indicators=analyzer.indicator_values,
                    support_resistance=analyzer.levels,
                    orderbook_analysis=analyzer.analyze_orderbook_levels(
                        fetch_orderbook(symbol, CONFIG["orderbook_limit"], logger),
                        current_price
                    ),
                    config=CONFIG,
                    df=df,
                    indicators_df=analyzer.indicator_values
                )
                format_signal_output(signal, {"Symbol": symbol, "Interval": interval, "Exchange": "Bybit"})

                time.sleep(analysis_interval * 60) # Sleep for analysis interval minutes

            except requests.exceptions.RequestException as e:
                logger.error(f"[bold red]Network error: {e}. Retrying in {retry_delay} seconds...[/bold red]")
                time.sleep(retry_delay)
            except KeyboardInterrupt:
                console.print("[bold yellow]Analysis stopped by user.[/bold yellow]")
                logger.info("Analysis stopped by user.")
                break
            except Exception as e:
                logger.exception(f"[bold red]An unexpected error occurred: {e}. Retrying in {retry_delay} seconds...[/bold red]")
                time.sleep(retry_delay)

    except Exception as e:
        logger.exception(f"[bold red]Fatal error in main execution: {e}[/bold red]")
        console.print(Panel(f"[bold red]Fatal error: {str(e)}[/bold red]", border_style="red"))
        raise

    # Text-based Mermaid Diagram using Rich for styling
    diagram_panel = Panel(
        """
[bold cyan]Signal Generation Process[/bold cyan]
[process]Market Data[/process] --> [process]Orderbook Analysis[/process]
[process]Market Data[/process] --> [process]Indicator Analysis[/process]

    [bold blue]Orderbook Analysis[/bold blue]
    [process]Orderbook Analysis[/process] --> [decision]Significant Volume?[/decision]
    [decision]Significant Volume?[/decision] -- Yes --> [decision]Near Support/Resistance?[/decision]
    [decision]Significant Volume?[/decision] -- No --> [decision]Continue Analysis?[/decision]

    [bold blue]Indicator Analysis[/bold blue]
    [process]Indicator Analysis[/process] --> [decision]Multiple Confirmations?[/decision]
    [decision]Multiple Confirmations?[/decision] -- Yes --> [output]Generate Signal[/output]
    [decision]Multiple Confirmations?[/decision] -- No --> [decision]Continue Analysis?[/decision]

    [decision]Near Support/Resistance?[/decision] -- Yes --> [output]Generate Signal[/output]
    [decision]Near Support/Resistance?[/decision] -- No --> [decision]Continue Analysis?[/decision]

    [decision]Continue Analysis?[/decision] -- Yes --> [process]Orderbook Analysis[/process]
    [decision]Continue Analysis?[/decision] -- No --> [output]No Signal[/output]

    [output]Generate Signal[/output] --> [process]Set Stop Loss[/process]
    [output]Generate Signal[/output] --> [process]Set Take Profit[/process]
    [output]Generate Signal[/output] --> [process]Set Confidence Level[/process]

    [process]Set Stop Loss[/process] & [process]Set Take Profit[/process] & [process]Set Confidence Level[/process] --> [output]Signal Output[/output]

[bold magenta]Legend:[/bold magenta]
[process]Pink Box: Processing Step[/process]
[decision]Blue Diamond: Decision Point[/decision]
[output]Green Box: Output/Final State[/output]
        """,
        title="[bold green]Mystical Signal Flow Diagram[/bold green]",
        border_style="green"
    )
    console.print(diagram_panel)


if __name__ == "__main__":
    main()
  
#completion, the text-based signal flow diagram will be revealed, offering a visual summary of the bot's analytical journey. The digital magic is now even more potent and visually enchanting!
