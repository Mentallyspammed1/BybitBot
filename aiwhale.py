import os
from typing import List, Literal, Optional, Dict, Any
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.progress import Progress
from rich.table import Table
from rich.markdown import Markdown
from datetime import datetime

console = Console()

class TradingAnalysis:
    """Base class for trading analysis functions with common functionality."""
    
    def __init__(self):
        self.console = Console()
        self.progress = Progress()
        
    def _validate_symbol(self, symbol: str) -> None:
        """Validate the trading symbol."""
        if not isinstance(symbol, str) or not symbol.isalnum():
            raise ValueError("Invalid symbol format")
            
    def _validate_timeframe(self, timeframe: str) -> None:
        """Validate the timeframe parameter."""
        valid_timeframes = ["1d", "5d", "1mo", "3mo", "6mo", "1y"]
        if timeframe not in valid_timeframes:
            raise ValueError(f"Invalid timeframe. Must be one of: {', '.join(valid_timeframes)}")
            
    def _format_value(self, value: Any) -> str:
        """Format values consistently."""
        if isinstance(value, float):
            return f"{value:.4f}"
        return str(value)
        
    def _create_panel(self, title: str, content: Text, border_style: str = "blue") -> Panel:
        """Create a consistent panel style."""
        return Panel(
            title=title,
            renderable=content,
            border_style=border_style,
            padding=(1, 2)
        )

def analyze_market_sentiment(
    symbol: str,
    timeframe: Literal["1d", "5d", "1mo", "3mo", "6mo", "1y"],
    indicators: List[str],
    sentiment_threshold: float,
    news_sources: Optional[List[str]] = None,
    technical_analysis: Optional[bool] = False,
) -> None:
    """Analyze market sentiment using AI and technical indicators."""
    
    # Input validation
    analysis = TradingAnalysis()
    analysis._validate_symbol(symbol)
    analysis._validate_timeframe(timeframe)
    
    # Create progress bar for analysis
    with analysis.progress as progress:
        task = progress.add_task("[green]Analyzing Market Sentiment...", total=100)
        
        # Simulate analysis steps
        progress.update(task, advance=30)
        
        # Create output text
        output_text = Text()
        output_text.append(f"Symbol: {symbol}\n", style="bold magenta")
        output_text.append(f"Timeframe: {timeframe}\n", style="bold magenta")
        output_text.append(f"Sentiment Threshold: {analysis._format_value(sentiment_threshold)}\n", style="bold magenta")
        output_text.append(f"Technical Analysis: {analysis._format_value(technical_analysis)}\n", style="bold magenta")
        output_text.append(f"Indicators: {', '.join(indicators)}\n", style="bold magenta")
        output_text.append(f"News Sources: {analysis._format_value(news_sources)}\n", style="bold magenta")
        
        # Add environment variables
        env_vars_text = Text("\nEnvironment Echoes:\n", style="bold green")
        has_env_vars = False
        for key, value in os.environ.items():
            if key.startswith("LLM_"):
                env_vars_text.append(f"  {key}: ", style="bold cyan")
                env_vars_text.append(f"{value}\n", style="italic green")
                has_env_vars = True
        
        if has_env_vars:
            output_text.append(env_vars_text)
        
        # Create and display panel
        panel = analysis._create_panel(
            title="Market Sentiment Analysis",
            content=output_text,
            border_style="blue"
        )
        analysis.console.print(panel)
        
        # Complete progress bar
        progress.update(task, advance=70)

def generate_trading_strategy(
    strategy_type: Literal["trend", "range", "scalping"],
    risk_level: Literal["conservative", "moderate", "aggressive"],
    capital_allocation: float,
    max_positions: int,
    stop_loss_percentage: Optional[float] = None,
    take_profit_multiplier: Optional[float] = None,
) -> None:
    """Generate trading strategy based on AI guidance."""
    
    # Input validation
    analysis = TradingAnalysis()
    if not 0 <= capital_allocation <= 1:
        raise ValueError("Capital allocation must be between 0 and 1")
    if max_positions <= 0:
        raise ValueError("Max positions must be positive")
    
    # Create progress bar for strategy generation
    with analysis.progress as progress:
        task = progress.add_task("[yellow]Generating Strategy...", total=100)
        
        # Simulate strategy generation
        progress.update(task, advance=30)
        
        # Create output text
        output_text = Text()
        output_text.append(f"Strategy Type: {strategy_type}\n", style="bold magenta")
        output_text.append(f"Risk Level: {risk_level}\n", style="bold magenta")
        output_text.append(f"Capital Allocation: {analysis._format_value(capital_allocation)}\n", style="bold magenta")
        output_text.append(f"Max Positions: {max_positions}\n", style="bold magenta")
        output_text.append(f"Stop Loss Percentage: {analysis._format_value(stop_loss_percentage)}\n", style="bold magenta")
        output_text.append(f"Take Profit Multiplier: {analysis._format_value(take_profit_multiplier)}\n", style="bold magenta")
        
        # Add environment variables
        env_vars_text = Text("\nEnvironment Echoes:\n", style="bold green")
        has_env_vars = False
        for key, value in os.environ.items():
            if key.startswith("LLM_"):
                env_vars_text.append(f"  {key}: ", style="bold cyan")
                env_vars_text.append(f"{value}\n", style="italic green")
                has_env_vars = True
        
        if has_env_vars:
            output_text.append(env_vars_text)
        
        # Create and display panel
        panel = analysis._create_panel(
            title="Trading Strategy Configuration",
            content=output_text,
            border_style="yellow"
        )
        analysis.console.print(panel)
        
        # Complete progress bar
        progress.update(task, advance=70)

def process_trade_recommendation(
    recommendation: str,
    confidence_score: float,
    risk_assessment: Literal["low", "medium", "high"],
    supporting_data: List[str],
    trader_preference: Optional[str] = None,
    market_context: Optional[str] = None,
) -> None:
    """Process trade recommendation with AI-assessed insights."""
    
    # Input validation
    analysis = TradingAnalysis()
    if not 0 <= confidence_score <= 1:
        raise ValueError("Confidence score must be between 0 and 1")
    
    # Create progress bar for recommendation processing
    with analysis.progress as progress:
        task = progress.add_task("[cyan]Processing Recommendation...", total=100)
        
        # Simulate processing
        progress.update(task, advance=30)
        
        # Create output text
        output_text = Text()
        output_text.append(f"Recommendation: {recommendation}\n", style="bold magenta")
        output_text.append(f"Confidence Score: {analysis._format_value(confidence_score)}\n", style="bold magenta")
        output_text.append(f"Risk Assessment: {risk_assessment}\n", style="bold magenta")
        output_text.append(f"Supporting Data: {', '.join(supporting_data)}\n", style="bold magenta")
        output_text.append(f"Trader Preference: {analysis._format_value(trader_preference)}\n", style="bold magenta")
        output_text.append(f"Market Context: {analysis._format_value(market_context)}\n", style="bold magenta")
        
        # Add environment variables
        env_vars_text = Text("\nEnvironment Echoes:\n", style="bold green")
        has_env_vars = False
        for key, value in os.environ.items():
            if key.startswith("LLM_"):
                env_vars_text.append(f"  {key}: ", style="bold cyan")
                env_vars_text.append(f"{value}\n", style="italic green")
                has_env_vars = True
        
        if has_env_vars:
            output_text.append(env_vars_text)
        
        # Create and display panel
        panel = analysis._create_panel(
            title="Trade Recommendation Analysis",
            content=output_text,
            border_style="cyan"
        )
        analysis.console.print(panel)
        
        # Complete progress bar
        progress.update(task, advance=70)

if __name__ == "__main__":
    try:
        # Example usage with progress tracking
        with Progress() as progress:
            task = progress.add_task("[bold red]Running Trading Analysis...", total=100)
            
            # Analyze market sentiment
            progress.update(task, advance=30)
            analyze_market_sentiment(
                symbol="BTCUSD",
                timeframe="1d",
                indicators=["RSI", "MACD"],
                sentiment_threshold=0.6,
                news_sources=["Crypto News", "Market Watch"],
                technical_analysis=True,
            )
            
            # Generate trading strategy
            progress.update(task, advance=40)
            generate_trading_strategy(
                strategy_type="trend",
                risk_level="moderate",
                capital_allocation=0.3,
                max_positions=5,
                stop_loss_percentage=0.02,
                take_profit_multiplier=2.0,
            )
            
            # Process trade recommendation
            progress.update(task, advance=30)
            process_trade_recommendation(
                recommendation="Long position recommended based on bullish indicators.",
                confidence_score=0.85,
                risk_assessment="medium",
                supporting_data=["RSI divergence", "MACD crossover"],
                trader_preference="aggressive",
                market_context="Positive market momentum observed.",
            )
            
            progress.update(task, advance=100)
            
    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {str(e)}", style="bold red")
