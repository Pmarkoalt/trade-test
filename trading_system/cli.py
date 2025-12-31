"""Command-line interface for the trading system with rich output."""

import argparse
import sys
import logging
from pathlib import Path
from typing import Optional, Dict, Any, Callable
from contextlib import contextmanager
from datetime import datetime

try:
    from rich.console import Console
    from rich.table import Table
    from rich.progress import (
        Progress, SpinnerColumn, TextColumn, BarColumn, 
        TimeElapsedColumn, TimeRemainingColumn, TaskID
    )
    from rich.panel import Panel
    from rich.prompt import Prompt, Confirm
    from rich.text import Text
    from rich.layout import Layout
    from rich import box
    from rich.markdown import Markdown
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False
    # Fallback if rich is not available
    Console = None
    Table = None
    Progress = None
    Panel = None
    Prompt = None
    Confirm = None
    Text = None
    Layout = None
    box = None
    Markdown = None
    TaskID = None

from .configs.run_config import RunConfig
from .integration.runner import run_backtest, run_validation, run_holdout, run_sensitivity_analysis
from .reporting.report_generator import ReportGenerator
from .logging import setup_logging as setup_enhanced_logging
from .exceptions import (
    TradingSystemError,
    ConfigurationError,
    DataError,
    StrategyError,
    BacktestError
)

# Global console instance
console = Console() if RICH_AVAILABLE else None


def print_success(message: str) -> None:
    """Print success message with green color."""
    if console:
        console.print(f"[bold green]✓[/bold green] {message}")
    else:
        print(f"✓ {message}")


def print_error(message: str) -> None:
    """Print error message with red color."""
    if console:
        console.print(f"[bold red]✗[/bold red] {message}")
    else:
        print(f"✗ {message}")


def print_warning(message: str) -> None:
    """Print warning message with yellow color."""
    if console:
        console.print(f"[bold yellow]⚠[/bold yellow] {message}")
    else:
        print(f"⚠ {message}")


def print_info(message: str) -> None:
    """Print info message."""
    if console:
        console.print(f"[cyan]ℹ[/cyan] {message}")
    else:
        print(f"ℹ {message}")


def print_banner(title: str, subtitle: Optional[str] = None) -> None:
    """Print a visual banner for command start."""
    if console:
        title_text = f"[bold cyan]{title}[/bold cyan]"
        if subtitle:
            title_text += f"\n[dim]{subtitle}[/dim]"
        console.print(Panel(
            title_text,
            border_style="cyan",
            padding=(1, 2)
        ))
    else:
        print(f"\n{'=' * 60}")
        print(f"  {title}")
        if subtitle:
            print(f"  {subtitle}")
        print(f"{'=' * 60}\n")


def print_section(title: str) -> None:
    """Print a section header."""
    if console:
        console.print(f"\n[bold cyan]▶[/bold cyan] [bold]{title}[/bold]")
    else:
        print(f"\n▶ {title}\n")


@contextmanager
def progress_context(description: str, total: Optional[int] = None, update_callback: Optional[Callable] = None):
    """Context manager for progress bars with optional update callback.
    
    Args:
        description: Description of the operation
        total: Total number of steps (None for indeterminate)
        update_callback: Optional callback function(progress, task_id, current, total) for manual updates
    """
    if console and Progress:
        # Build columns list, filtering out None values
        columns = [
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
        ]
        if total:
            columns.append(BarColumn())
            columns.append(TextColumn("[progress.percentage]{task.percentage:>3.0f}%"))
        columns.append(TimeElapsedColumn())
        if total:
            columns.append(TimeRemainingColumn())

        with Progress(
            *columns,
            console=console,
            transient=False,
            expand=True
        ) as progress:
            task_id = progress.add_task(description, total=total)
            if update_callback:
                update_callback(progress, task_id, 0, total)
            yield progress, task_id
            # Ensure completion
            if total:
                progress.update(task_id, completed=total if total else 0)
    else:
        # Fallback: just print the description
        print(f"Starting: {description}")
        yield None, None
        print(f"Completed: {description}")


@contextmanager
def multi_progress_context(steps: list[tuple[str, Optional[int]]]):
    """Context manager for multi-step progress tracking.
    
    Args:
        steps: List of (description, total) tuples for each step
    """
    if console and Progress:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            console=console,
            transient=False,
            expand=True
        ) as progress:
            task_ids = {}
            for desc, total in steps:
                task_ids[desc] = progress.add_task(desc, total=total or 0)
            yield progress, task_ids
            # Mark all as complete
            for task_id in task_ids.values():
                progress.update(task_id, completed=progress.tasks[task_id].total or 0)
    else:
        # Fallback
        for desc, _ in steps:
            print(f"Starting: {desc}")
        yield None, {}
        for desc, _ in steps:
            print(f"Completed: {desc}")


def display_results_table(results: Dict[str, Any], title: str = "Results") -> None:
    """Display results in a formatted table.
    
    Args:
        results: Dictionary of results to display
        title: Title for the table
    """
    if not console or not Table:
        # Fallback: print as formatted text
        print(f"\n{title}:")
        for key, value in results.items():
            print(f"  {key}: {value}")
        return
    
    table = Table(title=title, box=box.ROUNDED, show_header=True, header_style="bold magenta")
    table.add_column("Metric", style="cyan", no_wrap=True)
    table.add_column("Value", style="green")
    
    for key, value in results.items():
        # Format value based on type
        if isinstance(value, float):
            if abs(value) < 0.01:
                formatted_value = f"{value:.6f}"
            elif abs(value) < 1:
                formatted_value = f"{value:.4f}"
            else:
                formatted_value = f"{value:.2f}"
        elif isinstance(value, (int, str)):
            formatted_value = str(value)
        else:
            formatted_value = str(value)
        
        table.add_row(key.replace('_', ' ').title(), formatted_value)
    
    console.print(table)


def display_validation_results(results: Dict[str, Any]) -> None:
    """Display validation suite results in formatted tables.
    
    Args:
        results: Validation results dictionary
    """
    if not console:
        # Fallback
        print("\nValidation Results:")
        print(f"Status: {results.get('status', 'unknown')}")
        if results.get('rejections'):
            print(f"Rejections: {', '.join(results['rejections'])}")
        if results.get('warnings'):
            print(f"Warnings: {len(results['warnings'])} warnings")
        return
    
    # Status panel
    status = results.get('status', 'unknown')
    status_color = 'green' if status == 'passed' else 'red'
    status_text = Text(status.upper(), style=f"bold {status_color}")
    
    console.print(Panel(
        status_text,
        title="Validation Status",
        border_style=status_color
    ))
    
    # Summary table
    summary_table = Table(title="Summary", box=box.ROUNDED, show_header=True)
    summary_table.add_column("Metric", style="cyan")
    summary_table.add_column("Value", style="green")
    
    summary_table.add_row("Total Trades", str(results.get('total_trades', 0)))
    summary_table.add_row("R-Multiples Count", str(results.get('r_multiples_count', 0)))
    summary_table.add_row("Avg R-Multiple", f"{results.get('avg_r_multiple', 0):.3f}")
    
    console.print(summary_table)
    
    # Rejections and warnings
    if results.get('rejections'):
        console.print("\n[bold red]Rejections:[/bold red]")
        for rejection in results['rejections']:
            console.print(f"  • {rejection}")
    
    if results.get('warnings'):
        console.print(f"\n[bold yellow]Warnings ({len(results['warnings'])}):[/bold yellow]")
        for warning in results['warnings'][:10]:  # Show first 10
            console.print(f"  • {warning}")
        if len(results['warnings']) > 10:
            console.print(f"  ... and {len(results['warnings']) - 10} more warnings")


def display_sensitivity_results(results: Dict[str, Any]) -> None:
    """Display sensitivity analysis results.
    
    Args:
        results: Sensitivity analysis results dictionary
    """
    if not console:
        # Fallback
        print("\nSensitivity Analysis Results:")
        if 'results' in results:
            for asset_class, asset_results in results['results'].items():
                print(f"\n{asset_class}:")
                print(f"  Best params: {asset_results['analysis']['best_params']}")
        return
    
    console.print("\n[bold cyan]Sensitivity Analysis Results[/bold cyan]\n")
    
    if 'results' in results:
        for asset_class, asset_results in results['results'].items():
            analysis = asset_results['analysis']
            
            # Best parameters table
            best_params_table = Table(
                title=f"{asset_class.upper()} - Best Parameters",
                box=box.ROUNDED,
                show_header=True
            )
            best_params_table.add_column("Parameter", style="cyan")
            best_params_table.add_column("Value", style="green")
            
            best_params = analysis.get('best_params', {})
            for param, value in best_params.items():
                best_params_table.add_row(param, str(value))
            
            console.print(best_params_table)
            
            # Best metric value
            if analysis.get('results'):
                best_metric = max(r['metric'] for r in analysis['results'])
                console.print(f"\n[bold green]Best {results.get('metric_name', 'metric')}: {best_metric:.4f}[/bold green]")


def interactive_config_setup() -> Optional[str]:
    """Interactive configuration setup.
    
    Returns:
        Path to config file, or None if cancelled
    """
    if not console:
        print("Interactive mode requires rich library. Please install: pip install rich")
        return None
    
    console.print(Panel(
        "[bold cyan]Interactive Configuration Setup[/bold cyan]\n"
        "This will help you create or modify a configuration file.",
        title="Configuration Wizard",
        border_style="cyan"
    ))
    
    # Check if user wants to create new or use existing
    use_existing = Confirm.ask("Do you have an existing config file?")
    
    if use_existing:
        config_path = Prompt.ask("Enter path to config file", default="EXAMPLE_CONFIGS/run_config.yaml")
        if not Path(config_path).exists():
            print_error(f"Config file not found: {config_path}")
            return None
        return config_path
    else:
        # Guide user to create config
        console.print("\n[bold yellow]To create a new config file:[/bold yellow]")
        console.print("1. Copy EXAMPLE_CONFIGS/run_config.yaml to your desired location")
        console.print("2. Edit the file with your settings")
        console.print("3. Run this command again with --config <path>\n")
        
        create_now = Confirm.ask("Would you like to open the example config now?")
        if create_now:
            example_path = Path("EXAMPLE_CONFIGS/run_config.yaml")
            if example_path.exists():
                console.print(f"\n[green]Example config location:[/green] {example_path.absolute()}")
                return str(example_path)
            else:
                print_error("Example config not found in EXAMPLE_CONFIGS/run_config.yaml")
                return None
    
    return None


def setup_logging(config: RunConfig) -> None:
    """Setup logging configuration.
    
    Args:
        config: RunConfig instance with output settings
    """
    # Get log level
    log_level = getattr(logging, config.output.log_level.upper(), logging.INFO)
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Setup console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    
    # Setup file handler
    output_dir = config.get_output_dir()
    output_dir.mkdir(parents=True, exist_ok=True)
    log_file = output_dir / config.output.log_file
    
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)  # Always log DEBUG to file
    file_handler.setFormatter(formatter)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)
    
    logging.info(f"Logging initialized. Log file: {log_file}")


def cmd_backtest(args: argparse.Namespace) -> int:
    """Run backtest command.
    
    Args:
        args: Parsed command-line arguments
    
    Returns:
        Exit code (0 for success, non-zero for error)
    """
    try:
        period = getattr(args, 'period', 'train')
        config_path = Path(args.config)
        
        # Print banner
        print_banner(
            "Running Backtest",
            f"Period: {period.upper()} | Config: {config_path.name}"
        )
        
        # Validate config file exists
        if not config_path.exists():
            print_error(f"Config file not found: {config_path}")
            if console:
                console.print("\n[yellow]💡 Tips:[/yellow]")
                console.print(f"  • Check the path: {config_path.absolute()}")
                console.print(f"  • Generate a template: [cyan]python -m trading_system config template[/cyan]")
                console.print(f"  • Use interactive wizard: [cyan]python -m trading_system config wizard[/cyan]")
            return 1
        
        # Load config
        print_section("Loading Configuration")
        config = RunConfig.from_yaml(str(config_path))
        setup_enhanced_logging(config)
        print_success(f"Configuration loaded: {config_path.name}")
        
        # Run with progress indication
        print_section("Running Backtest")
        with progress_context(f"Processing {period} period", total=None):
            results = run_backtest(str(config_path), period=period)
        
        # Display key results if available
        print_section("Results")
        if isinstance(results, dict):
            key_metrics = {
                'Total Return': results.get('total_return', 0) * 100 if results.get('total_return') is not None else 0,
                'Sharpe Ratio': results.get('sharpe_ratio', 0),
                'Max Drawdown': results.get('max_drawdown', 0) * 100 if results.get('max_drawdown') is not None else 0,
                'Total Trades': results.get('total_trades', 0),
                'Win Rate': results.get('win_rate', 0) * 100 if results.get('win_rate') is not None else 0,
            }
            display_results_table(key_metrics, title=f"Backtest Results - {period.upper()}")
            
            # Get run_id if available for helpful next steps
            run_id = results.get('run_id')
            if run_id and console:
                console.print("\n[dim]Next steps:[/dim]")
                console.print(f"  • View dashboard: [cyan]python -m trading_system dashboard --run-id {run_id}[/cyan]")
                console.print(f"  • Generate report: [cyan]python -m trading_system report --run-id {run_id}[/cyan]")
        
        print_success("Backtest completed successfully")
        return 0
        
    except FileNotFoundError as e:
        print_error(f"Config file not found: {e}")
        if console:
            console.print("\n[yellow]💡 Tips:[/yellow]")
            console.print(f"  • Verify the config path exists")
            console.print(f"  • Generate a template: [cyan]python -m trading_system config template[/cyan]")
            console.print_exception(show_locals=False)
        return 1
    except ConfigurationError as e:
        print_error(f"Configuration error: {e}")
        if console:
            if hasattr(e, 'config_path') and e.config_path:
                console.print(f"\n[cyan]Configuration file:[/cyan] {e.config_path}")
            if hasattr(e, 'format_errors'):
                console.print(e.format_errors())
            else:
                console.print(f"\n[yellow]💡 Troubleshooting:[/yellow]")
                console.print(f"  • Validate config: [cyan]python -m trading_system config validate --path {e.config_path or args.config}[/cyan]")
                console.print(f"  • Generate template: [cyan]python -m trading_system config template[/cyan]")
                console.print(f"  • Review example configs in [cyan]EXAMPLE_CONFIGS/[/cyan]")
        return 1
    except (DataError, DataNotFoundError) as e:
        print_error(f"Data error: {e}")
        if console:
            console.print(f"\n[yellow]💡 Troubleshooting:[/yellow]")
            if hasattr(e, 'file_path') and e.file_path:
                console.print(f"  • File path: [cyan]{e.file_path}[/cyan]")
            if hasattr(e, 'data_path') and e.data_path:
                console.print(f"  • Data path: [cyan]{e.data_path}[/cyan]")
            if hasattr(e, 'symbol') and e.symbol:
                console.print(f"  • Symbol: [cyan]{e.symbol}[/cyan]")
            console.print(f"  • Check data file paths in config")
            console.print(f"  • Verify data files exist and are valid")
            console.print(f"  • Verify file format (CSV with: date, open, high, low, close, volume)")
            if hasattr(e, 'file_path') and e.file_path:
                console.print(f"  • Check file: [cyan]ls -la {e.file_path}[/cyan]")
        return 1
    except StrategyError as e:
        print_error(f"Strategy error: {e}")
        if console:
            console.print(f"\n[yellow]💡 Troubleshooting:[/yellow]")
            if hasattr(e, 'strategy_name') and e.strategy_name:
                console.print(f"  • Strategy: [cyan]{e.strategy_name}[/cyan]")
            if hasattr(e, 'symbol') and e.symbol:
                console.print(f"  • Symbol: [cyan]{e.symbol}[/cyan]")
            console.print(f"  • Check strategy configuration in config file")
            console.print(f"  • Verify strategy type and asset class match (equity vs crypto)")
            console.print(f"  • Verify strategy parameters are within valid ranges")
            console.print(f"  • Review example configs in [cyan]EXAMPLE_CONFIGS/[/cyan]")
        return 1
    except BacktestError as e:
        print_error(f"Backtest error: {e}")
        if console:
            console.print(f"\n[yellow]💡 Troubleshooting:[/yellow]")
            console.print(f"  • Check logs for detailed error information")
            if hasattr(e, 'date'):
                console.print(f"  • Error occurred at date: {e.date}")
            if hasattr(e, 'step'):
                console.print(f"  • Error occurred in step: {e.step}")
        return 1
    except KeyboardInterrupt:
        print_warning("\nBacktest cancelled by user")
        return 130
    except TradingSystemError as e:
        print_error(f"Trading system error: {e}")
        if console:
            console.print_exception(show_locals=False)
        return 1
    except Exception as e:
        print_error(f"Unexpected error: {e}")
        if console:
            console.print("\n[yellow]💡 Troubleshooting:[/yellow]")
            console.print(f"  • Check logs in the output directory")
            console.print(f"  • Validate config: [cyan]python -m trading_system config validate --path {args.config}[/cyan]")
            console.print_exception(show_locals=False)
        else:
            logging.exception("Backtest failed")
        return 1


def cmd_validate(args: argparse.Namespace) -> int:
    """Run validation suite command.
    
    Args:
        args: Parsed command-line arguments
    
    Returns:
        Exit code (0 for success, non-zero for error)
    """
    try:
        config_path = Path(args.config)
        
        # Print banner
        print_banner(
            "Running Validation Suite",
            f"Config: {config_path.name} | Comprehensive statistical and stress testing"
        )
        
        # Validate config file exists
        if not config_path.exists():
            print_error(f"Config file not found: {config_path}")
            if console:
                console.print("\n[yellow]💡 Tips:[/yellow]")
                console.print(f"  • Check the path: {config_path.absolute()}")
                console.print(f"  • Validate config first: [cyan]python -m trading_system config validate --path {args.config}[/cyan]")
            return 1
        
        # Load config
        print_section("Loading Configuration")
        config = RunConfig.from_yaml(str(config_path))
        setup_enhanced_logging(config)
        print_success(f"Configuration loaded: {config_path.name}")
        
        # Run with progress indication
        print_section("Running Validation Tests")
        with progress_context("Validating strategy performance", total=None):
            results = run_validation(str(config_path))
        
        # Display validation results
        print_section("Validation Results")
        display_validation_results(results)
        
        status = results.get('status', 'unknown')
        if status == 'passed':
            print_success("Validation suite passed - Strategy meets all criteria")
            if console:
                console.print("\n[green]✓[/green] Your strategy is ready for production testing")
            return 0
        else:
            print_error("Validation suite failed - Strategy did not meet all criteria")
            if console:
                console.print("\n[yellow]💡 Next Steps:[/yellow]")
                console.print(f"  • Review rejection reasons above")
                console.print(f"  • Adjust strategy parameters and retry")
                console.print(f"  • Run backtest: [cyan]python -m trading_system backtest --config {args.config}[/cyan]")
            return 1
        
    except FileNotFoundError as e:
        print_error(f"Config file not found: {e}")
        if console:
            console.print("\n[yellow]💡 Tips:[/yellow]")
            console.print(f"  • Verify the config path exists")
            console.print(f"  • Generate a template: [cyan]python -m trading_system config template[/cyan]")
            console.print_exception(show_locals=False)
        return 1
    except KeyboardInterrupt:
        print_warning("\nValidation cancelled by user")
        return 130
    except Exception as e:
        print_error(f"Validation failed: {e}")
        if console:
            console.print("\n[yellow]💡 Troubleshooting:[/yellow]")
            console.print(f"  • Check logs in the output directory")
            console.print(f"  • Validate config: [cyan]python -m trading_system config validate --path {args.config}[/cyan]")
            console.print_exception(show_locals=False)
        else:
            logging.exception("Validation failed")
        return 1


def cmd_holdout(args: argparse.Namespace) -> int:
    """Run holdout evaluation command.
    
    Args:
        args: Parsed command-line arguments
    
    Returns:
        Exit code (0 for success, non-zero for error)
    """
    try:
        config_path = Path(args.config)
        
        # Print banner
        print_banner(
            "Holdout Evaluation",
            f"Out-of-sample testing on unseen data | Config: {config_path.name}"
        )
        
        # Validate config file exists
        if not config_path.exists():
            print_error(f"Config file not found: {config_path}")
            if console:
                console.print("\n[yellow]💡 Tips:[/yellow]")
                console.print(f"  • Check the path: {config_path.absolute()}")
                console.print(f"  • Ensure holdout period is configured in your config")
            return 1
        
        # Load config
        print_section("Loading Configuration")
        config = RunConfig.from_yaml(str(config_path))
        setup_enhanced_logging(config)
        print_success(f"Configuration loaded: {config_path.name}")
        
        # Run with progress indication
        print_section("Running Holdout Evaluation")
        with progress_context("Processing holdout period", total=None):
            results = run_holdout(str(config_path))
        
        # Display key results if available
        print_section("Holdout Results")
        if isinstance(results, dict):
            key_metrics = {
                'Total Return': results.get('total_return', 0) * 100 if results.get('total_return') is not None else 0,
                'Sharpe Ratio': results.get('sharpe_ratio', 0),
                'Max Drawdown': results.get('max_drawdown', 0) * 100 if results.get('max_drawdown') is not None else 0,
                'Total Trades': results.get('total_trades', 0),
                'Win Rate': results.get('win_rate', 0) * 100 if results.get('win_rate') is not None else 0,
            }
            display_results_table(key_metrics, title="Holdout Evaluation Results")
            
            # Get run_id if available
            run_id = results.get('run_id')
            if run_id and console:
                console.print("\n[dim]Next steps:[/dim]")
                console.print(f"  • Compare with train/validation: [cyan]python -m trading_system report --run-id {run_id}[/cyan]")
        
        print_success("Holdout evaluation completed successfully")
        return 0
        
    except FileNotFoundError as e:
        print_error(f"Config file not found: {e}")
        if console:
            console.print("\n[yellow]💡 Tips:[/yellow]")
            console.print(f"  • Verify the config path exists")
            console.print_exception(show_locals=False)
        return 1
    except KeyboardInterrupt:
        print_warning("\nHoldout evaluation cancelled by user")
        return 130
    except Exception as e:
        print_error(f"Holdout evaluation failed: {e}")
        if console:
            console.print("\n[yellow]💡 Troubleshooting:[/yellow]")
            console.print(f"  • Ensure holdout period is properly configured")
            console.print(f"  • Check logs in the output directory")
            console.print_exception(show_locals=False)
        else:
            logging.exception("Holdout evaluation failed")
        return 1


def cmd_sensitivity(args: argparse.Namespace) -> int:
    """Run sensitivity analysis command.
    
    Args:
        args: Parsed command-line arguments
    
    Returns:
        Exit code (0 for success, non-zero for error)
    """
    try:
        period = getattr(args, 'period', 'train')
        metric = getattr(args, 'metric', 'sharpe_ratio')
        asset_class = getattr(args, 'asset_class', None)
        config_path = Path(args.config)
        
        # Print banner
        subtitle_parts = [f"Period: {period}", f"Metric: {metric}"]
        if asset_class:
            subtitle_parts.append(f"Asset: {asset_class}")
        print_banner(
            "Parameter Sensitivity Analysis",
            " | ".join(subtitle_parts) + f" | Config: {config_path.name}"
        )
        
        # Validate config file exists
        if not config_path.exists():
            print_error(f"Config file not found: {config_path}")
            if console:
                console.print("\n[yellow]💡 Tips:[/yellow]")
                console.print(f"  • Check the path: {config_path.absolute()}")
            return 1
        
        # Load config
        print_section("Loading Configuration")
        config = RunConfig.from_yaml(str(config_path))
        setup_enhanced_logging(config)
        print_success(f"Configuration loaded: {config_path.name}")
        
        if console:
            console.print(f"[dim]This may take several minutes depending on parameter grid size...[/dim]")
        
        # Run with progress indication
        print_section("Running Sensitivity Analysis")
        with progress_context("Exploring parameter space", total=None):
            results = run_sensitivity_analysis(
                str(config_path),
                period=period,
                metric_name=metric,
                asset_class=asset_class
            )
        
        # Display sensitivity results
        print_section("Optimal Parameters")
        display_sensitivity_results(results)
        
        print_success("Sensitivity analysis completed successfully")
        if console:
            console.print("\n[dim]Next steps:[/dim]")
            console.print(f"  • Update your config with optimal parameters")
            console.print(f"  • Run backtest with new parameters")
        return 0
        
    except FileNotFoundError as e:
        print_error(f"Config file not found: {e}")
        if console:
            console.print("\n[yellow]💡 Tips:[/yellow]")
            console.print(f"  • Verify the config path exists")
            console.print_exception(show_locals=False)
        return 1
    except ValueError as e:
        print_error(f"Configuration error: {e}")
        if console:
            console.print("\n[yellow]💡 Tips:[/yellow]")
            console.print(f"  • Validate config: [cyan]python -m trading_system config validate --path {args.config}[/cyan]")
            console.print_exception(show_locals=False)
        return 1
    except KeyboardInterrupt:
        print_warning("\nSensitivity analysis cancelled by user")
        return 130
    except Exception as e:
        print_error(f"Sensitivity analysis failed: {e}")
        if console:
            console.print("\n[yellow]💡 Troubleshooting:[/yellow]")
            console.print(f"  • Ensure parameter grids are configured in strategy config")
            console.print(f"  • Check logs in the output directory")
            console.print_exception(show_locals=False)
        else:
            logging.exception("Sensitivity analysis failed")
        return 1


def cmd_config_template(args: argparse.Namespace) -> int:
    """Generate configuration template command.
    
    Args:
        args: Parsed command-line arguments
    
    Returns:
        Exit code (0 for success, non-zero for error)
    """
    try:
        from .configs.template_generator import generate_run_config_template, generate_strategy_config_template
        
        config_type = getattr(args, 'type', 'run')
        output_path = getattr(args, 'output', None)
        
        if config_type == "run":
            content = generate_run_config_template(output_path=output_path, include_comments=True)
            if not output_path:
                print(content)
        elif config_type == "strategy":
            asset_class = getattr(args, 'asset_class', 'equity')
            content = generate_strategy_config_template(
                asset_class=asset_class,
                output_path=output_path,
                include_comments=True
            )
            if not output_path:
                print(content)
        else:
            logging.error(f"Unknown config type: {config_type}")
            return 1
        
        if output_path:
            logging.info(f"Template saved to: {output_path}")
        
        return 0
    except Exception as e:
        logging.error(f"Template generation failed: {e}", exc_info=True)
        return 1


def cmd_config_validate(args: argparse.Namespace) -> int:
    """Validate configuration file command.
    
    Args:
        args: Parsed command-line arguments
    
    Returns:
        Exit code (0 for success, non-zero for error)
    """
    try:
        from .configs.validation import validate_config_file
        
        config_path = Path(args.path)
        config_type = getattr(args, 'type', 'auto')
        
        # Print banner
        print_banner(
            "Configuration Validation",
            f"Validating: {config_path.name}"
        )
        
        # Check if file exists
        if not config_path.exists():
            print_error(f"Configuration file not found: {config_path}")
            if console:
                console.print("\n[yellow]💡 Tips:[/yellow]")
                console.print(f"  • Check the path: {config_path.absolute()}")
                console.print(f"  • Generate a template: [cyan]python -m trading_system config template --type run[/cyan]")
            return 1
        
        # Use the improved validation function
        print_section("Validating Configuration File")
        is_valid, error_message, config = validate_config_file(str(config_path), config_type)
        
        if is_valid:
            print_success(f"Configuration is valid: {config_path.name}")
            
            # Additional validation: check referenced strategy configs for run configs
            if config_type == "run" or (config_type == "auto" and hasattr(config, 'strategies')):
                from .configs.run_config import RunConfig
                if isinstance(config, RunConfig):
                    if config.strategies.equity and config.strategies.equity.enabled:
                        print_section("Validating Equity Strategy Config")
                        equity_path = Path(config.strategies.equity.config_path)
                        if not equity_path.exists():
                            print_error(f"Equity strategy config file not found: {equity_path}")
                            if console:
                                console.print(f"  • Referenced from: {config_path}")
                                console.print(f"  • Resolve path: {equity_path.absolute()}")
                            return 1
                        equity_valid, equity_error, _ = validate_config_file(
                            str(equity_path), "strategy"
                        )
                        if equity_valid:
                            print_success(f"Equity strategy config is valid: {equity_path.name}")
                        else:
                            print_error(f"Equity strategy config invalid: {equity_error}")
                            if console:
                                console.print(f"  • File: {equity_path}")
                            return 1
                    
                    if config.strategies.crypto and config.strategies.crypto.enabled:
                        print_section("Validating Crypto Strategy Config")
                        crypto_path = Path(config.strategies.crypto.config_path)
                        if not crypto_path.exists():
                            print_error(f"Crypto strategy config file not found: {crypto_path}")
                            if console:
                                console.print(f"  • Referenced from: {config_path}")
                                console.print(f"  • Resolve path: {crypto_path.absolute()}")
                            return 1
                        crypto_valid, crypto_error, _ = validate_config_file(
                            str(crypto_path), "strategy"
                        )
                        if crypto_valid:
                            print_success(f"Crypto strategy config is valid: {crypto_path.name}")
                        else:
                            print_error(f"Crypto strategy config invalid: {crypto_error}")
                            if console:
                                console.print(f"  • File: {crypto_path}")
                            return 1
            
            print_success("All configurations are valid ✓")
            return 0
        else:
            print_error("Configuration validation failed")
            if console:
                console.print(f"\n[red]Validation Errors:[/red]")
                console.print(Panel(error_message, border_style="red"))
                console.print("\n[yellow]💡 Tips:[/yellow]")
                console.print(f"  • Review the error messages above")
                console.print(f"  • Check configuration schema in docs")
                console.print(f"  • Generate template: [cyan]python -m trading_system config template --type run[/cyan]")
            else:
                print(f"\n{error_message}")
            return 1
        
    except KeyboardInterrupt:
        print_warning("\nValidation cancelled by user")
        return 130
    except Exception as e:
        print_error(f"Validation failed: {e}")
        if console:
            console.print("\n[yellow]💡 Troubleshooting:[/yellow]")
            console.print(f"  • Check that the file is valid YAML")
            console.print(f"  • Verify file permissions")
            console.print_exception(show_locals=False)
        else:
            logging.exception("Validation failed")
        return 1


def cmd_config_docs(args: argparse.Namespace) -> int:
    """Generate configuration documentation command.
    
    Args:
        args: Parsed command-line arguments
    
    Returns:
        Exit code (0 for success, non-zero for error)
    """
    try:
        from .configs.doc_generator import generate_config_docs
        
        output_path = getattr(args, 'output', None)
        content = generate_config_docs(output_path=output_path)
        
        if not output_path:
            print(content)
        else:
            print_success(f"Documentation saved to: {output_path}")
        
        return 0
    except Exception as e:
        print_error(f"Documentation generation failed: {e}")
        if console:
            console.print_exception()
        else:
            logging.exception("Documentation generation failed")
        return 1


def cmd_config_migrate(args: argparse.Namespace) -> int:
    """Migrate configuration file to a newer version.
    
    Args:
        args: Parsed command-line arguments
    
    Returns:
        Exit code (0 for success, non-zero for error)
    """
    try:
        from .configs.migration import migrate_config, backup_config, CURRENT_CONFIG_VERSION
        
        config_path = Path(args.path)
        target_version = getattr(args, 'target_version', None) or CURRENT_CONFIG_VERSION
        output_path = getattr(args, 'output', None)
        create_backup = getattr(args, 'backup', False)
        dry_run = getattr(args, 'dry_run', False)
        
        # Print banner
        print_banner(
            "Configuration Migration",
            f"Migrating: {config_path.name}"
        )
        
        # Check if file exists
        if not config_path.exists():
            print_error(f"Configuration file not found: {config_path}")
            return 1
        
        # Create backup if requested
        if create_backup and not dry_run:
            print_section("Creating Backup")
            try:
                backup_path = backup_config(str(config_path))
                print_success(f"Backup created: {backup_path}")
            except Exception as e:
                print_error(f"Failed to create backup: {e}")
                return 1
        
        # Perform migration
        print_section("Migrating Configuration")
        success, message, migrated_data = migrate_config(
            str(config_path),
            target_version=target_version,
            output_path=output_path,
            dry_run=dry_run
        )
        
        if success:
            print_success(message)
            if dry_run and console:
                console.print("\n[yellow]Note:[/yellow] This was a dry run. Use without --dry-run to apply changes.")
            return 0
        else:
            print_error(f"Migration failed: {message}")
            if console:
                console.print("\n[yellow]💡 Tips:[/yellow]")
                console.print("  • Check the error message above")
                console.print("  • Ensure the config file is valid YAML")
                console.print("  • Use --backup to create a backup before migrating")
            return 1
        
    except KeyboardInterrupt:
        print_warning("\nMigration cancelled by user")
        return 130
    except Exception as e:
        print_error(f"Migration failed: {e}")
        if console:
            console.print_exception()
        else:
            logging.exception("Migration failed")
        return 1


def cmd_config_version(args: argparse.Namespace) -> int:
    """Check version of a configuration file.
    
    Args:
        args: Parsed command-line arguments
    
    Returns:
        Exit code (0 for success, non-zero for error)
    """
    try:
        from .configs.migration import check_config_version, CURRENT_CONFIG_VERSION
        
        config_path = Path(args.path)
        
        # Print banner
        print_banner(
            "Configuration Version Check",
            f"Checking: {config_path.name}"
        )
        
        # Check if file exists
        if not config_path.exists():
            print_error(f"Configuration file not found: {config_path}")
            return 1
        
        # Check version
        version, is_current = check_config_version(str(config_path))
        
        print_section("Version Information")
        if console:
            table = Table(box=box.ROUNDED, show_header=True)
            table.add_column("Property", style="cyan")
            table.add_column("Value", style="green")
            table.add_row("Config Version", version)
            table.add_row("Current Version", CURRENT_CONFIG_VERSION)
            table.add_row("Status", "✅ Current" if is_current else "⚠️  Outdated")
            console.print(table)
            
            if not is_current:
                console.print("\n[yellow]💡 Tip:[/yellow]")
                console.print(f"  • Migrate to current version: [cyan]python -m trading_system config migrate --path {config_path}[/cyan]")
        else:
            print(f"Config Version: {version}")
            print(f"Current Version: {CURRENT_CONFIG_VERSION}")
            print(f"Status: {'Current' if is_current else 'Outdated'}")
        
        return 0 if is_current else 1
        
    except Exception as e:
        print_error(f"Version check failed: {e}")
        if console:
            console.print_exception()
        else:
            logging.exception("Version check failed")
        return 1


def cmd_config_schema(args: argparse.Namespace) -> int:
    """Export JSON Schema for configuration models.
    
    Args:
        args: Parsed command-line arguments
    
    Returns:
        Exit code (0 for success, non-zero for error)
    """
    try:
        from .configs.validation import export_json_schema
        from .configs.run_config import RunConfig
        from .configs.strategy_config import StrategyConfig
        
        config_type = getattr(args, 'type', 'run')
        output_path = getattr(args, 'output', None)
        
        if config_type == "run":
            schema = export_json_schema(RunConfig, output_path=output_path)
            schema_name = "RunConfig"
        elif config_type == "strategy":
            schema = export_json_schema(StrategyConfig, output_path=output_path)
            schema_name = "StrategyConfig"
        else:
            print_error(f"Unknown config type: {config_type}")
            return 1
        
        if not output_path:
            import json
            print(json.dumps(schema, indent=2))
        else:
            print_success(f"JSON Schema for {schema_name} saved to: {output_path}")
        
        return 0
    except Exception as e:
        print_error(f"Schema export failed: {e}")
        if console:
            console.print_exception()
        else:
            logging.exception("Schema export failed")
        return 1


def cmd_config_wizard(args: argparse.Namespace) -> int:
    """Interactive configuration wizard command.
    
    Args:
        args: Parsed command-line arguments
    
    Returns:
        Exit code (0 for success, non-zero for error)
    """
    try:
        from .cli.config_wizard import run_wizard
        
        config_type = getattr(args, 'type', 'run')
        output_path = getattr(args, 'output', None)
        
        run_wizard(config_type=config_type, output_path=output_path)
        return 0
    except Exception as e:
        logging.error(f"Wizard failed: {e}", exc_info=True)
        return 1


def cmd_strategy_template(args: argparse.Namespace) -> int:
    """Generate strategy class template command.
    
    Args:
        args: Parsed command-line arguments
    
    Returns:
        Exit code (0 for success, non-zero for error)
    """
    try:
        from .strategies.strategy_template_generator import generate_strategy_template
        
        strategy_name = args.name
        strategy_type = getattr(args, 'type', 'custom')
        asset_class = getattr(args, 'asset_class', 'equity')
        output_path = getattr(args, 'output', None)
        directory = getattr(args, 'directory', None)
        
        # Print banner
        print_banner(
            "Generating Strategy Template",
            f"Name: {strategy_name} | Type: {strategy_type} | Asset: {asset_class}"
        )
        
        # Generate template
        print_section("Generating Strategy Class")
        content = generate_strategy_template(
            strategy_name=strategy_name,
            strategy_type=strategy_type,
            asset_class=asset_class,
            output_path=output_path,
            directory=directory
        )
        
        # Determine output path for display
        if output_path:
            file_path = Path(output_path)
        else:
            if directory:
                strategy_dir = Path(directory)
            else:
                if strategy_type == "custom":
                    strategy_dir = Path("trading_system/strategies/custom")
                else:
                    strategy_dir = Path(f"trading_system/strategies/{strategy_type}")
            filename = f"{strategy_name}_{asset_class}.py"
            file_path = strategy_dir / filename
        
        print_success(f"Strategy template generated: {file_path}")
        
        if console:
            console.print("\n[dim]Next steps:[/dim]")
            console.print(f"  1. Review and implement the TODO sections in {file_path.name}")
            console.print(f"  2. Register the strategy in strategy_registry.py")
            console.print(f"  3. Create a strategy config YAML file")
            console.print(f"  4. Test the strategy with a backtest")
        
        return 0
    except ValueError as e:
        print_error(f"Invalid arguments: {e}")
        if console:
            console.print("\n[yellow]💡 Tips:[/yellow]")
            console.print(f"  • Strategy name should be lowercase with underscores (e.g., 'my_custom_strategy')")
            console.print(f"  • Strategy type must be one of: momentum, mean_reversion, factor, multi_timeframe, pairs, custom")
            console.print(f"  • Asset class must be 'equity' or 'crypto'")
        return 1
    except Exception as e:
        print_error(f"Strategy template generation failed: {e}")
        if console:
            console.print_exception()
        else:
            logging.exception("Strategy template generation failed")
        return 1


def cmd_strategy_create(args: argparse.Namespace) -> int:
    """Create strategy template command (with optional interactive wizard).
    
    Args:
        args: Parsed command-line arguments
    
    Returns:
        Exit code (0 for success, non-zero for error)
    """
    try:
        from .strategies.strategy_template_generator import generate_strategy_template
        from .cli.strategy_wizard import run_strategy_wizard
        
        # If name is provided, use non-interactive mode
        # Otherwise, run interactive wizard
        strategy_name = getattr(args, 'name', None)
        if strategy_name:
            # Non-interactive mode
            strategy_type = getattr(args, 'type', 'custom')
            asset_class = getattr(args, 'asset_class', 'equity')
            output_path = getattr(args, 'output', None)
            directory = getattr(args, 'directory', None)
        else:
            # Interactive wizard mode
            print_banner("Strategy Creation Wizard", "Interactive strategy template generator")
            wizard_params = run_strategy_wizard()
            strategy_name = wizard_params['strategy_name']
            strategy_type = wizard_params['strategy_type']
            asset_class = wizard_params['asset_class']
            output_path = wizard_params['output_path']
            directory = wizard_params['directory']
        
        # Print banner
        print_banner(
            "Generating Strategy Template",
            f"Name: {strategy_name} | Type: {strategy_type} | Asset: {asset_class}"
        )
        
        # Generate template
        print_section("Generating Strategy Class")
        content = generate_strategy_template(
            strategy_name=strategy_name,
            strategy_type=strategy_type,
            asset_class=asset_class,
            output_path=output_path,
            directory=directory
        )
        
        # Determine output path for display
        if output_path:
            file_path = Path(output_path)
        else:
            if directory:
                strategy_dir = Path(directory)
            else:
                if strategy_type == "custom":
                    strategy_dir = Path("trading_system/strategies/custom")
                else:
                    strategy_dir = Path(f"trading_system/strategies/{strategy_type}")
            filename = f"{strategy_name}_{asset_class}.py"
            file_path = strategy_dir / filename
        
        print_success(f"Strategy template generated: {file_path}")
        
        if console:
            console.print("\n[dim]Next steps:[/dim]")
            console.print(f"  1. Review and implement the TODO sections in {file_path.name}")
            console.print(f"  2. Register the strategy in strategy_registry.py")
            console.print(f"  3. Create a strategy config YAML file")
            console.print(f"  4. Test the strategy with a backtest")
        
        return 0
    except ValueError as e:
        print_error(f"Invalid arguments: {e}")
        if console:
            console.print("\n[yellow]💡 Tips:[/yellow]")
            console.print(f"  • Strategy name should be lowercase with underscores (e.g., 'my_custom_strategy')")
            console.print(f"  • Strategy type must be one of: momentum, mean_reversion, factor, multi_timeframe, pairs, custom")
            console.print(f"  • Asset class must be 'equity' or 'crypto'")
        return 1
    except KeyboardInterrupt:
        print_warning("\nStrategy creation cancelled by user")
        return 130
    except Exception as e:
        print_error(f"Strategy template generation failed: {e}")
        if console:
            console.print_exception()
        else:
            logging.exception("Strategy template generation failed")
        return 1


def cmd_report(args: argparse.Namespace) -> int:
    """Generate report command.
    
    Args:
        args: Parsed command-line arguments
    
    Returns:
        Exit code (0 for success, non-zero for error)
    """
    try:
        # Setup basic logging if not already configured
        if not logging.getLogger().handlers:
            logging.basicConfig(
                level=logging.INFO,
                format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
        
        base_path = getattr(args, 'base_path', 'results/')
        run_id = args.run_id
        base_path_obj = Path(base_path)
        run_dir = base_path_obj / run_id
        
        # Print banner
        print_banner(
            "Generating Reports",
            f"Run ID: {run_id} | Output: {base_path}"
        )
        
        # Validate run directory exists
        if not run_dir.exists():
            print_error(f"Run directory not found: {run_dir}")
            if console:
                console.print("\n[yellow]💡 Tips:[/yellow]")
                console.print(f"  • Check if run_id is correct: {run_id}")
                console.print(f"  • Available runs in: {base_path_obj.absolute()}")
                if base_path_obj.exists():
                    try:
                        runs = [d.name for d in base_path_obj.iterdir() if d.is_dir()]
                        if runs:
                            console.print(f"  • Available run IDs: {', '.join(runs[:5])}")
                            if len(runs) > 5:
                                console.print(f"  • ... and {len(runs) - 5} more")
                    except:
                        pass
                console.print(f"  • Run a backtest first: [cyan]python -m trading_system backtest -c <config>[/cyan]")
            return 1
        
        # Create report generator
        print_section("Loading Report Data")
        with progress_context("Loading backtest results", total=None):
            report_gen = ReportGenerator(base_path=base_path, run_id=run_id)
        print_success("Data loaded successfully")
        
        # Generate summary report
        print_section("Generating Reports")
        with progress_context("Generating summary report", total=None):
            summary_path = report_gen.generate_summary_report()
        print_success(f"Summary report: {Path(summary_path).relative_to(Path.cwd())}")
        
        # Generate comparison report (if multiple periods available)
        try:
            with progress_context("Generating comparison report", total=None):
                comparison_path = report_gen.generate_comparison_report()
            print_success(f"Comparison report: {Path(comparison_path).relative_to(Path.cwd())}")
        except ValueError as e:
            print_warning(f"Could not generate comparison report: {e}")
        
        # Print summary to console
        print_section("Report Summary")
        report_gen.print_summary()
        
        print_success("Report generation completed successfully")
        if console:
            console.print(f"\n[dim]View reports:[/dim] {run_dir.absolute()}")
            console.print(f"Launch dashboard: [cyan]python -m trading_system dashboard --run-id {run_id}[/cyan]")
        return 0
        
    except FileNotFoundError as e:
        print_error(f"Run directory not found: {e}")
        if console:
            console.print("\n[yellow]💡 Tips:[/yellow]")
            console.print(f"  • Verify the run_id exists in the base_path")
            console.print(f"  • Run a backtest first to generate results")
            console.print_exception(show_locals=False)
        return 1
    except KeyboardInterrupt:
        print_warning("\nReport generation cancelled by user")
        return 130
    except Exception as e:
        print_error(f"Report generation failed: {e}")
        if console:
            console.print("\n[yellow]💡 Troubleshooting:[/yellow]")
            console.print(f"  • Ensure the run completed successfully")
            console.print(f"  • Check logs in the run directory")
            console.print_exception(show_locals=False)
        else:
            logging.exception("Report generation failed")
        return 1


def cmd_dashboard(args: argparse.Namespace) -> int:
    """Launch interactive dashboard command.
    
    Args:
        args: Parsed command-line arguments
    
    Returns:
        Exit code (0 for success, non-zero for error)
    """
    try:
        import subprocess
        import sys
        
        base_path = getattr(args, 'base_path', 'results/')
        run_id = args.run_id
        
        if console:
            console.print(f"\n[bold cyan]Launching Dashboard[/bold cyan]")
            console.print(f"Run ID: {run_id}")
            console.print(f"Base Path: {base_path}\n")
        
        # Check if streamlit is available
        try:
            import streamlit
        except ImportError:
            print_error("Streamlit is not installed. Install it with: pip install streamlit")
            if console:
                console.print("\n[yellow]Tip:[/yellow] Streamlit is an optional dependency.")
                console.print("Add it to requirements.txt and install it to use the dashboard.")
            return 1
        
        # Get the dashboard file path
        from . import reporting
        import os
        dashboard_file = os.path.join(
            os.path.dirname(reporting.__file__),
            "dashboard.py"
        )
        
        # Build command - streamlit run <file> -- <args>
        cmd = [
            sys.executable,
            "-m",
            "streamlit",
            "run",
            dashboard_file,
            "--",
            "--base_path",
            base_path,
            "--run_id",
            run_id
        ]
        
        print_info("Starting Streamlit dashboard...")
        print_info("The dashboard will open in your browser automatically.")
        print_info("Press Ctrl+C to stop the dashboard.\n")
        
        # Run streamlit
        subprocess.run(cmd)
        
        return 0
        
    except KeyboardInterrupt:
        print_warning("\nDashboard stopped by user")
        return 130
    except FileNotFoundError as e:
        print_error(f"Run directory not found: {e}")
        if console:
            console.print(f"\n[yellow]Tip:[/yellow] Make sure the run_id exists in the base_path directory.")
            console.print(f"Available runs can be found in: {Path(base_path).absolute()}")
        return 1
    except Exception as e:
        print_error(f"Dashboard launch failed: {e}")
        if console:
            console.print_exception()
        else:
            logging.exception("Dashboard launch failed")
        return 1


def main() -> int:
    """Main CLI entry point.
    
    Returns:
        Exit code (0 for success, non-zero for error)
    """
    parser = argparse.ArgumentParser(
        description="Trading System CLI - Advanced backtesting and validation framework",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
📚 Examples:

  # Run backtest (alias: bt)
  python -m trading_system backtest --config EXAMPLE_CONFIGS/run_config.yaml
  python -m trading_system bt -c EXAMPLE_CONFIGS/run_config.yaml --period validation

  # Run validation suite (alias: val)
  python -m trading_system validate --config EXAMPLE_CONFIGS/run_config.yaml
  python -m trading_system val -c EXAMPLE_CONFIGS/run_config.yaml

  # Run holdout evaluation (alias: ho)
  python -m trading_system holdout --config EXAMPLE_CONFIGS/run_config.yaml

  # Parameter sensitivity analysis (alias: sens)
  python -m trading_system sensitivity -c EXAMPLE_CONFIGS/run_config.yaml --metric sharpe_ratio

  # Generate reports (alias: rep)
  python -m trading_system report --run-id <run_id>
  python -m trading_system rep -r <run_id> -b results/

  # Launch interactive dashboard (alias: dash)
  python -m trading_system dashboard --run-id <run_id>
  python -m trading_system dash -r <run_id>

  # Configuration management (alias: cfg)
  python -m trading_system config template --type run --output my_config.yaml
  python -m trading_system config validate --path my_config.yaml
  python -m trading_system config wizard --type run
  python -m trading_system config docs --output CONFIG_DOCS.md

  # Strategy template generation
  python -m trading_system strategy create --name my_custom_strategy --type momentum --asset-class equity
  python -m trading_system strategy create  # Interactive wizard
  python -m trading_system strategy-template --name my_strategy -t custom -a crypto  # Legacy alias

💡 Tips:
  • Use aliases for faster commands (bt, val, ho, sens, rep, dash, cfg)
  • Use --help with any command for detailed options
  • Run validation before backtest to check strategy health
  • Use config wizard for interactive configuration setup

📖 For more information, see the documentation in agent-files/.
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to run', metavar='COMMAND')
    
    # Backtest command (with alias 'bt')
    backtest_parser = subparsers.add_parser(
        'backtest',
        aliases=['bt'],
        help='Run backtest on specified period',
        description='Run a backtest on train, validation, or holdout period'
    )
    backtest_parser.add_argument(
        '--config',
        '-c',
        type=str,
        required=True,
        help='Path to run_config.yaml file'
    )
    backtest_parser.add_argument(
        '--period',
        '-p',
        type=str,
        choices=['train', 'validation', 'holdout'],
        default='train',
        help='Period to run (default: train)'
    )
    backtest_parser.set_defaults(func=cmd_backtest)
    
    # Validate command (with alias 'val')
    validate_parser = subparsers.add_parser(
        'validate',
        aliases=['val'],
        help='Run validation suite',
        description='Run full validation suite including statistical tests and stress tests'
    )
    validate_parser.add_argument(
        '--config',
        '-c',
        type=str,
        required=True,
        help='Path to run_config.yaml file'
    )
    validate_parser.set_defaults(func=cmd_validate)
    
    # Holdout command (with alias 'ho')
    holdout_parser = subparsers.add_parser(
        'holdout',
        aliases=['ho'],
        help='Run holdout evaluation',
        description='Run backtest on holdout period (out-of-sample test)'
    )
    holdout_parser.add_argument(
        '--config',
        '-c',
        type=str,
        required=True,
        help='Path to run_config.yaml file'
    )
    holdout_parser.set_defaults(func=cmd_holdout)
    
    # Sensitivity command (with alias 'sens')
    sensitivity_parser = subparsers.add_parser(
        'sensitivity',
        aliases=['sens'],
        help='Run parameter sensitivity analysis',
        description='Run grid search to find optimal parameters'
    )
    sensitivity_parser.add_argument(
        '--config',
        '-c',
        type=str,
        required=True,
        help='Path to run_config.yaml file'
    )
    sensitivity_parser.add_argument(
        '--period',
        '-p',
        type=str,
        choices=['train', 'validation', 'holdout'],
        default='train',
        help='Period to run backtests on (default: train)'
    )
    sensitivity_parser.add_argument(
        '--metric',
        '-m',
        type=str,
        default='sharpe_ratio',
        help='Metric to optimize (default: sharpe_ratio)'
    )
    sensitivity_parser.add_argument(
        '--asset-class',
        '-a',
        type=str,
        choices=['equity', 'crypto'],
        default=None,
        help='Asset class to analyze (default: all enabled)'
    )
    sensitivity_parser.set_defaults(func=cmd_sensitivity)
    
    # Report command (with alias 'rep')
    report_parser = subparsers.add_parser(
        'report',
        aliases=['rep'],
        help='Generate reports from completed run',
        description='Generate summary and comparison reports from a completed backtest run'
    )
    report_parser.add_argument(
        '--run-id',
        '-r',
        type=str,
        required=True,
        help='Run ID to generate report for'
    )
    report_parser.add_argument(
        '--base-path',
        '-b',
        type=str,
        default='results/',
        help='Base path for results directory (default: results/)'
    )
    report_parser.set_defaults(func=cmd_report)
    
    # Dashboard command (with alias 'dash')
    dashboard_parser = subparsers.add_parser(
        'dashboard',
        aliases=['dash'],
        help='Launch interactive dashboard',
        description='Launch Streamlit interactive dashboard for visualizing backtest results'
    )
    dashboard_parser.add_argument(
        '--run-id',
        '-r',
        type=str,
        required=True,
        help='Run ID to visualize'
    )
    dashboard_parser.add_argument(
        '--base-path',
        '-b',
        type=str,
        default='results/',
        help='Base path for results directory (default: results/)'
    )
    dashboard_parser.set_defaults(func=cmd_dashboard)
    
    # Config command group
    config_parser = subparsers.add_parser(
        'config',
        aliases=['cfg'],
        help='Configuration management commands',
        description='Configuration management: templates, validation, docs, and interactive wizard'
    )
    config_subparsers = config_parser.add_subparsers(dest='config_command', help='Config command to run')
    
    # Config template command
    template_parser = config_subparsers.add_parser(
        'template',
        help='Generate configuration template file'
    )
    template_parser.add_argument(
        '--type',
        type=str,
        choices=['run', 'strategy'],
        default='run',
        help='Type of configuration template (default: run)'
    )
    template_parser.add_argument(
        '--asset-class',
        type=str,
        choices=['equity', 'crypto'],
        default='equity',
        help='Asset class for strategy template (default: equity)'
    )
    template_parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output file path (default: print to stdout)'
    )
    template_parser.set_defaults(func=cmd_config_template)
    
    # Config validate command
    validate_config_parser = config_subparsers.add_parser(
        'validate',
        help='Validate a configuration file'
    )
    validate_config_parser.add_argument(
        '--path',
        type=str,
        required=True,
        help='Path to configuration file to validate'
    )
    validate_config_parser.add_argument(
        '--type',
        type=str,
        choices=['run', 'strategy', 'auto'],
        default='auto',
        help='Configuration type (default: auto-detect)'
    )
    validate_config_parser.set_defaults(func=cmd_config_validate)
    
    # Config docs command
    docs_parser = config_subparsers.add_parser(
        'docs',
        help='Generate configuration documentation'
    )
    docs_parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output file path (default: print to stdout)'
    )
    docs_parser.set_defaults(func=cmd_config_docs)
    
    # Config schema command
    schema_parser = config_subparsers.add_parser(
        'schema',
        help='Export JSON Schema for configuration models'
    )
    schema_parser.add_argument(
        '--type',
        type=str,
        choices=['run', 'strategy'],
        default='run',
        help='Type of configuration schema (default: run)'
    )
    schema_parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output file path (default: print to stdout)'
    )
    schema_parser.set_defaults(func=cmd_config_schema)
    
    # Config wizard command
    wizard_parser = config_subparsers.add_parser(
        'wizard',
        help='Interactive configuration wizard'
    )
    wizard_parser.add_argument(
        '--type',
        type=str,
        choices=['run', 'strategy'],
        default='run',
        help='Type of configuration to create (default: run)'
    )
    wizard_parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output file path (default: prompt for path)'
    )
    wizard_parser.set_defaults(func=cmd_config_wizard)
    
    # Config migrate command
    migrate_parser = config_subparsers.add_parser(
        'migrate',
        help='Migrate configuration file to a newer version'
    )
    migrate_parser.add_argument(
        '--path',
        '-p',
        type=str,
        required=True,
        help='Path to configuration file to migrate'
    )
    migrate_parser.add_argument(
        '--target-version',
        '-t',
        type=str,
        default=None,
        help='Target version (defaults to current version)'
    )
    migrate_parser.add_argument(
        '--output',
        '-o',
        type=str,
        default=None,
        help='Output path for migrated config (defaults to overwrite original)'
    )
    migrate_parser.add_argument(
        '--backup',
        '-b',
        action='store_true',
        help='Create backup before migrating'
    )
    migrate_parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be migrated without making changes'
    )
    migrate_parser.set_defaults(func=cmd_config_migrate)
    
    # Config version command
    version_parser = config_subparsers.add_parser(
        'version',
        help='Check version of a configuration file'
    )
    version_parser.add_argument(
        '--path',
        '-p',
        type=str,
        required=True,
        help='Path to configuration file to check'
    )
    version_parser.set_defaults(func=cmd_config_version)
    
    # Strategy command group
    strategy_parser = subparsers.add_parser(
        'strategy',
        help='Strategy management commands',
        description='Strategy management: create templates and manage strategies'
    )
    strategy_subparsers = strategy_parser.add_subparsers(dest='strategy_command', help='Strategy command to run')
    
    # Strategy create command
    strategy_create_parser = strategy_subparsers.add_parser(
        'create',
        help='Create a new strategy template',
        description='Create a new strategy class template (interactive wizard or direct)'
    )
    strategy_create_parser.add_argument(
        '--name',
        '-n',
        type=str,
        default=None,
        help='Strategy name (e.g., "my_custom_strategy"). If not provided, launches interactive wizard.'
    )
    strategy_create_parser.add_argument(
        '--type',
        '-t',
        type=str,
        choices=['momentum', 'mean_reversion', 'factor', 'multi_timeframe', 'pairs', 'custom'],
        default='custom',
        help='Strategy type (default: custom, only used with --name)'
    )
    strategy_create_parser.add_argument(
        '--asset-class',
        '-a',
        type=str,
        choices=['equity', 'crypto'],
        default='equity',
        help='Asset class (default: equity, only used with --name)'
    )
    strategy_create_parser.add_argument(
        '--output',
        '-o',
        type=str,
        default=None,
        help='Output file path (default: auto-generate in strategies directory)'
    )
    strategy_create_parser.add_argument(
        '--directory',
        '-d',
        type=str,
        default=None,
        help='Directory to create strategy in (default: strategies/{type})'
    )
    strategy_create_parser.set_defaults(func=cmd_strategy_create)
    
    # Strategy template command (kept for backward compatibility)
    strategy_template_parser = subparsers.add_parser(
        'strategy-template',
        aliases=['st'],
        help='Generate strategy class template file (deprecated: use "strategy create")',
        description='Generate a new strategy class template with all required methods'
    )
    strategy_template_parser.add_argument(
        '--name',
        '-n',
        type=str,
        required=True,
        help='Strategy name (e.g., "my_custom_strategy")'
    )
    strategy_template_parser.add_argument(
        '--type',
        '-t',
        type=str,
        choices=['momentum', 'mean_reversion', 'factor', 'multi_timeframe', 'pairs', 'custom'],
        default='custom',
        help='Strategy type (default: custom)'
    )
    strategy_template_parser.add_argument(
        '--asset-class',
        '-a',
        type=str,
        choices=['equity', 'crypto'],
        default='equity',
        help='Asset class (default: equity)'
    )
    strategy_template_parser.add_argument(
        '--output',
        '-o',
        type=str,
        default=None,
        help='Output file path (default: auto-generate in strategies directory)'
    )
    strategy_template_parser.add_argument(
        '--directory',
        '-d',
        type=str,
        default=None,
        help='Directory to create strategy in (default: strategies/{type})'
    )
    strategy_template_parser.set_defaults(func=cmd_strategy_template)
    
    # Parse arguments
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        if console:
            console.print("\n[yellow]Tip:[/yellow] Use '--help' with any command for more information.")
        return 1
    
    # Handle config subcommands
    if args.command == 'config' or args.command == 'cfg':
        if not args.config_command:
            config_parser.print_help()
            return 1
        return args.func(args)
    
    # Handle strategy subcommands
    if args.command == 'strategy':
        if not args.strategy_command:
            strategy_parser.print_help()
            return 1
        return args.func(args)
    
    # Run command
    try:
        return args.func(args)
    except KeyboardInterrupt:
        print_warning("\nOperation cancelled by user")
        return 130
    except Exception as e:
        print_error(f"Unexpected error: {e}")
        if console:
            console.print_exception()
        return 1


if __name__ == '__main__':
    sys.exit(main())
