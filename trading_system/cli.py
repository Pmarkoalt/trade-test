"""Command-line interface for the trading system."""

import argparse
import sys
import logging
from pathlib import Path
from typing import Optional

from .configs.run_config import RunConfig
from .integration.runner import run_backtest, run_validation, run_holdout


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
        config = RunConfig.from_yaml(args.config)
        setup_logging(config)
        
        period = getattr(args, 'period', 'train')
        logging.info(f"Running backtest: period={period}")
        
        results = run_backtest(args.config, period=period)
        
        logging.info("Backtest completed successfully")
        return 0
        
    except Exception as e:
        logging.error(f"Backtest failed: {e}", exc_info=True)
        return 1


def cmd_validate(args: argparse.Namespace) -> int:
    """Run validation suite command.
    
    Args:
        args: Parsed command-line arguments
    
    Returns:
        Exit code (0 for success, non-zero for error)
    """
    try:
        config = RunConfig.from_yaml(args.config)
        setup_logging(config)
        
        logging.info("Running validation suite")
        
        results = run_validation(args.config)
        
        logging.info("Validation suite completed successfully")
        return 0
        
    except Exception as e:
        logging.error(f"Validation failed: {e}", exc_info=True)
        return 1


def cmd_holdout(args: argparse.Namespace) -> int:
    """Run holdout evaluation command.
    
    Args:
        args: Parsed command-line arguments
    
    Returns:
        Exit code (0 for success, non-zero for error)
    """
    try:
        config = RunConfig.from_yaml(args.config)
        setup_logging(config)
        
        logging.info("Running holdout evaluation")
        
        results = run_holdout(args.config)
        
        logging.info("Holdout evaluation completed successfully")
        return 0
        
    except Exception as e:
        logging.error(f"Holdout evaluation failed: {e}", exc_info=True)
        return 1


def cmd_report(args: argparse.Namespace) -> int:
    """Generate report command.
    
    Args:
        args: Parsed command-line arguments
    
    Returns:
        Exit code (0 for success, non-zero for error)
    """
    try:
        # TODO: Implement report generation
        # This should load results from run_id and generate reports
        logging.warning("Report generation not yet fully implemented")
        logging.info(f"Report generation requested for run_id: {args.run_id}")
        return 0
        
    except Exception as e:
        logging.error(f"Report generation failed: {e}", exc_info=True)
        return 1


def main() -> int:
    """Main CLI entry point.
    
    Returns:
        Exit code (0 for success, non-zero for error)
    """
    parser = argparse.ArgumentParser(
        description="Trading system CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Backtest command
    backtest_parser = subparsers.add_parser(
        'backtest',
        help='Run backtest'
    )
    backtest_parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to run_config.yaml'
    )
    backtest_parser.add_argument(
        '--period',
        type=str,
        choices=['train', 'validation', 'holdout'],
        default='train',
        help='Period to run (default: train)'
    )
    backtest_parser.set_defaults(func=cmd_backtest)
    
    # Validate command
    validate_parser = subparsers.add_parser(
        'validate',
        help='Run validation suite'
    )
    validate_parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to run_config.yaml'
    )
    validate_parser.set_defaults(func=cmd_validate)
    
    # Holdout command
    holdout_parser = subparsers.add_parser(
        'holdout',
        help='Run holdout evaluation'
    )
    holdout_parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to run_config.yaml'
    )
    holdout_parser.set_defaults(func=cmd_holdout)
    
    # Report command
    report_parser = subparsers.add_parser(
        'report',
        help='Generate reports from a completed run'
    )
    report_parser.add_argument(
        '--run-id',
        type=str,
        required=True,
        help='Run ID to generate report for'
    )
    report_parser.set_defaults(func=cmd_report)
    
    # Parse arguments
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    # Run command
    return args.func(args)


if __name__ == '__main__':
    sys.exit(main())

