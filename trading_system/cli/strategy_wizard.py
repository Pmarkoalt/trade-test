"""Interactive strategy creation wizard."""

import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Optional

if TYPE_CHECKING:
    from rich.console import Console

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.prompt import Confirm, Prompt
    from rich.text import Text

    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False


class StrategyWizard:
    """Interactive wizard for creating strategy templates."""

    def __init__(self, use_rich: bool = True):
        """Initialize the wizard.

        Args:
            use_rich: Whether to use rich for enhanced output (if available)
        """
        self.use_rich = use_rich and RICH_AVAILABLE
        if self.use_rich:
            self.console: Optional["Console"] = Console()  # type: ignore[assignment]
        else:
            self.console: Optional["Console"] = None

    def print(self, message: str, style: Optional[str] = None):
        """Print a message.

        Args:
            message: Message to print
            style: Optional rich style
        """
        if self.use_rich and style:
            self.console.print(message, style=style)
        else:
            print(message)

    def prompt(self, question: str, default: Optional[str] = None, required: bool = True) -> str:
        """Prompt for user input.

        Args:
            question: Question to ask
            default: Default value
            required: Whether input is required

        Returns:
            User input string
        """
        if self.use_rich:
            return Prompt.ask(question, default=default or "")
        else:
            prompt_text = question
            if default:
                prompt_text += f" [{default}]"
            prompt_text += ": "

            while True:
                value = input(prompt_text).strip()
                if value or not required:
                    return value or default or ""
                print("This field is required. Please enter a value.")

    def confirm(self, question: str, default: bool = True) -> bool:
        """Ask a yes/no question.

        Args:
            question: Question to ask
            default: Default value

        Returns:
            Boolean answer
        """
        if self.use_rich:
            return Confirm.ask(question, default=default)
        else:
            prompt_text = question
            prompt_text += " (y/n) [y]" if default else " (y/n) [n]"
            prompt_text += ": "

            while True:
                value = input(prompt_text).strip().lower()
                if not value:
                    return default
                if value in ["y", "yes"]:
                    return True
                if value in ["n", "no"]:
                    return False
                print("Please enter 'y' or 'n'")

    def choose_option(self, question: str, options: list[str], default: Optional[str] = None) -> str:
        """Choose from a list of options.

        Args:
            question: Question to ask
            options: List of option strings
            default: Default option index or value

        Returns:
            Selected option
        """
        if self.use_rich:
            # Display options
            self.print(f"\n{question}", style="bold cyan")
            for i, option in enumerate(options, 1):
                marker = "→" if option == default else " "
                self.print(f"  {marker} {i}. {option}")

            while True:
                choice = self.prompt("\nEnter choice", default=str(options.index(default) + 1) if default else None)
                try:
                    idx = int(choice) - 1
                    if 0 <= idx < len(options):
                        return options[idx]
                    self.print(f"Please enter a number between 1 and {len(options)}", style="red")
                except ValueError:
                    # Try to match by name
                    if choice in options:
                        return choice
                    self.print(f"Invalid choice. Please enter a number or option name.", style="red")
        else:
            # Fallback for non-rich
            print(f"\n{question}")
            for i, option in enumerate(options, 1):
                marker = "→" if option == default else " "
                print(f"  {marker} {i}. {option}")

            while True:
                choice = input(f"\nEnter choice [1-{len(options)}]: ").strip()
                try:
                    idx = int(choice) - 1
                    if 0 <= idx < len(options):
                        return options[idx]
                    print(f"Please enter a number between 1 and {len(options)}")
                except ValueError:
                    if choice in options:
                        return choice
                    print(f"Invalid choice. Please enter a number or option name.")

    def run_wizard(self) -> Dict[str, Any]:
        """Run the strategy creation wizard.

        Returns:
            Dictionary with strategy creation parameters
        """
        if self.use_rich:
            self.console.print(
                Panel(
                    "[bold cyan]Strategy Template Generator[/bold cyan]\n"
                    "This wizard will help you create a new strategy class template.",
                    title="Welcome",
                    border_style="cyan",
                )
            )
        else:
            self.print("\n" + "=" * 60)
            self.print("Strategy Template Generator")
            self.print("=" * 60 + "\n")

        # Strategy name
        self.print("\nStrategy Information", style="bold yellow")
        self.print("-" * 40)

        strategy_name = self.prompt("Strategy name (lowercase with underscores, e.g., 'my_custom_strategy')", required=True)

        # Validate strategy name format
        if not strategy_name.replace("_", "").replace("-", "").isalnum():
            self.print("Warning: Strategy name should contain only letters, numbers, underscores, and hyphens", style="yellow")

        # Strategy type
        strategy_types = ["momentum", "mean_reversion", "factor", "multi_timeframe", "pairs", "custom"]

        self.print("\nStrategy Type", style="bold yellow")
        self.print("-" * 40)
        self.print("Choose the base strategy type:")
        self.print("  • momentum: Momentum/breakout strategies")
        self.print("  • mean_reversion: Mean reversion strategies")
        self.print("  • factor: Factor-based strategies")
        self.print("  • multi_timeframe: Multi-timeframe strategies")
        self.print("  • pairs: Pairs trading strategies")
        self.print("  • custom: Custom strategy from scratch")

        strategy_type = self.choose_option("Select strategy type", strategy_types, default="custom")

        # Asset class
        self.print("\nAsset Class", style="bold yellow")
        self.print("-" * 40)
        asset_classes = ["equity", "crypto"]
        asset_class = self.choose_option("Select asset class", asset_classes, default="equity")

        # Output options
        self.print("\nOutput Options", style="bold yellow")
        self.print("-" * 40)

        use_custom_path = self.confirm(
            "Use custom output path? (otherwise auto-generate in strategies directory)", default=False
        )

        output_path = None
        directory = None

        if use_custom_path:
            output_path = self.prompt("Output file path", required=True)
        else:
            use_custom_dir = self.confirm(
                "Use custom directory? (otherwise use default based on strategy type)", default=False
            )
            if use_custom_dir:
                directory = self.prompt("Directory path", required=True)

        return {
            "strategy_name": strategy_name,
            "strategy_type": strategy_type,
            "asset_class": asset_class,
            "output_path": output_path,
            "directory": directory,
        }


def run_strategy_wizard() -> Dict[str, Any]:
    """Run the strategy creation wizard.

    Returns:
        Dictionary with strategy creation parameters
    """
    wizard = StrategyWizard()

    try:
        return wizard.run_wizard()
    except KeyboardInterrupt:
        wizard.print("\n\nStrategy wizard cancelled.", style="yellow")
        sys.exit(1)
    except Exception as e:
        wizard.print(f"\n✗ Error: {e}", style="bold red")
        sys.exit(1)
