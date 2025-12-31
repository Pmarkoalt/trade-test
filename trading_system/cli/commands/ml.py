"""CLI commands for ML functionality."""

import argparse
from datetime import datetime, timedelta
from typing import Optional

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box

from trading_system.ml_refinement.config import MLConfig, ModelType
from trading_system.ml_refinement.storage.feature_db import FeatureDatabase
from trading_system.ml_refinement.training.trainer import ModelTrainer
from trading_system.scheduler.jobs.ml_retrain_job import MLRetrainJob

# Try to import ModelRegistry, create stub if not available
try:
    from trading_system.ml_refinement.models.model_registry import ModelRegistry
except ImportError:
    # Create minimal ModelRegistry stub
    class ModelRegistry:
        """Minimal model registry stub."""

        def __init__(self, model_dir: str, feature_db: FeatureDatabase):
            self.model_dir = model_dir
            self.feature_db = feature_db

        def get_active(self, model_type: ModelType):
            """Get active model metadata for type."""
            return self.feature_db.get_active_model(model_type.value)

        def activate(self, model_id: str):
            """Activate a model."""
            return self.feature_db.activate_model(model_id)


console = Console()


def setup_parser(ml_parser):
    """Set up ML CLI commands.

    Args:
        ml_parser: The ML parser to add subcommands to.
    """
    ml_subparsers = ml_parser.add_subparsers(dest="ml_command", help="ML command to run")

    # Train command
    train_parser = ml_subparsers.add_parser(
        "train",
        help="Train a new model",
    )
    train_parser.add_argument(
        "--model-type",
        choices=["signal_quality"],
        default="signal_quality",
        help="Type of model to train",
    )
    train_parser.add_argument(
        "--start-date",
        help="Training data start date (YYYY-MM-DD)",
    )
    train_parser.add_argument(
        "--end-date",
        help="Training data end date (YYYY-MM-DD)",
    )
    train_parser.add_argument(
        "--feature-db",
        default="features.db",
        help="Path to feature database",
    )
    train_parser.add_argument(
        "--model-dir",
        default="models/",
        help="Directory for model storage",
    )

    # Status command
    status_parser = ml_subparsers.add_parser(
        "status",
        help="Show ML system status",
    )
    status_parser.add_argument(
        "--feature-db",
        default="features.db",
        help="Path to feature database",
    )
    status_parser.add_argument(
        "--model-dir",
        default="models/",
        help="Directory for model storage",
    )

    # Models command
    models_parser = ml_subparsers.add_parser(
        "models",
        help="List trained models",
    )
    models_parser.add_argument(
        "--model-type",
        choices=["signal_quality", "all"],
        default="all",
        help="Filter by model type",
    )
    models_parser.add_argument(
        "--feature-db",
        default="features.db",
        help="Path to feature database",
    )

    # Features command
    features_parser = ml_subparsers.add_parser(
        "features",
        help="Show feature statistics",
    )
    features_parser.add_argument(
        "--feature-db",
        default="features.db",
        help="Path to feature database",
    )

    # Retrain command
    retrain_parser = ml_subparsers.add_parser(
        "retrain",
        help="Run retraining job",
    )
    retrain_parser.add_argument(
        "--force",
        action="store_true",
        help="Force retraining regardless of sample count",
    )
    retrain_parser.add_argument(
        "--feature-db",
        default="features.db",
        help="Path to feature database",
    )
    retrain_parser.add_argument(
        "--model-dir",
        default="models/",
        help="Directory for model storage",
    )


def handle_command(args):
    """Handle ML commands."""
    if args.ml_command == "train":
        return cmd_train(args)
    elif args.ml_command == "status":
        return cmd_status(args)
    elif args.ml_command == "models":
        return cmd_models(args)
    elif args.ml_command == "features":
        return cmd_features(args)
    elif args.ml_command == "retrain":
        return cmd_retrain(args)
    else:
        console.print("[yellow]Use --help to see available commands[/yellow]")
        return 1


def cmd_train(args):
    """Train a new model."""
    config = MLConfig()
    feature_db = FeatureDatabase(args.feature_db)
    feature_db.initialize()

    trainer = ModelTrainer(config, feature_db, args.model_dir)

    model_type = ModelType(args.model_type)

    with console.status("Training model..."):
        result = trainer.train(
            model_type=model_type,
            start_date=args.start_date,
            end_date=args.end_date,
        )

    if result.success:
        console.print(
            Panel(
                "[green]Model trained successfully![/green]\n\n"
                f"Model ID: {result.model_id}\n"
                f"Samples: {result.train_samples}\n"
                f"CV AUC: {result.cv_metrics.get('auc', 0):.3f}\n"
                f"Time: {result.total_time_seconds:.1f}s",
                title="Training Complete",
                box=box.ROUNDED,
            )
        )

        # Show top features
        if result.top_features:
            table = Table(title="Top Features", box=box.SIMPLE)
            table.add_column("Feature")
            table.add_column("Importance", justify="right")

            for name, importance in result.top_features[:10]:
                table.add_row(name, f"{importance:.4f}")

            console.print(table)

    else:
        console.print(f"[red]Training failed: {result.error_message}[/red]")
        feature_db.close()
        return 1

    feature_db.close()
    return 0


def cmd_status(args):
    """Show ML system status."""
    config = MLConfig()
    feature_db = FeatureDatabase(args.feature_db)
    feature_db.initialize()
    model_registry = ModelRegistry(args.model_dir, feature_db)

    # Feature statistics
    total_features = feature_db.count_samples(require_target=False)
    labeled_features = feature_db.count_samples(require_target=True)

    console.print(
        Panel(
            f"Total feature vectors: {total_features}\n"
            f"With labels: {labeled_features}\n"
            f"Unlabeled: {total_features - labeled_features}",
            title="Feature Database",
            box=box.ROUNDED,
        )
    )

    # Model status
    table = Table(title="Active Models", box=box.ROUNDED)
    table.add_column("Model Type")
    table.add_column("Model ID")
    table.add_column("AUC")
    table.add_column("Deployed")

    for model_type in ModelType:
        model = model_registry.get_active(model_type)
        if model:
            table.add_row(
                model_type.value,
                model.model_id[:20] + "..." if len(model.model_id) > 20 else model.model_id,
                f"{model.validation_metrics.get('auc', 0):.3f}",
                model.train_end_date or "Unknown",
            )
        else:
            table.add_row(
                model_type.value,
                "[dim]None[/dim]",
                "-",
                "-",
            )

    console.print(table)

    # Retrain status
    job = MLRetrainJob(config, feature_db, model_registry, args.model_dir)
    for model_type in ModelType:
        status = job.check_retrain_needed(model_type)
        if status["needed"]:
            console.print(f"[yellow]Retrain recommended for {model_type.value}: " f"{status['reason']}[/yellow]")

    feature_db.close()
    return 0


def cmd_models(args):
    """List trained models."""
    feature_db = FeatureDatabase(args.feature_db)
    feature_db.initialize()

    model_types = [ModelType.SIGNAL_QUALITY] if args.model_type != "all" else list(ModelType)

    for model_type in model_types:
        history = feature_db.get_model_history(model_type.value, limit=10)

        if not history:
            console.print(f"[dim]No models for {model_type.value}[/dim]")
            continue

        table = Table(title=f"{model_type.value} Models", box=box.ROUNDED)
        table.add_column("Model ID")
        table.add_column("Version")
        table.add_column("AUC")
        table.add_column("Samples")
        table.add_column("Active")

        for model in history:
            active_mark = "[green]Yes[/green]" if model.is_active else "[dim]No[/dim]"
            table.add_row(
                model.model_id[:24] if len(model.model_id) > 24 else model.model_id,
                model.version,
                f"{model.validation_metrics.get('auc', 0):.3f}",
                str(model.train_samples),
                active_mark,
            )

        console.print(table)

    feature_db.close()
    return 0


def cmd_features(args):
    """Show feature statistics."""
    feature_db = FeatureDatabase(args.feature_db)
    feature_db.initialize()

    # Get sample of features to analyze
    X, y, feature_names = feature_db.get_training_data(
        start_date="2020-01-01",
        end_date=datetime.now().strftime("%Y-%m-%d"),
    )

    if len(X) == 0:
        console.print("[yellow]No features in database[/yellow]")
        feature_db.close()
        return 0

    import numpy as np

    console.print(f"\nTotal samples: {len(X)}")
    console.print(f"Features: {len(feature_names)}")
    if len(y) > 0:
        console.print(f"Target win rate: {(y > 0).mean():.1%}")

    # Feature statistics
    table = Table(title="Feature Statistics", box=box.SIMPLE)
    table.add_column("Feature")
    table.add_column("Mean", justify="right")
    table.add_column("Std", justify="right")
    table.add_column("Min", justify="right")
    table.add_column("Max", justify="right")

    for i, name in enumerate(feature_names[:20]):  # Show first 20
        col = X[:, i]
        table.add_row(
            name[:25] if len(name) > 25 else name,
            f"{np.mean(col):.3f}",
            f"{np.std(col):.3f}",
            f"{np.min(col):.3f}",
            f"{np.max(col):.3f}",
        )

    console.print(table)

    if len(feature_names) > 20:
        console.print(f"[dim]... and {len(feature_names) - 20} more features[/dim]")

    feature_db.close()
    return 0


def cmd_retrain(args):
    """Run retraining job."""
    config = MLConfig()
    feature_db = FeatureDatabase(args.feature_db)
    feature_db.initialize()
    model_registry = ModelRegistry(args.model_dir, feature_db)

    job = MLRetrainJob(config, feature_db, model_registry, args.model_dir)

    with console.status("Running retrain job..."):
        results = job.run(force=args.force)

    # Show results
    if results["models_retrained"]:
        table = Table(title="Models Retrained", box=box.ROUNDED)
        table.add_column("Model Type")
        table.add_column("Model ID")
        table.add_column("AUC")
        table.add_column("Samples")

        for model in results["models_retrained"]:
            table.add_row(
                model["model_type"],
                model["model_id"][:24] if len(model["model_id"]) > 24 else model["model_id"],
                f"{model['cv_auc']:.3f}",
                str(model["samples"]),
            )

        console.print(table)

    if results["models_skipped"]:
        console.print(f"[dim]Skipped: {', '.join(results['models_skipped'])}[/dim]")

    if results["errors"]:
        for error in results["errors"]:
            console.print(f"[red]Error ({error['model_type']}): {error['error']}[/red]")
        feature_db.close()
        return 1

    console.print(f"\nCompleted in {results['elapsed_seconds']:.1f}s")

    feature_db.close()
    return 0
