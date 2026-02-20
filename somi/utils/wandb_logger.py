"""
Weights & Biases Logger for SOMI Experiments
============================================

Centralized W&B logging utility for all SOMI experiments.
Provides consistent logging across all experiment scripts.
"""

import wandb
import json
import os
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime
import torch


class SOMIWandbLogger:
    """
    Centralized W&B logger for SOMI experiments.
    
    Usage:
        logger = SOMIWandbLogger(
            experiment_name="v5_combined_params",
            config={"learning_rate": 0.0002, "beta1": 0.95}
        )
        
        # Log metrics during training
        logger.log({"loss": 0.5, "accuracy": 0.9})
        
        # Log final results
        logger.log_final_results({"best_accuracy": 0.9})
    """
    
    def __init__(
        self,
        experiment_name: str,
        config: Optional[Dict[str, Any]] = None,
        project: str = "somi-aimo3",
        entity: str = "johnandersonclayton-somi",
        tags: Optional[list] = None,
        resume: Optional[str] = None,
        mode: str = "online"
    ):
        """
        Initialize W&B logger.
        
        Args:
            experiment_name: Name of the experiment
            config: Experiment configuration dictionary
            project: W&B project name
            entity: W&B entity/username
            tags: List of tags for the experiment
            resume: Run ID to resume (optional)
            mode: "online", "offline", or "disabled"
        """
        self.experiment_name = experiment_name
        self.project = project
        self.entity = entity
        self.config = config or {}
        self.tags = tags or []
        self.mode = mode
        
        # Add default tags
        if "somi" not in [t.lower() for t in self.tags]:
            self.tags.append("somi")
        if "aimo3" not in [t.lower() for t in self.tags]:
            self.tags.append("aimo3")
        
        # Initialize W&B run
        self.run = wandb.init(
            project=project,
            entity=entity,
            name=experiment_name,
            config=self.config,
            tags=self.tags,
            resume=resume,
            mode=mode,
            reinit=True
        )
        
        # Track step counter
        self.step = 0
        
    def log(self, metrics: Dict[str, Any], step: Optional[int] = None, commit: bool = True):
        """
        Log metrics to W&B.
        
        Args:
            metrics: Dictionary of metrics to log
            step: Step number (auto-increments if None)
            commit: Whether to commit the log (default True)
        """
        if step is None:
            step = self.step
            self.step += 1
        
        # Convert torch tensors to Python scalars
        processed_metrics = {}
        for key, value in metrics.items():
            if isinstance(value, torch.Tensor):
                if value.numel() == 1:
                    processed_metrics[key] = value.item()
                else:
                    # For multi-element tensors, log summary stats
                    processed_metrics[f"{key}_mean"] = value.mean().item()
                    processed_metrics[f"{key}_std"] = value.std().item()
                    processed_metrics[f"{key}_min"] = value.min().item()
                    processed_metrics[f"{key}_max"] = value.max().item()
            else:
                processed_metrics[key] = value
        
        self.run.log(processed_metrics, step=step, commit=commit)
    
    def log_final_results(self, results: Dict[str, Any]):
        """
        Log final experiment results.
        
        Args:
            results: Dictionary of final results
        """
        # Add prefix to distinguish final results
        final_metrics = {f"final/{k}": v for k, v in results.items()}
        self.log(final_metrics)
        
        # Also log as summary for easy comparison
        for key, value in results.items():
            self.run.summary[key] = value
    
    def log_model_info(self, model_info: Dict[str, Any]):
        """
        Log model information (architecture, parameters, etc.).
        
        Args:
            model_info: Dictionary of model information
        """
        model_metrics = {f"model/{k}": v for k, v in model_info.items()}
        self.log(model_metrics)
        
        # Also add to config
        self.run.config.update({"model_info": model_info})
    
    def log_somi_metrics(self, somi_metrics: Dict[str, Any]):
        """
        Log SOMI-specific metrics (coordination, stress, frequencies, etc.).
        
        Args:
            somi_metrics: Dictionary of SOMI metrics
        """
        somi_metrics_prefixed = {f"somi/{k}": v for k, v in somi_metrics.items()}
        self.log(somi_metrics_prefixed)
    
    def log_experiment_config(self, config: Dict[str, Any]):
        """
        Update experiment configuration.
        
        Args:
            config: Configuration dictionary
        """
        self.run.config.update(config)
        self.config.update(config)
    
    def log_artifact(self, file_path: str, artifact_name: str, artifact_type: str = "dataset"):
        """
        Log a file as a W&B artifact.
        
        Args:
            file_path: Path to the file
            artifact_name: Name for the artifact
            artifact_type: Type of artifact (dataset, model, etc.)
        """
        artifact = wandb.Artifact(artifact_name, type=artifact_type)
        artifact.add_file(file_path)
        self.run.log_artifact(artifact)
    
    def log_image(self, image, caption: str = "", name: str = None):
        """
        Log an image to W&B.
        
        Args:
            image: Image tensor or numpy array
            caption: Caption for the image
            name: Name for the image (auto-generated if None)
        """
        if name is None:
            name = f"image_{self.step}"
        
        if isinstance(image, torch.Tensor):
            image = image.detach().cpu().numpy()
        
        self.run.log({name: wandb.Image(image, caption=caption)})
    
    def log_table(self, table_data: list, columns: list, table_name: str = "results"):
        """
        Log a table to W&B.
        
        Args:
            table_data: List of rows (each row is a list of values)
            columns: List of column names
            table_name: Name for the table
        """
        table = wandb.Table(columns=columns, data=table_data)
        self.run.log({table_name: table})
    
    def finish(self):
        """Finish the W&B run."""
        self.run.finish()
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.finish()


def log_existing_results_to_wandb(
    results_file: str,
    experiment_name: str,
    config: Optional[Dict[str, Any]] = None
):
    """
    Log existing results from a JSON file to W&B.
    
    Args:
        results_file: Path to JSON results file
        experiment_name: Name for the experiment
        config: Optional configuration dictionary
    """
    # Load results
    with open(results_file, 'r') as f:
        results = json.load(f)
    
    # Initialize logger
    logger = SOMIWandbLogger(
        experiment_name=experiment_name,
        config=config or {}
    )
    
    # Log all results
    if "metadata" in results:
        logger.log_experiment_config(results["metadata"])
    
    if "model_info" in results:
        logger.log_model_info(results["model_info"])
    
    if "somi_metrics" in results:
        logger.log_somi_metrics(results["somi_metrics"])
    
    if "math_problems" in results:
        # Log math problem results as a table
        problems = []
        for problem, result in results["math_problems"].items():
            problems.append([
                problem,
                result.get("correct", False),
                result.get("answer", ""),
                result.get("expected", ""),
                result.get("time", 0)
            ])
        
        logger.log_table(
            problems,
            columns=["problem", "correct", "answer", "expected", "time"],
            table_name="math_problems"
        )
    
    # Log any other metrics
    for key, value in results.items():
        if key not in ["metadata", "model_info", "somi_metrics", "math_problems"]:
            if isinstance(value, (int, float)):
                logger.log_final_results({key: value})
    
    logger.finish()
    print(f"Logged results from {results_file} to W&B")


def setup_wandb_environment():
    """
    Set up W&B environment variables.
    Call this at the start of experiment scripts.
    """
    # Set W&B project and entity
    os.environ.setdefault("WANDB_PROJECT", "somi-aimo3")
    os.environ.setdefault("WANDB_ENTITY", "johnandersonclayton-somi")
    
    # Disable W&B code saving (optional, reduces clutter)
    os.environ.setdefault("WANDB_DISABLE_CODE", "false")
    
    print(f"W&B configured: project={os.environ['WANDB_PROJECT']}, entity={os.environ['WANDB_ENTITY']}")


if __name__ == "__main__":
    # Example usage
    setup_wandb_environment()
    
    # Example: Log a simple experiment
    logger = SOMIWandbLogger(
        experiment_name="test_experiment",
        config={
            "learning_rate": 0.001,
            "batch_size": 32,
            "beta1": 0.95
        },
        tags=["test", "debug"]
    )
    
    # Log some metrics
    for step in range(10):
        logger.log({
            "loss": 1.0 / (step + 1),
            "accuracy": step * 0.1,
            "coordination": 0.5 + step * 0.05
        })
    
    # Log final results
    logger.log_final_results({
        "best_accuracy": 0.9,
        "final_loss": 0.1
    })
    
    logger.finish()
    print("Example W&B logging complete!")
