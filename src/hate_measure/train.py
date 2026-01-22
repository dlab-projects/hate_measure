"""Training script for hate speech scorer using HuggingFace Trainer."""

import os

from transformers import AutoTokenizer, Trainer, TrainingArguments

from hate_measure.data import load_hate_speech_data
from hate_measure.nn import HateSpeechScorer, HateSpeechScorerConfig


def train(
    output_dir: str,
    base_model: str = "AnswerDotAI/ModernBERT-base",
    num_epochs: int = 3,
    batch_size: int = 32,
    learning_rate: float = 2e-5,
    warmup_ratio: float = 0.1,
    weight_decay: float = 0.01,
    max_length: int = 512,
    logging_steps: int = 100,
    save_strategy: str = "epoch",
    bf16: bool = True,
    report_to: str = "none",
    run_name: str | None = None,
    wandb_project: str | None = None,
    push_to_hub: bool = False,
    hub_model_id: str | None = None,
):
    """Train a hate speech scoring model.

    Parameters
    ----------
    output_dir : str
        Directory to save model checkpoints and final model.
    base_model : str
        HuggingFace model identifier for the base encoder.
    num_epochs : int
        Number of training epochs.
    batch_size : int
        Training batch size per device.
    learning_rate : float
        Peak learning rate.
    warmup_ratio : float
        Fraction of training steps for learning rate warmup.
    weight_decay : float
        Weight decay for AdamW optimizer.
    max_length : int
        Maximum sequence length for tokenization.
    logging_steps : int
        Log metrics every N steps.
    save_strategy : str
        When to save checkpoints ("epoch", "steps", or "no").
    bf16 : bool
        Use bfloat16 mixed precision if available.
    report_to : str
        Integration to report metrics to. Options: "wandb", "tensorboard",
        "mlflow", "neptune", "clearml", "comet_ml", "all", or "none".
    run_name : str, optional
        Name for the training run (used by wandb, tensorboard, etc.).
    wandb_project : str, optional
        W&B project name. If not specified, uses default or WANDB_PROJECT env var.
    push_to_hub : bool, optional
        Whether to push the model to the Hugging Face hub.
    hub_model_id : str, optional
        HuggingFace Hub model ID (e.g., "username/model-name"). Required if push_to_hub is True.
    """
    # Model
    config = HateSpeechScorerConfig(encoder_model_name_or_path=base_model)
    model = HateSpeechScorer(config)
    tokenizer = AutoTokenizer.from_pretrained(config.encoder_model_name_or_path)

    # Data
    train_dataset = load_hate_speech_data(
        tokenizer,
        split="train",
        max_length=max_length
    )

    # Set wandb project via environment variable if specified
    if wandb_project is not None:
        os.environ["WANDB_PROJECT"] = wandb_project

    # Training arguments
    training_args = {
        "output_dir": output_dir,
        "num_train_epochs": num_epochs,
        "per_device_train_batch_size": batch_size,
        "learning_rate": learning_rate,
        "warmup_ratio": warmup_ratio,
        "weight_decay": weight_decay,
        "logging_steps": logging_steps,
        "save_strategy": save_strategy,
        "bf16": bf16,
        "report_to": report_to,
        "push_to_hub": push_to_hub,
    }

    # Add optional run name
    if run_name is not None:
        training_args["run_name"] = run_name

    # Add hub model ID if pushing to hub
    if hub_model_id is not None:
        training_args["hub_model_id"] = hub_model_id

    args = TrainingArguments(**training_args)

    # Train
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
    )
    trainer.train()

    # Save final model and tokenizer
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Model saved to {output_dir}")

    return model, tokenizer
