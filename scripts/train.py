"""Command-line script for training hate speech scorer model."""

import argparse

from hate_measure.train import train


def main(args):
    # Train the model
    train(
        output_dir=args.output_dir,
        base_model=args.base_model,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        max_length=args.max_length,
        logging_steps=args.logging_steps,
        save_strategy=args.save_strategy,
        bf16=args.bf16,
        report_to=args.report_to,
        run_name=args.run_name,
        wandb_project=args.wandb_project,
        push_to_hub=args.push_to_hub,
        hub_model_id=args.hub_model_id,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a hate speech scoring model with HuggingFace Trainer")

    # Model and output
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save model checkpoints and final model",
    )
    parser.add_argument(
        "--base_model",
        type=str,
        default="AnswerDotAI/ModernBERT-large",
        help="HuggingFace model identifier for the base encoder",
    )

    # Training hyperparameters
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=3,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Training batch size per device",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=2e-5,
        help="Peak learning rate",
    )
    parser.add_argument(
        "--warmup_ratio",
        type=float,
        default=0.1,
        help="Fraction of training steps for learning rate warmup",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.01,
        help="Weight decay for AdamW optimizer",
    )

    # Data and tokenization
    parser.add_argument(
        "--max_length",
        type=int,
        default=512,
        help="Maximum sequence length for tokenization",
    )

    # Logging and checkpointing
    parser.add_argument(
        "--logging_steps",
        type=int,
        default=100,
        help="Log metrics every N steps",
    )

    parser.add_argument(
        "--save_strategy",
        type=str,
        default="epoch",
        choices=["epoch", "steps", "no"],
        help="When to save checkpoints",
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="wandb",
        help='Integration to report metrics to (e.g., "wandb", "tensorboard", "mlflow", "none")',
    )
    parser.add_argument(
        "--run_name",
        type=str,
        default=None,
        help="Name for the training run (visible in wandb/tensorboard)",
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default=None,
        help="W&B project name (if not set, uses WANDB_PROJECT env var or wandb default)",
    )
    parser.add_argument(
        "--push_to_hub",
        action="store_true",
        default=False,
        help="Push the trained model to HuggingFace Hub",
    )
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help='HuggingFace Hub model ID (e.g., "username/hate-speech-scorer")',
    )

    # Training settings
    parser.add_argument(
        "--bf16",
        action="store_true",
        default=True,
        help="Use bfloat16 mixed precision if available",
    )

    parser.add_argument(
        "--no-bf16",
        dest="bf16",
        action="store_false",
        help="Disable bfloat16 mixed precision",
    )

    args = parser.parse_args()
    main(args)
