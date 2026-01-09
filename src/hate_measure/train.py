"""Training script for hate speech scorer using HuggingFace Trainer."""

from transformers import AutoTokenizer, Trainer, TrainingArguments

from hate_measure.data import load_hate_speech_data
from hate_measure.nn import HateSpeechScorer, HateSpeechScorerConfig


def train(
    output_dir: str = "./hate_speech_model",
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
    """
    # Model
    config = HateSpeechScorerConfig(base_model_name=base_model)
    model = HateSpeechScorer(config)
    tokenizer = AutoTokenizer.from_pretrained(config.encoder_model_name_or_path)

    # Data
    train_dataset = load_hate_speech_data(
        tokenizer, split="train", max_length=max_length
    )

    # Training arguments
    args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        learning_rate=learning_rate,
        warmup_ratio=warmup_ratio,
        weight_decay=weight_decay,
        logging_steps=logging_steps,
        save_strategy=save_strategy,
        bf16=bf16,
        report_to="none",  # Disable wandb/tensorboard by default
    )

    # Train
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
    )
    trainer.train()

    # Save final model and tokenizer
    final_path = f"{output_dir}/final"
    model.save_pretrained(final_path)
    tokenizer.save_pretrained(final_path)
    print(f"Model saved to {final_path}")

    return model, tokenizer


def main():
    """Entry point for command-line training."""
    train()


if __name__ == "__main__":
    main()
