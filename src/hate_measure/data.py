"""Data loading utilities for hate speech scoring."""

from datasets import load_dataset


def load_hate_speech_data(
    tokenizer,
    split="train",
    dataset_name="ucberkeley-dlab/measuring-hate-speech",
    target_col="hate_speech_score",
    max_length=512,
):
    """Load and tokenize the measuring hate speech dataset.

    Returns a HuggingFace Dataset ready for use with HF Trainer.

    Parameters
    ----------
    tokenizer : PreTrainedTokenizer
        Tokenizer to use for encoding text.
    split : str
        Dataset split to load (e.g., "train", "test").
    dataset_name : str
        HuggingFace dataset identifier.
    target_col : str
        Column name containing the target scores.
    max_length : int
        Maximum sequence length for tokenization.

    Returns
    -------
    datasets.Dataset
        Tokenized dataset with columns: input_ids, attention_mask, labels.
    """
    ds = load_dataset(dataset_name, split=split)

    def tokenize(batch):
        tokens = tokenizer(
            batch["text"],
            padding="max_length",
            truncation=True,
            max_length=max_length,
        )
        # HF Trainer expects "labels" key
        tokens["labels"] = batch[target_col]
        return tokens

    # Remove all original columns, keep only tokenized outputs
    ds = ds.map(tokenize, batched=True, remove_columns=ds.column_names)
    ds.set_format("torch")
    return ds
