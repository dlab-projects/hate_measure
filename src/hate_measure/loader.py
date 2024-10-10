import datasets
import lightning as L
import numpy as np
import pandas as pd
import torch

from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer


class MeasuringHateSpeechDataset(Dataset):
    """Measuring Hate Speech Dataset object for PyTorch DataLoader.

    Parameters
    ----------
    input_ids : array-like
        Input IDs produced by tokenizer for Measuring Hate Speech samples.

    attention_mask : array-like
        Attention mask produced by tokenizer for Measuring Hate Speech samples.

    scores : array-like
        Hate speech scores.
    """
    def __init__(self, input_ids, attention_mask, weights, scores):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.weights = weights
        self.scores = scores

    def __getitem__(self, idx):
        return {
            "input_ids": torch.tensor(self.input_ids[idx]),
            "attention_mask": torch.tensor(self.attention_mask[idx]),
            "weights": torch.tensor(self.weights[idx]),
            "hate_speech_scores": torch.tensor(self.scores[idx])
        }

    def __len__(self):
        return len(self.scores)


class MeasuringHateSpeechGeneric(Dataset):
    def __init__(self, path, text_col='selftext', tokenizer='roberta-base'):
        self.path = path
        self.text_col = text_col
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        self.setup()

    def setup(self):
        def tokenize_function(examples):
            return self.tokenizer(
                examples[self.text_col], truncation=True, padding="max_length"
            )

        df = pd.read_csv(self.path)
        data = datasets.Dataset.from_pandas(df)
        tokenized = data.map(tokenize_function, batched=True)
        self.input_ids = tokenized['input_ids']
        self.attention_mask = tokenized['attention_mask']

    def __getitem__(self, idx):
        return {
            "input_ids": torch.tensor(self.input_ids[idx]),
            "attention_mask": torch.tensor(self.attention_mask[idx]),
        }

    def __len__(self):
        return len(self.input_ids)


class MeasuringHateSpeechModule(L.LightningDataModule):
    """Measuring Hate Speech DataModule for PyTorch Lightning.

    Parameters
    ----------
    config : dict
        Configuration dictionary. Specifices:
            - dataset_name, a string pointing to 
            - batch_size, an int indicating batch size
            - num_workers, an int indicating number of workers for the DataLoader
    """
    def __init__(self, config=None):
        super().__init__()
        # Default configuration
        default_config = {
            'dataset_name': 'ucberkeley-dlab/measuring-hate-speech',
            'batch_size': 32,
            'num_workers': 4,
            'tokenizer': 'roberta-base'
        }
        # Update default config with any values provided by the user
        if config is not None:
            default_config.update(config)
        self.cfg = default_config
        self.tokenizer = AutoTokenizer.from_pretrained(self.cfg['tokenizer'])

    def prepare_data(self):
        """Only called from the main process for downloading the dataset"""
        datasets.load_dataset(self.cfg['dataset_name'], split="train")

    def setup(self, stage: str):
        def tokenize_function(examples):
            return self.tokenizer(
                examples['text'], truncation=True, padding="max_length"
            )

        if stage == "fit":
            dataset = datasets.load_dataset(self.cfg['dataset_name'], split="train")
            df = dataset.to_pandas()
            comments = df.drop_duplicates('comment_id').sort_values('comment_id')
            weights = np.sqrt(df['comment_id'].value_counts().sort_index().values)
            dataset = datasets.Dataset.from_pandas(comments)
            tokenized = dataset.map(tokenize_function, batched=True)

            self.trn_dset = MeasuringHateSpeechDataset(
                tokenized['input_ids'],
                tokenized['attention_mask'],
                weights,
                tokenized['hate_speech_score']
            )
 
    def train_dataloader(self):
        return DataLoader(
            self.trn_dset,
            num_workers=self.cfg['num_workers'],
            batch_size=self.cfg['batch_size'],
        )

