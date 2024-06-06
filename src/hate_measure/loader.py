import pytorch_lightning as pl
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer
import torch

class MeasuringHateSpeechDataset(Dataset):
    def __init__(self, input_ids, attention_mask, scores):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.scores = scores

    def __getitem__(self, idx):
        return {
            "input_ids": torch.tensor(self.input_ids[idx]),
            "attention_mask": torch.tensor(self.attention_mask[idx]),
            "hate_speech_scores": torch.tensor(self.scores[idx])
        }

    def __len__(self):
        return len(self.scores)


class MeasuringHateSpeechModule(pl.LightningDataModule):
    def __init__(self, config=None, tokenizer='roberta-base'):
        super().__init__()
        # Default configuration
        default_config = {
            'dataset_name': 'ucberkeley-dlab/measuring-hate-speech',
            'batch_size': 32,
            'num_workers': 4
        }
        # Update default config with any values provided by the user
        if config is not None:
            default_config.update(config)
        self.cfg = default_config

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)

    def prepare_data(self):
        """Only called from the main process for downloading the dataset"""
        load_dataset(self.cfg['dataset_name'], split="train")

    def setup(self, stage: str):
        def tokenize_function(examples):
            return self.tokenizer(
                examples['text'], truncation=True, padding="max_length"
            )

        if stage == "fit":
            dataset = load_dataset(self.cfg['dataset_name'], split="train")
            tokenized = dataset.map(tokenize_function, batched=True)

            self.trn_dset = MeasuringHateSpeechDataset(
                tokenized['input_ids'], 
                tokenized['attention_mask'], 
                tokenized['hate_speech_score']
            )
 
    def train_dataloader(self):
        return DataLoader(
            self.trn_dset,
            num_workers=self.cfg['num_workers'],
            batch_size=self.cfg['batch_size'],
        )

