# train.py
import torch
import torch.nn as nn

from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from tqdm import tqdm

from hate_measure.nn.regressors import HateSpeechScorer
from hate_measure.data import build_measuring_hate_speech_dataset
from hate_measure.nn.config import HateSpeechScorerConfig


class HateSpeechTrainer:
    """
    Train-only regression trainer for HateSpeechScorer.
    """
    def __init__(self, config: dict):
        self.config = config
        self.device = torch.device(
            "cuda" if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available()
            else "cpu"
        )
        hf_config = HateSpeechScorerConfig(
            encoder_model_name_or_path=config['base_model'],
            n_dense=config.get('n_dense', 128),
            dropout_rate=config.get('dropout_rate', 0.4),
            num_labels=1,
        )
        self.model = HateSpeechScorer(hf_config).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(hf_config.encoder_model_name_or_path)
        # Loss / Optimizer
        self.criterion = nn.MSELoss()
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=config['lr'],
            weight_decay=config.get('weight_decay', 0.0),
            betas=config.get('betas', (0.9, 0.999)),
            eps=config.get('eps', 1e-8),
        )
        self.train_loader: DataLoader | None = None
        self.scheduler = None

        if config.get('WANDB_API_KEY', None) is not None:
            import wandb
            wandb.login(key=config['WANDB_API_KEY'])
            wandb.init(project="hate-speech-scorer", name="hate-speech-scorer")
            wandb.config.update(config)
            self.wandb = wandb
        else:
            self.wandb = None

    def prepare_data(self):
        """
        Build the training DataLoader via data.py factory and set up the LR scheduler.
        """
        self.train_loader = build_measuring_hate_speech_dataset(
            tokenizer=self.tokenizer,
            split=self.config.get('split', 'train'),
            dataset_name=self.config.get('dataset_name', 'ucberkeley-dlab/measuring-hate-speech'),
            target_col=self.config.get('target_col', 'hate_speech_score'),
            max_length=self.config.get('max_length', 512),
            padding=self.config.get('padding', 'max_length'),
            truncation=self.config.get('truncation', True),
            as_dataloader=True,
            batch_size=self.config['batch_size'],
            num_workers=self.config.get('num_workers', 0),
            shuffle=True,
        )

        total_steps = len(self.train_loader) * self.config['epochs']
        warmup_steps = int(total_steps * self.config.get('warmup_ratio', 0.1))
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )

    def train_epoch(self, epoch_idx: int):
        self.model.train()
        running = 0.0
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch_idx+1}/{self.config['epochs']}")
        for batch in pbar:
            input_ids = batch['input_ids'].to(self.device, non_blocking=True)
            attention_mask = batch['attention_mask'].to(self.device, non_blocking=True)
            targets = batch['hate_speech_scores'].to(self.device, non_blocking=True)
            # targets may come as [B] or [B,1]; make them [B]
            targets = targets.squeeze(-1)

            self.optimizer.zero_grad(set_to_none=True)
            preds = self.model(input_ids, attention_mask).squeeze(-1)
            loss = self.criterion(preds, targets)

            loss.backward()
            if self.config.get('max_grad_norm') is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config['max_grad_norm'])
            self.optimizer.step()
            if self.scheduler is not None:
                self.scheduler.step()

            running += loss.item()
            if self.wandb is not None:
                self.wandb.log({
                    'epoch': epoch_idx,
                    'batch_loss': loss.item(),
                    'learning_rate': self.scheduler.get_last_lr()[0] if self.scheduler else self.config['lr']
                })
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        return running / len(self.train_loader)

    def train(self):
        if self.train_loader is None:
            raise RuntimeError("Call prepare_data() before train().")

        print("Starting training...")
        for epoch in range(self.config['epochs']):
            train_loss = self.train_epoch(epoch)
            print(f"Epoch {epoch+1}/{self.config['epochs']} | Train Loss: {train_loss:.4f}")

    @torch.no_grad()
    def predict(self, texts):
        self.model.eval()
        input_ids, attention_mask = self.tokenize(texts)
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)
        preds = self.model(input_ids, attention_mask).squeeze(-1)
        return preds.detach().cpu().numpy()

    def save_model(self, path='best_model.pt'):
        torch.save(self.model.state_dict(), path)
        print(f"Model saved to {path}")

    def load_model(self, path='best_model.pt'):
        state = torch.load(path, map_location=self.device)
        self.model.load_state_dict(state)
        self.model.to(self.device)
        print(f"Model loaded from {path}")


def create_default_config():
    return {
        'dataset_name': 'ucberkeley-dlab/measuring-hate-speech',
        'target_col': 'hate_speech_score',
        'split': 'train',
        'base_model': 'AnswerDotAI/ModernBERT-base',
        'lr': 2e-5,
        'weight_decay': 0.01,
        'betas': (0.9, 0.999),
        'eps': 1e-8,
        'epochs': 2,
        'batch_size': 32,
        'n_dense': 128,
        'dropout_rate': 0.4,
        'warmup_ratio': 0.1,
        'num_workers': 4,
        'max_length': 512,
        'padding': 'max_length',
        'truncation': True,
        'max_grad_norm': 1.0,
    }