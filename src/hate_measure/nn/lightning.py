import torch
import lightning as L
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from .regressors import HateSpeechMeasurerModule


class HateSpeechMeasurer(L.LightningModule):
    def __init__(self, config, tokenizer):
        """method used to define our model parameters"""
        super().__init__()
        default_config = {
            'base': 'roberta-base',
            'n_dense': 128,
            'dropout_rate': 0.4,
            'accumulate_grad_batches': 1,
            'warmup_ratio': 0.1,
            'lr': 2.5e-5
        }
        # Update default config with any values provided by the user
        if config is not None:
            default_config.update(config)
        self.cfg = default_config
        self.tokenizer = tokenizer
        self.model = HateSpeechMeasurerModule(
            base=self.cfg['base'],
            n_dense=self.cfg['n_dense'],
            dropout_rate=self.cfg['dropout_rate'])
        self.save_hyperparameters()
        self.loss = torch.nn.MSELoss(reduction='none')


    def configure_optimizers(self):
        # Optimizer
        optimizer = AdamW(self.parameters(), lr=float(self.cfg['lr']))

        num_devices = 1
        total_steps = (
            len(self.trainer.datamodule.train_dataloader())
            // self.cfg['accumulate_grad_batches']
            // num_devices
            * self.cfg['epoch']
        )
        warmup_steps = int(total_steps * self.cfg['warmup_ratio'])
        scheduler = {
            "scheduler": get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=total_steps
            ),
            "interval": "step",  # runs per batch rather than per epoch
            "frequency": 1,
            "name": "learning_rate",
        }

        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        output = self(batch['input_ids'], batch['attention_mask']).squeeze()
        loss = self.loss(output, batch['hate_speech_scores'])
        loss = batch['weights'] * loss
        loss = loss.mean()
        self.log("train_loss", loss)
        return {"loss": loss}

    def forward(self, input_ids, attention_mask):
        return self.model(input_ids, attention_mask)

