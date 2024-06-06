import torch
from pytorch_lightning import LightningModule
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from .regressors import HateSpeechMeasurerModule


class HateSpeechMeasurer(LightningModule):
    def __init__(self, config, tokenizer):
        """method used to define our model parameters"""
        super().__init__()
        self.cfg = config
        self.tokenizer = tokenizer
        self.model = HateSpeechMeasurerModule()
        self.save_hyperparameters()
        self.loss = torch.nn.MSELoss()


    def configure_optimizers(self):
        # Optimizer
        optimizer = AdamW(self.parameters(), lr=float(self.cfg['lr']))

        # Scheduler with warmup
        #num_devices = (
        #    torch.cuda.device_count()
        #    if self.trainer.devices == -1
        #    else int(self.trainer.devices)
        #)
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
                optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
            ),
            "interval": "step",  # runs per batch rather than per epoch
            "frequency": 1,
            "name": "learning_rate",
        }

        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        output = self(batch['input_ids'], batch['attention_mask']).squeeze()
        loss = self.loss(output, batch['hate_speech_scores'])
        self.log("train_loss", loss)
        return {"loss": loss}

    def forward(self, input_ids, attention_mask):
        return self.model(input_ids, attention_mask)

