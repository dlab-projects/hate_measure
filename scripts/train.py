import lightning as L
import os

from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from hate_measure.loader import MeasuringHateSpeechModule
from hate_measure.nn.lightning import HateSpeechMeasurer
from transformers import AutoTokenizer


def run(config):
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    TOKENIZER = 'roberta-large'
    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER)
    # DataModule
    data_module_config = {
        'batch_size': 8,
        'tokenizer': TOKENIZER
    }
    data_module = MeasuringHateSpeechModule(config=data_module_config)
    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        monitor='train_loss',
        dirpath='checkpoints',
        every_n_epochs=1,
        save_on_train_epoch_end=True)

    # Model
    model = HateSpeechMeasurer(config, tokenizer)
    logger = TensorBoardLogger("logs", name="mhs_version2")
    # Trainer
    trainer = L.Trainer(
        accelerator="cuda",
        max_epochs=config['epoch'],
        devices=1,
        logger=logger,
        callbacks=[checkpoint_callback])


    # Train the model
    trainer.fit(model, data_module)

if __name__ == "__main__":
    # Configuration dictionary
    config = {
        'base': 'roberta-large',
        'n_dense': 256,
        'dropout_rate': 0.1,
        'lr': 2e-5,
        'epoch': 3,
        'accumulate_grad_batches': 1,
        'warmup_ratio': 0.1
    }
    run(config)
