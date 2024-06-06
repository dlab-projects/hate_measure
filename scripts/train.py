from pytorch_lightning import Trainer
from transformers import AutoTokenizer
from hate_measure.loader import MeasuringHateSpeechModule
from hate_measure.nn.lightning import HateSpeechMeasurer
import os
from lightning.pytorch.loggers import TensorBoardLogger


def run(config):
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained('roberta-base')

    # DataModule
    data_module = MeasuringHateSpeechModule()

    # Model
    model = HateSpeechMeasurer(config, tokenizer)
    logger = TensorBoardLogger("logs", name="my_model")
    # Trainer
    trainer = Trainer(
        accelerator="mps",
        max_epochs=config['epoch'],
        logger=logger)

    # Train the model
    trainer.fit(model, data_module)

if __name__ == "__main__":
    # Configuration dictionary
    config = {
        'lr': 2e-5,
        'epoch': 3,
        'accumulate_grad_batches': 1,
        'warmup_ratio': 0.1
    }
    run(config)
