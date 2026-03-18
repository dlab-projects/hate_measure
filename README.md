# `hate_measure`: Neural Networks for Measuring Hate Speech

`hate_measure` is a package for training and serving continuous hate-speech scorers built on top of encoder models. 

The package uses the Hugging Face dataset [`ucberkeley-dlab/measuring-hate-speech`](https://huggingface.co/datasets/ucberkeley-dlab/measuring-hate-speech) and follows the measurement-based framing described in [Kennedy et al. (2019), *Constructing interval variables via faceted Rasch measurement and multitask deep learning: a hate speech application*](https://arxiv.org/abs/2009.10277) and  [Sachdeva et al. (2022), *The Measuring Hate Speech Corpus: Leveraging Rasch Measurement Theory for Data Perspectivism*](https://aclanthology.org/2022.nlperspectives-1.11/).

The repository centers on a custom regression head over ModernBERT and is aligned with the published Hugging Face checkpoint [`ucberkeley-dlab/mhs-scorer-modernbert-large`](https://huggingface.co/ucberkeley-dlab/mhs-scorer-modernbert-large). 

## Overview

This repository currently provides:

- A dataset loader for the Measuring Hate Speech dataset on Hugging Face
- A custom `transformers`-compatible regression model and config
- A training entry point built on Hugging Face `Trainer`
- Hub-facing model/config files for publishing custom code models

At a high level, the model predicts a **continuous hate speech score** rather than a binary hate/not-hate label.

## Measuring Hate Speech: Background

The Measuring Hate Speech project frames hate speech as a measurement problem rather than a simple hard classification task. The *Measuring Hate Speech* corpus consists of **50,070 unique social media comments** from YouTube, Reddit, and Twitter, labeled by **11,143 annotators**, with labels aggregated using **faceted Rasch measurement theory** to derive a continuous `hate_speech_score`.

The dataset includes:

- Text for each comment
- A continuous `hate_speech_score`
- Construct-level labels such as sentiment, respect, insult, humiliation, dehumanization, violence, genocide, and related dimensions
- Target identity annotations
- Annotator demographic and perspective-related metadata

## Installation

Python `3.11+` is required.

### With `uv`

```bash
uv sync
```

Optional Weights & Biases support:

```
uv sync --extra wandb
```

With pip
```
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

Optional Weights & Biases support:

pip install -e ".[wandb]"

## Training

The main training entry point is:

```bash
python scripts/train.py --output_dir outputs/mhs-modernbert-large
```

More options can be specified:

```bash
uv run python scripts/train.py \
  --output_dir outputs/mhs-modernbert-large \
  --base_model AnswerDotAI/ModernBERT-large \
  --num_epochs 3 \
  --batch_size 32 \
  --learning_rate 2e-5 \
  --warmup_ratio 0.1 \
  --weight_decay 0.01 \
  --max_length 512 \
  --report_to none
```

**Notes**
- The script defaults to --report_to wandb, so pass --report_to none unless you installed the optional wandb extra.
- Mixed precision is enabled by default with --bf16; use --no-bf16 if your hardware does not support it.
- The script supports --push_to_hub and --hub_model_id for publishing trained checkpoints.

## Using the Published Hugging Face Model

The published model card for [ucberkeley-dlab/mhs-scorer-modernbert-large](https://huggingface.co/ucberkeley-dlab/mhs-scorer-modernbert-large) shows how to load the hosted checkpoint with remote custom code.

```python
import torch
from transformers import AutoModel, AutoTokenizer

model_id = "ucberkeley-dlab/mhs-scorer-modernbert-large"
tokenizer = AutoTokenizer.from_pretrained("AnswerDotAI/ModernBERT-large")
model = AutoModel.from_pretrained(model_id, trust_remote_code=True)

text = "your text here"
inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)

with torch.no_grad():
    outputs = model(**inputs)

score = outputs.logits.squeeze(-1).item()
print(score)
```

Treat the output as a continuous relative score, not as a calibrated probability or a substitute for a full moderation policy.

## Repository Layout

```text
.
├── src/hate_measure/
│   ├── data.py               # Hugging Face dataset loading/tokenization
│   ├── train.py              # Training function
│   ├── constants.py          # Construct/target/annotator column groups
│   └── nn/
│       ├── config.py         # Custom HF config
│       └── scorer.py         # Custom regression model
├── scripts/
│   └── train.py              # CLI entry point
├── hub/
│   ├── config.py             # Hub remote-code config
│   └── scorer.py             # Hub remote-code model
├── notebooks/                # Exploratory data notebooks
└── data/                     # Local research data artifacts
```


