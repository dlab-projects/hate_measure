import torch

from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader

class MeasuringHateSpeechDataset(Dataset):
    def __init__(self, input_ids, attention_mask, scores,
                 ids_dtype=torch.long, mask_dtype=torch.long, y_dtype=torch.float32):
        self.input_ids = torch.as_tensor(input_ids, dtype=ids_dtype)
        self.attention_mask = torch.as_tensor(attention_mask, dtype=mask_dtype)
        self.scores = torch.as_tensor(scores, dtype=y_dtype)

    def __len__(self):
        return self.scores.shape[0]

    def __getitem__(self, idx):
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
            "labels": self.scores[idx],
        }

    @classmethod
    def from_hf(cls, ds_split, target_col="hate_speech_score",
                ids_dtype=torch.long, mask_dtype=torch.long, y_dtype=torch.float32):
        """
        Build directly from a HF split (e.g., ds['train']) that already has
        'input_ids', 'attention_mask', and target_col columns.
        """
        return cls(
            input_ids=ds_split["input_ids"],
            attention_mask=ds_split["attention_mask"],
            scores=ds_split[target_col],
            ids_dtype=ids_dtype,
            mask_dtype=mask_dtype,
            y_dtype=y_dtype,
        )

def build_measuring_hate_speech_dataset(
    tokenizer,
    split="train",
    dataset_name="ucberkeley-dlab/measuring-hate-speech",
    target_col="hate_speech_score",
    max_length=512,
    padding="max_length",
    truncation=True,
    as_dataloader=False,
    batch_size=32,
    num_workers=0,
    shuffle=True,
):
    # 1) Load HF split
    ds = load_dataset(dataset_name, split=split)

    # 2) Tokenize (return lists for HF Datasets)
    def tok_fn(batch):
        out = tokenizer(
            batch["text"],
            padding=padding,
            truncation=truncation,
            max_length=max_length,
        )
        return {"input_ids": out["input_ids"], "attention_mask": out["attention_mask"]}

    ds = ds.map(tok_fn, batched=True)

    # 3) Tensorify once
    input_ids = torch.tensor(ds["input_ids"], dtype=torch.long)
    attention_mask = torch.tensor(ds["attention_mask"], dtype=torch.long)
    scores = torch.tensor(ds[target_col], dtype=torch.float32)

    dataset = MeasuringHateSpeechDataset(
        input_ids=input_ids,
        attention_mask=attention_mask,
        scores=scores,
    )

    if not as_dataloader:
        return dataset

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )
    return loader