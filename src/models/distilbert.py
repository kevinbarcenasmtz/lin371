"""DistilBERT fine-tuning for binary mention classification.

Input format: "{target_word} [SEP] {company_context_window}"
Context window = most recent 2 historical transcripts (newest-first concat),
head-truncated to 512 tokens by the tokenizer.

Default hyperparameters (overridable via the CLI in scripts/run_distilbert.py):
  lr=2e-5, batch_size=16, epochs=3, warmup_ratio=0.1, weight_decay=0.01, max_seq_len=512

Checkpoints saved to outputs/models/distilbert/.
"""
from __future__ import annotations

import logging
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import f1_score
from torch.utils.data import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

from src.constants import MODELS_DIR, RANDOM_SEED

logger = logging.getLogger(__name__)

MODEL_NAME = "distilbert-base-uncased"
DEFAULT_OUTPUT_DIR = MODELS_DIR / "distilbert"


class MentionDataset(Dataset):
    """Lightweight torch Dataset wrapping pre-tokenized encodings + labels."""

    def __init__(self, encodings: dict[str, Any], labels: list[int] | None = None):
        self.encodings = encodings
        self.labels = labels

    def __len__(self) -> int:
        return len(self.encodings["input_ids"])

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        item = {k: torch.as_tensor(v[idx]) for k, v in self.encodings.items()}
        if self.labels is not None:
            item["labels"] = torch.as_tensor(int(self.labels[idx]), dtype=torch.long)
        return item


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def format_inputs(words: list[str], contexts: list[str]) -> list[str]:
    """Pair target word with context using the tokenizer's [SEP] token as separator.

    We use the string literal `[SEP]` here rather than the tokenizer's special-
    token id so this function has no tokenizer dependency. The tokenizer
    treats text pairs via the `text_pair` arg in `tokenize_texts`, so the
    literal is defensive fallback only when the caller passes already-joined
    strings (not used by the current run script).
    """
    if len(words) != len(contexts):
        raise ValueError(f"words ({len(words)}) and contexts ({len(contexts)}) length mismatch")
    return [f"{w} [SEP] {c}" for w, c in zip(words, contexts)]


def tokenize_texts(
    tokenizer: AutoTokenizer,
    words: list[str],
    contexts: list[str],
    max_seq_len: int = 512,
) -> dict[str, Any]:
    """Tokenize (word, context) pairs as a sentence-pair task.

    Context is passed as `text_pair`, so truncation trims context tokens first
    and preserves the target word intact — which matches our input design.
    """
    return tokenizer(
        words,
        contexts,
        padding=True,
        truncation="only_second",
        max_length=max_seq_len,
        return_tensors=None,
    )


def _compute_metrics(eval_pred) -> dict[str, float]:
    logits, labels = eval_pred
    preds = np.asarray(logits).argmax(axis=-1)
    return {
        "f1_macro": float(f1_score(labels, preds, labels=[0, 1], average="macro", zero_division=0)),
        "accuracy": float((preds == labels).mean()),
    }


def train_distilbert(
    train_words: list[str],
    train_contexts: list[str],
    train_labels: list[int],
    dev_words: list[str],
    dev_contexts: list[str],
    dev_labels: list[int],
    output_dir: str | Path | None = None,
    epochs: int = 3,
    lr: float = 2e-5,
    batch_size: int = 16,
    max_seq_len: int = 512,
    warmup_ratio: float = 0.1,
    weight_decay: float = 0.01,
    seed: int = RANDOM_SEED,
) -> tuple[AutoModelForSequenceClassification, AutoTokenizer, dict]:
    """Fine-tune distilbert-base-uncased for binary mention classification.

    Loads best-dev-F1 checkpoint at end of training and saves to output_dir.
    Returns (model, tokenizer, best_metrics).
    """
    _seed_everything(seed)
    output_dir = Path(output_dir) if output_dir else DEFAULT_OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading tokenizer + model: %s", MODEL_NAME)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)

    logger.info("Tokenizing: train=%d dev=%d (max_seq_len=%d)",
                len(train_words), len(dev_words), max_seq_len)
    train_enc = tokenize_texts(tokenizer, train_words, train_contexts, max_seq_len)
    dev_enc = tokenize_texts(tokenizer, dev_words, dev_contexts, max_seq_len)
    train_ds = MentionDataset(train_enc, train_labels)
    dev_ds = MentionDataset(dev_enc, dev_labels)

    training_args = TrainingArguments(
        output_dir=str(output_dir / "trainer"),
        num_train_epochs=epochs,
        learning_rate=lr,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size * 2,
        warmup_ratio=warmup_ratio,
        weight_decay=weight_decay,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        save_total_limit=2,
        logging_steps=25,
        seed=seed,
        report_to=[],
        fp16=torch.cuda.is_available(),
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=dev_ds,
        tokenizer=tokenizer,
        compute_metrics=_compute_metrics,
    )
    logger.info("Starting training (device=%s, epochs=%d)",
                "cuda" if torch.cuda.is_available() else "cpu", epochs)
    trainer.train()

    best_metrics = trainer.evaluate()
    logger.info("Best dev metrics: %s", best_metrics)

    save_dir = output_dir / "best"
    trainer.save_model(str(save_dir))
    tokenizer.save_pretrained(str(save_dir))
    logger.info("Saved best checkpoint to %s", save_dir)

    return model, tokenizer, best_metrics


@torch.no_grad()
def predict_distilbert(
    model: AutoModelForSequenceClassification,
    tokenizer: AutoTokenizer,
    words: list[str],
    contexts: list[str],
    batch_size: int = 32,
    max_seq_len: int = 512,
    device: str | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Run inference. Returns (preds, probs_positive)."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    enc = tokenize_texts(tokenizer, words, contexts, max_seq_len=max_seq_len)
    ds = MentionDataset(enc)
    n = len(ds)
    all_probs: list[np.ndarray] = []
    for start in range(0, n, batch_size):
        batch = [ds[i] for i in range(start, min(start + batch_size, n))]
        input_ids = torch.stack([b["input_ids"] for b in batch]).to(device)
        attn = torch.stack([b["attention_mask"] for b in batch]).to(device)
        out = model(input_ids=input_ids, attention_mask=attn)
        probs = torch.softmax(out.logits, dim=-1).detach().cpu().numpy()
        all_probs.append(probs)
    probs = np.concatenate(all_probs, axis=0)
    preds = probs.argmax(axis=-1)
    return preds, probs[:, 1]
