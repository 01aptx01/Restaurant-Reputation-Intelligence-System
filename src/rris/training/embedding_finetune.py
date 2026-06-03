# -*- coding: utf-8 -*-
"""Optional SentenceTransformer fine-tuning before classifier training."""

from __future__ import annotations

import os

import numpy as np
from sentence_transformers import InputExample, SentenceTransformer, losses
from torch.utils.data import DataLoader

from rris import config


def _supervised_examples(texts: list[str], labels: np.ndarray) -> list[InputExample]:
    return [
        InputExample(texts=[str(t)], label=int(l) - 1)
        for t, l in zip(texts, labels)
    ]


def _contrastive_examples(
    texts: list[str],
    labels: np.ndarray,
    rng: np.random.RandomState,
    pairs_per_row: int = 1,
) -> list[InputExample]:
    by_star: dict[int, list[str]] = {}
    for text, label in zip(texts, labels):
        by_star.setdefault(int(label), []).append(str(text))
    examples: list[InputExample] = []
    stars = sorted(by_star.keys())
    for text, label in zip(texts, labels):
        star = int(label)
        pos_pool = by_star.get(star, [str(text)])
        neg_stars = [s for s in stars if s != star]
        if not neg_stars:
            continue
        for _ in range(pairs_per_row):
            pos = rng.choice(pos_pool)
            neg_star = rng.choice(neg_stars)
            neg = rng.choice(by_star[neg_star])
            examples.append(InputExample(texts=[str(text), pos, neg]))
    return examples


def finetune_embedding_model(
    texts: list[str],
    labels: np.ndarray,
    *,
    base_model: str | None = None,
    mode: str | None = None,
    output_dir: str | None = None,
    epochs: int | None = None,
    batch_size: int | None = None,
) -> str:
    """Fine-tune a SentenceTransformer and return the saved model directory."""
    base_model = base_model or config.EMBEDDING_FINETUNE_MODEL
    mode = (mode or config.EMBEDDING_FINETUNE_MODE).lower()
    output_dir = output_dir or config.EMBEDDING_FINETUNE_OUTPUT
    epochs = epochs if epochs is not None else config.EMBEDDING_FINETUNE_EPOCHS
    batch_size = batch_size if batch_size is not None else config.EMBEDDING_FINETUNE_BATCH_SIZE
    device = config.TORCH_DEVICE if config.TORCH_DEVICE != "cpu" else "cpu"

    print(f"Fine-tuning embedding: {base_model} mode={mode} epochs={epochs}")
    model = SentenceTransformer(base_model, device=device)
    rng = np.random.RandomState(config.RANDOM_STATE)

    if mode == "contrastive":
        train_examples = _contrastive_examples(texts, labels, rng)
        train_loss = losses.TripletLoss(model)
    else:
        train_examples = _supervised_examples(texts, labels)
        train_loss = losses.SoftmaxLoss(
            model=model,
            sentence_embedding_dimension=model.get_sentence_embedding_dimension(),
            num_labels=5,
        )

    if not train_examples:
        print("No fine-tune examples; skipping.")
        return base_model

    loader = DataLoader(train_examples, shuffle=True, batch_size=batch_size)
    warmup = int(len(loader) * epochs * 0.1)
    model.fit(
        train_objectives=[(loader, train_loss)],
        epochs=epochs,
        warmup_steps=max(1, warmup),
        show_progress_bar=True,
    )
    os.makedirs(output_dir, exist_ok=True)
    model.save(output_dir)
    print(f"Saved fine-tuned embedding model to {output_dir}")
    return output_dir


def resolve_embedding_model_path(meta: dict | None = None) -> str:
    """Return path or name of embedding model for encode (fine-tuned if available)."""
    meta = meta or {}
    finetuned = meta.get("finetuned_model_path")
    if finetuned and os.path.isdir(finetuned):
        return finetuned
    if getattr(config, "EMBEDDING_FINETUNE", False) and os.path.isdir(
        config.EMBEDDING_FINETUNE_OUTPUT
    ):
        return config.EMBEDDING_FINETUNE_OUTPUT
    return meta.get("embedding_model", config.EMBEDDING_MODEL_NAME)
