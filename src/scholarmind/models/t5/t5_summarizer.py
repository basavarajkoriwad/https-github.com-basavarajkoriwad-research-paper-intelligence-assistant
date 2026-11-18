"""Experiment D: T5 Transformer Encoder-Decoder Scientific Paper Summarization.

Mathematical Foundation:
T5 frames summarization as sequence-to-sequence conditional text generation over an input
scientific paper x = (x_1, ..., x_N) and target abstract y = (y_1, ..., y_M):

    1. Encoder: Bidirectional self-attention with relative position bias maps x -> H_enc.
    2. Decoder: Causal masked self-attention over y_{<t}, followed by encoder-decoder
       cross-attention (queries from decoder, keys/values from H_enc):
           CrossAttn(Q_dec, K_enc, V_enc) = softmax((Q_dec K_enc^T) / sqrt(d_k)) V_enc
    3. Training Objective (Teacher-Forced Negative Log-Likelihood):
           L(theta) = - (1 / M) * sum_{t=1}^{M} log P_theta(y_t | y_{<t}, x)
    4. Evaluation (ROUGE-N & ROUGE-L):
           ROUGE-N_F1 = 2 * Precision_N * Recall_N / (Precision_N + Recall_N)
           ROUGE-L    = Longest Common Subsequence (LCS) F1 capturing structural fluency.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from rouge_score import rouge_scorer
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from scholarmind.core.device import count_parameters
from scholarmind.core.logging import get_logger
from scholarmind.nlp.segmentation import ScientificSentenceSegmenter

logger = get_logger(__name__)


def lead_n_baseline(
    articles: list[str],
    n_sentences: int = 3,
) -> list[str]:
    """Strong extractive Lead-N baseline using our scientific sentence segmenter."""
    segmenter = ScientificSentenceSegmenter(backend="auto")
    summaries: list[str] = []
    for article in articles:
        sents = segmenter.segment(article)
        summaries.append(" ".join(sents[:n_sentences]) if sents else article[:400])
    return summaries


def compute_rouge_metrics(
    predictions: list[str],
    references: list[str],
) -> dict[str, float]:
    """Compute mean ROUGE-1, ROUGE-2, and ROUGE-L F1 scores using Google's rouge-score."""
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    r1, r2, rl = [], [], []
    for pred, ref in zip(predictions, references):
        scores = scorer.score(ref or "", pred or "")
        r1.append(scores["rouge1"].fmeasure)
        r2.append(scores["rouge2"].fmeasure)
        rl.append(scores["rougeL"].fmeasure)

    return {
        "rouge1": round(float(np.mean(r1)) if r1 else 0.0, 4),
        "rouge2": round(float(np.mean(r2)) if r2 else 0.0, 4),
        "rougeL": round(float(np.mean(rl)) if rl else 0.0, 4),
    }


def _condense_scientific_article(article: str, max_words: int = 420) -> str:
    """Front+back extractive condensation preserving Introduction (first 70%) + Conclusion (last 30%).

    Because EDA showed 100% of ArXiv articles exceed 512 tokens, keeping both the lead
    motivation/method and concluding findings preserves far more summary-salient content
    than blind prefix truncation.
    """
    words = (article or "").split()
    if len(words) <= max_words:
        return " ".join(words)
    head_n = int(max_words * 0.70)
    tail_n = max_words - head_n
    return " ".join(words[:head_n]) + " ... " + " ".join(words[-tail_n:])


class ArxivSummarizationDataset(Dataset):
    """PyTorch Dataset preparing prefix-prompted `(article, abstract)` pairs for T5 fine-tuning."""

    def __init__(
        self,
        df: pd.DataFrame,
        tokenizer: Any,
        article_col: str = "article_clean",
        abstract_col: str = "abstract_clean",
        max_source_len: int = 512,
        max_target_len: int = 128,
        prefix: str = "summarize scientific paper: ",
    ) -> None:
        sources = [prefix + _condense_scientific_article(t, max_words=380) for t in df[article_col].astype(str)]
        targets = df[abstract_col].astype(str).tolist()

        model_inputs = tokenizer(
            sources,
            max_length=max_source_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        labels = tokenizer(
            text_target=targets,
            max_length=max_target_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )["input_ids"]

        # Mask padding positions with -100 so CrossEntropyLoss ignores pad tokens
        labels[labels == tokenizer.pad_token_id] = -100

        self.input_ids = model_inputs["input_ids"]
        self.attention_mask = model_inputs["attention_mask"]
        self.labels = labels

    def __len__(self) -> int:
        return self.input_ids.size(0)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
            "labels": self.labels[idx],
        }


class T5ScientificSummarizer:
    """Wrapper for T5 / Flan-T5 inference, hierarchical map-reduce, and PyTorch fine-tuning."""

    def __init__(
        self,
        model_name: str = "google/flan-t5-small",
        device: torch.device | None = None,
        tokenizer: Any | None = None,
        model: Any | None = None,
    ) -> None:
        self.model_name = model_name
        self.device = device or torch.device("cpu")
        self.tokenizer = tokenizer or AutoTokenizer.from_pretrained(model_name)
        self.model = (model or AutoModelForSeq2SeqLM.from_pretrained(model_name)).to(self.device)

    @torch.no_grad()
    def summarize_batch(
        self,
        articles: list[str],
        max_source_len: int = 512,
        max_new_tokens: int = 110,
        min_length: int = 35,
        num_beams: int = 2,
        batch_size: int = 4,
        prefix: str = "summarize scientific paper: ",
    ) -> list[str]:
        """Generate abstractive summaries with beam search and length normalization."""
        self.model.eval()
        outputs: list[str] = []

        for start in range(0, len(articles), batch_size):
            batch_texts = [
                prefix + _condense_scientific_article(a, max_words=380)
                for a in articles[start : start + batch_size]
            ]
            enc = self.tokenizer(
                batch_texts,
                max_length=max_source_len,
                truncation=True,
                padding=True,
                return_tensors="pt",
            ).to(self.device)

            gen_ids = self.model.generate(
                **enc,
                max_new_tokens=max_new_tokens,
                min_length=min_length,
                num_beams=num_beams,
                length_penalty=1.4,
                no_repeat_ngram_size=3,
                early_stopping=True,
            )
            decoded = self.tokenizer.batch_decode(gen_ids, skip_special_tokens=True)
            outputs.extend(s.strip() for s in decoded)

        return outputs

    @torch.no_grad()
    def summarize_hierarchical_map_reduce(
        self,
        article: str,
        max_chunks: int = 4,
        chunk_words: int = 320,
    ) -> str:
        """Map-Reduce summarization for ultra-long papers on CPU / 512-token models.

        Splits the paper into evenly spaced representative blocks (Intro, Early Method,
        Late Method/Results, Conclusion), summarizes each block (Map), and synthesizes
        the intermediate digests into a final cohesive summary (Reduce).
        """
        words = (article or "").split()
        if len(words) <= chunk_words:
            return self.summarize_batch([article], num_beams=2)[0]

        # Select evenly spaced windows covering beginning, middle, and end of the paper
        total_words = len(words)
        stride = max(1, (total_words - chunk_words) // max(1, max_chunks - 1))
        chunks = [
            " ".join(words[i * stride : i * stride + chunk_words])
            for i in range(max_chunks)
            if i * stride < total_words
        ]
        # Map step
        partial_summaries = self.summarize_batch(
            chunks,
            max_source_len=448,
            max_new_tokens=64,
            min_length=20,
            num_beams=2,
            batch_size=len(chunks),
            prefix="summarize this research paper section: ",
        )
        # Reduce step
        combined = " ".join(partial_summaries)
        final_summary = self.summarize_batch(
            [combined],
            max_source_len=448,
            max_new_tokens=110,
            min_length=35,
            num_beams=2,
            batch_size=1,
            prefix="synthesize a cohesive scientific abstract from these section summaries: ",
        )[0]
        return final_summary

    @torch.no_grad()
    def evaluate_loss(self, loader: DataLoader) -> float:
        """Compute token-level teacher-forced cross-entropy loss on a DataLoader."""
        self.model.eval()
        total_loss = 0.0
        total_samples = 0
        for batch in loader:
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = batch["labels"].to(self.device)
            out = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            bs = input_ids.size(0)
            total_loss += float(out.loss.item()) * bs
            total_samples += bs
        return round(total_loss / max(1, total_samples), 4)

    def fine_tune(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        checkpoint_path: Path,
        epochs: int = 3,
        batch_size: int = 4,
        lr: float = 1e-4,
        max_source_len: int = 384,
        max_target_len: int = 112,
    ) -> dict[str, Any]:
        """Fine-tune T5 in PyTorch, recording train/val loss per epoch and saving best checkpoint."""
        train_ds = ArxivSummarizationDataset(
            train_df,
            self.tokenizer,
            max_source_len=max_source_len,
            max_target_len=max_target_len,
        )
        val_ds = ArxivSummarizationDataset(
            val_df,
            self.tokenizer,
            max_source_len=max_source_len,
            max_target_len=max_target_len,
        )
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

        total_params, trainable_params = count_parameters(self.model)
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=0.01)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs))

        history: list[dict[str, float]] = []
        best_val_loss = float("inf")
        best_epoch = 0
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

        # Record Epoch 0 (pretrained baseline loss before any gradient step)
        init_val_loss = self.evaluate_loss(val_loader)
        logger.info("T5 Pretrained (Epoch 0) | val_loss=%.4f", init_val_loss)

        train_start = time.perf_counter()
        for epoch in range(1, epochs + 1):
            self.model.train()
            running_loss = 0.0
            seen = 0

            for batch in train_loader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)

                optimizer.zero_grad()
                out = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = out.loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()

                bs = input_ids.size(0)
                running_loss += float(loss.item()) * bs
                seen += bs

            scheduler.step()
            train_loss = round(running_loss / max(1, seen), 4)
            val_loss = self.evaluate_loss(val_loader)
            history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
            logger.info("T5 Epoch %02d/%02d | train_loss=%.4f | val_loss=%.4f", epoch, epochs, train_loss, val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch
                torch.save(
                    {
                        "model_name": self.model_name,
                        "model_state_dict": self.model.state_dict(),
                        "best_epoch": best_epoch,
                        "best_val_loss": best_val_loss,
                        "pretrained_val_loss": init_val_loss,
                        "history": history,
                    },
                    checkpoint_path,
                )

        train_seconds = round(time.perf_counter() - train_start, 2)

        # Reload best weights for subsequent test evaluation
        best_ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(best_ckpt["model_state_dict"])

        return {
            "model_name": self.model_name,
            "device": str(self.device),
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "pretrained_val_loss": init_val_loss,
            "best_val_loss": best_val_loss,
            "best_epoch": best_epoch,
            "train_seconds": train_seconds,
            "checkpoint_path": str(checkpoint_path),
            "history": history,
        }
