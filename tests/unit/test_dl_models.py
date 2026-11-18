"""Unit tests for all PyTorch DL & NLP model architectures (offline, deterministic, CPU-fast)."""

from __future__ import annotations

import math
from pathlib import Path

import pandas as pd
import pytest
import torch
from transformers import BertConfig, BertModel, T5Config, T5ForConditionalGeneration

from scholarmind.core.device import count_parameters, set_seed
from scholarmind.models.bert.sci_bert_classifier import SciBERTClassifier
from scholarmind.models.embeddings.retrieval_benchmark import (
    TFIDFEmbedder,
    evaluate_retrieval,
)
from scholarmind.models.rnn_lstm.dataset import (
    ArxivSequenceDataset,
    build_label_mapping,
    collate_padded_sequences,
    compute_classical_baselines,
)
from scholarmind.models.rnn_lstm.lstm_models import (
    BiLSTMAttentionClassifier,
    LSTMClassifier,
)
from scholarmind.models.rnn_lstm.vanilla_rnn import VanillaRNNClassifier
from scholarmind.models.t5.t5_summarizer import (
    ArxivSummarizationDataset,
    T5ScientificSummarizer,
    compute_rouge_metrics,
    lead_n_baseline,
)
from scholarmind.nlp.tokenization import Vocabulary


def test_vanilla_rnn_forward_backward_both_poolings() -> None:
    set_seed(42)
    input_ids = torch.tensor([[1, 2, 3, 0, 0], [4, 5, 6, 7, 8]], dtype=torch.long)
    lengths = torch.tensor([3, 5], dtype=torch.long)
    targets = torch.tensor([0, 2], dtype=torch.long)

    for pooling in ("last", "mean"):
        model = VanillaRNNClassifier(vocab_size=20, num_classes=4, embed_dim=16, hidden_dim=24, pooling=pooling)
        logits, h_n = model(input_ids, lengths, return_hidden=True)
        assert logits.shape == (2, 4)
        assert h_n.shape == (2, 24)
        loss = torch.nn.functional.cross_entropy(logits, targets)
        loss.backward()
        assert model.embedding.weight.grad is not None


def test_lstm_and_bilstm_attention_states_and_pad_masking() -> None:
    set_seed(42)
    input_ids = torch.tensor([[2, 3, 4, 0, 0], [5, 6, 7, 8, 9]], dtype=torch.long)
    lengths = torch.tensor([3, 5], dtype=torch.long)

    lstm = LSTMClassifier(vocab_size=20, num_classes=4, embed_dim=16, hidden_dim=20)
    logits, states = lstm(input_ids, lengths, return_states=True)
    assert logits.shape == (2, 4)
    assert states["h_n"].shape == (2, 20)
    assert states["c_n"].shape == (2, 20)

    bilstm = BiLSTMAttentionClassifier(vocab_size=20, num_classes=4, embed_dim=16, hidden_dim=20, num_layers=2)
    b_logits, b_states = bilstm(input_ids, lengths, return_states=True)
    attn = b_states["attention_weights"]
    assert b_logits.shape == (2, 4)
    assert b_states["h_final"].shape == (2, 40)
    assert b_states["c_final"].shape == (2, 40)
    assert attn.shape == (2, 5)
    # Attention weights must sum to 1.0 and be strictly 0.0 on <PAD> tokens (positions 3, 4 of row 0)
    assert torch.allclose(attn.sum(dim=-1), torch.ones(2), atol=1e-5)
    assert float(attn[0, 3].item()) == pytest.approx(0.0, abs=1e-6)
    assert float(attn[0, 4].item()) == pytest.approx(0.0, abs=1e-6)


def test_arxiv_sequence_dataset_collate_and_classical_baselines() -> None:
    df = pd.DataFrame(
        [
            {"combined_text": "transformer attention neural language model", "primary_category": "cs.CL"},
            {"combined_text": "convolutional vision object detection image", "primary_category": "cs.CV"},
            {"combined_text": "language generation transformer decoding", "primary_category": "cs.CL"},
            {"combined_text": "image segmentation visual encoder", "primary_category": "cs.CV"},
        ]
    )
    vocab = Vocabulary(min_freq=1).fit([row.split() for row in df["combined_text"]])
    l2i, _ = build_label_mapping(df)
    ds = ArxivSequenceDataset(df, vocab, l2i, max_len=16)
    batch = [ds[0], ds[1]]
    padded, lengths, labels = collate_padded_sequences(batch, pad_id=vocab.pad_id)
    assert padded.shape[0] == 2
    assert lengths.shape == (2,)
    assert labels.shape == (2,)

    baselines = compute_classical_baselines(df, df, df)
    assert "majority_baseline" in baselines
    assert "tfidf_logreg_baseline" in baselines
    assert 0.0 <= baselines["tfidf_logreg_baseline"]["test_accuracy"] <= 1.0


def test_scibert_classifier_selective_freezing_and_pooling_offline() -> None:
    set_seed(42)
    tiny_cfg = BertConfig(
        vocab_size=100,
        hidden_size=32,
        num_hidden_layers=4,
        num_attention_heads=4,
        intermediate_size=64,
    )
    backbone = BertModel(tiny_cfg)
    clf = SciBERTClassifier(
        model_name="tiny-offline-bert",
        num_classes=3,
        pooling="mean",
        unfreeze_last_n_layers=1,
        backbone=backbone,
    )
    total_p, trainable_p = count_parameters(clf)
    assert total_p > trainable_p > 0  # Confirmed lower layers frozen, top layer + head trainable

    input_ids = torch.randint(1, 90, (2, 10))
    mask = torch.ones((2, 10), dtype=torch.long)
    mask[0, 7:] = 0

    embs = clf.extract_embeddings(input_ids, mask, normalize=True)
    assert embs.shape == (2, 32)
    assert torch.allclose(embs.norm(dim=-1), torch.ones(2), atol=1e-5)

    logits = clf(input_ids, mask)
    assert logits.shape == (2, 3)


def test_t5_summarizer_rouge_and_dataset_masking_offline(tmp_path: Path) -> None:
    # Test ROUGE + Lead-3
    articles = [
        "We propose SparseFormer for scientific NLP. It lowers memory by 50%. Experiments show strong ROUGE gains. Extra detail here."
    ]
    refs = ["We propose SparseFormer for scientific NLP achieving strong ROUGE gains."]
    lead = lead_n_baseline(articles, n_sentences=2)
    rouge = compute_rouge_metrics(lead, refs)
    assert rouge["rouge1"] > 0.5
    assert 0.0 <= rouge["rougeL"] <= 1.0

    # Tiny offline T5 model + minimal mock tokenizer to verify label -100 masking & fine_tune
    class _DummyTok:
        pad_token_id = 0

        def __call__(self, texts=None, text_target=None, max_length=16, padding=True, truncation=True, return_tensors="pt"):
            items = text_target if text_target is not None else texts
            n = len(items)
            ids = torch.randint(1, 50, (n, min(max_length, 12)))
            ids[:, -2:] = 0  # trailing pads
            mask = (ids != 0).long()
            return {"input_ids": ids, "attention_mask": mask}

        def batch_decode(self, ids, skip_special_tokens=True):
            return ["generated summary text" for _ in range(len(ids))]

    tok = _DummyTok()
    df = pd.DataFrame(
        [
            {"article_clean": "scientific paper body text " * 10, "abstract_clean": "concise scientific abstract"},
            {"article_clean": "another research paper body " * 10, "abstract_clean": "second scientific abstract"},
        ]
    )
    ds = ArxivSummarizationDataset(df, tok, max_source_len=12, max_target_len=12)
    assert (ds[0]["labels"][-2:] == -100).all()

    t5_cfg = T5Config(
        vocab_size=64,
        d_model=32,
        d_kv=8,
        d_ff=64,
        num_layers=1,
        num_heads=4,
        decoder_start_token_id=0,
        pad_token_id=0,
        eos_token_id=1,
    )
    tiny_t5 = T5ForConditionalGeneration(t5_cfg)
    summarizer = T5ScientificSummarizer(model_name="tiny-t5", tokenizer=tok, model=tiny_t5)
    ckpt = tmp_path / "t5_ckpt.pt"
    ft_res = summarizer.fine_tune(df, df, checkpoint_path=ckpt, epochs=1, batch_size=2, max_source_len=12, max_target_len=12)
    assert ckpt.exists()
    assert len(ft_res["history"]) == 1


def test_tfidf_retrieval_metrics_math() -> None:
    corpus = [
        "Transformer attention mechanism for neural machine translation",
        "Mitochondria generate cellular ATP through oxidative phosphorylation",
        "Convolutional neural networks for image object detection",
    ]
    queries = [
        "neural machine translation attention",
        "mitochondria ATP phosphorylation",
    ]
    gold = [0, 1]
    emb = TFIDFEmbedder()
    metrics, qv, cv = evaluate_retrieval(emb, queries, corpus, gold)
    assert metrics["recall_at_1"] == pytest.approx(1.0)
    assert metrics["mrr_at_10"] == pytest.approx(1.0)
    assert metrics["ndcg_at_10"] == pytest.approx(1.0 / math.log2(2.0))
    assert qv.shape[0] == 2 and cv.shape[0] == 3
