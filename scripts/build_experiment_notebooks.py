"""Generates the 5 structured Jupyter notebooks for Experiments A-E under `notebooks/`."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _md(text: str) -> dict[str, Any]:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in text.strip().splitlines()],
    }


def _code(text: str) -> dict[str, Any]:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in text.strip().splitlines()],
    }


def _make_nb(cells: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3 (.venv)",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "version": "3.12",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def build_all_notebooks(notebooks_dir: Path) -> list[Path]:
    """Create all 5 clean, runnable Jupyter notebooks documenting architecture, math, and code."""
    notebooks_dir.mkdir(parents=True, exist_ok=True)
    created: list[Path] = []

    # --- Notebook 01: Experiment A (RNN) ---
    nb_a = _make_nb(
        [
            _md(
                """
# Experiment A: Sequential Text Processing & Vanilla RNN (PyTorch)

## 1. Objective
Implement a basic Elman **Recurrent Neural Network (RNN)** in PyTorch from scratch (`VanillaRNNClassifier`), train it on our leakage-free ArXiv scientific abstracts dataset, record training/validation losses, and compare against Majority-Class and TF-IDF + Logistic Regression baselines.

## 2. Model Architecture
```
Input Token IDs (B, T) -> nn.Embedding(V, 128, pad=0) -> Dropout(0.3)
  -> pack_padded_sequence -> nn.RNN(128, 128, nonlinearity='tanh')
  -> Pooling (Last Valid h_T  vs.  Masked Mean Pool)
  -> Dropout(0.3) -> Linear(128, 8 ArXiv Classes)
```

## 3. Mathematical Foundation
At each timestep $t \\in \\{1, \\dots, T\\}$, the hidden state $h_t \\in \\mathbb{R}^H$ updates sequentially:
$$h_t = \\tanh(W_{ih} x_t + b_{ih} + W_{hh} h_{t-1} + b_{hh}), \\quad h_0 = \\mathbf{0}$$

During **Backpropagation Through Time (BPTT)**, gradients flow backward via repeated Jacobian products:
$$\\frac{\\partial \\mathcal{L}}{\\partial h_t} = \\frac{\\partial \\mathcal{L}}{\\partial h_T} \\prod_{k=t+1}^{T} \\text{diag}(1 - h_k^2) W_{hh}$$
Because $|1 - h_k^2| \\le 1$, gradients decay exponentially across 200-word scientific abstracts (**vanishing gradients**), which we mitigate partly via gradient norm clipping (`max_norm=1.0`) and compare `last` vs. `mean` pooling.
                """
            ),
            _code(
                """
import json
from pathlib import Path
from scholarmind.core.config import load_config
from scholarmind.experiments.exp_a_rnn import run_experiment_a

cfg = load_config()
results = run_experiment_a(config=cfg, epochs=12, batch_size=32, seed=42)
print("Baselines:", json.dumps(results["baselines"], indent=2))
print("Vanilla RNN (last h_T) Test:", results["vanilla_rnn_last"]["test"])
print("Vanilla RNN (mean pool) Test:", results["vanilla_rnn_mean"]["test"])
                """
            ),
            _code(
                """
import pandas as pd
from IPython.display import Image, display

display(pd.DataFrame(results["vanilla_rnn_last"]["history"]))
display(Image(filename=results["loss_curve_plot"]))
                """
            ),
        ]
    )

    # --- Notebook 02: Experiment B (LSTM & BiLSTM) ---
    nb_b = _make_nb(
        [
            _md(
                """
# Experiment B: LSTM and BiLSTM + Additive Self-Attention (PyTorch)

## 1. Objective
Implement both a **Unidirectional LSTM** (`LSTMClassifier`) and a **2-Layer Bidirectional LSTM with Additive Self-Attention** (`BiLSTMAttentionClassifier`), inspect hidden states $h_T$ and memory cells $c_T$, visualize top-attended scientific tokens, and compare directly against Experiment A (Vanilla RNN) on identical splits and random seed.

## 2. Mathematical Foundation — Gated Memory Cells & Attention
An LSTM separates short-term exposed state $h_t$ from a long-term **additive memory cell highway** $c_t$:
$$f_t = \\sigma(W_f [h_{t-1}; x_t] + b_f), \\quad i_t = \\sigma(W_i [h_{t-1}; x_t] + b_i)$$
$$g_t = \\tanh(W_g [h_{t-1}; x_t] + b_g), \\quad c_t = f_t \\odot c_{t-1} + i_t \\odot g_t$$
$$o_t = \\sigma(W_o [h_{t-1}; x_t] + b_o), \\quad h_t = o_t \\odot \\tanh(c_t)$$

Because $\\frac{\\partial c_t}{\\partial c_{t-1}} = f_t$ (additive highway rather than repeated matrix multiplication), LSTMs retain long-range dependencies across scientific abstracts.

A **BiLSTM** concatenates forward and backward context $h_t = [\\overrightarrow{h}_t; \\overleftarrow{h}_t] \\in \\mathbb{R}^{2H}$, and **Additive Self-Attention** weights every token:
$$u_t = \\tanh(W_a h_t + b_a), \\quad \\alpha_t = \\text{softmax}_t(v_a^\\top u_t), \\quad z = \\sum_{t=1}^{T} \\alpha_t h_t$$
                """
            ),
            _code(
                """
import json
from scholarmind.core.config import load_config
from scholarmind.experiments.exp_b_lstm_bilstm import run_experiment_b

cfg = load_config()
results = run_experiment_b(config=cfg, epochs=12, batch_size=32, seed=42)
print("LSTM Test:", results["lstm"]["test"])
print("BiLSTM + Attention Test:", results["bilstm_attention"]["test"])
print("State & Attention Diagnostics:", json.dumps(results["state_and_attention_diagnostics"], indent=2))
                """
            ),
            _code(
                """
import pandas as pd
from IPython.display import Image, display

display(pd.DataFrame(results["bilstm_attention"]["history"]))
display(Image(filename=results["loss_curve_plot"]))
                """
            ),
        ]
    )

    # --- Notebook 03: Experiment C (BERT / SciBERT) ---
    nb_c = _make_nb(
        [
            _md(
                """
# Experiment C: Pretrained BERT (`SciBERT`) — Contextual Representations & Fine-Tuning

## 1. Objective
Load pretrained **`allenai/scibert_scivocab_uncased`**, empirically verify context-dependent token embeddings (disambiguating polysemous words like `kernel` and `attention`), and fine-tune the top Transformer layers + classification head on the exact same ArXiv classification benchmark as Experiments A & B.

## 2. Mathematical Foundation — Scaled Dot-Product Multi-Head Self-Attention
Every BERT layer lets every token attend directly to all other tokens in $O(1)$ sequential path length:
$$\\text{Attention}(Q, K, V) = \\text{softmax}\\left(\\frac{Q K^\\top}{\\sqrt{d_k}}\\right) V$$
$$\\text{MultiHead}(H) = \\text{Concat}(\\text{head}_1, \\dots, \\text{head}_h) W^O$$

Attention-masked mean pooling aggregates final layer hidden states into a fixed scientific paper vector:
$$e_{\\text{doc}} = \\frac{\\sum_{i=1}^{T} m_i h_i^{(L)}}{\\sum_{i=1}^{T} m_i}$$
                """
            ),
            _code(
                """
import json
from scholarmind.core.config import load_config
from scholarmind.experiments.exp_c_bert import run_experiment_c

cfg = load_config()
results = run_experiment_c(config=cfg, epochs=4, batch_size=16, unfreeze_last_n_layers=2, seed=42)
print("Contextual Probes:", json.dumps(results["contextual_embedding_probe"], indent=2))
print("SciBERT Fine-Tuned Test:", results["scibert_finetune"]["test"])
                """
            ),
            _code(
                """
import pandas as pd
from IPython.display import Image, display

display(pd.DataFrame(results["scibert_finetune"]["history"]))
display(Image(filename=results["loss_curve_plot"]))
                """
            ),
        ]
    )

    # --- Notebook 04: Experiment D (T5 Summarization) ---
    nb_d = _make_nb(
        [
            _md(
                """
# Experiment D: T5 Transformer Summarization on ArXiv Papers

## 1. Objective
Evaluate **`google/flan-t5-small`** on the processed **`ccdv/arxiv-summarization`** dataset across four regimes:
1. Extractive **Lead-3** baseline
2. Pretrained zero-shot single-pass T5 inference
3. Pretrained **Hierarchical Map-Reduce** T5 inference (handling >4,096-token papers on CPU)
4. **PyTorch Fine-Tuned T5** (recording train/val cross-entropy loss and checkpointing best weights)

## 2. Mathematical Foundation — Encoder-Decoder Seq2Seq & ROUGE
T5 models conditional sequence generation via causal masked decoder self-attention + encoder-decoder cross-attention:
$$\\mathcal{L}(\\theta) = -\\frac{1}{M} \\sum_{t=1}^{M} \\log P_\\theta(y_t \\mid y_{<t}, x)$$
Padding positions in `labels` are masked with `-100`. Summarization quality is evaluated using **ROUGE-1**, **ROUGE-2** (n-gram overlap F1), and **ROUGE-L** (Longest Common Subsequence F1).
                """
            ),
            _code(
                """
import json
from scholarmind.core.config import load_config
from scholarmind.experiments.exp_d_t5 import run_experiment_d

cfg = load_config()
results = run_experiment_d(config=cfg, epochs=3, batch_size=4, seed=42)
print("Lead-3 ROUGE:", results["lead3_baseline_rouge"])
print("Pretrained T5 ROUGE:", results["pretrained_t5_rouge"])
print("Hierarchical Map-Reduce ROUGE:", results["hierarchical_map_reduce_rouge"])
print("Fine-Tuned T5 ROUGE:", results["finetuned_t5_rouge"])
                """
            ),
            _code(
                """
import pandas as pd
from IPython.display import Image, display

display(pd.DataFrame(results["finetuning"]["history"]))
display(Image(filename=results["loss_curve_plot"]))
print(json.dumps(results["sample_summaries"][0], indent=2))
                """
            ),
        ]
    )

    # --- Notebook 05: Experiment E (Embedding Comparison) ---
    nb_e = _make_nb(
        [
            _md(
                """
# Experiment E: Embedding Comparison for Scientific Document Retrieval

## 1. Objective
Empirically benchmark **TF-IDF**, **Pretrained BERT (`SciBERT` `[CLS]` vs. Mean pooling)**, and **Sentence Transformers (`BAAI/bge-small-en-v1.5`)** across two real scientific retrieval tasks:
1. **ArXiv Title → Abstract Retrieval**
2. **SciQ Question → Supporting Scientific Passage Retrieval**

## 2. Mathematical Foundation — Vector Spaces & Ranking Metrics
- **TF-IDF:** $\\text{tfidf}(t, d) = (1 + \\log \\text{tf}(t, d)) \\cdot \\log \\frac{1 + N}{1 + \\text{df}(t)}$
- **Bi-Encoder Contrastive Loss (Sentence Transformers):** $\\mathcal{L}_{\\text{InfoNCE}} = -\\log \\frac{\\exp(\\text{sim}(q, d^+)/\\tau)}{\\sum_j \\exp(\\text{sim}(q, d_j)/\\tau)}$
- **Ranking Metrics:**
$$\\text{Recall}@k = \\frac{1}{Q}\\sum_{i=1}^{Q} \\mathbb{1}[r_i \\le k], \\quad \\text{MRR}@10 = \\frac{1}{Q}\\sum_{i=1}^{Q} \\frac{\\mathbb{1}[r_i \\le 10]}{r_i}, \\quad \\text{nDCG}@10 = \\frac{1}{Q}\\sum_{i=1}^{Q} \\frac{\\mathbb{1}[r_i \\le 10]}{\\log_2(r_i + 1)}$$
                """
            ),
            _code(
                """
import pandas as pd
from scholarmind.core.config import load_config
from scholarmind.experiments.exp_e_embeddings import run_experiment_e

cfg = load_config()
results = run_experiment_e(config=cfg, seed=42)
print("=== ArXiv Title -> Abstract Retrieval ===")
display(pd.DataFrame(results["arxiv_title_to_abstract"]))
print("=== SciQ Question -> Supporting Passage Retrieval ===")
display(pd.DataFrame(results["sciq_question_to_support"]))
                """
            ),
            _code(
                """
from IPython.display import Image, display

display(Image(filename=results["comparison_plot"]))
                """
            ),
        ]
    )

    mapping = [
        ("01_experiment_a_rnn.ipynb", nb_a),
        ("02_experiment_b_lstm_bilstm.ipynb", nb_b),
        ("03_experiment_c_bert.ipynb", nb_c),
        ("04_experiment_d_t5.ipynb", nb_d),
        ("05_experiment_e_embeddings.ipynb", nb_e),
    ]
    for filename, nb in mapping:
        out = notebooks_dir / filename
        out.write_text(json.dumps(nb, indent=2), encoding="utf-8")
        created.append(out)
    return created


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    paths = build_all_notebooks(root / "notebooks")
    for p in paths:
        print(f"Created {p}")
