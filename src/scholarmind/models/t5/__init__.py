"""T5 encoder-decoder scientific summarization, fine-tuning, and ROUGE evaluation."""

from scholarmind.models.t5.t5_summarizer import (
    ArxivSummarizationDataset,
    T5ScientificSummarizer,
    compute_rouge_metrics,
    lead_n_baseline,
)

__all__ = [
    "ArxivSummarizationDataset",
    "T5ScientificSummarizer",
    "compute_rouge_metrics",
    "lead_n_baseline",
]
