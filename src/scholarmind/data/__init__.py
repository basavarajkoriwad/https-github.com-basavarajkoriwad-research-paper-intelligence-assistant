"""Dataset acquisition, EDA, leakage-free splitting, and preprocessing pipelines."""

from scholarmind.data.arxiv_loader import (
    fetch_arxiv_api_sample,
    get_kaggle_manual_instructions,
    load_arxiv_jsonl,
)
from scholarmind.data.arxiv_summ_loader import load_arxiv_summarization
from scholarmind.data.eda import DatasetEDAAnalyzer
from scholarmind.data.sciq_loader import load_sciq
from scholarmind.data.splits import (
    compute_text_hash,
    deduplicate_dataframe,
    group_aware_split,
    verify_no_leakage,
)
from scholarmind.data.squad_loader import load_squad_v2
from scholarmind.data.storage import load_manifest, load_parquet, save_dataset_splits

__all__ = [
    "DatasetEDAAnalyzer",
    "compute_text_hash",
    "deduplicate_dataframe",
    "fetch_arxiv_api_sample",
    "get_kaggle_manual_instructions",
    "group_aware_split",
    "load_arxiv_jsonl",
    "load_arxiv_summarization",
    "load_manifest",
    "load_parquet",
    "load_sciq",
    "load_squad_v2",
    "save_dataset_splits",
    "verify_no_leakage",
]
