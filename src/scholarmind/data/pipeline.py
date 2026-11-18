"""End-to-end preprocessing pipeline orchestrating cleaning, NLP features, chunking, splits, and storage."""

from __future__ import annotations

from typing import Any

import pandas as pd

from scholarmind.core.config import AppConfig
from scholarmind.core.logging import get_logger
from scholarmind.core.schemas import SplitManifest
from scholarmind.data.splits import (
    compute_text_hash,
    deduplicate_dataframe,
    group_aware_split,
    remove_cross_split_leakage,
    verify_no_leakage,
)
from scholarmind.data.storage import save_dataset_splits
from scholarmind.nlp.chunking import HierarchicalSectionChunker
from scholarmind.nlp.cleaning import (
    CleaningProfile,
    ScientificTextCleaner,
    clean_qa_example,
)
from scholarmind.nlp.normalization import ScientificNormalizer
from scholarmind.nlp.segmentation import ScientificSentenceSegmenter
from scholarmind.nlp.tokenization import (
    ScientificTokenizer,
    Vocabulary,
    estimate_subword_tokens,
)

logger = get_logger(__name__)


class DataPreprocessingPipeline:
    """Applies task-appropriate NLP preprocessing, leakage-free splitting, and Parquet export."""

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.config.ensure_directories()

        self.rag_cleaner = ScientificTextCleaner(profile=CleaningProfile.RAG_SEMANTIC)
        self.summ_cleaner = ScientificTextCleaner(profile=CleaningProfile.SUMMARIZATION_SCIENTIFIC)
        self.classical_cleaner = ScientificTextCleaner(profile=CleaningProfile.CLASSIFICATION_CLASSICAL)
        self.segmenter = ScientificSentenceSegmenter(
            backend="auto",
            spacy_model=config.spacy_model,
        )
        self.tokenizer = ScientificTokenizer(
            backend="scientific_regex",
            lowercase=True,
            strip_punctuation=True,
        )
        self.normalizer = ScientificNormalizer(
            backend="auto",
            preserve_negations=config.preserve_negations,
            spacy_model=config.spacy_model,
        )
        self.chunker = HierarchicalSectionChunker(
            max_tokens=config.chunking.max_tokens,
            min_tokens=config.chunking.min_tokens,
            overlap_sentences=config.chunking.overlap_sentences,
            parent_max_tokens=config.chunking.parent_max_tokens,
            include_breadcrumb=config.chunking.include_breadcrumb,
            segmenter=self.segmenter,
        )

    def process_arxiv_metadata(
        self,
        df: pd.DataFrame,
        scale: str,
        source: str,
    ) -> SplitManifest:
        """Clean, normalize, deduplicate, group-split, and build train-only RNN vocabulary for ArXiv."""
        logger.info("Preprocessing ArXiv metadata (%d raw rows)...", len(df))
        working = df.copy()

        # 1. Task-specific cleaning:
        #    - `title_clean` & `abstract_clean`: preserve case/punctuation for SciBERT & embeddings
        #    - `classical_text`: normalized/lemmatized with negations preserved for BM25/EDA/RNN baselines
        working["title_clean"] = working["title"].map(self.rag_cleaner.clean)
        working["abstract_clean"] = working["abstract"].map(self.rag_cleaner.clean)
        working["combined_text"] = working["title_clean"] + ". " + working["abstract_clean"]
        working["classical_tokens"] = working["combined_text"].map(
            lambda t: self.normalizer.normalize_text(
                self.classical_cleaner.clean(t),
                remove_stopwords=True,
                lemmatize=True,
            )
        )
        working["classical_text"] = working["classical_tokens"].map(" ".join)
        working["sentence_count"] = working["abstract_clean"].map(lambda t: len(self.segmenter.segment(t)))
        working["word_count"] = working["abstract_clean"].str.split().str.len()
        working["estimated_tokens"] = working["abstract_clean"].map(estimate_subword_tokens)
        working["text_hash"] = working["combined_text"].map(compute_text_hash)
        working["title_group"] = working["title_clean"].map(compute_text_hash)

        # Drop empty rows after cleaning
        working = working[(working["title_clean"] != "") & (working["abstract_clean"] != "")].reset_index(drop=True)

        # 2. Deduplicate BEFORE splitting
        working, removed_dups = deduplicate_dataframe(
            working,
            text_cols=["title_clean", "abstract_clean"],
            id_col="paper_id",
        )

        # 3. Leakage-free stratified group-aware split (grouped by normalized title hash)
        splits = group_aware_split(
            working,
            group_col="title_group",
            train_ratio=self.config.train_ratio,
            val_ratio=self.config.val_ratio,
            test_ratio=self.config.test_ratio,
            seed=self.config.seed,
            stratify_col="primary_category",
        )

        # 4. Strict leakage audit
        audit = verify_no_leakage(
            dataset_name="arxiv_metadata",
            splits=splits,
            id_col="paper_id",
            group_col="title_group",
            text_hash_col="text_hash",
            strict=True,
        )

        # 5. Fit RNN/BiLSTM Vocabulary STRICTLY on the training split (zero test/val vocab leakage!)
        vocab = Vocabulary(min_freq=2, max_size=30000)
        train_tokenized = [
            self.tokenizer.tokenize(self.classical_cleaner.clean(text))
            for text in splits["train"]["combined_text"]
        ]
        vocab.fit(train_tokenized)
        vocab_path = self.config.processed_dir / "arxiv_metadata" / "rnn_vocab.json"
        vocab.save(vocab_path)
        logger.info("Built train-only RNN/LSTM vocabulary (%d tokens) -> %s", len(vocab), vocab_path)

        return save_dataset_splits(
            dataset_name="arxiv_metadata",
            splits=splits,
            output_dir=self.config.processed_dir,
            source=source,
            scale=scale,
            preprocessing_profile="rag_semantic + classification_classical (train-only vocab)",
            leakage_audit=audit,
            seed=self.config.seed,
            notes={
                "deduplicated_rows": removed_dups,
                "vocab_size_train_only": len(vocab),
                "vocab_file": vocab_path.name,
                "segmentation_backend": self.segmenter.active_backend,
                "lemmatizer_backend": self.normalizer.active_backend,
            },
        )

    def process_squad_v2(
        self,
        raw_splits: dict[str, pd.DataFrame],
        scale: str,
    ) -> SplitManifest:
        """Offset-safe clean SQuAD 2.0, create leakage-free train/val/test by Wikipedia title."""
        logger.info("Preprocessing SQuAD 2.0 splits...")
        processed_raw: dict[str, pd.DataFrame] = {}
        invalid_offsets_dropped = 0

        for split_name, df in raw_splits.items():
            cleaned_rows = []
            for rec in df.to_dict(orient="records"):
                qa_clean = clean_qa_example(
                    context=rec["context"],
                    question=rec["question"],
                    answer_text=rec["answer_text"] if not rec["is_impossible"] else None,
                    answer_start=rec["answer_start"] if not rec["is_impossible"] else None,
                )
                if not qa_clean["offset_valid"]:
                    invalid_offsets_dropped += 1
                    continue

                cleaned_rows.append(
                    {
                        "id": rec["id"],
                        "title": str(rec["title"]).strip(),
                        "title_group": compute_text_hash(rec["title"]),
                        "context": qa_clean["context"],
                        "question": qa_clean["question"],
                        "answer_text": qa_clean["answer_text"],
                        "answer_start": qa_clean["answer_start"],
                        "is_impossible": bool(rec["is_impossible"]),
                        "context_words": len(qa_clean["context"].split()),
                        "context_est_tokens": estimate_subword_tokens(qa_clean["context"]),
                        "question_words": len(qa_clean["question"].split()),
                        "text_hash": compute_text_hash(qa_clean["context"] + " || " + qa_clean["question"]),
                    }
                )
            df_clean = pd.DataFrame(cleaned_rows)
            df_clean, _ = deduplicate_dataframe(df_clean, text_cols=["context", "question"], id_col="id")
            processed_raw[split_name] = df_clean

        # Hold official SQuAD 2.0 validation set strictly as `test`, and create a group-aware
        # `train` / `val` partition from official train grouped by `title_group` so no Wikipedia
        # article appears in both training and validation.
        official_train = processed_raw["train"]
        official_val_as_test = processed_raw["val"]

        if official_train["title_group"].nunique() >= 3:
            from sklearn.model_selection import GroupShuffleSplit

            gss = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=self.config.seed)
            tr_idx, va_idx = next(gss.split(official_train, groups=official_train["title_group"]))
            splits = {
                "train": official_train.iloc[tr_idx].reset_index(drop=True),
                "val": official_train.iloc[va_idx].reset_index(drop=True),
                "test": official_val_as_test.reset_index(drop=True),
            }
        else:
            splits = {
                "train": official_train.reset_index(drop=True),
                "val": official_val_as_test.reset_index(drop=True),
            }

        # Enforce zero leakage between official val (`test`) and `train`/`val`
        splits, cross_removed = remove_cross_split_leakage(
            splits,
            group_col="title_group",
            text_hash_col="text_hash",
            priority_order=("test", "val", "train"),
        )

        audit = verify_no_leakage(
            dataset_name="squad_v2",
            splits=splits,
            id_col="id",
            group_col="title_group",
            text_hash_col="text_hash",
            strict=True,
        )

        return save_dataset_splits(
            dataset_name="squad_v2",
            splits=splits,
            output_dir=self.config.processed_dir,
            source="huggingface:rajpurkar/squad_v2",
            scale=scale,
            preprocessing_profile="qa_preserve_offsets (no stopword removal, no lemmatization)",
            leakage_audit=audit,
            seed=self.config.seed,
            notes={
                "invalid_offsets_dropped": invalid_offsets_dropped,
                "cross_split_leaks_removed": cross_removed,
                "split_policy": "Official val held out as test; train split 85/15 by article title_group.",
            },
        )

    def process_arxiv_summarization(
        self,
        raw_splits: dict[str, pd.DataFrame],
        scale: str,
    ) -> tuple[SplitManifest, SplitManifest]:
        """Clean ArXiv Summarization splits, verify leakage, and generate scientific RAG chunks."""
        logger.info("Preprocessing ArXiv Summarization splits...")
        cleaned_splits: dict[str, pd.DataFrame] = {}

        for split_name, df in raw_splits.items():
            working = df.copy()
            working["article_clean"] = working["article"].map(self.summ_cleaner.clean)
            working["abstract_clean"] = working["abstract"].map(self.summ_cleaner.clean)
            working["article_words"] = working["article_clean"].str.split().str.len()
            working["abstract_words"] = working["abstract_clean"].str.split().str.len()
            working["article_est_tokens"] = working["article_clean"].map(estimate_subword_tokens)
            working["abstract_est_tokens"] = working["abstract_clean"].map(estimate_subword_tokens)
            working["compression_ratio"] = (
                working["article_words"] / working["abstract_words"].replace(0, 1)
            ).round(2)
            working["has_math"] = working["article_clean"].map(ScientificTextCleaner.has_math)
            working["has_citation"] = working["article_clean"].map(ScientificTextCleaner.has_citation)
            working["text_hash"] = working["abstract_clean"].map(compute_text_hash)

            # Filter degenerate records with very short articles or empty abstracts
            working = working[
                (working["article_words"] >= 50) & (working["abstract_words"] >= 15)
            ].reset_index(drop=True)
            working, _ = deduplicate_dataframe(
                working,
                text_cols=["abstract_clean"],
                id_col="paper_id",
            )
            cleaned_splits[split_name] = working

        cleaned_splits, cross_removed = remove_cross_split_leakage(
            cleaned_splits,
            group_col="paper_id",
            text_hash_col="text_hash",
            priority_order=("test", "val", "train"),
        )

        summ_audit = verify_no_leakage(
            dataset_name="arxiv_summarization",
            splits=cleaned_splits,
            id_col="paper_id",
            group_col="paper_id",
            text_hash_col="text_hash",
            strict=True,
        )

        summ_manifest = save_dataset_splits(
            dataset_name="arxiv_summarization",
            splits=cleaned_splits,
            output_dir=self.config.processed_dir,
            source="huggingface:ccdv/arxiv-summarization",
            scale=scale,
            preprocessing_profile="summarization_scientific (@xmath/@xcite normalized, case preserved)",
            leakage_audit=summ_audit,
            seed=self.config.seed,
            notes={"cross_split_leaks_removed": cross_removed},
        )

        # Scientific document chunking: chunk each split independently (preserving split isolation!)
        chunk_splits: dict[str, pd.DataFrame] = {}
        for split_name, df in cleaned_splits.items():
            chunk_rows: list[dict[str, Any]] = []
            for rec in df.to_dict(orient="records"):
                chunks = self.chunker.chunk_document(
                    text=rec["article_clean"],
                    paper_id=rec["paper_id"],
                    paper_title=rec["paper_id"],
                    include_parents=True,
                    extra_metadata={"source_split": split_name},
                )
                for ch in chunks:
                    row = ch.model_dump()
                    row["text_hash"] = compute_text_hash(ch.text)
                    row.pop("metadata", None)
                    chunk_rows.append(row)
            chunk_splits[split_name] = pd.DataFrame(chunk_rows)
            logger.info(
                "Chunked ArXiv Summarization '%s': %d papers -> %d parent+child chunks.",
                split_name,
                len(df),
                len(chunk_rows),
            )

        chunk_audit = verify_no_leakage(
            dataset_name="arxiv_scientific_chunks",
            splits=chunk_splits,
            id_col="chunk_id",
            group_col="paper_id",
            strict=True,
        )

        chunk_manifest = save_dataset_splits(
            dataset_name="arxiv_scientific_chunks",
            splits=chunk_splits,
            output_dir=self.config.processed_dir,
            source="derived:ccdv/arxiv-summarization",
            scale=scale,
            preprocessing_profile=f"hierarchical_section_chunking (max_tokens={self.config.chunking.max_tokens}, overlap_sents={self.config.chunking.overlap_sentences})",
            leakage_audit=chunk_audit,
            seed=self.config.seed,
            notes={
                "chunking_config": self.config.chunking.model_dump(),
            },
        )

        return summ_manifest, chunk_manifest

    def process_sciq(
        self,
        raw_splits: dict[str, pd.DataFrame],
        scale: str,
    ) -> SplitManifest:
        """Clean SciQ splits, add classical lexical views, and verify zero support/question leakage."""
        logger.info("Preprocessing SciQ splits...")
        cleaned_splits: dict[str, pd.DataFrame] = {}

        for split_name, df in raw_splits.items():
            working = df.copy()
            working["question_clean"] = working["question"].map(self.rag_cleaner.clean)
            working["correct_answer_clean"] = working["correct_answer"].map(self.rag_cleaner.clean)
            working["support_clean"] = working["support"].map(self.rag_cleaner.clean)
            working["has_support"] = working["support_clean"] != ""
            working["question_classical"] = working["question_clean"].map(
                lambda t: self.normalizer.normalize_to_string(self.classical_cleaner.clean(t))
            )
            working["support_classical"] = working["support_clean"].map(
                lambda t: self.normalizer.normalize_to_string(self.classical_cleaner.clean(t))
            )
            working["support_words"] = working["support_clean"].str.split().str.len()
            working["support_est_tokens"] = working["support_clean"].map(estimate_subword_tokens)
            working["support_group"] = working["support_clean"].map(compute_text_hash)
            working["text_hash"] = working["question_clean"].map(compute_text_hash)

            working = working[
                (working["question_clean"] != "") & (working["correct_answer_clean"] != "")
            ].reset_index(drop=True)
            working, _ = deduplicate_dataframe(working, text_cols=["question_clean"], id_col="id")
            cleaned_splits[split_name] = working

        # Prevent same support passage or duplicate question from leaking between train & val/test
        cleaned_splits, cross_removed = remove_cross_split_leakage(
            cleaned_splits,
            group_col="support_group",
            text_hash_col="text_hash",
            priority_order=("test", "val", "train"),
        )

        audit = verify_no_leakage(
            dataset_name="sciq",
            splits=cleaned_splits,
            id_col="id",
            group_col="support_group",
            text_hash_col="text_hash",
            strict=True,
        )

        return save_dataset_splits(
            dataset_name="sciq",
            splits=cleaned_splits,
            output_dir=self.config.processed_dir,
            source="huggingface:allenai/sciq",
            scale=scale,
            preprocessing_profile="rag_semantic + classical_lexical (negations preserved)",
            leakage_audit=audit,
            seed=self.config.seed,
            notes={"cross_split_leaks_removed": cross_removed},
        )
