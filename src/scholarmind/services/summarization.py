"""Long-document T5 summarization service reusing Experiment D's `T5ScientificSummarizer`."""

from __future__ import annotations

from pathlib import Path

import torch

from scholarmind.core.logging import get_logger
from scholarmind.core.schemas import DocumentChunk, ParsedPaper, SummaryResponse
from scholarmind.models.t5.t5_summarizer import T5ScientificSummarizer

logger = get_logger(__name__)


class PaperSummarizationService:
    """Summarizes short and long research papers via our existing T5 model and chunked Map-Reduce."""

    def __init__(
        self,
        model_name: str = "google/flan-t5-small",
        finetuned_checkpoint: Path | None = None,
        hf_cache_dir: Path | None = None,
        summarizer: T5ScientificSummarizer | None = None,
    ) -> None:
        self.model_name = model_name
        self.finetuned_checkpoint = finetuned_checkpoint
        self.hf_cache_dir = hf_cache_dir
        self._summarizer: T5ScientificSummarizer | None = summarizer
        self.used_finetuned_checkpoint = False

    def _get_summarizer(self) -> T5ScientificSummarizer:
        if self._summarizer is not None:
            return self._summarizer

        kwargs: dict = {}
        if self.hf_cache_dir and (self.hf_cache_dir / "hub").exists():
            kwargs["cache_dir"] = str(self.hf_cache_dir / "hub")
            kwargs["local_files_only"] = True

        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        tok = AutoTokenizer.from_pretrained(self.model_name, **kwargs)
        mdl = AutoModelForSeq2SeqLM.from_pretrained(self.model_name, **kwargs)
        summarizer = T5ScientificSummarizer(
            model_name=self.model_name,
            device=torch.device("cpu"),
            tokenizer=tok,
            model=mdl,
        )
        if self.finetuned_checkpoint and self.finetuned_checkpoint.exists():
            ckpt = torch.load(self.finetuned_checkpoint, map_location="cpu", weights_only=False)
            if "model_state_dict" in ckpt:
                summarizer.model.load_state_dict(ckpt["model_state_dict"])
                self.used_finetuned_checkpoint = True
                logger.info(
                    "Loaded Experiment D fine-tuned T5 checkpoint -> %s",
                    self.finetuned_checkpoint,
                )

        self._summarizer = summarizer
        return summarizer

    def summarize_paper(
        self,
        paper: ParsedPaper,
        chunks: list[DocumentChunk],
        strategy: str = "auto",
        max_chunks: int = 4,
    ) -> SummaryResponse:
        """Generate a paper summary using T5 single-pass or chunk-based hierarchical Map-Reduce."""
        summarizer = self._get_summarizer()
        full_text = "\n\n".join(p.text for p in paper.pages if p.text.strip())

        chosen_strategy = strategy.lower().strip()
        if chosen_strategy == "auto":
            chosen_strategy = "hierarchical" if (paper.word_count > 600 and len(chunks) > 2) else "single_pass"

        if chosen_strategy == "hierarchical" and len(chunks) > 1:
            # Select representative chunks spanning Intro, Methodology, Results, and Conclusion
            # (skipping pure references)
            content_chunks = [c for c in chunks if c.section_type != "references"] or chunks
            if len(content_chunks) <= max_chunks:
                selected = content_chunks
            else:
                step = (len(content_chunks) - 1) / max(1, max_chunks - 1)
                selected = [content_chunks[int(round(i * step))] for i in range(max_chunks)]

            partial_summaries = summarizer.summarize_batch(
                [c.text for c in selected],
                max_source_len=448,
                max_new_tokens=72,
                min_length=20,
                num_beams=2,
                batch_size=len(selected),
                prefix="summarize this research paper section: ",
            )
            combined_digest = " ".join(partial_summaries)
            final_summary = summarizer.summarize_batch(
                [combined_digest],
                max_source_len=448,
                max_new_tokens=130,
                min_length=35,
                num_beams=2,
                batch_size=1,
                prefix="summarize scientific paper: ",
            )[0]
            chunks_processed = len(selected)
        else:
            final_summary = summarizer.summarize_batch(
                [full_text or paper.abstract],
                max_source_len=512,
                max_new_tokens=130,
                min_length=35,
                num_beams=2,
                batch_size=1,
            )[0]
            chosen_strategy = "single_pass"
            chunks_processed = 1

        return SummaryResponse(
            paper_id=paper.paper_id,
            paper_title=paper.title,
            strategy=chosen_strategy,
            model_name=self.model_name,
            used_finetuned_checkpoint=self.used_finetuned_checkpoint,
            page_count=paper.page_count,
            chunks_processed=chunks_processed,
            summary=final_summary,
        )
