"""Grounded Research Analysis Service extracting objectives, methods, datasets, models, findings, limitations & future work.

Strict anti-fabrication policy:
- Every extracted objective, methodology, finding, limitation, and author future-work statement
  is verbatim-grounded in the paper's actual extracted pages and sections.
- Derived future directions are explicitly labeled `source_type="derived_from_limitation"` and
  anchored to a specific limitation sentence and page number present in the paper.
- If a category (e.g., limitations or datasets) does not appear in the paper text, the service
  reports that honestly in `extraction_notes` rather than inventing content.
"""

from __future__ import annotations

import re

from scholarmind.core.schemas import (
    GroundedFindingItem,
    ParsedPaper,
    ResearchAnalysisReport,
)
from scholarmind.nlp.segmentation import ScientificSentenceSegmenter

# Curated catalog of well-known ML/NLP/CV/Scientific datasets & benchmarks (matched verbatim only!)
_KNOWN_DATASETS: tuple[str, ...] = (
    "ArXiv", "SQuAD", "SQuAD 2.0", "SciQ", "PubMed", "PubMedQA", "BioASQ", "SciFact",
    "GLUE", "SuperGLUE", "MNLI", "SST-2", "CoNLL", "WMT14", "WMT16", "WMT", "ImageNet",
    "CIFAR-10", "CIFAR-100", "MNIST", "COCO", "MS MARCO", "BEIR", "MTEB", "MMLU",
    "GSM8K", "MATH", "HumanEval", "CNN/DailyMail", "XSum", "BookCorpus", "Wikipedia",
    "Common Crawl", "C4", "The Pile", "Natural Questions", "HotpotQA", "TriviaQA",
    "NFCorpus", "TREC-COVID", "CiteSeer", "Cora", "Semantic Scholar", "S2ORC",
)

# Curated catalog of model architectures (matched verbatim only!)
_KNOWN_MODELS: tuple[str, ...] = (
    "Transformer", "BERT", "SciBERT", "RoBERTa", "DeBERTa", "DeBERTa-v3", "DistilBERT",
    "ALBERT", "ELECTRA", "T5", "Flan-T5", "LongT5", "LED", "Longformer", "BART",
    "Pegasus", "GPT-2", "GPT-3", "GPT-4", "LLaMA", "Llama-2", "Llama-3", "Mistral",
    "Qwen", "Gemma", "Phi-3", "RNN", "LSTM", "BiLSTM", "GRU", "CNN", "ResNet",
    "ViT", "CLIP", "BGE", "BGE-M3", "SPECTER", "SPECTER2", "MiniLM", "LoRA", "QLoRA",
    "RAG", "FAISS", "BM25", "CRF", "SVM", "GNN", "GCN", "GAT", "Diffusion",
)

_OBJECTIVE_PAT = re.compile(
    r"\b(?:we\s+(?:propose|introduce|present|develop|investigate|study|design|address|explore|aim\s+to)|"
    r"our\s+(?:goal|objective|aim|contribution|work\s+focuses)|this\s+paper\s+(?:presents|proposes|introduces|investigates|studies|addresses))\b",
    re.IGNORECASE,
)

_METHOD_PAT = re.compile(
    r"\b(?:architecture|framework|pipeline|algorithm|mechanism|objective\s+function|loss\s+function|"
    r"fine-?tun(?:e|ing|ed)|pre-?train(?:ed|ing)|attention|encoder|decoder|embedding|chunking|"
    r"retrieval|cross-entropy|optimizer|AdamW|trained\s+using|formulate|consists\s+of|combining)\b",
    re.IGNORECASE,
)

_FINDING_PAT = re.compile(
    r"\b(?:outperforms?|achieves?|improves?|reduces?|surpasses?|demonstrates?|results\s+show|"
    r"empirical(?:ly)?|ROUGE|BLEU|F1|accuracy|precision|recall|nDCG|MRR|perplexity|gain(?:s)?\s+of|"
    r"\d+(?:\.\d+)?%)\b",
    re.IGNORECASE,
)

_LIMITATION_PAT = re.compile(
    r"\b(?:limitation(?:s)?|drawback(?:s)?|bottleneck(?:s)?|shortcoming(?:s)?|weakness(?:es)?|"
    r"fails?\s+to|struggles?\s+(?:to|with)|cannot|does\s+not\s+(?:generalize|scale|support|handle)|"
    r"has\s+not\s+been\s+(?:evaluated|tested)|limited\s+to|computational\s+(?:cost|overhead|burden)|"
    r"high\s+(?:memory|latency|compute)|threats?\s+to\s+validity|however,?\s+.*?(?:limit|cost|fail|challeng|remain|degrad))\b",
    re.IGNORECASE,
)

_FUTURE_WORK_PAT = re.compile(
    r"\b(?:future\s+work|in\s+the\s+future|we\s+leave\s+.*?\s+for\s+future|promising\s+(?:avenue|direction)|"
    r"extend(?:ing)?\s+(?:this|our)\s+(?:work|method|approach)|open\s+(?:question|problem|direction)|"
    r"plan\s+to\s+(?:explore|investigate|extend))\b",
    re.IGNORECASE,
)

_CUSTOM_DATASET_PAT = re.compile(
    r"\b(?:on|using|from)\s+the\s+([A-Z][A-Za-z0-9\-/]{2,28})\s+(?:dataset|benchmark|corpus)\b"
)


class ResearchAnalysisService:
    """Extracts structured, verifiable scientific insights strictly from parsed paper pages."""

    def __init__(self) -> None:
        self.segmenter = ScientificSentenceSegmenter(backend="auto")

    @staticmethod
    def _extract_catalog_matches(full_text: str, catalog: tuple[str, ...]) -> list[str]:
        found: list[str] = []
        seen_lower: set[str] = set()
        for item in catalog:
            pattern = re.compile(r"(?<![A-Za-z0-9])" + re.escape(item) + r"(?![A-Za-z0-9])", re.IGNORECASE)
            if pattern.search(full_text) and item.lower() not in seen_lower:
                found.append(item)
                seen_lower.add(item.lower())
        return found

    @staticmethod
    def _dedup_items(items: list[GroundedFindingItem], limit: int = 5) -> list[GroundedFindingItem]:
        unique: list[GroundedFindingItem] = []
        seen: set[str] = set()
        for item in items:
            key = re.sub(r"\s+", " ", item.statement.lower()).strip()[:120]
            if key not in seen and len(item.statement.split()) >= 6:
                seen.add(key)
                unique.append(item)
            if len(unique) >= limit:
                break
        return unique

    def analyze(self, paper: ParsedPaper) -> ResearchAnalysisReport:
        """Analyze a `ParsedPaper` into grounded objectives, methods, datasets, models, findings & limitations."""
        full_text = "\n\n".join(p.text for p in paper.pages if p.text.strip())

        # Extract explicitly mentioned datasets & model architectures
        datasets = self._extract_catalog_matches(full_text, _KNOWN_DATASETS)
        for match in _CUSTOM_DATASET_PAT.finditer(full_text):
            candidate = match.group(1).strip()
            if candidate.lower() not in {d.lower() for d in datasets}:
                datasets.append(candidate)

        models = self._extract_catalog_matches(full_text, _KNOWN_MODELS)

        objectives: list[GroundedFindingItem] = []
        methodologies: list[GroundedFindingItem] = []
        findings: list[GroundedFindingItem] = []
        limitations: list[GroundedFindingItem] = []
        future_dirs: list[GroundedFindingItem] = []
        notes: list[str] = []

        # Walk every parsed section (excluding pure bibliography) with exact page provenance
        for sec in paper.sections:
            if sec.section_type == "references":
                continue
            sentences = self.segmenter.segment(sec.text)
            for sent in sentences:
                clean_s = " ".join(sent.split())
                if len(clean_s.split()) < 6 or len(clean_s) > 480:
                    continue

                if sec.section_type in ("abstract", "introduction", "body") and _OBJECTIVE_PAT.search(clean_s):
                    objectives.append(
                        GroundedFindingItem(
                            statement=clean_s,
                            page_number=sec.page_number,
                            section_title=sec.section_title,
                            source_type="author_stated",
                        )
                    )

                if sec.section_type in ("methodology", "abstract", "body") and _METHOD_PAT.search(clean_s):
                    methodologies.append(
                        GroundedFindingItem(
                            statement=clean_s,
                            page_number=sec.page_number,
                            section_title=sec.section_title,
                            source_type="author_stated",
                        )
                    )

                if sec.section_type in ("results", "experiments", "abstract", "conclusion", "body") and _FINDING_PAT.search(clean_s):
                    findings.append(
                        GroundedFindingItem(
                            statement=clean_s,
                            page_number=sec.page_number,
                            section_title=sec.section_title,
                            source_type="author_stated",
                        )
                    )

                if sec.section_type == "limitations" or _LIMITATION_PAT.search(clean_s):
                    limitations.append(
                        GroundedFindingItem(
                            statement=clean_s,
                            page_number=sec.page_number,
                            section_title=sec.section_title,
                            source_type="author_stated",
                        )
                    )

                if _FUTURE_WORK_PAT.search(clean_s):
                    future_dirs.append(
                        GroundedFindingItem(
                            statement=clean_s,
                            page_number=sec.page_number,
                            section_title=sec.section_title,
                            source_type="author_stated",
                        )
                    )

        objectives = self._dedup_items(objectives, limit=4)
        methodologies = self._dedup_items(methodologies, limit=5)
        findings = self._dedup_items(findings, limit=5)
        limitations = self._dedup_items(limitations, limit=5)
        future_dirs = self._dedup_items(future_dirs, limit=4)

        # Derive grounded follow-up directions ONLY from actual stated limitations (never invented!)
        if len(future_dirs) < 3 and limitations:
            for lim in limitations[: 3 - len(future_dirs)]:
                future_dirs.append(
                    GroundedFindingItem(
                        statement=(
                            f"Address stated limitation (p. {lim.page_number}, {lim.section_title}): "
                            f"\"{lim.statement}\""
                        ),
                        page_number=lim.page_number,
                        section_title=lim.section_title,
                        source_type="derived_from_limitation",
                    )
                )

        if not datasets:
            notes.append("No recognized dataset or benchmark names were explicitly detected in extracted text.")
        if not models:
            notes.append("No recognized model architecture names were explicitly detected in extracted text.")
        if not limitations:
            notes.append("No explicit limitations section or limitation statements were detected in extracted text.")
        if not future_dirs:
            notes.append("No explicit future-work statements or limitation-derived directions could be extracted.")

        return ResearchAnalysisReport(
            paper_id=paper.paper_id,
            paper_title=paper.title,
            objectives=objectives,
            methodologies=methodologies,
            datasets=datasets,
            model_architectures=models,
            key_findings=findings,
            explicit_limitations=limitations,
            future_directions=future_dirs,
            extraction_notes=notes,
        )
