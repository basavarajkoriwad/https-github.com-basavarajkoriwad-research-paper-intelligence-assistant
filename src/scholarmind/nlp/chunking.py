"""Scientific document section detection and hierarchical chunking for RAG and long-context NLP."""

from __future__ import annotations

import hashlib
import re

from scholarmind.core.schemas import DocumentChunk, ParsedSection
from scholarmind.nlp.cleaning import ScientificTextCleaner
from scholarmind.nlp.segmentation import ScientificSentenceSegmenter
from scholarmind.nlp.tokenization import estimate_subword_tokens

# Canonical scientific paper section taxonomy
_SECTION_TYPE_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("abstract", re.compile(r"\babstract\b", re.IGNORECASE)),
    ("introduction", re.compile(r"\b(?:introduction|overview|motivation)\b", re.IGNORECASE)),
    ("related_work", re.compile(r"\b(?:related\s+work|background|literature\s+review|prior\s+work)\b", re.IGNORECASE)),
    ("methodology", re.compile(r"\b(?:methods?|methodology|approach(?:es)?|architectures?|models?|frameworks?|formulations?|algorithms?|proposed)\b", re.IGNORECASE)),
    ("experiments", re.compile(r"\b(?:experiments?|experimental\s+setup|implementation|evaluation\s+setup|datasets?|training\s+details)\b", re.IGNORECASE)),
    ("results", re.compile(r"\b(?:results?|ablations?|empirical|performance|analysis|findings?)\b", re.IGNORECASE)),
    ("limitations", re.compile(r"\b(?:limitations?|threats\s+to\s+validity|broader\s+impacts?|ethical|failure\s+cases?)\b", re.IGNORECASE)),
    ("discussion", re.compile(r"\b(?:discussions?|future\s+work|open\s+problems?)\b", re.IGNORECASE)),
    ("conclusion", re.compile(r"\b(?:conclusions?|concluding\s+remarks?|summary)\b", re.IGNORECASE)),
    ("references", re.compile(r"\b(?:references?|bibliography|acknowledg(?:e?ments?)?)\b", re.IGNORECASE)),
]

# Matches Markdown headers (`## 3. Methodology`), numbered academic headers (`1. Introduction`,
# `3.2 Experimental Setup`), or standalone capitalized section titles on their own line.
_HEADER_LINE_PATTERN = re.compile(
    r"^(?:"
    r"#{1,4}\s+(?P<md_title>[^\n]{2,100})"
    r"|(?:(?:\d+(?:\.\d+)*|[IVX]+)\.?\s+)(?P<num_title>[A-Z][A-Za-z0-9\s,&:\-/]{2,80})"
    r"|(?P<bare_title>(?:Abstract|Introduction|Background|Related Work|Methodology|Methods|Approach|Experiments|Experimental Setup|Results|Discussion|Limitations|Future Work|Conclusion|Conclusions|References|Acknowledgments?))\.?"
    r")\s*$",
    re.IGNORECASE,
)


class ScientificSectionDetector:
    """Detects and normalizes canonical scientific sections from raw or parsed paper text."""

    @staticmethod
    def classify_section_type(title: str) -> str:
        """Map a free-form section heading onto the canonical scientific section taxonomy."""
        for section_type, pattern in _SECTION_TYPE_PATTERNS:
            if pattern.search(title):
                return section_type
        return "body"

    def split_into_sections(
        self,
        text: str,
        paper_id: str,
        default_title: str = "Body",
    ) -> list[ParsedSection]:
        """Segment a scientific paper string into ordered `ParsedSection` blocks."""
        if not text or not text.strip():
            return []

        lines = text.split("\n")
        sections: list[ParsedSection] = []
        current_title = default_title
        current_lines: list[str] = []
        current_start = 0
        cursor = 0

        def _flush(end_pos: int) -> None:
            body = "\n".join(current_lines).strip()
            if not body:
                return
            sec_order = len(sections)
            sec_type = self.classify_section_type(current_title)
            sections.append(
                ParsedSection(
                    section_id=f"{paper_id}::sec_{sec_order:02d}",
                    paper_id=paper_id,
                    section_title=current_title.strip(),
                    section_type=sec_type,
                    order=sec_order,
                    text=body,
                    char_start=current_start,
                    char_end=end_pos,
                )
            )

        for line in lines:
            stripped = line.strip()
            match = _HEADER_LINE_PATTERN.match(stripped) if stripped and len(stripped) <= 105 else None
            # Avoid misclassifying long prose sentences ending with a period as numbered headers
            is_valid_header = bool(match) and not (
                match.group("num_title") and stripped.endswith(".") and len(stripped.split()) > 10
            )
            if is_valid_header and match is not None:
                _flush(cursor)
                title = (
                    match.group("md_title")
                    or match.group("num_title")
                    or match.group("bare_title")
                    or default_title
                ).strip()
                current_title = title
                current_lines = []
                current_start = cursor
            else:
                current_lines.append(line)
            cursor += len(line) + 1

        _flush(cursor)

        if not sections:
            sections.append(
                ParsedSection(
                    section_id=f"{paper_id}::sec_00",
                    paper_id=paper_id,
                    section_title=default_title,
                    section_type=self.classify_section_type(default_title),
                    order=0,
                    text=text.strip(),
                    char_start=0,
                    char_end=len(text),
                )
            )

        return sections


class HierarchicalSectionChunker:
    """Parent-Child Section-Aware Chunker designed for scientific RAG & QA.

    Why this chunker outperforms naive character splitting on research papers:
    1. Respects section boundaries (never mixes Methodology with Limitations in one chunk).
    2. Aligns chunk boundaries to complete scientific sentences (using `ScientificSentenceSegmenter`).
    3. Adds sentence-aligned overlap to preserve cross-sentence claims and coreferences.
    4. Prefixes each chunk with a structured breadcrumb (`[Paper: ... | Section: ...]`) so
       embeddings retain document and section identity during retrieval.
    5. Links every fine-grained child chunk to a broader parent section chunk for context expansion.
    """

    def __init__(
        self,
        max_tokens: int = 384,
        min_tokens: int = 48,
        overlap_sentences: int = 2,
        parent_max_tokens: int = 1500,
        include_breadcrumb: bool = True,
        segmenter: ScientificSentenceSegmenter | None = None,
        section_detector: ScientificSectionDetector | None = None,
    ) -> None:
        if max_tokens <= 0:
            raise ValueError("max_tokens must be positive.")
        if overlap_sentences < 0:
            raise ValueError("overlap_sentences must be >= 0.")

        self.max_tokens = max_tokens
        self.min_tokens = min_tokens
        self.overlap_sentences = overlap_sentences
        self.parent_max_tokens = parent_max_tokens
        self.include_breadcrumb = include_breadcrumb
        self.segmenter = segmenter or ScientificSentenceSegmenter(backend="auto")
        self.section_detector = section_detector or ScientificSectionDetector()

    def _build_breadcrumb(self, paper_title: str, section_title: str) -> str:
        short_title = (paper_title[:90] + "...") if len(paper_title) > 93 else paper_title
        if short_title:
            return f"[Paper: {short_title} | Section: {section_title}]"
        return f"[Section: {section_title}]"

    def _pack_sentences(
        self,
        sentences: list[str],
        max_tokens: int,
        overlap_sentences: int,
    ) -> list[list[str]]:
        """Greedily pack sentences into chunks within `max_tokens` with sentence overlap."""
        if not sentences:
            return []

        chunks: list[list[str]] = []
        idx = 0
        n = len(sentences)

        while idx < n:
            current: list[str] = []
            current_tokens = 0
            start_idx = idx

            while idx < n:
                sent = sentences[idx]
                sent_tokens = estimate_subword_tokens(sent)

                if current and (current_tokens + sent_tokens) > max_tokens:
                    break

                # If a single very long sentence exceeds max_tokens, split by words safely
                if not current and sent_tokens > max_tokens:
                    words = sent.split()
                    max_words = max(20, int(max_tokens / 1.30))
                    for w_start in range(0, len(words), max_words):
                        sub_sent = " ".join(words[w_start : w_start + max_words])
                        chunks.append([sub_sent])
                    idx += 1
                    current = []
                    break

                current.append(sent)
                current_tokens += sent_tokens
                idx += 1

            if current:
                chunks.append(current)
                if idx < n and overlap_sentences > 0:
                    step_back = min(overlap_sentences, max(0, len(current) - 1))
                    next_idx = idx - step_back
                    # Always guarantee forward progress
                    idx = next_idx if next_idx > start_idx else idx

        return chunks

    def chunk_document(
        self,
        text: str,
        paper_id: str,
        paper_title: str = "",
        sections: list[ParsedSection] | None = None,
        include_parents: bool = True,
        extra_metadata: dict | None = None,
    ) -> list[DocumentChunk]:
        """Chunk a scientific document into parent and child `DocumentChunk` records."""
        parsed_sections = (
            sections
            if sections is not None
            else self.section_detector.split_into_sections(text=text, paper_id=paper_id)
        )

        all_chunks: list[DocumentChunk] = []
        global_child_idx = 0
        base_meta = dict(extra_metadata or {})

        for sec in parsed_sections:
            # Skip pure bibliography sections during RAG chunking if configured externally;
            # keep them tagged clearly as section_type="references" so downstream can filter.
            sentences = self.segmenter.segment(sec.text)
            if not sentences:
                continue

            parent_groups = self._pack_sentences(
                sentences=sentences,
                max_tokens=self.parent_max_tokens,
                overlap_sentences=0,
            )

            for p_idx, parent_sents in enumerate(parent_groups):
                parent_text = " ".join(parent_sents).strip()
                parent_id = f"{sec.section_id}::p{p_idx:02d}"
                breadcrumb = self._build_breadcrumb(paper_title, sec.section_title)
                parent_ctx = f"{breadcrumb}\n{parent_text}" if self.include_breadcrumb else parent_text

                if include_parents:
                    all_chunks.append(
                        DocumentChunk(
                            chunk_id=parent_id,
                            paper_id=paper_id,
                            parent_id=None,
                            chunk_level="parent",
                            paper_title=paper_title,
                            section_title=sec.section_title,
                            section_type=sec.section_type,
                            chunk_index=p_idx,
                            text=parent_text,
                            contextualized_text=parent_ctx,
                            char_start=sec.char_start,
                            char_end=sec.char_end,
                            word_count=len(parent_text.split()),
                            estimated_tokens=estimate_subword_tokens(parent_text),
                            sentence_count=len(parent_sents),
                            has_math=ScientificTextCleaner.has_math(parent_text),
                            has_citation=ScientificTextCleaner.has_citation(parent_text),
                            metadata={**base_meta, "section_order": sec.order},
                        )
                    )

                child_groups = self._pack_sentences(
                    sentences=parent_sents,
                    max_tokens=self.max_tokens,
                    overlap_sentences=self.overlap_sentences,
                )

                for c_sents in child_groups:
                    child_text = " ".join(c_sents).strip()
                    if not child_text:
                        continue
                    child_ctx = (
                        f"{breadcrumb}\n{child_text}" if self.include_breadcrumb else child_text
                    )
                    content_hash = hashlib.md5(
                        f"{paper_id}:{global_child_idx}:{child_text[:64]}".encode("utf-8")
                    ).hexdigest()[:8]
                    child_id = f"{paper_id}::c{global_child_idx:04d}_{content_hash}"

                    all_chunks.append(
                        DocumentChunk(
                            chunk_id=child_id,
                            paper_id=paper_id,
                            parent_id=parent_id,
                            chunk_level="child",
                            paper_title=paper_title,
                            section_title=sec.section_title,
                            section_type=sec.section_type,
                            chunk_index=global_child_idx,
                            text=child_text,
                            contextualized_text=child_ctx,
                            char_start=sec.char_start,
                            char_end=sec.char_end,
                            word_count=len(child_text.split()),
                            estimated_tokens=estimate_subword_tokens(child_text),
                            sentence_count=len(c_sents),
                            has_math=ScientificTextCleaner.has_math(child_text),
                            has_citation=ScientificTextCleaner.has_citation(child_text),
                            metadata={**base_meta, "section_order": sec.order},
                        )
                    )
                    global_child_idx += 1

        return all_chunks


class FixedWindowChunker:
    """Simple token-window baseline chunker retained for RAG ablation comparisons."""

    def __init__(self, window_words: int = 250, overlap_words: int = 50) -> None:
        if overlap_words >= window_words:
            raise ValueError("overlap_words must be strictly less than window_words.")
        self.window_words = window_words
        self.overlap_words = overlap_words

    def chunk_document(
        self,
        text: str,
        paper_id: str,
        paper_title: str = "",
    ) -> list[DocumentChunk]:
        words = (text or "").split()
        if not words:
            return []

        step = self.window_words - self.overlap_words
        chunks: list[DocumentChunk] = []
        for idx, start in enumerate(range(0, len(words), step)):
            window = words[start : start + self.window_words]
            chunk_text = " ".join(window)
            chunks.append(
                DocumentChunk(
                    chunk_id=f"{paper_id}::fixed_{idx:04d}",
                    paper_id=paper_id,
                    parent_id=None,
                    chunk_level="child",
                    paper_title=paper_title,
                    section_title="Body",
                    section_type="body",
                    chunk_index=idx,
                    text=chunk_text,
                    contextualized_text=chunk_text,
                    word_count=len(window),
                    estimated_tokens=estimate_subword_tokens(chunk_text),
                    sentence_count=max(1, chunk_text.count(".")),
                    has_math=ScientificTextCleaner.has_math(chunk_text),
                    has_citation=ScientificTextCleaner.has_citation(chunk_text),
                )
            )
            if start + self.window_words >= len(words):
                break
        return chunks
