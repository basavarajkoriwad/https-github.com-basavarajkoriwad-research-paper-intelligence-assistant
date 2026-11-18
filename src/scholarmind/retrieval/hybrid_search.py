"""Hybrid semantic + BM25 lexical search engine using Reciprocal Rank Fusion (RRF)."""

from __future__ import annotations

from typing import Literal

from rank_bm25 import BM25Okapi

from scholarmind.core.schemas import RetrievedPassage
from scholarmind.nlp.tokenization import ScientificTokenizer
from scholarmind.retrieval.vector_store import FAISSVectorStore


class HybridSearchEngine:
    """Combines FAISS dense semantic retrieval with BM25 lexical matching via RRF.

    Grounded directly in Experiment E findings: dense Sentence Transformers capture
    paraphrases and semantic concepts while lexical scoring preserves exact scientific
    acronyms, dataset names, and metric identifiers.
    """

    def __init__(self, vector_store: FAISSVectorStore, rrf_k: int = 60) -> None:
        self.vector_store = vector_store
        self.rrf_k = rrf_k
        self.tokenizer = ScientificTokenizer(
            backend="scientific_regex",
            lowercase=True,
            strip_punctuation=True,
        )

    def _bm25_search(
        self,
        query: str,
        top_k: int,
        paper_ids: list[str] | None = None,
        section_types: list[str] | None = None,
    ) -> list[RetrievedPassage]:
        allowed_papers = set(paper_ids) if paper_ids else None
        allowed_sections = {s.lower() for s in section_types} if section_types else None

        filtered_meta = [
            m
            for m in self.vector_store.metadata
            if (allowed_papers is None or m["paper_id"] in allowed_papers)
            and (allowed_sections is None or m["section_type"].lower() in allowed_sections)
        ]
        if not filtered_meta:
            return []

        tokenized_docs = [self.tokenizer.tokenize(m["contextualized_text"]) for m in filtered_meta]
        q_tokens = self.tokenizer.tokenize(query)
        if not q_tokens:
            return []

        bm25 = BM25Okapi(tokenized_docs)
        raw_scores = bm25.get_scores(q_tokens)
        ranked_idx = sorted(range(len(raw_scores)), key=lambda i: raw_scores[i], reverse=True)

        results: list[RetrievedPassage] = []
        for idx in ranked_idx[:top_k]:
            sc = float(raw_scores[idx])
            m = filtered_meta[idx]
            results.append(
                RetrievedPassage(
                    rank=len(results) + 1,
                    chunk_id=m["chunk_id"],
                    paper_id=m["paper_id"],
                    document_name=m.get("document_name") or m["paper_title"],
                    paper_title=m["paper_title"],
                    section_title=m["section_title"],
                    section_type=m["section_type"],
                    page_number=int(m["page_number"]),
                    score=round(sc, 4),
                    dense_score=0.0,
                    lexical_score=round(sc, 4),
                    text=m["text"],
                )
            )
        return results

    def search(
        self,
        query: str,
        top_k: int = 5,
        mode: Literal["hybrid", "dense", "lexical"] = "hybrid",
        paper_ids: list[str] | None = None,
        section_types: list[str] | None = None,
    ) -> list[RetrievedPassage]:
        """Run dense, lexical, or RRF-fused hybrid retrieval over indexed scientific chunks."""
        if not query or not query.strip() or self.vector_store.size == 0:
            return []

        if mode == "dense":
            return self.vector_store.search(
                query=query,
                top_k=top_k,
                paper_ids=paper_ids,
                section_types=section_types,
            )

        if mode == "lexical":
            return self._bm25_search(
                query=query,
                top_k=top_k,
                paper_ids=paper_ids,
                section_types=section_types,
            )

        candidate_k = min(self.vector_store.size, max(top_k * 4, 20))
        dense_hits = self.vector_store.search(
            query=query,
            top_k=candidate_k,
            paper_ids=paper_ids,
            section_types=section_types,
        )
        bm25_hits = self._bm25_search(
            query=query,
            top_k=candidate_k,
            paper_ids=paper_ids,
            section_types=section_types,
        )

        # Reciprocal Rank Fusion preserving exact dense & lexical component scores
        fused: dict[str, dict] = {}
        for rank, hit in enumerate(dense_hits, start=1):
            fused[hit.chunk_id] = {
                "passage": hit,
                "rrf": 1.0 / (self.rrf_k + rank),
                "dense": hit.dense_score,
                "lexical": 0.0,
            }

        for rank, hit in enumerate(bm25_hits, start=1):
            if hit.chunk_id in fused:
                # Only boost via BM25 if lexical score is strictly positive
                if hit.lexical_score > 0:
                    fused[hit.chunk_id]["rrf"] += 1.0 / (self.rrf_k + rank)
                fused[hit.chunk_id]["lexical"] = hit.lexical_score
            else:
                fused[hit.chunk_id] = {
                    "passage": hit,
                    "rrf": (1.0 / (self.rrf_k + rank)) if hit.lexical_score > 0 else 0.0,
                    "dense": 0.0,
                    "lexical": hit.lexical_score,
                }

        ranked = sorted(
            fused.values(),
            key=lambda item: (item["rrf"], item["dense"], item["lexical"]),
            reverse=True,
        )

        output: list[RetrievedPassage] = []
        for rank_idx, item in enumerate(ranked[:top_k], start=1):
            p: RetrievedPassage = item["passage"]
            output.append(
                p.model_copy(
                    update={
                        "rank": rank_idx,
                        "score": round(float(item["rrf"]), 5),
                        "dense_score": round(float(item["dense"]), 4),
                        "lexical_score": round(float(item["lexical"]), 4),
                    }
                )
            )
        return output
