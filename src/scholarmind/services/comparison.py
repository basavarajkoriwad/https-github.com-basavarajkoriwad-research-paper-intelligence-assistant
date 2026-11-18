"""Grounded Multi-Paper Comparison Service comparing methods, datasets, models, and results."""

from __future__ import annotations

from scholarmind.core.schemas import (
    CitationReference,
    PaperComparisonEntry,
    PaperComparisonResponse,
    ParsedPaper,
    ResearchAnalysisReport,
)


class PaperComparisonService:
    """Builds a structured comparison matrix across multiple research papers strictly from extracted evidence."""

    def compare(
        self,
        papers: list[ParsedPaper],
        analyses: list[ResearchAnalysisReport],
    ) -> PaperComparisonResponse:
        """Compare 2+ papers across objectives, methods, datasets, architectures, results, and limitations."""
        if len(papers) < 2:
            raise ValueError("At least 2 papers are required for multi-paper comparison.")

        matrix: list[PaperComparisonEntry] = []
        citations: list[CitationReference] = []
        dataset_sets: list[set[str]] = []
        model_sets: list[set[str]] = []

        for paper, analysis in zip(papers, analyses):
            primary_obj = (
                analysis.objectives[0].statement
                if analysis.objectives
                else "Not explicitly extracted from paper text."
            )
            method_sum = (
                " ".join(m.statement for m in analysis.methodologies[:2])
                if analysis.methodologies
                else "Not explicitly extracted from paper text."
            )
            reported_results = [
                f"[p. {f.page_number}] {f.statement}" for f in analysis.key_findings[:3]
            ]
            stated_lims = [
                f"[p. {lim.page_number}] {lim.statement}" for lim in analysis.explicit_limitations[:3]
            ]

            matrix.append(
                PaperComparisonEntry(
                    paper_id=paper.paper_id,
                    paper_title=paper.title,
                    page_count=paper.page_count,
                    primary_objective=primary_obj,
                    methodology_summary=method_sum,
                    datasets=analysis.datasets,
                    model_architectures=analysis.model_architectures,
                    reported_results=reported_results,
                    explicit_limitations=stated_lims,
                )
            )

            dataset_sets.append({d.lower() for d in analysis.datasets})
            model_sets.append({m.lower() for m in analysis.model_architectures})

            # Attach verifiable citations for key evidence from each paper
            for item in (analysis.objectives[:1] + analysis.key_findings[:1] + analysis.explicit_limitations[:1]):
                citations.append(
                    CitationReference(
                        paper_id=paper.paper_id,
                        paper_title=paper.title,
                        page_number=item.page_number,
                        section_title=item.section_title,
                        chunk_id=f"{paper.paper_id}::p{item.page_number:03d}",
                        quote=item.statement[:220],
                    )
                )

        # Preserve canonical casing for shared datasets and models
        canon_ds = {d.lower(): d for a in analyses for d in a.datasets}
        canon_mod = {m.lower(): m for a in analyses for m in a.model_architectures}

        shared_ds_keys = set.intersection(*dataset_sets) if all(dataset_sets) else set()
        shared_mod_keys = set.intersection(*model_sets) if all(model_sets) else set()
        shared_datasets = sorted(canon_ds[k] for k in shared_ds_keys)
        shared_models = sorted(canon_mod[k] for k in shared_mod_keys)

        # Build concise, 100% factual comparative summary (never fabricating benchmarks/gaps)
        summary_parts: list[str] = [
            f"Compared {len(papers)} research papers: "
            + "; ".join(f"'{p.title}' ({p.page_count} pp.)" for p in papers)
            + "."
        ]
        if shared_datasets:
            summary_parts.append(f"Shared datasets/benchmarks across all papers: {', '.join(shared_datasets)}.")
        else:
            summary_parts.append(
                "No single dataset/benchmark is shared across all compared papers (direct metric comparisons require caution)."
            )
        if shared_models:
            summary_parts.append(f"Shared model architectures/techniques: {', '.join(shared_models)}.")

        for entry in matrix:
            ds_label = ", ".join(entry.datasets) if entry.datasets else "none explicitly detected"
            mod_label = ", ".join(entry.model_architectures) if entry.model_architectures else "none explicitly detected"
            summary_parts.append(
                f"Paper '{entry.paper_title}' references architectures [{mod_label}] and datasets [{ds_label}]."
            )

        return PaperComparisonResponse(
            paper_ids=[p.paper_id for p in papers],
            comparison_matrix=matrix,
            shared_datasets=shared_datasets,
            shared_models=shared_models,
            comparative_summary=" ".join(summary_parts),
            citations=citations,
        )
