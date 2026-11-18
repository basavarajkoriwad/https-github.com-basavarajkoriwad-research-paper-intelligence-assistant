"""Task-aware stopword handling and lemmatization for scientific NLP."""

from __future__ import annotations

import re
from typing import Literal

from scholarmind.core.logging import get_logger
from scholarmind.nlp.tokenization import ScientificTokenizer

logger = get_logger(__name__)

# Critical negations and limitation/contrast discourse cues that MUST NOT be removed
# because stripping them flips scientific claims ("does NOT outperform") and erases
# limitation signals ("however", "limited", "without", "fails").
PRESERVED_NEGATIONS_AND_CUES: frozenset[str] = frozenset(
    {
        "no",
        "not",
        "nor",
        "never",
        "neither",
        "none",
        "nothing",
        "nowhere",
        "cannot",
        "can't",
        "won't",
        "don't",
        "doesn't",
        "didn't",
        "isn't",
        "aren't",
        "wasn't",
        "weren't",
        "haven't",
        "hasn't",
        "hadn't",
        "shouldn't",
        "wouldn't",
        "couldn't",
        "without",
        "however",
        "although",
        "though",
        "but",
        "yet",
        "despite",
        "except",
        "only",
        "few",
        "less",
        "least",
        "more",
        "most",
        "against",
        "under",
        "over",
    }
)

# Built-in fallback English stopword list in case NLTK stopwords corpus is not downloaded
_FALLBACK_ENGLISH_STOPWORDS: frozenset[str] = frozenset(
    {
        "a", "about", "above", "after", "again", "all", "am", "an", "and", "any", "are",
        "as", "at", "be", "because", "been", "before", "being", "below", "between", "both",
        "by", "can", "did", "do", "does", "doing", "down", "during", "each", "for", "from",
        "further", "had", "has", "have", "having", "he", "her", "here", "hers", "herself",
        "him", "himself", "his", "how", "i", "if", "in", "into", "is", "it", "its", "itself",
        "just", "me", "my", "myself", "now", "of", "off", "on", "once", "or", "other", "our",
        "ours", "ourselves", "out", "own", "s", "same", "she", "should", "so", "some", "such",
        "t", "than", "that", "the", "their", "theirs", "them", "themselves", "then", "there",
        "these", "they", "this", "those", "through", "to", "too", "until", "up", "very", "was",
        "we", "were", "what", "when", "where", "which", "while", "who", "whom", "why", "will",
        "with", "you", "your", "yours", "yourself", "yourselves",
    }
)


def should_apply_classical_normalization(task: str) -> bool:
    """Explicit engineering policy indicating when stopword removal / lemmatization is appropriate.

    - True  for: lexical BM25 indexing, TF-IDF baselines, EDA n-gram frequency analysis,
                 classical topic analysis, and optional compact RNN vocabulary reduction.
    - False for: BERT/SciBERT/DeBERTa extractive QA, T5/LongT5 abstractive summarization,
                 dense semantic embeddings (BGE-M3), and LLM RAG generation (where syntax,
                 function words, and morphology carry essential semantic signal).
    """
    classical_tasks = {
        "bm25",
        "tfidf",
        "eda_keywords",
        "topic_modeling",
        "classical_classification",
    }
    return task.lower().strip() in classical_tasks


class ScientificNormalizer:
    """Stopword filter and lemmatizer preserving scientific negations and technical terms."""

    def __init__(
        self,
        backend: Literal["auto", "spacy", "nltk", "rule"] = "auto",
        preserve_negations: bool = True,
        extra_stopwords: set[str] | None = None,
        spacy_model: str = "en_core_web_sm",
    ) -> None:
        self.preserve_negations = preserve_negations
        self.spacy_model = spacy_model
        self.tokenizer = ScientificTokenizer(
            backend="scientific_regex",
            lowercase=True,
            strip_punctuation=True,
        )
        self.stopwords = self._load_stopwords(extra_stopwords)
        self._spacy_nlp = None
        self._nltk_lemmatizer = None
        self._nltk_pos_ready = False
        self.active_backend = self._init_lemmatizer(backend)

    def _load_stopwords(self, extra_stopwords: set[str] | None) -> set[str]:
        try:
            from nltk.corpus import stopwords as nltk_stopwords

            sw = set(nltk_stopwords.words("english"))
        except (ImportError, LookupError):
            sw = set(_FALLBACK_ENGLISH_STOPWORDS)

        if extra_stopwords:
            sw.update(w.lower() for w in extra_stopwords)

        if self.preserve_negations:
            sw -= PRESERVED_NEGATIONS_AND_CUES

        return sw

    def _init_lemmatizer(self, backend: str) -> str:
        if backend in ("auto", "spacy"):
            try:
                import spacy

                try:
                    self._spacy_nlp = spacy.load(self.spacy_model, disable=["ner", "parser"])
                    return "spacy"
                except OSError:
                    if backend == "spacy":
                        logger.warning(
                            "spaCy model '%s' not installed; falling back to NLTK/rule lemmatizer.",
                            self.spacy_model,
                        )
            except ImportError:
                if backend == "spacy":
                    raise

        if backend in ("auto", "nltk"):
            try:
                import nltk
                from nltk.stem import WordNetLemmatizer

                nltk.data.find("corpora/wordnet")
                self._nltk_lemmatizer = WordNetLemmatizer()
                try:
                    nltk.data.find("taggers/averaged_perceptron_tagger_eng")
                    self._nltk_pos_ready = True
                except LookupError:
                    try:
                        nltk.data.find("taggers/averaged_perceptron_tagger")
                        self._nltk_pos_ready = True
                    except LookupError:
                        self._nltk_pos_ready = False
                return "nltk"
            except (ImportError, LookupError):
                if backend == "nltk":
                    logger.warning(
                        "NLTK WordNet corpus unavailable locally; using lightweight rule fallback."
                    )

        return "rule"

    @staticmethod
    def _penn_to_wordnet_pos(tag: str) -> str:
        if tag.startswith("J"):
            return "a"
        if tag.startswith("V"):
            return "v"
        if tag.startswith("R"):
            return "r"
        return "n"

    @staticmethod
    def _rule_lemmatize(token: str) -> str:
        """Conservative suffix-based fallback lemmatizer that avoids mangling acronyms."""
        if len(token) <= 3 or "-" in token or not token.isalpha():
            return token
        if token.endswith("ies") and len(token) > 4:
            return token[:-3] + "y"
        if token.endswith("sses") or token.endswith("xes") or token.endswith("ches"):
            return token[:-2]
        if token.endswith("s") and not token.endswith(("ss", "us", "is")):
            return token[:-1]
        return token

    def remove_stopwords(self, tokens: list[str]) -> list[str]:
        """Filter stopwords while preserving negations and scientific discourse cues."""
        return [tok for tok in tokens if tok.lower() not in self.stopwords]

    def lemmatize_tokens(self, tokens: list[str]) -> list[str]:
        """Lemmatize a sequence of tokens using the active NLP backend."""
        if not tokens:
            return []

        if self.active_backend == "spacy" and self._spacy_nlp is not None:
            doc = self._spacy_nlp(" ".join(tokens))
            return [tok.lemma_.lower() if tok.lemma_ != "-PRON-" else tok.text.lower() for tok in doc if not tok.is_space]

        if self.active_backend == "nltk" and self._nltk_lemmatizer is not None:
            if self._nltk_pos_ready:
                import nltk

                tagged = nltk.pos_tag(tokens)
                return [
                    self._nltk_lemmatizer.lemmatize(tok, self._penn_to_wordnet_pos(pos))
                    for tok, pos in tagged
                ]
            return [self._nltk_lemmatizer.lemmatize(tok) for tok in tokens]

        return [self._rule_lemmatize(tok) for tok in tokens]

    def normalize_text(
        self,
        text: str | None,
        remove_stopwords: bool = True,
        lemmatize: bool = True,
        min_token_len: int = 2,
    ) -> list[str]:
        """Run full classical normalization (tokenize -> optional lemmatize -> optional stopwords)."""
        if not text or not isinstance(text, str):
            return []

        tokens = self.tokenizer.tokenize(text)
        if lemmatize:
            tokens = self.lemmatize_tokens(tokens)
        if remove_stopwords:
            tokens = self.remove_stopwords(tokens)

        return [
            tok
            for tok in tokens
            if len(tok) >= min_token_len and not re.fullmatch(r"\d+", tok)
        ]

    def normalize_to_string(
        self,
        text: str | None,
        remove_stopwords: bool = True,
        lemmatize: bool = True,
    ) -> str:
        """Convenience wrapper returning space-joined normalized tokens."""
        return " ".join(
            self.normalize_text(
                text,
                remove_stopwords=remove_stopwords,
                lemmatize=lemmatize,
            )
        )
