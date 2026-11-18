"""Open-source LLM clients for grounded RAG generation and synthesis.

Supports:
1. Ollama (`qwen2.5:3b`, `llama3.2:3b`, `mistral`, etc.) when an Ollama daemon is active.
2. Local PyTorch `google/flan-t5-small` (cached in `artifacts/hf_cache`) as a 100% self-contained
   open-source CPU fallback requiring zero API keys or external daemons.
"""

from __future__ import annotations

import os
import socket
from abc import ABC, abstractmethod
from pathlib import Path
from urllib.parse import urlparse

import requests
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from scholarmind.core.device import get_device
from scholarmind.core.logging import get_logger

logger = get_logger(__name__)

# Process-wide cache so instruction-tuned Flan-T5 loads ONCE and is reused across requests
_SHARED_FLAN_MODELS: dict[tuple[str, str], tuple] = {}


class BaseLLMClient(ABC):
    """Abstract interface for open-source text generation backends."""

    backend_name: str

    @abstractmethod
    def generate(self, prompt: str, max_new_tokens: int = 180) -> str:
        """Generate a grounded text response given a constructed prompt."""


class OllamaLLMClient(BaseLLMClient):
    """Client for locally served open-weights LLMs via Ollama REST API."""

    def __init__(
        self,
        model: str = "qwen2.5:3b",
        base_url: str = "http://localhost:11434",
        timeout: int = 60,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.backend_name = f"ollama:{model}"

    def is_available(self) -> bool:
        try:
            parsed = urlparse(self.base_url)
            host = "127.0.0.1" if parsed.hostname in ("localhost", None) else parsed.hostname
            port = parsed.port or 11434
            with socket.create_connection((host, port), timeout=0.25):
                pass
            resp = requests.get(f"{self.base_url}/api/tags", timeout=1.0)
            return resp.status_code == 200
        except OSError:
            return False
        except Exception:
            return False

    def generate(self, prompt: str, max_new_tokens: int = 256) -> str:
        resp = requests.post(
            f"{self.base_url}/api/generate",
            json={
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.1,
                    "num_predict": max_new_tokens,
                },
            },
            timeout=self.timeout,
        )
        resp.raise_for_status()
        return str(resp.json().get("response", "")).strip()


class LocalFlanT5LLMClient(BaseLLMClient):
    """Self-contained open-source instruction-tuned Flan-T5 client running locally in PyTorch."""

    def __init__(
        self,
        model_name: str = "google/flan-t5-base",
        hf_cache_dir: Path | None = None,
        device: torch.device | None = None,
    ) -> None:
        self.model_name = model_name
        self.hf_cache_dir = hf_cache_dir
        self.device = device or get_device("auto")
        self.backend_name = f"local-pytorch:{model_name}"
        self._tokenizer = None
        self._model = None

    def _ensure_loaded(self) -> None:
        if self._model is not None and self._tokenizer is not None:
            return
        cache_key = (self.model_name, str(self.device))
        if cache_key in _SHARED_FLAN_MODELS:
            self._tokenizer, self._model = _SHARED_FLAN_MODELS[cache_key]
            return

        kwargs: dict = {}
        if self.hf_cache_dir and (self.hf_cache_dir / "hub").exists():
            kwargs["cache_dir"] = str(self.hf_cache_dir / "hub")
            kwargs["local_files_only"] = True
        logger.info("Loading shared open-source local LLM (%s) on %s...", self.model_name, self.device)
        tok = AutoTokenizer.from_pretrained(self.model_name, **kwargs)
        mdl = AutoModelForSeq2SeqLM.from_pretrained(self.model_name, **kwargs).to(self.device)
        mdl.eval()
        _SHARED_FLAN_MODELS[cache_key] = (tok, mdl)
        self._tokenizer, self._model = tok, mdl

    @torch.no_grad()
    def generate(self, prompt: str, max_new_tokens: int = 180) -> str:
        self._ensure_loaded()
        assert self._tokenizer is not None and self._model is not None
        enc = self._tokenizer(
            prompt,
            max_length=768,
            truncation=True,
            return_tensors="pt",
        ).to(self.device)
        out_ids = self._model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            num_beams=2,
            no_repeat_ngram_size=3,
            early_stopping=True,
        )
        return self._tokenizer.decode(out_ids[0], skip_special_tokens=True).strip()


def create_open_llm_client(
    preferred_provider: str | None = None,
    ollama_model: str = "qwen2.5:3b",
    flan_model: str = "google/flan-t5-small",
    hf_cache_dir: Path | None = None,
) -> BaseLLMClient:
    """Instantiate Ollama if configured and reachable; otherwise use local PyTorch Flan-T5."""
    provider = (preferred_provider or os.getenv("SCHOLARMIND_LLM_PROVIDER", "local")).lower().strip()
    if provider == "ollama":
        ollama = OllamaLLMClient(model=ollama_model)
        if ollama.is_available():
            logger.info("Connected to local Ollama open-source LLM (%s).", ollama_model)
            return ollama
        logger.warning(
            "Ollama daemon not reachable at localhost:11434; falling back to local open-source %s.",
            flan_model,
        )
    return LocalFlanT5LLMClient(model_name=flan_model, hf_cache_dir=hf_cache_dir)
