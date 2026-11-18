"""Open-source LLM client abstraction (Ollama + local PyTorch Flan-T5 fallback)."""

from scholarmind.models.llm.open_llm import (
    BaseLLMClient,
    LocalFlanT5LLMClient,
    OllamaLLMClient,
    create_open_llm_client,
)

__all__ = [
    "BaseLLMClient",
    "LocalFlanT5LLMClient",
    "OllamaLLMClient",
    "create_open_llm_client",
]
