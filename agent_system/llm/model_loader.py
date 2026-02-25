"""LLM model and tokenizer loading utilities."""

from __future__ import annotations

import structlog
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from agent_system.config.settings import Settings

logger = structlog.get_logger(__name__)


class ModelLoader:
    """Loads and caches the Qwen model and tokenizer."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._tokenizer: AutoTokenizer | None = None
        self._model: AutoModelForCausalLM | None = None

    @property
    def tokenizer(self) -> AutoTokenizer:
        if self._tokenizer is None:
            raise RuntimeError("Model not loaded. Call `load()` first.")
        return self._tokenizer

    @property
    def model(self) -> AutoModelForCausalLM:
        if self._model is None:
            raise RuntimeError("Model not loaded. Call `load()` first.")
        return self._model

    def load(self) -> None:
        """Load tokenizer and model into memory."""
        model_name = self._settings.model_name

        logger.info("loading_tokenizer", model=model_name)
        self._tokenizer = AutoTokenizer.from_pretrained(model_name)

        logger.info("loading_model", model=model_name)
        self._model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
        )

        logger.info("model_loaded_successfully", model=model_name)