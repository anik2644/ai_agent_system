"""Generates text responses from the Qwen LLM.

Encapsulates the chat-template formatting, tokenisation, and
generation loop.
"""

from __future__ import annotations

from typing import Any

import structlog
import torch

from agent_system.config.settings import Settings
from agent_system.llm.model_loader import ModelLoader

logger = structlog.get_logger(__name__)


class ResponseGenerator:
    """Wraps Qwen generation behind a clean interface."""

    def __init__(self, model_loader: ModelLoader, settings: Settings) -> None:
        self._loader = model_loader
        self._settings = settings

    def generate(
        self,
        messages: list[dict[str, str]],
        tools: list[dict[str, Any]] | None = None,
        max_new_tokens: int | None = None,
    ) -> str:
        """Generate a text response given a chat *messages* list.

        Parameters
        ----------
        messages:
            OpenAI-style chat messages (system / user / assistant / tool).
        tools:
            Optional tool schema to enable function-calling.
        max_new_tokens:
            Override the default max_new_tokens from settings.
        """
        tokenizer = self._loader.tokenizer
        model = self._loader.model
        settings = self._settings
        max_tokens = max_new_tokens or settings.max_new_tokens

        # Build prompt using the chat template
        kwargs: dict[str, Any] = {
            "tokenize": False,
            "add_generation_prompt": True,
        }
        if tools:
            kwargs["tools"] = tools

        text = tokenizer.apply_chat_template(messages, **kwargs)

        inputs = tokenizer(text, return_tensors="pt").to(model.device)

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=True,
                temperature=settings.temperature,
                top_p=settings.top_p,
                repetition_penalty=settings.repetition_penalty,
            )

        # Decode only newly generated tokens
        generated_ids = output_ids[0][inputs["input_ids"].shape[1]:]
        response_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

        return response_text.strip()