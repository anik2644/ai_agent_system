"""Lightweight, rule-based query parser.

Extracts intent and named entities from the user's raw natural-language
input.  Can be replaced later with an LLM-based parser without changing
the rest of the pipeline.
"""

from __future__ import annotations

import re

import structlog

from agent_system.core.models import QueryContext, QueryIntent

logger = structlog.get_logger(__name__)

# Simple keyword → intent mapping (extend as needed)
_INTENT_PATTERNS: list[tuple[re.Pattern[str], QueryIntent]] = [
    (re.compile(r"\b(find|search|list|show)\b.*\bdoctor", re.I), QueryIntent.SEARCH_DOCTORS),
    (re.compile(r"\bdetail|info|about\b.*\b(dr\.?|doctor)\b", re.I), QueryIntent.DOCTOR_DETAIL),
    (re.compile(r"\bhealth|symptom|treatment\b", re.I), QueryIntent.GENERAL_HEALTH),
]

# Entity extraction patterns
_CITY_PATTERN = re.compile(r"\bin\s+([A-Z][a-z]+(?:\s[A-Z][a-z]+)*)", re.U)
_SPECIALTY_PATTERN = re.compile(
    r"\b(cardiologist|dermatologist|pediatrician|neurologist|orthopedic|"
    r"dentist|psychiatrist|oncologist|surgeon|general\s*practitioner)\b",
    re.I,
)


class QueryParser:
    """Stateless service that converts a raw string into a `QueryContext`."""

    def parse(self, raw_query: str) -> QueryContext:
        """Parse *raw_query* and return a rich `QueryContext`."""
        intent = self._detect_intent(raw_query)
        entities = self._extract_entities(raw_query)

        context = QueryContext(
            raw_query=raw_query,
            intent=intent,
            entities=entities,
        )
        logger.info(
            "query_parsed",
            query_id=str(context.query_id),
            intent=intent.value,
            entities=entities,
        )
        return context

    # -- private helpers --------------------------------------------------

    @staticmethod
    def _detect_intent(text: str) -> QueryIntent:
        for pattern, intent in _INTENT_PATTERNS:
            if pattern.search(text):
                return intent
        return QueryIntent.UNKNOWN

    @staticmethod
    def _extract_entities(text: str) -> dict[str, str]:
        entities: dict[str, str] = {}

        city_match = _CITY_PATTERN.search(text)
        if city_match:
            entities["city"] = city_match.group(1)

        spec_match = _SPECIALTY_PATTERN.search(text)
        if spec_match:
            entities["specialty"] = spec_match.group(1).lower()

        return entities