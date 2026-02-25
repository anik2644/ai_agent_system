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

# Simple keyword → intent mapping
_INTENT_PATTERNS: list[tuple[re.Pattern[str], QueryIntent]] = [
    (re.compile(r"\b(detail|info|about)\b.*\b(dr\.?|doctor)\s*(number|#|id)?\s*\d+", re.I), QueryIntent.DOCTOR_DETAIL),
    (re.compile(r"\b(find|search|list|show|need|looking)\b.*\bdoctor", re.I), QueryIntent.SEARCH_DOCTORS),
    (re.compile(r"\b(find|search|show|need|looking)\b.*\b(cardiolog|dermatolog|neurolog|dentist|orthoped|surgeon|pediatric|psychiatr|oncolog|general\s*practitioner)", re.I), QueryIntent.SEARCH_DOCTORS),
    (re.compile(r"\bwhat\b.*\b(location|cit)", re.I), QueryIntent.LIST_LOCATIONS),
    (re.compile(r"\bwhat\b.*\bspeciali[sz]ation", re.I), QueryIntent.LIST_SPECIALIZATIONS),
    (re.compile(r"\blocation", re.I), QueryIntent.LIST_LOCATIONS),
    (re.compile(r"\bspeciali[sz]ation", re.I), QueryIntent.LIST_SPECIALIZATIONS),
    (re.compile(r"\bhealth|symptom|treatment|pain|hurt|breaking out\b", re.I), QueryIntent.GENERAL_HEALTH),
]

# Entity extraction patterns
_CITY_PATTERN = re.compile(r"\bin\s+([A-Z][a-z]+(?:\s[A-Z][a-z]+)*)", re.U)
_SPECIALTY_PATTERN = re.compile(
    r"\b(cardiolog(?:ist|y)?|dermatolog(?:ist|y)?|pediatric(?:ian|s)?|neurolog(?:ist|y)?|orthoped(?:ic)?|"
    r"dentist(?:ry)?|psychiatr(?:ist|y)?|oncolog(?:ist|y)?|surgeon|general\s*practitioner|"
    r"eye\s*doctor|skin\s*specialist|brain\s*surgeon)\b",
    re.I,
)
_DOCTOR_ID_PATTERN = re.compile(r"\bdoctor\s*(?:number|#|id)?\s*(\d+)", re.I)
_FEE_UNDER_PATTERN = re.compile(r"(?:under|less\s*than|below|max(?:imum)?(?:\s*budget)?)\s*(\d+)", re.I)
_FEE_BETWEEN_PATTERN = re.compile(r"between\s*(\d+)\s*(?:and|to|-)\s*(\d+)", re.I)
_FEE_MIN_MAX_PATTERN = re.compile(r"(?:min|from)\s*(\d+).*(?:max|to)\s*(\d+)", re.I)


class QueryParser:
    """Stateless service that converts a raw string into a `QueryContext`."""

    def parse(self, raw_query: str) -> QueryContext:
        """Parse *raw_query* and return a rich `QueryContext`."""
        intent = self._detect_intent(raw_query)
        entities = self._extract_entities(raw_query)

        # Auto-upgrade intent based on entities
        if "doctor_id" in entities and intent == QueryIntent.UNKNOWN:
            intent = QueryIntent.DOCTOR_DETAIL
        if ("specialty" in entities or "city" in entities or "max_fee" in entities) and intent == QueryIntent.UNKNOWN:
            intent = QueryIntent.SEARCH_DOCTORS

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

        # Doctor ID
        id_match = _DOCTOR_ID_PATTERN.search(text)
        if id_match:
            entities["doctor_id"] = id_match.group(1)

        # City
        city_match = _CITY_PATTERN.search(text)
        if city_match:
            entities["city"] = city_match.group(1)

        # Specialty
        spec_match = _SPECIALTY_PATTERN.search(text)
        if spec_match:
            raw = spec_match.group(1).lower()
            # Normalise common aliases
            aliases = {
                "eye doctor": "ophthalmology",
                "skin specialist": "dermatology",
                "brain surgeon": "neurosurgery",
            }
            entities["specialty"] = aliases.get(raw, raw)

        # Fee range
        between_match = _FEE_BETWEEN_PATTERN.search(text)
        min_max_match = _FEE_MIN_MAX_PATTERN.search(text)
        under_match = _FEE_UNDER_PATTERN.search(text)

        if between_match:
            entities["min_fee"] = between_match.group(1)
            entities["max_fee"] = between_match.group(2)
        elif min_max_match:
            entities["min_fee"] = min_max_match.group(1)
            entities["max_fee"] = min_max_match.group(2)
        elif under_match:
            entities["max_fee"] = under_match.group(1)

        return entities