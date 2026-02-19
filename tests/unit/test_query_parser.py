"""Unit tests for the query parser."""

import pytest

from agent_system.core.models import QueryIntent
from agent_system.services.query_parser import QueryParser


@pytest.fixture()
def parser() -> QueryParser:
    return QueryParser()


class TestQueryParser:
    @pytest.mark.parametrize(
        ("query", "expected_intent"),
        [
            ("Find doctors in Austin", QueryIntent.SEARCH_DOCTORS),
            ("Search for a cardiologist in Dallas", QueryIntent.SEARCH_DOCTORS),
            ("Show me details about Dr. Smith", QueryIntent.DOCTOR_DETAIL),
            ("What are the symptoms of diabetes?", QueryIntent.GENERAL_HEALTH),
            ("Random nonsense", QueryIntent.UNKNOWN),
        ],
    )
    def test_intent_detection(
        self, parser: QueryParser, query: str, expected_intent: QueryIntent
    ) -> None:
        ctx = parser.parse(query)
        assert ctx.intent == expected_intent

    def test_city_extraction(self, parser: QueryParser) -> None:
        ctx = parser.parse("Find doctors in Austin")
        assert ctx.entities.get("city") == "Austin"

    def test_specialty_extraction(self, parser: QueryParser) -> None:
        ctx = parser.parse("I need a cardiologist")
        assert ctx.entities.get("specialty") == "cardiologist"

    def test_combined_extraction(self, parser: QueryParser) -> None:
        ctx = parser.parse("Find a dermatologist in Denver")
        assert ctx.entities["city"] == "Denver"
        assert ctx.entities["specialty"] == "dermatologist"