from rapidfuzz import process, fuzz


class BaseResolver:
    """
    Shared fuzzy matching engine used by all param resolvers.
    Subclasses provide the alias map and define what gets returned.
    """

    SCORE_THRESHOLD: int = 60

    def __init__(self):
        # Populated by subclasses via _build_index()
        self._alias_lookup: dict = {}
        self._all_aliases: list[str] = []

    def _build_index(self, alias_map: list[dict], alias_key: str = "aliases") -> None:
        """
        Builds the flat alias → metadata lookup from a structured alias map.

        Each entry in alias_map must have:
          - an 'aliases' list  (or whatever alias_key is)
          - additional fields stored as metadata (subclass decides)

        Subclasses call this from __init__ with their own map.
        """
        for entry in alias_map:
            metadata = self._extract_metadata(entry)
            for alias in entry[alias_key]:
                alias_lower = alias.lower()
                self._alias_lookup[alias_lower] = metadata
                self._all_aliases.append(alias_lower)

    def _extract_metadata(self, entry: dict) -> dict:
        """
        Override in subclasses to define what metadata gets stored per alias.
        """
        raise NotImplementedError

    def _fuzzy_match(self, query: str) -> tuple[str | None, int]:
        """
        Runs rapidfuzz token_set_ratio against all known aliases.
        Returns (matched_alias, score) or (None, 0).
        """
        if not query:
            return None, 0

        result = process.extractOne(
            query,
            self._all_aliases,
            scorer=fuzz.token_set_ratio,
        )

        if result is None:
            return None, 0

        matched_alias, score, _ = result
        return matched_alias, score

    def _build_response(
        self,
        user_input: str,
        matched_alias: str | None,
        score: int,
        metadata: dict | None,
    ) -> dict:
        """Builds a standard response dict. Subclasses merge in their own fields."""
        status = (
            "✅ Confident match"
            if score >= self.SCORE_THRESHOLD
            else "⚠️ Low confidence"
        )
        if matched_alias is None:
            status = "❌ No match found"

        return {
            "input": user_input,
            "matched_alias": matched_alias.title() if matched_alias else None,
            "score": score,
            "status": status,
            **(metadata or {}),
        }

    def resolve(self, user_input: str) -> dict:
        """Public entry point. Subclasses may override for custom logic."""
        query = user_input.strip().lower()
        matched_alias, score = self._fuzzy_match(query)

        if matched_alias is None or score < self.SCORE_THRESHOLD:
            metadata = None
        else:
            metadata = self._alias_lookup.get(matched_alias)

        return self._build_response(user_input, matched_alias, score, metadata)