from .base_resolver import BaseResolver
from ..data.thana_aliases import THANA_ALIAS_MAP


class LocationResolver(BaseResolver):
    """
    Resolves a user-provided location string (with typos / aliases)
    to a canonical thana name.

    Usage:
        resolver = LocationResolver()
        result = resolver.resolve("Danmondi")
        # → {"input": "Danmondi", "thana_name": "Dhanmondi", "score": 87, ...}
    """

    def __init__(self):
        super().__init__()
        self._build_index(THANA_ALIAS_MAP)

    def _extract_metadata(self, entry: dict) -> dict:
        return {"thana_name": entry["thana_name"]}