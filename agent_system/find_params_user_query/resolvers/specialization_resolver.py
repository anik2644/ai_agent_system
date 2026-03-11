from .base_resolver import BaseResolver
from ..data.specialization_aliases import SPECIALIZATION_ALIAS_MAP


class SpecializationResolver(BaseResolver):
    """
    Resolves a user-provided specialization string (with typos / layman terms /
    Bangla romanizations) to a canonical medical specialization name.

    Usage:
        resolver = SpecializationResolver()
        result = resolver.resolve("Hridoy Doctor")
        # → {"input": "Hridoy Doctor", "specialization_name": "Cardiology", "score": 90, ...}
    """

    def __init__(self):
        super().__init__()
        self._build_index(SPECIALIZATION_ALIAS_MAP)

    def _extract_metadata(self, entry: dict) -> dict:
        return {"specialization_name": entry["specialization_name"]}