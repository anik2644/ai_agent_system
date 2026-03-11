from .base_resolver import BaseResolver
from ..data.hospital_aliases import HOSPITAL_ALIAS_MAP


class HospitalResolver(BaseResolver):
    """
    Resolves a user-provided hospital name (with typos / abbreviations /
    Bangla romanizations) to a canonical hospital name, thana, and type.

    Usage:
        resolver = HospitalResolver()
        result = resolver.resolve("Apollo Dhaka")
        # → {
        #       "input": "Apollo Dhaka",
        #       "hospital_name": "Evercare Hospital Dhaka",
        #       "thana": "Ramna",
        #       "type": "non-govt",
        #       "score": 90, ...
        #   }
    """

    def __init__(self):
        super().__init__()
        self._build_index(HOSPITAL_ALIAS_MAP)

    def _extract_metadata(self, entry: dict) -> dict:
        return {
            "hospital_name": entry["hospital_name"],
            "thana": entry["thana"],
            "type": entry["type"],
        }