"""
query_param_extractor.py
------------------------
Orchestrates all resolvers to extract structured params from a raw user query.

This is the single entry point your pipeline/agents should call.

Example usage:
    from agent_system.find_params_user_query import extract_query_params

    params = extract_query_params(
        location="Danmondi",
        specialization="Hridoy Doctor",
        hospital_name="Apollo Dhaka",
    )
    # → QueryParams(thana="Dhanmondi", specialization="Cardiology",
    #               hospital="Evercare Hospital Dhaka", ...)
"""

from dataclasses import dataclass, field
from .resolvers import LocationResolver, SpecializationResolver, HospitalResolver


@dataclass
class ResolvedParam:
    """Holds the resolved value + confidence metadata for a single param."""
    raw_input: str
    resolved_value: str | None
    matched_alias: str | None
    score: int
    status: str
    confident: bool

    def __bool__(self) -> bool:
        return self.confident and self.resolved_value is not None


@dataclass
class QueryParams:
    """Structured output of all resolved query parameters."""
    thana: ResolvedParam | None = None
    specialization: ResolvedParam | None = None
    hospital: ResolvedParam | None = None
    # hospital also carries these when resolved
    hospital_thana: str | None = None
    hospital_type: str | None = None

    def to_dict(self) -> dict:
        return {
            "thana": self.thana.resolved_value if self.thana else None,
            "thana_score": self.thana.score if self.thana else None,
            "thana_status": self.thana.status if self.thana else None,
            "specialization": self.specialization.resolved_value if self.specialization else None,
            "specialization_score": self.specialization.score if self.specialization else None,
            "specialization_status": self.specialization.status if self.specialization else None,
            "hospital": self.hospital.resolved_value if self.hospital else None,
            "hospital_score": self.hospital.score if self.hospital else None,
            "hospital_status": self.hospital.status if self.hospital else None,
            "hospital_thana": self.hospital_thana,
            "hospital_type": self.hospital_type,
        }


# Module-level singleton resolvers (built once, reused across calls)
_location_resolver = LocationResolver()
_specialization_resolver = SpecializationResolver()
_hospital_resolver = HospitalResolver()


def _make_resolved_param(raw: str, result: dict, value_key: str) -> ResolvedParam:
    return ResolvedParam(
        raw_input=raw,
        resolved_value=result.get(value_key),
        matched_alias=result.get("matched_alias"),
        score=result.get("score", 0),
        status=result.get("status", ""),
        confident="✅" in result.get("status", ""),
    )


def extract_query_params(
    location: str | None = None,
    specialization: str | None = None,
    hospital_name: str | None = None,
) -> QueryParams:
    """
    Main function. Pass whichever params are available from the user query.
    Returns a QueryParams object with all resolved values.

    Args:
        location:       Raw location string from user (thana / area / alias)
        specialization: Raw specialization string from user (doctor type / alias)
        hospital_name:  Raw hospital name from user (full name / abbreviation / alias)

    Returns:
        QueryParams with resolved values and confidence metadata
    """
    params = QueryParams()

    if location:
        result = _location_resolver.resolve(location)
        params.thana = _make_resolved_param(location, result, "thana_name")

    if specialization:
        result = _specialization_resolver.resolve(specialization)
        params.specialization = _make_resolved_param(specialization, result, "specialization_name")

    if hospital_name:
        result = _hospital_resolver.resolve(hospital_name)
        params.hospital = _make_resolved_param(hospital_name, result, "hospital_name")
        # Also surface thana + type from the hospital match
        if params.hospital.confident:
            params.hospital_thana = result.get("thana")
            params.hospital_type = result.get("type")

    return params