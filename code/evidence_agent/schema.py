"""Shared schema, enums, and validation helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

OUTPUT_COLUMNS = [
    "user_id",
    "image_paths",
    "user_claim",
    "claim_object",
    "evidence_standard_met",
    "evidence_standard_met_reason",
    "risk_flags",
    "issue_type",
    "object_part",
    "claim_status",
    "claim_status_justification",
    "supporting_image_ids",
    "valid_image",
    "severity",
]

CLAIM_STATUSES = {"supported", "contradicted", "not_enough_information"}
ISSUE_TYPES = {
    "dent",
    "scratch",
    "crack",
    "glass_shatter",
    "broken_part",
    "missing_part",
    "torn_packaging",
    "crushed_packaging",
    "water_damage",
    "stain",
    "none",
    "unknown",
}
OBJECT_PARTS = {
    "car": {
        "front_bumper",
        "rear_bumper",
        "door",
        "hood",
        "windshield",
        "side_mirror",
        "headlight",
        "taillight",
        "fender",
        "quarter_panel",
        "body",
        "unknown",
    },
    "laptop": {
        "screen",
        "keyboard",
        "trackpad",
        "hinge",
        "lid",
        "corner",
        "port",
        "base",
        "body",
        "unknown",
    },
    "package": {
        "box",
        "package_corner",
        "package_side",
        "seal",
        "label",
        "contents",
        "item",
        "unknown",
    },
}
RISK_FLAGS = {
    "none",
    "blurry_image",
    "cropped_or_obstructed",
    "low_light_or_glare",
    "wrong_angle",
    "wrong_object",
    "wrong_object_part",
    "damage_not_visible",
    "claim_mismatch",
    "possible_manipulation",
    "non_original_image",
    "text_instruction_present",
    "user_history_risk",
    "manual_review_required",
}
SEVERITIES = {"none", "low", "medium", "high", "unknown"}
BOOL_TEXT = {"true", "false"}

PART_ALIASES = {
    "bumper": "front_bumper",
    "front": "front_bumper",
    "rear": "rear_bumper",
    "back": "rear_bumper",
    "back light": "taillight",
    "tail light": "taillight",
    "taillamp": "taillight",
    "mirror": "side_mirror",
    "side mirror": "side_mirror",
    "front glass": "windshield",
    "glass": "windshield",
    "display": "screen",
    "keys": "keyboard",
    "keycaps": "keyboard",
    "touchpad": "trackpad",
    "outer lid": "lid",
    "edge": "body",
    "box corner": "package_corner",
    "corner": "package_corner",
    "flap": "seal",
    "tape": "seal",
    "shipping label": "label",
    "inner item": "item",
    "product": "contents",
}

ISSUE_ALIASES = {
    "scrape": "scratch",
    "scuff": "scratch",
    "mark": "scratch",
    "shatter": "glass_shatter",
    "shattered": "glass_shatter",
    "broken": "broken_part",
    "broke": "broken_part",
    "toot": "broken_part",
    "missing": "missing_part",
    "faltan": "missing_part",
    "opened": "torn_packaging",
    "open": "torn_packaging",
    "torn": "torn_packaging",
    "crushed": "crushed_packaging",
    "crush": "crushed_packaging",
    "dab": "crushed_packaging",
    "wet": "water_damage",
    "water": "water_damage",
    "liquid": "stain",
    "coffee": "stain",
    "stain": "stain",
    "oily": "stain",
}


@dataclass
class ParsedClaim:
    issue_type: str = "unknown"
    object_part: str = "unknown"
    severity_hint: str = "unknown"
    constraints: list[str] = field(default_factory=list)
    adversarial_text: bool = False


@dataclass
class ImageObservation:
    image_id: str
    normalized_path: str
    valid_image: bool
    visible_object: str = "unknown"
    visible_parts: list[str] = field(default_factory=list)
    issue_type: str = "unknown"
    object_part: str = "unknown"
    severity: str = "unknown"
    risk_flags: list[str] = field(default_factory=list)
    description: str = ""
    confidence: float = 0.0


@dataclass
class EvidenceRequirement:
    """A single row from evidence_requirements.csv."""
    requirement_id: str
    claim_object: str
    applies_to: str
    minimum_image_evidence: str


@dataclass
class CrossImageAggregation:
    """Aggregated cross-image evidence computed before adjudication."""
    conflicting_evidence: bool = False
    partial_support: bool = False
    object_consistent: bool = True
    part_consistent: bool = True
    max_confidence: float = 0.0
    avg_confidence: float = 0.0
    supporting_confidence: float = 0.0
    aggregated_issues: list[str] = field(default_factory=list)
    aggregated_parts: list[str] = field(default_factory=list)
    image_count: int = 0
    valid_image_count: int = 0


@dataclass
class MatchedRequirement:
    """An evidence requirement matched against actual image evidence."""
    requirement_id: str
    text: str
    met: bool
    reason: str


# Maps requirement_id -> (set of issue_types that trigger matching)
REQUIREMENT_ISSUE_MAP: dict[str, set[str]] = {
    "REQ_CAR_BODY_PANEL": {"dent", "scratch"},
    "REQ_CAR_GLASS_LIGHT_MIRROR": {"crack", "broken_part", "missing_part", "glass_shatter"},
    "REQ_LAPTOP_SCREEN_KEYBOARD_TRACKPAD": set(),  # matched by part, not issue
    "REQ_LAPTOP_BODY_HINGE_PORT": set(),  # matched by part, not issue
    "REQ_PACKAGE_EXTERIOR": {"crushed_packaging", "torn_packaging"},
    "REQ_PACKAGE_LABEL_OR_STAIN": {"water_damage", "stain"},
    "REQ_PACKAGE_CONTENTS": set(),  # matched by part, not issue
}

# Maps requirement_id -> (set of object_parts that trigger matching)
REQUIREMENT_PART_MAP: dict[str, set[str]] = {
    "REQ_LAPTOP_SCREEN_KEYBOARD_TRACKPAD": {"screen", "keyboard", "trackpad"},
    "REQ_LAPTOP_BODY_HINGE_PORT": {"hinge", "lid", "corner", "body", "base", "port"},
    "REQ_PACKAGE_CONTENTS": {"contents", "item"},
}


def normalize_text(value: Any, default: str = "unknown") -> str:
    if value is None:
        return default
    text = str(value).strip().lower().replace(" ", "_").replace("-", "_")
    return text or default


def coerce_enum(value: Any, allowed: set[str], default: str) -> str:
    text = normalize_text(value, default)
    return text if text in allowed else default


def coerce_bool_text(value: Any, default: str = "false") -> str:
    text = normalize_text(value, default)
    if text in {"true", "yes", "1"}:
        return "true"
    if text in {"false", "no", "0"}:
        return "false"
    return default


def join_flags(flags: list[str] | set[str]) -> str:
    cleaned = [flag for flag in flags if flag in RISK_FLAGS and flag != "none"]
    if not cleaned:
        return "none"
    return ";".join(sorted(dict.fromkeys(cleaned)))


def split_flags(value: str) -> set[str]:
    if not value or value == "none":
        return set()
    return {part.strip() for part in value.split(";") if part.strip() and part.strip() != "none"}
