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
