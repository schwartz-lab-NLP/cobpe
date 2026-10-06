from __future__ import annotations

from typing import Iterable, Sequence


DEFAULT_PREPOSITION_PROFILE = "core"

PREPOSITION_PROFILES: dict[str, tuple[str, ...]] = {
    "core": (
        "by",
        "at",
        "of",
        "to",
        "in",
        "on",
        "with",
        "for",
        "from",
    ),
    "expanded": (
        "by",
        "at",
        "of",
        "to",
        "in",
        "on",
        "with",
        "for",
        "from",
        "about",
        "across",
        "after",
        "against",
        "around",
        "before",
        "between",
        "during",
        "inside",
        "into",
        "onto",
        "over",
        "through",
        "under",
        "without",
    ),
    "full": (
        "about",
        "above",
        "across",
        "after",
        "against",
        "along",
        "amid",
        "among",
        "around",
        "at",
        "before",
        "behind",
        "below",
        "beneath",
        "beside",
        "between",
        "beyond",
        "by",
        "despite",
        "during",
        "for",
        "from",
        "in",
        "inside",
        "into",
        "near",
        "of",
        "on",
        "onto",
        "outside",
        "over",
        "through",
        "throughout",
        "to",
        "toward",
        "towards",
        "under",
        "underneath",
        "until",
        "upon",
        "with",
        "within",
        "without",
        "per",
        "via",
    ),
}

DEFAULT_PREPOSITION_LIST = list(PREPOSITION_PROFILES[DEFAULT_PREPOSITION_PROFILE])


def normalize_preposition_list(preposition_list: Iterable[str] | None) -> list[str]:
    if preposition_list is None:
        return []
    out: list[str] = []
    seen: set[str] = set()
    for raw in preposition_list:
        val = str(raw).strip().lower()
        if not val or val in seen:
            continue
        seen.add(val)
        out.append(val)
    return out


def normalize_preposition_profile_name(profile: str | None) -> str:
    key = str(profile or DEFAULT_PREPOSITION_PROFILE).strip().lower()
    if key == "":
        key = DEFAULT_PREPOSITION_PROFILE
    return key


def resolve_preposition_profile(profile: str | None) -> list[str]:
    key = normalize_preposition_profile_name(profile)
    if key not in PREPOSITION_PROFILES:
        valid = ", ".join(sorted(PREPOSITION_PROFILES.keys()))
        raise ValueError(f"Unknown preposition profile '{profile}'. Expected one of: {valid}")
    return list(PREPOSITION_PROFILES[key])


def classify_preposition_list(preposition_list: Sequence[str] | None) -> str:
    if preposition_list is None:
        return DEFAULT_PREPOSITION_PROFILE
    normalized = normalize_preposition_list(preposition_list)
    if not normalized:
        return "custom"
    normalized_set = set(normalized)
    for profile_name, profile_values in PREPOSITION_PROFILES.items():
        canonical = list(profile_values)
        if normalized == canonical:
            return profile_name
        if len(normalized) == len(canonical) and normalized_set == set(canonical):
            return profile_name
    return "custom"


def resolve_preposition_settings(
    profile: str | None,
    preposition_list: Iterable[str] | None,
    *,
    allow_auto: bool = True,
) -> tuple[list[str], str]:
    if preposition_list is not None:
        normalized = normalize_preposition_list(preposition_list)
        return normalized, classify_preposition_list(normalized)

    key = str(profile or DEFAULT_PREPOSITION_PROFILE).strip().lower()
    if key == "" or (allow_auto and key in {"auto", "off", "unknown"}):
        key = DEFAULT_PREPOSITION_PROFILE
    key = normalize_preposition_profile_name(key)
    resolved = resolve_preposition_profile(key)
    return resolved, key
