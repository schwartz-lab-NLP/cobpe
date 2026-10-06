import pytest
import sys
from pathlib import Path

from cobpe.decomposition.prepositions import (
    PREPOSITION_PROFILES,
    classify_preposition_list,
    resolve_preposition_profile,
)


def test_resolve_core_profile_ordered_values():
    assert resolve_preposition_profile("core") == list(PREPOSITION_PROFILES["core"])


def test_resolve_expanded_profile_ordered_values():
    assert resolve_preposition_profile("expanded") == list(PREPOSITION_PROFILES["expanded"])


def test_resolve_full_profile_ordered_values():
    assert resolve_preposition_profile("full") == list(PREPOSITION_PROFILES["full"])


def test_unknown_profile_raises_clean_error():
    with pytest.raises(ValueError, match="Unknown preposition profile"):
        resolve_preposition_profile("does_not_exist")
    with pytest.raises(ValueError, match="Unknown preposition profile"):
        resolve_preposition_profile("existing")


def test_classifier_maps_known_lists():
    assert classify_preposition_list(PREPOSITION_PROFILES["core"]) == "core"
    assert classify_preposition_list(PREPOSITION_PROFILES["expanded"]) == "expanded"
    assert classify_preposition_list(PREPOSITION_PROFILES["full"]) == "full"


def test_classifier_marks_non_matching_lists_as_custom():
    assert classify_preposition_list(["by", "at"]) == "custom"
    assert classify_preposition_list(["outside", "for", "by"]) == "custom"
