"""Surface-form modifier helpers used by the published CoBPE metadata path."""

from __future__ import annotations

from collections import defaultdict

from .constants import *  # noqa: F401,F403
from .affixes import strip_all_affixes
from .prepositions import DEFAULT_PREPOSITION_LIST, normalize_preposition_list
def build_complete_transforms(base_transforms, space_transform, cap_transform,
                             prefix_punct=None, suffix_punct=None,
                             possessive=None,
                             article=None, preposition=None,
                             article_cap=None, prep_cap=None,
                             article_space_prefix=None, prep_space_prefix=None,
                             morphology_mode="atomic"):
    """Ensure transforms contain exactly one value from each transformation group."""
    complete_transforms = [t for t in base_transforms]

    morph_prefixes = tuple(f"{group}_" for group in MORPHOLOGICAL_GROUPS)

    # Ensure every word gets exactly one value from each transformation group
    # Check for inflection: either atomic (inflect_*) or compositional (morphological groups)
    has_inflection = any(
        t.startswith('inflect_') or t == NO_INFLECTION or t == NA_INFLECTION or
        t.startswith(morph_prefixes) or
        t in MORPHOLOGICAL_GROUP_NO_VALUES.values() or
        t in MORPHOLOGICAL_GROUP_NA_VALUES.values()
        for t in complete_transforms)
    has_derivation = any(
        t.startswith('deriv_') or t == NO_DERIVATION or t == NA_DERIVATION for t in complete_transforms)
    has_prefix_punct = any(
        t.startswith('punct_prefix_') or t == NO_PREFIX_PUNCTUATION or t == NA_PREFIX_PUNCTUATION for t in complete_transforms)
    has_suffix_punct = any(
        t.startswith('punct_suffix_') or t == NO_SUFFIX_PUNCTUATION or t == NA_SUFFIX_PUNCTUATION for t in complete_transforms)
    has_possessive = any(
        t in [
            POSSESSIVE_S,
            POSSESSIVE_S_CURLY,
            POSSESSIVE_PLURAL,
            POSSESSIVE_PLURAL_CURLY,
            NO_POSSESSIVE,
            NA_POSSESSIVE,
        ] for t in complete_transforms
    )
    has_article = any(is_determiner_transform_name(t) or is_determiner_no_or_na(t) for t in complete_transforms)
    has_prep = any(
        t.startswith('prep_') or t == NO_PREPOSITION or t == NA_PREPOSITION for t in complete_transforms)
    has_article_cap = any(
        t in [NO_ARTICLE_CAPITALIZATION, ADD_ARTICLE_CAPITALIZATION, NA_ARTICLE_CAPITALIZATION] for t in complete_transforms)
    has_prep_cap = any(
        t in [NO_PREP_CAPITALIZATION, ADD_PREP_CAPITALIZATION, NA_PREP_CAPITALIZATION] for t in complete_transforms)
    has_article_space = any(
        t in [NO_ARTICLE_SPACE_PREFIX, ADD_ARTICLE_SPACE_PREFIX, NA_ARTICLE_SPACE_PREFIX] for t in complete_transforms)
    has_prep_space = any(
        t in [NO_PREP_SPACE_PREFIX, ADD_PREP_SPACE_PREFIX, NA_PREP_SPACE_PREFIX] for t in complete_transforms)

    if morphology_mode == "atomic" and not has_inflection:
        complete_transforms.append(NO_INFLECTION)
    if not has_derivation:
        complete_transforms.append(NO_DERIVATION)

    # Add space and capitalization transforms
    complete_transforms.append(space_transform)
    complete_transforms.append(cap_transform)

    # Add new transformation groups if provided
    if not has_prefix_punct:
        complete_transforms.append(prefix_punct if prefix_punct is not None else NO_PREFIX_PUNCTUATION)
    if not has_suffix_punct:
        complete_transforms.append(suffix_punct if suffix_punct is not None else NO_SUFFIX_PUNCTUATION)
    if not has_possessive:
        complete_transforms.append(possessive if possessive is not None else NO_POSSESSIVE)
    if not has_article:
        complete_transforms.append(article if article is not None else NO_ARTICLE)
    if not has_prep:
        complete_transforms.append(preposition if preposition is not None else NO_PREPOSITION)
    if not has_article_cap:
        complete_transforms.append(article_cap if article_cap is not None else NO_ARTICLE_CAPITALIZATION)
    if not has_prep_cap:
        complete_transforms.append(prep_cap if prep_cap is not None else NO_PREP_CAPITALIZATION)
    if article_space_prefix is not None and not has_article_space:
        complete_transforms.append(article_space_prefix)
    if prep_space_prefix is not None and not has_prep_space:
        complete_transforms.append(prep_space_prefix)

    return complete_transforms

def get_variation_transformations(word, base_form=None, transforms=None, space_prefix=" ",
                                  decompose_spaces=True, decompose_capitalization=True,
                                  morphology_mode="atomic",
                                  use_relative_space_cap_transforms=False,
                                  include_all_caps_capitalization=False,
                                  article_space_prefix=None,
                                  prep_space_prefix=None):
    """
    Get variations of a word with transformation tags.

    Args:
        word: The word to transform
        base_form: The original base form for comparison
        space_prefix: The space prefix character
        decompose_spaces: Whether to treat space prefix as a transformation
        decompose_capitalization: Whether to treat capitalization as a transformation
        use_relative_space_cap_transforms: Whether to model space/cap changes as add/remove
        include_all_caps_capitalization: Whether to emit all-caps capitalization transforms

    Returns:
        Dict mapping word variations to their transformation tags
    """
    result = {}
    has_space_prefix = word.startswith(space_prefix)
    word_without_prefix = word[len(space_prefix):] if has_space_prefix else word

    def _is_multi_letter_all_caps(text):
        alpha_chars = [c for c in text if c.isalpha()]
        return len(alpha_chars) > 1 and all(c.isupper() for c in alpha_chars)

    def _cap_state(text):
        if not text:
            return "none"
        if _is_multi_letter_all_caps(text):
            # When all-caps transform is disabled, preserve literal all-caps words
            # and do not remap them to lowercase/no-cap variants.
            return "all_caps" if include_all_caps_capitalization else "all_caps_literal"
        if text[0].isupper():
            return "first_cap"
        return "none"

    def _apply_cap_state(text, state):
        if state == "all_caps":
            return text.upper()
        if state == "all_caps_literal":
            return text
        if state == "first_cap":
            return text.capitalize()
        return text.lower()

    def _relative_space_transform(base_has_space, var_has_space):
        if var_has_space == base_has_space:
            return NO_SPACE_PREFIX_TRANSFORM
        return WITH_SPACE_PREFIX_TRANSFORM if var_has_space else REMOVE_SPACE_PREFIX_TRANSFORM

    def _relative_cap_transform(base_cap_state, var_cap_state):
        if var_cap_state == base_cap_state:
            return NO_BASE_CAPITALIZATION_TRANSFORM
        if var_cap_state == "all_caps_literal":
            return NO_BASE_CAPITALIZATION_TRANSFORM
        if base_cap_state == "all_caps_literal":
            return NO_BASE_CAPITALIZATION_TRANSFORM
        if var_cap_state == "all_caps":
            return ADD_ALL_CAPS_BASE_CAPITALIZATION_TRANSFORM
        if var_cap_state == "first_cap":
            return ADD_BASE_CAPITALIZATION_TRANSFORM if base_cap_state == "none" else REMOVE_BASE_CAPITALIZATION_TRANSFORM
        return REMOVE_BASE_CAPITALIZATION_TRANSFORM

    def _absolute_cap_transform(var_cap_state):
        if var_cap_state == "all_caps_literal":
            return NO_BASE_CAPITALIZATION_TRANSFORM
        if var_cap_state == "all_caps":
            return ADD_ALL_CAPS_BASE_CAPITALIZATION_TRANSFORM
        if var_cap_state == "first_cap":
            return ADD_BASE_CAPITALIZATION_TRANSFORM
        return NO_BASE_CAPITALIZATION_TRANSFORM

    base_has_space_prefix = has_space_prefix
    current_cap_state = _cap_state(word_without_prefix)
    base_cap_state = current_cap_state

    # Set transformation types
    if base_form:
        base_has_space_prefix = base_form.startswith(space_prefix)
        base_word_without_prefix = base_form[len(space_prefix):] if base_has_space_prefix else base_form
        base_cap_state = _cap_state(base_word_without_prefix)

    if use_relative_space_cap_transforms:
        space_transform = _relative_space_transform(base_has_space_prefix, has_space_prefix)
        cap_transform = _relative_cap_transform(base_cap_state, current_cap_state)
    else:
        # Base-form-independent absolute capitalization transform.
        space_transform = WITH_SPACE_PREFIX_TRANSFORM if has_space_prefix else NO_SPACE_PREFIX_TRANSFORM
        cap_transform = _absolute_cap_transform(current_cap_state)
    # Add current word with transformations
    transforms = [] if transforms is None else list(transforms)
    # If it's the base form without transformations, add default transforms
    if not transforms:
        if morphology_mode == "atomic":
            transforms.extend([NO_INFLECTION, NO_DERIVATION])
        else:
            transforms.extend([NO_DERIVATION])

    base_transforms = [t for t in transforms]
    complete_transforms = build_complete_transforms(
        base_transforms, space_transform, cap_transform,
        prefix_punct=None, suffix_punct=None,
        article=None, preposition=None,
        article_cap=None, prep_cap=None,
        article_space_prefix=article_space_prefix,
        prep_space_prefix=prep_space_prefix,
        morphology_mode=morphology_mode
    )
    result[word] = complete_transforms

    # Generate additional space/capitalization variants.
    space_options = {has_space_prefix}
    if decompose_spaces:
        space_options.add(not has_space_prefix)
    cap_options = {current_cap_state}
    if decompose_capitalization:
        if current_cap_state != "all_caps_literal":
            cap_options.update({"none", "first_cap"})
            if include_all_caps_capitalization:
                cap_options.add("all_caps")

    normalized_core = word_without_prefix
    if decompose_capitalization and current_cap_state != "all_caps_literal":
        normalized_core = word_without_prefix.lower()
    for var_has_space in sorted(space_options):
        for var_cap_state in sorted(cap_options):
            variant_core = normalized_core
            if decompose_capitalization:
                variant_core = _apply_cap_state(normalized_core, var_cap_state)
            variant_str = (space_prefix if var_has_space else "") + variant_core
            if variant_str == word:
                continue
            if use_relative_space_cap_transforms:
                var_space_transform = _relative_space_transform(base_has_space_prefix, var_has_space)
                var_cap_transform = _relative_cap_transform(base_cap_state, var_cap_state)
            else:
                var_space_transform = WITH_SPACE_PREFIX_TRANSFORM if var_has_space else NO_SPACE_PREFIX_TRANSFORM
                var_cap_transform = _absolute_cap_transform(var_cap_state)
            result[variant_str] = build_complete_transforms(
                transforms, var_space_transform, var_cap_transform,
                prefix_punct=None, suffix_punct=None,
                article=None, preposition=None,
                article_cap=None, prep_cap=None,
                article_space_prefix=article_space_prefix,
                prep_space_prefix=prep_space_prefix,
                morphology_mode=morphology_mode
            )

    return result

def apply_capitalization_pattern(original, target):
    """Apply capitalization pattern from original to target string."""
    if original and len(original) > 0 and original[0].isupper() and len(target) > 0:
        return target[0].upper() + target[1:]
    return target

def get_label_maps_from_decomposition_map(
    decomposition_map,
    morphology_mode='atomic',
    active_morphological_groups=None,
    force_articles=False,
    force_possessive_determiners=False,
    force_demonstrative_determiners=False,
    force_quantifier_determiners=False,
    force_prepositions=False,
    force_punctuation=False,
    force_possessives=False,
    force_article_space_prefix=False,
    force_prep_space_prefix=False,
    preposition_list=None,
    include_na_types=True,
):
    """
    Extract unique transformation types from the decomposition map.

    Args:
        decomposition_map (dict): Decomposition map created by get_vocabulary_decomposition
        morphology_mode (str): 'atomic' or 'compositional'
        active_morphological_groups (list): List of active morphological groups (for compositional mode)
        force_articles (bool): Always include ARTICLE_* transforms, even if absent in the map
        force_demonstrative_determiners (bool): Include demonstrative determiners in determiners group
            when force_articles is enabled.
        force_quantifier_determiners (bool): Include quantifier determiners in determiners group
            when force_articles is enabled.
        force_prepositions (bool): Always include prep_* transforms, even if absent in the map
        force_punctuation (bool): Always include punct_* transforms, even if absent in the map
        force_possessives (bool): Always include possessive transforms in a dedicated group
        force_article_space_prefix (bool): Always include article space-prefix transforms
        force_prep_space_prefix (bool): Always include preposition space-prefix transforms
        preposition_list (list): List of prepositions to include (defaults to current "core" profile)
        include_na_types (bool): Whether to include NA_* labels for morphological groups.

    Returns:
        tuple: (transformations_to_int dict, loss_indices dict)
    """
    def _collect_types():
        types = set()
        for _, inflections in decomposition_map.items():
            for _, inflection_info in inflections.items():
                types.update(inflection_info)
        return types

    def _collect_morph_groups(types):
        morph_group_types = defaultdict(set)
        for t in types:
            group = _get_morph_group_from_transform(t)
            if group and t not in {MORPHOLOGICAL_GROUP_NO_VALUES[group], MORPHOLOGICAL_GROUP_NA_VALUES[group]}:
                morph_group_types[group].add(t)
        allowed_groups = active_morphological_groups or MORPHOLOGICAL_GROUPS
        active_groups = [g for g in allowed_groups if g in morph_group_types]
        return morph_group_types, active_groups

    transformation_types = _collect_types()

    morph_group_types = defaultdict(set)
    active_morph_groups = []
    if morphology_mode == "compositional":
        morph_group_types, active_morph_groups = _collect_morph_groups(transformation_types)

    def _order_article_types(article_types):
        ordered = (
            [ARTICLE_THE, ARTICLE_A, ARTICLE_AN]
            + POSSESSIVE_DETERMINER_TRANSFORMS
            + DEMONSTRATIVE_DETERMINER_TRANSFORMS
            + QUANTIFIER_DETERMINER_TRANSFORMS
        )
        present = [t for t in ordered if t in article_types]
        extra = [t for t in article_types if t not in ordered]
        return present + sorted(extra)

    def _order_prep_types(prep_types):
        ordered = [f"prep_{p}" for p in default_preps]
        present = [t for t in ordered if t in prep_types]
        extra = [t for t in prep_types if t not in ordered]
        return present + sorted(extra)

    def _order_possessive_types(possessive_types):
        ordered = [POSSESSIVE_S, POSSESSIVE_S_CURLY, POSSESSIVE_PLURAL, POSSESSIVE_PLURAL_CURLY]
        present = [t for t in ordered if t in possessive_types]
        extra = [t for t in possessive_types if t not in ordered]
        return present + sorted(extra)

    def _order_punct_types(punct_types, ordered):
        present = [t for t in ordered if t in punct_types]
        extra = [t for t in punct_types if t not in ordered]
        return present + sorted(extra)

    default_preps = (
        normalize_preposition_list(preposition_list)
        if preposition_list is not None
        else list(DEFAULT_PREPOSITION_LIST)
    )

    inflection_types = [t for t in transformation_types if t.startswith('inflect_')] if morphology_mode == "atomic" else []
    derivation_types = [t for t in transformation_types if t.startswith('deriv_')]
    prefix_punct_types = [t for t in transformation_types if t.startswith('punct_prefix_')]
    suffix_punct_types = [t for t in transformation_types if t.startswith('punct_suffix_')]
    possessive_types = [
        t for t in transformation_types
        if t in [POSSESSIVE_S, POSSESSIVE_S_CURLY, POSSESSIVE_PLURAL, POSSESSIVE_PLURAL_CURLY]
    ]
    article_types = [t for t in transformation_types if is_non_default_determiner_transform(t)]
    prep_types = [t for t in transformation_types if t.startswith('prep_') and t not in [NO_PREPOSITION, NA_PREPOSITION]]

    prefix_punct_order = [
        PREFIX_PUNCT_SINGLE_PAREN,
        "punct_prefix_\u2018(",
        "punct_prefix_\u2019(",
        PREFIX_PUNCT_DOUBLE_PAREN,
        "punct_prefix_\u201c(",
        "punct_prefix_\u201d(",
        PREFIX_PUNCT_SINGLE_SQUARE,
        "punct_prefix_\u2018[",
        "punct_prefix_\u2019[",
        PREFIX_PUNCT_DOUBLE_SQUARE,
        "punct_prefix_\u201c[",
        "punct_prefix_\u201d[",
        PREFIX_PUNCT_PAREN_SINGLE,
        "punct_prefix_(\u2018",
        "punct_prefix_(\u2019",
        PREFIX_PUNCT_HYPHEN_PAREN,
        PREFIX_PUNCT_HYPHEN_SQUARE,
        PREFIX_PUNCT_DOUBLE_QUOTE,
        PREFIX_PUNCT_DOUBLE_QUOTE_LEFT,
        PREFIX_PUNCT_DOUBLE_QUOTE_RIGHT,
        PREFIX_PUNCT_SINGLE_QUOTE,
        PREFIX_PUNCT_SINGLE_QUOTE_LEFT,
        PREFIX_PUNCT_SINGLE_QUOTE_RIGHT,
        PREFIX_PUNCT_BACKTICK,
        PREFIX_PUNCT_PAREN,
        PREFIX_PUNCT_SQUARE,
        PREFIX_PUNCT_CURLY,
        PREFIX_PUNCT_HYPHEN,
        "punct_prefix_\u2014",
    ]
    suffix_punct_order = [
        SUFFIX_PUNCT_PAREN_SINGLE,
        "punct_suffix_)\u2019",
        SUFFIX_PUNCT_PAREN_DOUBLE,
        "punct_suffix_)\u201d",
        SUFFIX_PUNCT_SQUARE_SINGLE,
        "punct_suffix_]\u2019",
        SUFFIX_PUNCT_SQUARE_DOUBLE,
        "punct_suffix_]\u201d",
        SUFFIX_PUNCT_PERIOD_SINGLE,
        "punct_suffix_.\u2019",
        SUFFIX_PUNCT_PERIOD_DOUBLE,
        "punct_suffix_.\u201d",
        SUFFIX_PUNCT_EXCLAIM_SINGLE,
        "punct_suffix_!\u2019",
        SUFFIX_PUNCT_EXCLAIM_DOUBLE,
        "punct_suffix_!\u201d",
        SUFFIX_PUNCT_QUESTION_SINGLE,
        "punct_suffix_?\u2019",
        SUFFIX_PUNCT_QUESTION_DOUBLE,
        "punct_suffix_?\u201d",
        SUFFIX_PUNCT_COMMA_SINGLE,
        "punct_suffix_,\u2019",
        SUFFIX_PUNCT_COMMA_DOUBLE,
        "punct_suffix_,\u201d",
        SUFFIX_PUNCT_SINGLE_COMMA,
        "punct_suffix_\u2019,",
        SUFFIX_PUNCT_DOUBLE_COMMA,
        "punct_suffix_\u201d,",
        SUFFIX_PUNCT_SINGLE_PERIOD,
        "punct_suffix_\u2019.",
        SUFFIX_PUNCT_DOUBLE_PERIOD,
        "punct_suffix_\u201d.",
        SUFFIX_PUNCT_PERIOD_PAREN,
        SUFFIX_PUNCT_PAREN_PERIOD,
        SUFFIX_PUNCT_EXCLAIM_PAREN,
        SUFFIX_PUNCT_QUESTION_PAREN,
        SUFFIX_PUNCT_COMMA_PAREN,
        SUFFIX_PUNCT_PAREN_HYPHEN,
        SUFFIX_PUNCT_SQUARE_HYPHEN,
        SUFFIX_PUNCT_DOUBLE_QUOTE,
        SUFFIX_PUNCT_DOUBLE_QUOTE_LEFT,
        SUFFIX_PUNCT_DOUBLE_QUOTE_RIGHT,
        SUFFIX_PUNCT_SINGLE_QUOTE,
        SUFFIX_PUNCT_SINGLE_QUOTE_LEFT,
        SUFFIX_PUNCT_SINGLE_QUOTE_RIGHT,
        SUFFIX_PUNCT_BACKTICK,
        SUFFIX_PUNCT_PAREN,
        SUFFIX_PUNCT_SQUARE,
        SUFFIX_PUNCT_CURLY,
        SUFFIX_PUNCT_PERIOD,
        SUFFIX_PUNCT_EXCLAIM,
        SUFFIX_PUNCT_QUESTION,
        SUFFIX_PUNCT_COMMA,
        SUFFIX_PUNCT_SEMICOLON,
        SUFFIX_PUNCT_COLON,
        SUFFIX_PUNCT_HYPHEN,
        "punct_suffix_\u2014",
    ]
    if not force_possessives:
        suffix_punct_order.append(SUFFIX_PUNCT_POSSESSIVE_S)
        suffix_punct_order.append(SUFFIX_PUNCT_POSSESSIVE_S_CURLY)
        suffix_punct_order.append(SUFFIX_PUNCT_POSSESSIVE_PLURAL)
        suffix_punct_order.append(SUFFIX_PUNCT_POSSESSIVE_PLURAL_CURLY)

    if force_punctuation:
        prefix_punct_types = prefix_punct_order
        suffix_punct_types = suffix_punct_order
    else:
        if prefix_punct_types:
            prefix_punct_types = _order_punct_types(prefix_punct_types, prefix_punct_order)
        if suffix_punct_types:
            suffix_punct_types = _order_punct_types(suffix_punct_types, suffix_punct_order)

    if force_articles:
        article_types = [ARTICLE_THE, ARTICLE_A, ARTICLE_AN]
        if force_possessive_determiners:
            article_types.extend(POSSESSIVE_DETERMINER_TRANSFORMS)
        if force_demonstrative_determiners:
            article_types.extend(DEMONSTRATIVE_DETERMINER_TRANSFORMS)
        if force_quantifier_determiners:
            article_types.extend(QUANTIFIER_DETERMINER_TRANSFORMS)
    elif article_types:
        article_types = _order_article_types(article_types)

    if force_prepositions:
        prep_types = [f"prep_{p}" for p in default_preps]
    elif prep_types:
        prep_types = _order_prep_types(prep_types)

    if force_possessives:
        possessive_types = [POSSESSIVE_S, POSSESSIVE_S_CURLY, POSSESSIVE_PLURAL, POSSESSIVE_PLURAL_CURLY]
    elif possessive_types:
        possessive_types = _order_possessive_types(possessive_types)

    article_space_active = force_article_space_prefix or ADD_ARTICLE_SPACE_PREFIX in transformation_types
    prep_space_active = force_prep_space_prefix or ADD_PREP_SPACE_PREFIX in transformation_types

    active_groups = {
        "inflections": morphology_mode == "atomic" and len(inflection_types) > 0,
        "derivations": len(derivation_types) > 0,
        "space_prefix": (WITH_SPACE_PREFIX_TRANSFORM in transformation_types or
                         REMOVE_SPACE_PREFIX_TRANSFORM in transformation_types),
        "base_capitalization": (ADD_BASE_CAPITALIZATION_TRANSFORM in transformation_types or
                                ADD_ALL_CAPS_BASE_CAPITALIZATION_TRANSFORM in transformation_types or
                                REMOVE_BASE_CAPITALIZATION_TRANSFORM in transformation_types),
        "prefix_punctuation": len(prefix_punct_types) > 0,
        "suffix_punctuation": len(suffix_punct_types) > 0,
        "possessives": len(possessive_types) > 0,
        "determiners": len(article_types) > 0,
        "prepositions": len(prep_types) > 0,
        "article_space_prefix": article_space_active,
        "prep_space_prefix": prep_space_active,
        "article_capitalization": ADD_ARTICLE_CAPITALIZATION in transformation_types,
        "preposition_capitalization": ADD_PREP_CAPITALIZATION in transformation_types,
    }

    allowed_transformations = set()
    if active_groups["inflections"]:
        allowed_transformations.update(inflection_types)
        allowed_transformations.add(NO_INFLECTION)
        if include_na_types:
            allowed_transformations.add(NA_INFLECTION)
    if morphology_mode == "compositional":
        for group in active_morph_groups:
            allowed_transformations.update(morph_group_types[group])
            allowed_transformations.add(MORPHOLOGICAL_GROUP_NO_VALUES[group])
            if include_na_types:
                allowed_transformations.add(MORPHOLOGICAL_GROUP_NA_VALUES[group])
    if active_groups["derivations"]:
        allowed_transformations.update(derivation_types)
        allowed_transformations.add(NO_DERIVATION)
        if include_na_types:
            allowed_transformations.add(NA_DERIVATION)
    if active_groups["space_prefix"]:
        allowed_transformations.update([
            NO_SPACE_PREFIX_TRANSFORM,
            WITH_SPACE_PREFIX_TRANSFORM,
            REMOVE_SPACE_PREFIX_TRANSFORM,
        ])
    if active_groups["base_capitalization"]:
        allowed_transformations.update([
            ADD_BASE_CAPITALIZATION_TRANSFORM,
            ADD_ALL_CAPS_BASE_CAPITALIZATION_TRANSFORM,
            REMOVE_BASE_CAPITALIZATION_TRANSFORM,
            NO_BASE_CAPITALIZATION_TRANSFORM,
        ])
    if active_groups["prefix_punctuation"]:
        allowed_transformations.update(prefix_punct_types)
        allowed_transformations.add(NO_PREFIX_PUNCTUATION)
    if active_groups["suffix_punctuation"]:
        allowed_transformations.update(suffix_punct_types)
        allowed_transformations.add(NO_SUFFIX_PUNCTUATION)
    if active_groups["possessives"]:
        allowed_transformations.update(possessive_types)
        allowed_transformations.add(NO_POSSESSIVE)
    if active_groups["determiners"]:
        allowed_transformations.update(article_types)
        allowed_transformations.add(NO_ARTICLE)
        allowed_transformations.add(LEGACY_NO_ARTICLE)
        if include_na_types:
            allowed_transformations.add(NA_ARTICLE)
            allowed_transformations.add(LEGACY_NA_ARTICLE)
    if active_groups["prepositions"]:
        allowed_transformations.update(prep_types)
        allowed_transformations.add(NO_PREPOSITION)
    if active_groups["article_space_prefix"]:
        allowed_transformations.update([NO_ARTICLE_SPACE_PREFIX, ADD_ARTICLE_SPACE_PREFIX])
    if active_groups["prep_space_prefix"]:
        allowed_transformations.update([NO_PREP_SPACE_PREFIX, ADD_PREP_SPACE_PREFIX])
    if active_groups["article_capitalization"]:
        allowed_transformations.update([ADD_ARTICLE_CAPITALIZATION, NO_ARTICLE_CAPITALIZATION])
    if active_groups["preposition_capitalization"]:
        allowed_transformations.update([ADD_PREP_CAPITALIZATION, NO_PREP_CAPITALIZATION])

    if allowed_transformations:
        for _, inflections in decomposition_map.items():
            for inflection_form, inflection_info in inflections.items():
                inflections[inflection_form] = [t for t in inflection_info if t in allowed_transformations]

    transformation_types = _collect_types()
    inflection_types = sorted([t for t in transformation_types if t.startswith('inflect_')]) if morphology_mode == "atomic" else []
    derivation_types = sorted([t for t in transformation_types if t.startswith('deriv_')])
    prefix_punct_types = [t for t in transformation_types if t.startswith('punct_prefix_')]
    suffix_punct_types = [t for t in transformation_types if t.startswith('punct_suffix_')]
    possessive_types = [
        t for t in transformation_types
        if t in [POSSESSIVE_S, POSSESSIVE_S_CURLY, POSSESSIVE_PLURAL, POSSESSIVE_PLURAL_CURLY]
    ]
    article_types = [t for t in transformation_types if is_non_default_determiner_transform(t)]
    prep_types = [t for t in transformation_types if t.startswith('prep_') and t not in [NO_PREPOSITION, NA_PREPOSITION]]

    if force_punctuation:
        prefix_punct_types = prefix_punct_order
        suffix_punct_types = suffix_punct_order
    else:
        if prefix_punct_types:
            prefix_punct_types = _order_punct_types(prefix_punct_types, prefix_punct_order)
        if suffix_punct_types:
            suffix_punct_types = _order_punct_types(suffix_punct_types, suffix_punct_order)

    if force_articles:
        article_types = [ARTICLE_THE, ARTICLE_A, ARTICLE_AN]
        if force_possessive_determiners:
            article_types.extend(POSSESSIVE_DETERMINER_TRANSFORMS)
        if force_demonstrative_determiners:
            article_types.extend(DEMONSTRATIVE_DETERMINER_TRANSFORMS)
        if force_quantifier_determiners:
            article_types.extend(QUANTIFIER_DETERMINER_TRANSFORMS)
    elif article_types:
        article_types = _order_article_types(article_types)

    if force_prepositions:
        prep_types = [f"prep_{p}" for p in default_preps]
    elif prep_types:
        prep_types = _order_prep_types(prep_types)

    if force_possessives:
        possessive_types = [POSSESSIVE_S, POSSESSIVE_S_CURLY, POSSESSIVE_PLURAL, POSSESSIVE_PLURAL_CURLY]
    elif possessive_types:
        possessive_types = _order_possessive_types(possessive_types)

    if morphology_mode == "compositional":
        morph_group_types, active_morph_groups = _collect_morph_groups(transformation_types)

    # Build transformations_names with consecutive groups
    # IMPORTANT: For UnifiedModifierArray, rel=0 must be "no transformation" (NO_*).
    transformations_names = []
    loss_indices = {}
    def _append_group(
        group_name,
        no_value,
        values,
        na_value=None,
        aliases=None,
    ):
        # Do not create groups that have no learnable/non-default values.
        # A group with only NO/NA labels is effectively empty and should be omitted.
        if not values:
            return None
        start = len(transformations_names)
        transformations_names.append(no_value)
        transformations_names.extend(values)
        if include_na_types and na_value is not None:
            transformations_names.append(na_value)
        end = len(transformations_names)
        if group_name is not None:
            loss_indices[group_name] = (start, end)
        for alias in aliases or []:
            loss_indices[alias] = (start, end)
        return start, end

    if morphology_mode == 'compositional':
        _append_group(
            "space_prefix",
            NO_SPACE_PREFIX_TRANSFORM,
            [WITH_SPACE_PREFIX_TRANSFORM, REMOVE_SPACE_PREFIX_TRANSFORM],
            aliases=["prefix"],
        )
        _append_group(
            "base_capitalization",
            NO_BASE_CAPITALIZATION_TRANSFORM,
            [
                ADD_BASE_CAPITALIZATION_TRANSFORM,
                ADD_ALL_CAPS_BASE_CAPITALIZATION_TRANSFORM,
                REMOVE_BASE_CAPITALIZATION_TRANSFORM,
            ],
        )
        if derivation_types:
            _append_group(
                "derivation",
                NO_DERIVATION,
                derivation_types,
                na_value=NA_DERIVATION,
                aliases=["derivations"],
            )

        for group in active_morph_groups:
            _append_group(
                group,
                MORPHOLOGICAL_GROUP_NO_VALUES[group],
                sorted(morph_group_types[group]),
                na_value=MORPHOLOGICAL_GROUP_NA_VALUES[group],
            )
    else:
        _append_group(
            "inflection",
            NO_INFLECTION,
            inflection_types,
            na_value=NA_INFLECTION,
            aliases=["inflections"],
        )
        _append_group(
            "derivation",
            NO_DERIVATION,
            derivation_types,
            na_value=NA_DERIVATION,
            aliases=["derivations"],
        )
        _append_group(
            "space_prefix",
            NO_SPACE_PREFIX_TRANSFORM,
            [WITH_SPACE_PREFIX_TRANSFORM, REMOVE_SPACE_PREFIX_TRANSFORM],
            aliases=["prefix"],
        )
        _append_group(
            "base_capitalization",
            NO_BASE_CAPITALIZATION_TRANSFORM,
            [
                ADD_BASE_CAPITALIZATION_TRANSFORM,
                ADD_ALL_CAPS_BASE_CAPITALIZATION_TRANSFORM,
                REMOVE_BASE_CAPITALIZATION_TRANSFORM,
            ],
        )

    _append_group(
        "prefix_punctuation",
        NO_PREFIX_PUNCTUATION,
        prefix_punct_types,
    )
    _append_group(
        "suffix_punctuation",
        NO_SUFFIX_PUNCTUATION,
        suffix_punct_types,
    )
    _append_group(
        "possessives",
        NO_POSSESSIVE,
        possessive_types,
    )
    _append_group(
        DETERMINER_GROUP_NAME,
        NO_ARTICLE,
        article_types,
        aliases=list(DETERMINER_GROUP_ALIASES),
    )
    if article_space_active:
        _append_group(
            "article_space_prefix",
            NO_ARTICLE_SPACE_PREFIX,
            [ADD_ARTICLE_SPACE_PREFIX],
        )
    _append_group(
        "prepositions",
        NO_PREPOSITION,
        prep_types,
    )
    if prep_space_active:
        _append_group(
            "prep_space_prefix",
            NO_PREP_SPACE_PREFIX,
            [ADD_PREP_SPACE_PREFIX],
        )
    _append_group(
        "article_capitalization",
        NO_ARTICLE_CAPITALIZATION,
        [ADD_ARTICLE_CAPITALIZATION],
    )
    _append_group(
        "prep_capitalization",
        NO_PREP_CAPITALIZATION,
        [ADD_PREP_CAPITALIZATION],
        aliases=["preposition_capitalization"],
    )

    transformations_to_int = dict(zip(transformations_names, range(len(transformations_names))))

    return transformations_to_int, loss_indices

__all__ = [
    "build_complete_transforms",
    "get_variation_transformations",
    "apply_capitalization_pattern",
    "get_label_maps_from_decomposition_map",
    "strip_all_affixes",
]
