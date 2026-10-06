"""Transformation constants for vocabulary decomposition."""

import re

# Constants for transformation types
# Space prefix transformations
WITH_SPACE_PREFIX_TRANSFORM = "with_space_prefix"
NO_SPACE_PREFIX_TRANSFORM = "no_space_prefix"
REMOVE_SPACE_PREFIX_TRANSFORM = "remove_space_prefix"
NA_SPACE_PREFIX_TRANSFORM = "NA_space_prefix"

# Base capitalization transformations (existing, renamed for clarity)
NO_BASE_CAPITALIZATION_TRANSFORM = "no_capitalization"
ADD_BASE_CAPITALIZATION_TRANSFORM = "add_capitalization"
ADD_ALL_CAPS_BASE_CAPITALIZATION_TRANSFORM = "add_all_caps"
REMOVE_BASE_CAPITALIZATION_TRANSFORM = "remove_capitalization"
NA_BASE_CAPITALIZATION_TRANSFORM = "NA_capitalization"

# Legacy names for backwards compatibility
NO_CAPITALIZATION_TRANSFORM = NO_BASE_CAPITALIZATION_TRANSFORM
ADD_CAPITALIZATION_TRANSFORM = ADD_BASE_CAPITALIZATION_TRANSFORM
ADD_ALL_CAPS_CAPITALIZATION_TRANSFORM = ADD_ALL_CAPS_BASE_CAPITALIZATION_TRANSFORM
REMOVE_CAPITALIZATION_TRANSFORM = REMOVE_BASE_CAPITALIZATION_TRANSFORM
NA_CAPITALIZATION_TRANSFORM = NA_BASE_CAPITALIZATION_TRANSFORM

# Inflection and derivation transformations
NO_INFLECTION = "none"
NA_INFLECTION = "NA_inflection"
NO_DERIVATION = "no_derivation"
NA_DERIVATION = "NA_derivation"

# Prefix punctuation transformations (combinations of opening marks)
NO_PREFIX_PUNCTUATION = "no_prefix_punct"
NA_PREFIX_PUNCTUATION = "NA_prefix_punct"
# Single marks
PREFIX_PUNCT_SINGLE_QUOTE = "punct_prefix_'"
PREFIX_PUNCT_SINGLE_QUOTE_LEFT = "punct_prefix_\u2018"
PREFIX_PUNCT_SINGLE_QUOTE_RIGHT = "punct_prefix_\u2019"
PREFIX_PUNCT_DOUBLE_QUOTE = "punct_prefix_\""
PREFIX_PUNCT_DOUBLE_QUOTE_LEFT = "punct_prefix_\u201c"
PREFIX_PUNCT_DOUBLE_QUOTE_RIGHT = "punct_prefix_\u201d"
PREFIX_PUNCT_BACKTICK = "punct_prefix_`"
PREFIX_PUNCT_PAREN = "punct_prefix_("
PREFIX_PUNCT_SQUARE = "punct_prefix_["
PREFIX_PUNCT_CURLY = "punct_prefix_{"
PREFIX_PUNCT_HYPHEN = "punct_prefix_-"
# Two-mark combinations (most common)
PREFIX_PUNCT_SINGLE_PAREN = "punct_prefix_'("
PREFIX_PUNCT_DOUBLE_PAREN = "punct_prefix_\"("
PREFIX_PUNCT_SINGLE_SQUARE = "punct_prefix_'["
PREFIX_PUNCT_DOUBLE_SQUARE = "punct_prefix_\"["
PREFIX_PUNCT_PAREN_SINGLE = "punct_prefix_('\""  # "(' special case
PREFIX_PUNCT_HYPHEN_PAREN = "punct_prefix_-("
PREFIX_PUNCT_HYPHEN_SQUARE = "punct_prefix_-["

# Suffix punctuation transformations (combinations of closing marks)
NO_SUFFIX_PUNCTUATION = "no_suffix_punct"
NA_SUFFIX_PUNCTUATION = "NA_suffix_punct"
# Single marks - quotes
SUFFIX_PUNCT_SINGLE_QUOTE = "punct_suffix_'"
SUFFIX_PUNCT_SINGLE_QUOTE_LEFT = "punct_suffix_\u2018"
SUFFIX_PUNCT_SINGLE_QUOTE_RIGHT = "punct_suffix_\u2019"
SUFFIX_PUNCT_DOUBLE_QUOTE = "punct_suffix_\""
SUFFIX_PUNCT_DOUBLE_QUOTE_LEFT = "punct_suffix_\u201c"
SUFFIX_PUNCT_DOUBLE_QUOTE_RIGHT = "punct_suffix_\u201d"
SUFFIX_PUNCT_BACKTICK = "punct_suffix_`"
# Single marks - brackets
SUFFIX_PUNCT_PAREN = "punct_suffix_)"
SUFFIX_PUNCT_SQUARE = "punct_suffix_]"
SUFFIX_PUNCT_CURLY = "punct_suffix_}"
# Single marks - terminal
SUFFIX_PUNCT_PERIOD = "punct_suffix_."
SUFFIX_PUNCT_EXCLAIM = "punct_suffix_!"
SUFFIX_PUNCT_QUESTION = "punct_suffix_?"
SUFFIX_PUNCT_COMMA = "punct_suffix_,"
SUFFIX_PUNCT_SEMICOLON = "punct_suffix_;"
SUFFIX_PUNCT_COLON = "punct_suffix_:"
# Single marks - possessive and hyphen
SUFFIX_PUNCT_POSSESSIVE_S = "punct_suffix_'s"
SUFFIX_PUNCT_POSSESSIVE_S_CURLY = "punct_suffix_\u2019s"
SUFFIX_PUNCT_POSSESSIVE_PLURAL = "punct_suffix_s'"
SUFFIX_PUNCT_POSSESSIVE_PLURAL_CURLY = "punct_suffix_s\u2019"
SUFFIX_PUNCT_HYPHEN = "punct_suffix_-"
# Two-mark combinations (most common)
SUFFIX_PUNCT_PAREN_SINGLE = "punct_suffix_)'"
SUFFIX_PUNCT_PAREN_DOUBLE = "punct_suffix_)\""
SUFFIX_PUNCT_SQUARE_SINGLE = "punct_suffix_]'"
SUFFIX_PUNCT_SQUARE_DOUBLE = "punct_suffix_]\""
SUFFIX_PUNCT_PERIOD_SINGLE = "punct_suffix_.'"
SUFFIX_PUNCT_PERIOD_DOUBLE = "punct_suffix_.\""
SUFFIX_PUNCT_EXCLAIM_SINGLE = "punct_suffix_!'"
SUFFIX_PUNCT_EXCLAIM_DOUBLE = "punct_suffix_!\""
SUFFIX_PUNCT_QUESTION_SINGLE = "punct_suffix_?'"
SUFFIX_PUNCT_QUESTION_DOUBLE = "punct_suffix_?\""
SUFFIX_PUNCT_COMMA_SINGLE = "punct_suffix_,'"
SUFFIX_PUNCT_COMMA_DOUBLE = "punct_suffix_,\""
SUFFIX_PUNCT_SINGLE_COMMA = "punct_suffix_',"
SUFFIX_PUNCT_DOUBLE_COMMA = "punct_suffix_\","
SUFFIX_PUNCT_SINGLE_PERIOD = "punct_suffix_'."
SUFFIX_PUNCT_DOUBLE_PERIOD = "punct_suffix_\"."
SUFFIX_PUNCT_PERIOD_PAREN = "punct_suffix_.)"
SUFFIX_PUNCT_PAREN_PERIOD = "punct_suffix_)."
SUFFIX_PUNCT_EXCLAIM_PAREN = "punct_suffix_!)"
SUFFIX_PUNCT_QUESTION_PAREN = "punct_suffix_?)"
SUFFIX_PUNCT_COMMA_PAREN = "punct_suffix_,)"

# Optional separate possessive-suffix group
NO_POSSESSIVE = "no_possessive"
NA_POSSESSIVE = "NA_possessive"
POSSESSIVE_S = "possessive_'s"
POSSESSIVE_PLURAL = "possessive_s'"
POSSESSIVE_S_CURLY = "possessive_\u2019s"
POSSESSIVE_PLURAL_CURLY = "possessive_s\u2019"

# Pronoun possessive forms that should not be treated as inflections.
POSSESSIVE_PRONOUN_SKIP = {
    ("it", "its"),
}
SUFFIX_PUNCT_PAREN_HYPHEN = "punct_suffix_)-"
SUFFIX_PUNCT_SQUARE_HYPHEN = "punct_suffix_]-"

# Determiner transformations
NO_ARTICLE = "no_det"
NA_ARTICLE = "NA_det"
ARTICLE_THE = "det_the"
ARTICLE_A = "det_a"
ARTICLE_AN = "det_an"
ARTICLE_MY = "det_my"
ARTICLE_YOUR = "det_your"
ARTICLE_HIS = "det_his"
ARTICLE_HER = "det_her"
ARTICLE_OUR = "det_our"
ARTICLE_THEIR = "det_their"
ARTICLE_ITS = "det_its"
ARTICLE_THIS = "det_this"
ARTICLE_THAT = "det_that"
ARTICLE_THESE = "det_these"
ARTICLE_THOSE = "det_those"
ARTICLE_SOME = "det_some"
ARTICLE_ANY = "det_any"
ARTICLE_NO = "det_no"
ARTICLE_ALL = "det_all"
ARTICLE_BOTH = "det_both"
ARTICLE_EACH = "det_each"
ARTICLE_EVERY = "det_every"
ARTICLE_SEVERAL = "det_several"
ARTICLE_MANY = "det_many"
ARTICLE_MUCH = "det_much"
ARTICLE_MORE = "det_more"
ARTICLE_MOST = "det_most"
ARTICLE_FEW = "det_few"
ARTICLE_FEWER = "det_fewer"
ARTICLE_LITTLE = "det_little"
ARTICLE_LESS = "det_less"
ARTICLE_ANOTHER = "det_another"
LEGACY_NO_ARTICLE = "no_article"
LEGACY_NA_ARTICLE = "NA_article"
POSSESSIVE_DETERMINER_TRANSFORMS = [
    ARTICLE_MY,
    ARTICLE_YOUR,
    ARTICLE_HIS,
    ARTICLE_HER,
    ARTICLE_OUR,
    ARTICLE_THEIR,
    ARTICLE_ITS,
]
DEMONSTRATIVE_DETERMINER_TRANSFORMS = [
    ARTICLE_THIS,
    ARTICLE_THAT,
    ARTICLE_THESE,
    ARTICLE_THOSE,
]
QUANTIFIER_DETERMINER_TRANSFORMS = [
    ARTICLE_SOME,
    ARTICLE_ANY,
    ARTICLE_NO,
    ARTICLE_ALL,
    ARTICLE_BOTH,
    ARTICLE_EACH,
    ARTICLE_EVERY,
    ARTICLE_SEVERAL,
    ARTICLE_MANY,
    ARTICLE_MUCH,
    ARTICLE_MORE,
    ARTICLE_MOST,
    ARTICLE_FEW,
    ARTICLE_FEWER,
    ARTICLE_LITTLE,
    ARTICLE_LESS,
    ARTICLE_ANOTHER,
]
DETERMINER_GROUP_NAME = "determiners"
DETERMINER_GROUP_ALIASES = ("article_det", "articles")


def is_determiner_transform_name(name: str) -> bool:
    name = str(name or "")
    return name.startswith("det_") or name.startswith("article_")


def is_determiner_no_or_na(name: str) -> bool:
    return name in {NO_ARTICLE, NA_ARTICLE, LEGACY_NO_ARTICLE, LEGACY_NA_ARTICLE}


def is_non_default_determiner_transform(name: str) -> bool:
    return is_determiner_transform_name(name) and not is_determiner_no_or_na(name)

# Preposition transformations
NO_PREPOSITION = "no_prep"
NA_PREPOSITION = "NA_prep"
# Most frequent prepositions (default list)
PREP_BY = "prep_by"
PREP_AT = "prep_at"
PREP_OF = "prep_of"
PREP_TO = "prep_to"
PREP_IN = "prep_in"
PREP_ON = "prep_on"
PREP_WITH = "prep_with"
PREP_FOR = "prep_for"
PREP_FROM = "prep_from"
# Additional prepositions (can be enabled by extending the list)
# PREP_ABOUT = "prep_about"
# PREP_INTO = "prep_into"
# PREP_THROUGH = "prep_through"
# PREP_DURING = "prep_during"
# PREP_BEFORE = "prep_before"
# PREP_AFTER = "prep_after"
# PREP_ABOVE = "prep_above"
# PREP_BELOW = "prep_below"
# PREP_BETWEEN = "prep_between"
# PREP_UNDER = "prep_under"
# PREP_OVER = "prep_over"

# Preposition capitalization transformations
NO_PREP_CAPITALIZATION = "no_prep_cap"
ADD_PREP_CAPITALIZATION = "add_prep_cap"
NA_PREP_CAPITALIZATION = "NA_prep_cap"

# Spanish enclitic pronouns (for disambiguating UniMorph duplicate tags)
SPANISH_CLITICS = [
    "los", "las", "les",
    "lo", "la", "le",
    "me", "te", "se", "nos", "os",
]

# Morphological feature groups (compositional mode).
# These replace the atomic 'inflection' group when morphology_mode="compositional".
MORPHOLOGICAL_GROUPS = [
    "pos",              # V, N, ADJ, ADV, PRON, DET
    "tense",            # PRS, PST, FUT, IMPF
    "mood",             # IND, SBJV, COND, IMP
    "aspect",           # PFV, IPFV, PRF, PROG
    "person",           # 1, 2, 3
    "number",           # SG, PL, DU
    "gender",           # MASC, FEM, NEUT
    "case",             # NOM, ACC, GEN, DAT, ABL, LOC, INS, VOC
    "formality",        # FORM, INFM
    "voice",            # ACT, PASS, MID
    "polarity",         # POS, NEG
    "definiteness",     # DEF, INDF, SPEC
    "finiteness",       # FIN, NFIN, INF, PTCP, CVB
    "animacy",          # ANIM, INAN
    "clitic1_surface",  # me, te, se, lo, la, le, etc.
    "clitic2_surface",  # second clitic surface (if present)
    "clitic_person",    # 1, 2, 3
    "clitic_number",    # SG, PL
    "clitic_gender",    # MASC, FEM
    "clitic_case",      # ACC, DAT, GEN
    "clitic_type",      # PRO, REFL
    "clitic_reflexive", # REFL
    "degree",           # CMPR, SPRL
    "lgspec",           # LGSPEC1-10
]

MORPHOLOGICAL_GROUP_NO_VALUES = {group: f"no_{group}" for group in MORPHOLOGICAL_GROUPS}
MORPHOLOGICAL_GROUP_NA_VALUES = {group: f"NA_{group}" for group in MORPHOLOGICAL_GROUPS}


def _group_value_sort_key(value: str):
    if value.isdigit():
        return (0, int(value))
    match = re.match(r"^([A-Za-z_]+)(\\d+)$", value)
    if match:
        return (1, match.group(1), int(match.group(2)))
    return (2, value)


def _normalize_group_value(value: str) -> str:
    if value == "none" or "+" not in value:
        return value
    parts = [part for part in value.split("+") if part]
    unique = sorted(set(parts), key=_group_value_sort_key)
    return "+".join(unique)

# UniMorph tag to morphological group category mapping.
UNIMORPH_TAG_CATEGORY_MAP = {
    # Tense
    "PRS": "tense", "PST": "tense", "FUT": "tense", "IMPF": "tense",
    # Mood
    "IND": "mood", "SBJV": "mood", "COND": "mood", "IMP": "mood", "OPT": "mood",
    # Aspect
    "PFV": "aspect", "IPFV": "aspect", "PRF": "aspect", "PROG": "aspect",
    # Person
    "1": "person", "2": "person", "3": "person", "0": "person",
    # Number
    "SG": "number", "PL": "number", "DU": "number", "PAU": "number",
    # Gender
    "MASC": "gender", "FEM": "gender", "NEUT": "gender",
    # Case
    "NOM": "case", "ACC": "case", "GEN": "case", "DAT": "case",
    "ABL": "case", "LOC": "case", "INS": "case", "VOC": "case",
    "ESS": "case", "TRANS": "case", "COM": "case",
    # Formality
    "FORM": "formality", "INFM": "formality",
    # Voice
    "ACT": "voice", "PASS": "voice", "MID": "voice", "CAUS": "voice",
    # Polarity
    "POS": "polarity", "NEG": "polarity",
    # Definiteness
    "DEF": "definiteness", "INDF": "definiteness", "SPEC": "definiteness",
    # Finiteness
    "FIN": "finiteness", "NFIN": "finiteness", "INF": "finiteness",
    # Animacy
    "ANIM": "animacy", "INAN": "animacy", "HUM": "animacy",
    # Degree
    "CMPR": "degree", "SPRL": "degree",
    # Clitic marker (handled explicitly when surface tags are injected)
}

# Article capitalization transformations
NO_ARTICLE_CAPITALIZATION = "no_article_cap"
ADD_ARTICLE_CAPITALIZATION = "add_article_cap"
NA_ARTICLE_CAPITALIZATION = "NA_article_cap"

# Article/preposition space-prefix transformations (optional)
NO_ARTICLE_SPACE_PREFIX = "no_article_space_prefix"
ADD_ARTICLE_SPACE_PREFIX = "add_article_space_prefix"
NA_ARTICLE_SPACE_PREFIX = "NA_article_space_prefix"
NO_PREP_SPACE_PREFIX = "no_prep_space_prefix"
ADD_PREP_SPACE_PREFIX = "add_prep_space_prefix"
NA_PREP_SPACE_PREFIX = "NA_prep_space_prefix"

# Update NO_EMBEDDING_TYPES and NA_EMBEDDING_TYPES to include new groups
BASE_NO_EMBEDDING_TYPES = [
    NO_INFLECTION,
    NO_DERIVATION,
    NO_SPACE_PREFIX_TRANSFORM,
    NO_BASE_CAPITALIZATION_TRANSFORM,
    NO_PREFIX_PUNCTUATION,
    NO_SUFFIX_PUNCTUATION,
    NO_POSSESSIVE,
    NO_ARTICLE,
    NO_PREPOSITION,
    NO_PREP_CAPITALIZATION,
    NO_ARTICLE_CAPITALIZATION,
    NO_ARTICLE_SPACE_PREFIX,
    NO_PREP_SPACE_PREFIX,
]

BASE_NA_EMBEDDING_TYPES = [
    NA_INFLECTION,
    NA_DERIVATION,
    NA_SPACE_PREFIX_TRANSFORM,
    NA_BASE_CAPITALIZATION_TRANSFORM,
    NA_PREFIX_PUNCTUATION,
    NA_SUFFIX_PUNCTUATION,
    NA_POSSESSIVE,
    NA_ARTICLE,
    NA_PREPOSITION,
    NA_PREP_CAPITALIZATION,
    NA_ARTICLE_CAPITALIZATION,
    NA_ARTICLE_SPACE_PREFIX,
    NA_PREP_SPACE_PREFIX,
]
NO_EMBEDDING_TYPES = BASE_NO_EMBEDDING_TYPES + list(MORPHOLOGICAL_GROUP_NO_VALUES.values())
NA_EMBEDDING_TYPES = BASE_NA_EMBEDDING_TYPES + list(MORPHOLOGICAL_GROUP_NA_VALUES.values())
IGNORE_MULTI_TYPES_IN_INIT = []
FORBID_MULTI_TYPES_IN_INIT = [ADD_CAPITALIZATION_TRANSFORM, ADD_ALL_CAPS_CAPITALIZATION_TRANSFORM]

# Unified transformation groups for dual-stream tokenization
# This defines the canonical order of groups in the 2D modifier array
# Each position in the modifier array corresponds to one group
UNIFIED_TRANSFORM_GROUPS = [
    'space_prefix',           # NO_SPACE, ADD_SPACE
    'base_capitalization',    # NO_CAP, FIRST_CAP, ALL_CAPS (for base word)
    'inflection',             # NONE, GERUND, PAST, PLURAL, etc.
    'derivation',             # NONE, AGENT_ER, ABLE, etc.
    'determiners',            # NONE, THE, A, AN, MY, YOUR, ...
    'article_capitalization', # NO_CAP, CAP
    'prepositions',           # NONE, BY, AT, OF, TO, IN, ON, WITH, FOR, FROM
    'prep_capitalization',    # NO_CAP, CAP
    'prefix_punctuation',     # NONE, OPEN_PAREN, OPEN_BRACKET, QUOTE, etc.
    'suffix_punctuation',     # NONE, PERIOD, COMMA, EXCLAIM, CLOSE_PAREN, APOSTROPHE_S, etc.
    'possessives',            # NONE, 's, s'
]

# Mapping from group names to their NO_* constants (index 0 in each group)
UNIFIED_GROUP_NO_VALUES = {
    'space_prefix': NO_SPACE_PREFIX_TRANSFORM,
    'base_capitalization': NO_BASE_CAPITALIZATION_TRANSFORM,
    'inflection': NO_INFLECTION,
    'derivation': NO_DERIVATION,
    'determiners': NO_ARTICLE,
    'article_det': NO_ARTICLE,
    'articles': NO_ARTICLE,  # backward-compatible alias
    'article_capitalization': NO_ARTICLE_CAPITALIZATION,
    'prepositions': NO_PREPOSITION,
    'prep_capitalization': NO_PREP_CAPITALIZATION,
    'article_space_prefix': NO_ARTICLE_SPACE_PREFIX,
    'prep_space_prefix': NO_PREP_SPACE_PREFIX,
    'prefix_punctuation': NO_PREFIX_PUNCTUATION,
    'suffix_punctuation': NO_SUFFIX_PUNCTUATION,
    'possessives': NO_POSSESSIVE,
}

# Mapping from group names to their NA_* constants (for inapplicable cases)
UNIFIED_GROUP_NA_VALUES = {
    'space_prefix': NA_SPACE_PREFIX_TRANSFORM,
    'base_capitalization': NA_BASE_CAPITALIZATION_TRANSFORM,
    'inflection': NA_INFLECTION,
    'derivation': NA_DERIVATION,
    'determiners': NA_ARTICLE,
    'article_det': NA_ARTICLE,
    'articles': NA_ARTICLE,  # backward-compatible alias
    'article_capitalization': NA_ARTICLE_CAPITALIZATION,
    'prepositions': NA_PREPOSITION,
    'prep_capitalization': NA_PREP_CAPITALIZATION,
    'article_space_prefix': NA_ARTICLE_SPACE_PREFIX,
    'prep_space_prefix': NA_PREP_SPACE_PREFIX,
    'prefix_punctuation': NA_PREFIX_PUNCTUATION,
    'suffix_punctuation': NA_SUFFIX_PUNCTUATION,
    'possessives': NA_POSSESSIVE,
}

__all__ = [
    'WITH_SPACE_PREFIX_TRANSFORM',
    'NO_SPACE_PREFIX_TRANSFORM',
    'REMOVE_SPACE_PREFIX_TRANSFORM',
    'NA_SPACE_PREFIX_TRANSFORM',
    'NO_BASE_CAPITALIZATION_TRANSFORM',
    'ADD_BASE_CAPITALIZATION_TRANSFORM',
    'ADD_ALL_CAPS_BASE_CAPITALIZATION_TRANSFORM',
    'REMOVE_BASE_CAPITALIZATION_TRANSFORM',
    'NA_BASE_CAPITALIZATION_TRANSFORM',
    'NO_CAPITALIZATION_TRANSFORM',
    'ADD_CAPITALIZATION_TRANSFORM',
    'ADD_ALL_CAPS_CAPITALIZATION_TRANSFORM',
    'REMOVE_CAPITALIZATION_TRANSFORM',
    'NA_CAPITALIZATION_TRANSFORM',
    'NO_INFLECTION',
    'NA_INFLECTION',
    'NO_DERIVATION',
    'NA_DERIVATION',
    'NO_PREFIX_PUNCTUATION',
    'NA_PREFIX_PUNCTUATION',
    'PREFIX_PUNCT_SINGLE_QUOTE',
    'PREFIX_PUNCT_SINGLE_QUOTE_LEFT',
    'PREFIX_PUNCT_SINGLE_QUOTE_RIGHT',
    'PREFIX_PUNCT_DOUBLE_QUOTE',
    'PREFIX_PUNCT_DOUBLE_QUOTE_LEFT',
    'PREFIX_PUNCT_DOUBLE_QUOTE_RIGHT',
    'PREFIX_PUNCT_BACKTICK',
    'PREFIX_PUNCT_PAREN',
    'PREFIX_PUNCT_SQUARE',
    'PREFIX_PUNCT_CURLY',
    'PREFIX_PUNCT_HYPHEN',
    'PREFIX_PUNCT_SINGLE_PAREN',
    'PREFIX_PUNCT_DOUBLE_PAREN',
    'PREFIX_PUNCT_SINGLE_SQUARE',
    'PREFIX_PUNCT_DOUBLE_SQUARE',
    'PREFIX_PUNCT_PAREN_SINGLE',
    'PREFIX_PUNCT_HYPHEN_PAREN',
    'PREFIX_PUNCT_HYPHEN_SQUARE',
    'NO_SUFFIX_PUNCTUATION',
    'NA_SUFFIX_PUNCTUATION',
    'SUFFIX_PUNCT_SINGLE_QUOTE',
    'SUFFIX_PUNCT_SINGLE_QUOTE_LEFT',
    'SUFFIX_PUNCT_SINGLE_QUOTE_RIGHT',
    'SUFFIX_PUNCT_DOUBLE_QUOTE',
    'SUFFIX_PUNCT_DOUBLE_QUOTE_LEFT',
    'SUFFIX_PUNCT_DOUBLE_QUOTE_RIGHT',
    'SUFFIX_PUNCT_BACKTICK',
    'SUFFIX_PUNCT_PAREN',
    'SUFFIX_PUNCT_SQUARE',
    'SUFFIX_PUNCT_CURLY',
    'SUFFIX_PUNCT_PERIOD',
    'SUFFIX_PUNCT_EXCLAIM',
    'SUFFIX_PUNCT_QUESTION',
    'SUFFIX_PUNCT_COMMA',
    'SUFFIX_PUNCT_SEMICOLON',
    'SUFFIX_PUNCT_COLON',
    'SUFFIX_PUNCT_POSSESSIVE_S',
    'SUFFIX_PUNCT_POSSESSIVE_S_CURLY',
    'SUFFIX_PUNCT_POSSESSIVE_PLURAL',
    'SUFFIX_PUNCT_POSSESSIVE_PLURAL_CURLY',
    'SUFFIX_PUNCT_HYPHEN',
    'SUFFIX_PUNCT_PAREN_SINGLE',
    'SUFFIX_PUNCT_PAREN_DOUBLE',
    'SUFFIX_PUNCT_SQUARE_SINGLE',
    'SUFFIX_PUNCT_SQUARE_DOUBLE',
    'SUFFIX_PUNCT_PERIOD_SINGLE',
    'SUFFIX_PUNCT_PERIOD_DOUBLE',
    'SUFFIX_PUNCT_EXCLAIM_SINGLE',
    'SUFFIX_PUNCT_EXCLAIM_DOUBLE',
    'SUFFIX_PUNCT_QUESTION_SINGLE',
    'SUFFIX_PUNCT_QUESTION_DOUBLE',
    'SUFFIX_PUNCT_COMMA_SINGLE',
    'SUFFIX_PUNCT_COMMA_DOUBLE',
    'SUFFIX_PUNCT_SINGLE_COMMA',
    'SUFFIX_PUNCT_DOUBLE_COMMA',
    'SUFFIX_PUNCT_SINGLE_PERIOD',
    'SUFFIX_PUNCT_DOUBLE_PERIOD',
    'SUFFIX_PUNCT_PERIOD_PAREN',
    'SUFFIX_PUNCT_PAREN_PERIOD',
    'SUFFIX_PUNCT_EXCLAIM_PAREN',
    'SUFFIX_PUNCT_QUESTION_PAREN',
    'SUFFIX_PUNCT_COMMA_PAREN',
    'NO_POSSESSIVE',
    'NA_POSSESSIVE',
    'POSSESSIVE_S',
    'POSSESSIVE_PLURAL',
    'POSSESSIVE_S_CURLY',
    'POSSESSIVE_PLURAL_CURLY',
    'POSSESSIVE_PRONOUN_SKIP',
    'SUFFIX_PUNCT_PAREN_HYPHEN',
    'SUFFIX_PUNCT_SQUARE_HYPHEN',
    'NO_ARTICLE',
    'NA_ARTICLE',
    'ARTICLE_THE',
    'ARTICLE_A',
    'ARTICLE_AN',
    'ARTICLE_MY',
    'ARTICLE_YOUR',
    'ARTICLE_HIS',
    'ARTICLE_HER',
    'ARTICLE_OUR',
    'ARTICLE_THEIR',
    'ARTICLE_ITS',
    'ARTICLE_THIS',
    'ARTICLE_THAT',
    'ARTICLE_THESE',
    'ARTICLE_THOSE',
    'ARTICLE_SOME',
    'ARTICLE_ANY',
    'ARTICLE_NO',
    'ARTICLE_ALL',
    'ARTICLE_BOTH',
    'ARTICLE_EACH',
    'ARTICLE_EVERY',
    'ARTICLE_SEVERAL',
    'ARTICLE_MANY',
    'ARTICLE_MUCH',
    'ARTICLE_MORE',
    'ARTICLE_MOST',
    'ARTICLE_FEW',
    'ARTICLE_FEWER',
    'ARTICLE_LITTLE',
    'ARTICLE_LESS',
    'ARTICLE_ANOTHER',
    'LEGACY_NO_ARTICLE',
    'LEGACY_NA_ARTICLE',
    'POSSESSIVE_DETERMINER_TRANSFORMS',
    'DEMONSTRATIVE_DETERMINER_TRANSFORMS',
    'QUANTIFIER_DETERMINER_TRANSFORMS',
    'DETERMINER_GROUP_NAME',
    'DETERMINER_GROUP_ALIASES',
    'is_determiner_transform_name',
    'is_determiner_no_or_na',
    'is_non_default_determiner_transform',
    'NO_PREPOSITION',
    'NA_PREPOSITION',
    'PREP_BY',
    'PREP_AT',
    'PREP_OF',
    'PREP_TO',
    'PREP_IN',
    'PREP_ON',
    'PREP_WITH',
    'PREP_FOR',
    'PREP_FROM',
    'NO_PREP_CAPITALIZATION',
    'ADD_PREP_CAPITALIZATION',
    'NA_PREP_CAPITALIZATION',
    'SPANISH_CLITICS',
    'MORPHOLOGICAL_GROUPS',
    'MORPHOLOGICAL_GROUP_NO_VALUES',
    'MORPHOLOGICAL_GROUP_NA_VALUES',
    '_group_value_sort_key',
    '_normalize_group_value',
    'UNIMORPH_TAG_CATEGORY_MAP',
    'NO_ARTICLE_CAPITALIZATION',
    'ADD_ARTICLE_CAPITALIZATION',
    'NA_ARTICLE_CAPITALIZATION',
    'NO_ARTICLE_SPACE_PREFIX',
    'ADD_ARTICLE_SPACE_PREFIX',
    'NA_ARTICLE_SPACE_PREFIX',
    'NO_PREP_SPACE_PREFIX',
    'ADD_PREP_SPACE_PREFIX',
    'NA_PREP_SPACE_PREFIX',
    'BASE_NO_EMBEDDING_TYPES',
    'BASE_NA_EMBEDDING_TYPES',
    'NO_EMBEDDING_TYPES',
    'NA_EMBEDDING_TYPES',
    'IGNORE_MULTI_TYPES_IN_INIT',
    'FORBID_MULTI_TYPES_IN_INIT',
    'UNIFIED_TRANSFORM_GROUPS',
    'UNIFIED_GROUP_NO_VALUES',
    'UNIFIED_GROUP_NA_VALUES',
]
