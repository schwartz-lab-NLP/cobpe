"""Affix, punctuation, determiner, and preposition detection."""

from .constants import *  # type: ignore  # noqa: F401,F403
from .prepositions import DEFAULT_PREPOSITION_LIST, normalize_preposition_list

# ==============================================================================
# Punctuation, Article, and Preposition Detection Functions
# ==============================================================================

def detect_prefix_punctuation(text):
    """
    Detect prefix punctuation combinations at the start of text.

    Returns:
        tuple: (punctuation_type_constant, remaining_text) or (None, text) if no match
    """
    if not text:
        return None, text

    # Define prefix patterns in order of priority (longest first)
    # Format: (pattern, constant)
    prefix_patterns = [
        # Two-character combinations
        ("'(", PREFIX_PUNCT_SINGLE_PAREN),
        ("\u2018(", "punct_prefix_\u2018("),
        ("\u2019(", "punct_prefix_\u2019("),
        ("\"(", PREFIX_PUNCT_DOUBLE_PAREN),
        ("\u201c(", "punct_prefix_\u201c("),
        ("\u201d(", "punct_prefix_\u201d("),
        ("'[", PREFIX_PUNCT_SINGLE_SQUARE),
        ("\u2018[", "punct_prefix_\u2018["),
        ("\u2019[", "punct_prefix_\u2019["),
        ("\"[", PREFIX_PUNCT_DOUBLE_SQUARE),
        ("\u201c[", "punct_prefix_\u201c["),
        ("\u201d[", "punct_prefix_\u201d["),
        ("('", PREFIX_PUNCT_PAREN_SINGLE),
        ("(\u2018", "punct_prefix_(\u2018"),
        ("(\u2019", "punct_prefix_(\u2019"),
        ("-(", PREFIX_PUNCT_HYPHEN_PAREN),
        ("-[", PREFIX_PUNCT_HYPHEN_SQUARE),
        # Single characters
        ("'", PREFIX_PUNCT_SINGLE_QUOTE),
        ("\u2018", PREFIX_PUNCT_SINGLE_QUOTE_LEFT),
        ("\u2019", PREFIX_PUNCT_SINGLE_QUOTE_RIGHT),
        ("\"", PREFIX_PUNCT_DOUBLE_QUOTE),
        ("\u201c", PREFIX_PUNCT_DOUBLE_QUOTE_LEFT),
        ("\u201d", PREFIX_PUNCT_DOUBLE_QUOTE_RIGHT),
        ("`", PREFIX_PUNCT_BACKTICK),
        ("(", PREFIX_PUNCT_PAREN),
        ("[", PREFIX_PUNCT_SQUARE),
        ("{", PREFIX_PUNCT_CURLY),
        ("-", PREFIX_PUNCT_HYPHEN),
    ]

    for pattern, const in prefix_patterns:
        if text.startswith(pattern):
            return const, text[len(pattern):]

    return None, text


def detect_suffix_punctuation(text, include_possessives=True):
    """
    Detect suffix punctuation combinations at the end of text.

    Returns:
        tuple: (punctuation_type_constant, remaining_text) or (None, text) if no match
    """
    if not text:
        return None, text

    # Define suffix patterns in order of priority (longest first)
    # Format: (pattern, constant)
    suffix_patterns = [
        # Two-character combinations
        (")'", SUFFIX_PUNCT_PAREN_SINGLE),
        (")\u2019", "punct_suffix_)\u2019"),
        (")\"", SUFFIX_PUNCT_PAREN_DOUBLE),
        (")\u201d", "punct_suffix_)\u201d"),
        ("]'", SUFFIX_PUNCT_SQUARE_SINGLE),
        ("]\u2019", "punct_suffix_]\u2019"),
        ("]\"", SUFFIX_PUNCT_SQUARE_DOUBLE),
        ("]\u201d", "punct_suffix_]\u201d"),
        (".'", SUFFIX_PUNCT_PERIOD_SINGLE),
        (".\u2019", "punct_suffix_.\u2019"),
        (".\"", SUFFIX_PUNCT_PERIOD_DOUBLE),
        (".\u201d", "punct_suffix_.\u201d"),
        ("!'", SUFFIX_PUNCT_EXCLAIM_SINGLE),
        ("!\u2019", "punct_suffix_!\u2019"),
        ("!\"", SUFFIX_PUNCT_EXCLAIM_DOUBLE),
        ("!\u201d", "punct_suffix_!\u201d"),
        ("?'", SUFFIX_PUNCT_QUESTION_SINGLE),
        ("?\u2019", "punct_suffix_?\u2019"),
        ("?\"", SUFFIX_PUNCT_QUESTION_DOUBLE),
        ("?\u201d", "punct_suffix_?\u201d"),
        (",'", SUFFIX_PUNCT_COMMA_SINGLE),
        (",\u2019", "punct_suffix_,\u2019"),
        (",\"", SUFFIX_PUNCT_COMMA_DOUBLE),
        (",\u201d", "punct_suffix_,\u201d"),
        ("',", SUFFIX_PUNCT_SINGLE_COMMA),
        ("\u2019,", "punct_suffix_\u2019,"),
        ("\",", SUFFIX_PUNCT_DOUBLE_COMMA),
        ("\u201d,", "punct_suffix_\u201d,"),
        ("'.", SUFFIX_PUNCT_SINGLE_PERIOD),
        ("\u2019.", "punct_suffix_\u2019."),
        ("\".", SUFFIX_PUNCT_DOUBLE_PERIOD),
        ("\u201d.", "punct_suffix_\u201d."),
        (".)", SUFFIX_PUNCT_PERIOD_PAREN),
        (").", SUFFIX_PUNCT_PAREN_PERIOD),
        ("!)", SUFFIX_PUNCT_EXCLAIM_PAREN),
        ("?)", SUFFIX_PUNCT_QUESTION_PAREN),
        (",)", SUFFIX_PUNCT_COMMA_PAREN),
        (")-", SUFFIX_PUNCT_PAREN_HYPHEN),
        ("]-", SUFFIX_PUNCT_SQUARE_HYPHEN),
    ]
    if include_possessives:
        suffix_patterns.extend([
            ("'s", SUFFIX_PUNCT_POSSESSIVE_S),  # Must come before single quote
            ("s'", SUFFIX_PUNCT_POSSESSIVE_PLURAL),
            ("\u2019s", SUFFIX_PUNCT_POSSESSIVE_S_CURLY),
            ("s\u2019", SUFFIX_PUNCT_POSSESSIVE_PLURAL_CURLY),
        ])
    suffix_patterns.extend([
        # Single characters
        ("'", SUFFIX_PUNCT_SINGLE_QUOTE),  # Can be both regular quote or possessive
        ("\u2018", SUFFIX_PUNCT_SINGLE_QUOTE_LEFT),
        ("\u2019", SUFFIX_PUNCT_SINGLE_QUOTE_RIGHT),
        ("\"", SUFFIX_PUNCT_DOUBLE_QUOTE),
        ("\u201c", SUFFIX_PUNCT_DOUBLE_QUOTE_LEFT),
        ("\u201d", SUFFIX_PUNCT_DOUBLE_QUOTE_RIGHT),
        ("`", SUFFIX_PUNCT_BACKTICK),
        (")", SUFFIX_PUNCT_PAREN),
        ("]", SUFFIX_PUNCT_SQUARE),
        ("}", SUFFIX_PUNCT_CURLY),
        (".", SUFFIX_PUNCT_PERIOD),
        ("!", SUFFIX_PUNCT_EXCLAIM),
        ("?", SUFFIX_PUNCT_QUESTION),
        (",", SUFFIX_PUNCT_COMMA),
        (";", SUFFIX_PUNCT_SEMICOLON),
        (":", SUFFIX_PUNCT_COLON),
        ("-", SUFFIX_PUNCT_HYPHEN),
    ])

    for pattern, const in suffix_patterns:
        if text.endswith(pattern):
            return const, text[:-len(pattern)]

    return None, text


def detect_possessive_suffix(text):
    """Detect English possessive suffix at the end of text."""
    if not text:
        return None, text

    patterns = [
        ("'s", POSSESSIVE_S),
        ("\u2019s", POSSESSIVE_S_CURLY),
        ("s'", POSSESSIVE_PLURAL),
        ("s\u2019", POSSESSIVE_PLURAL_CURLY),
    ]
    for pattern, const in patterns:
        if text.endswith(pattern):
            return const, text[:-len(pattern)]
    return None, text


def detect_article_prefix(
    text,
    articles=None,
    include_possessive_determiners=False,
    include_demonstrative_determiners=False,
    include_quantifier_determiners=False,
):
    """
    Detect article prefix at the start of text.

    Args:
        text: Input text to check
        articles: List of articles to check for (default: ["the", "a", "an"])

    Returns:
        tuple: (article_constant, is_capitalized, remaining_text) or (None, False, text) if no match
    """
    if not text:
        return None, False, text

    article_to_const = {
        "the": ARTICLE_THE,
        "a": ARTICLE_A,
        "an": ARTICLE_AN,
        "my": ARTICLE_MY,
        "your": ARTICLE_YOUR,
        "his": ARTICLE_HIS,
        "her": ARTICLE_HER,
        "our": ARTICLE_OUR,
        "their": ARTICLE_THEIR,
        "its": ARTICLE_ITS,
        "this": ARTICLE_THIS,
        "that": ARTICLE_THAT,
        "these": ARTICLE_THESE,
        "those": ARTICLE_THOSE,
        "some": ARTICLE_SOME,
        "any": ARTICLE_ANY,
        "no": ARTICLE_NO,
        "all": ARTICLE_ALL,
        "both": ARTICLE_BOTH,
        "each": ARTICLE_EACH,
        "every": ARTICLE_EVERY,
        "several": ARTICLE_SEVERAL,
        "many": ARTICLE_MANY,
        "much": ARTICLE_MUCH,
        "more": ARTICLE_MORE,
        "most": ARTICLE_MOST,
        "few": ARTICLE_FEW,
        "fewer": ARTICLE_FEWER,
        "little": ARTICLE_LITTLE,
        "less": ARTICLE_LESS,
        "another": ARTICLE_ANOTHER,
    }

    if articles is None:
        articles = ["the", "a", "an"]
        if include_possessive_determiners:
            articles.extend(["my", "your", "his", "her", "our", "their", "its"])
        if include_demonstrative_determiners:
            articles.extend(["this", "that", "these", "those"])
        if include_quantifier_determiners:
            articles.extend(
                [
                    "some",
                    "any",
                    "no",
                    "all",
                    "both",
                    "each",
                    "every",
                    "several",
                    "many",
                    "much",
                    "more",
                    "most",
                    "few",
                    "fewer",
                    "little",
                    "less",
                    "another",
                ]
            )

    text_lower = text.lower()

    for article in sorted(articles, key=len, reverse=True):
        # Check if text starts with article followed by space
        if text_lower.startswith(article + " "):
            is_cap = text[:len(article)][0].isupper()
            remaining = text[len(article) + 1:]  # +1 to skip the space

            article_const = article_to_const.get(article)
            if article_const is not None:
                return article_const, is_cap, remaining

    return None, False, text


def detect_preposition_prefix(text, prepositions=None):
    """
    Detect preposition prefix at the start of text.

    Args:
        text: Input text to check
        prepositions: List of prepositions to check for (default: common prepositions)

    Returns:
        tuple: (preposition_constant, is_capitalized, remaining_text) or (None, False, text) if no match
    """
    if not text:
        return None, False, text

    if prepositions is None:
        prepositions = list(DEFAULT_PREPOSITION_LIST)
    else:
        prepositions = normalize_preposition_list(prepositions)

    text_lower = text.lower()

    # Sort by length (longest first) to match longer prepositions first
    for prep in sorted(prepositions, key=len, reverse=True):
        # Check if text starts with preposition followed by space
        if text_lower.startswith(prep + " "):
            is_cap = text[:len(prep)][0].isupper()
            remaining = text[len(prep) + 1:]  # +1 to skip the space

            # Return the appropriate constant dynamically
            prep_const = f"prep_{prep}"
            return prep_const, is_cap, remaining

    return None, False, text


def strip_all_affixes(text, space_prefix=" ", decompose_punctuation=False,
                      decompose_articles=False, decompose_prepositions=False,
                      prepositions=None, track_article_prep_space_prefix=False,
                      decompose_possessive_determiners=False,
                      decompose_demonstrative_determiners=False,
                      decompose_quantifier_determiners=False,
                      decompose_possessive_suffixes=False):
    """
    Strip all affixes (space, punctuation, articles, prepositions) from text.

    Returns:
        tuple: (base_text, transformations_dict) where transformations_dict contains:
            - 'prefix_punct': prefix punctuation constant or None
            - 'suffix_punct': suffix punctuation constant or None
            - 'possessive': possessive suffix constant or None
            - 'article': article constant or None
            - 'article_cap': whether article is capitalized
            - 'prep': preposition constant or None
            - 'prep_cap': whether preposition is capitalized
            - 'space_prefix': whether has space prefix
            - 'article_space_prefix': whether article had a leading space
            - 'prep_space_prefix': whether preposition had a leading space
    """
    transformations = {
        'prefix_punct': None,
        'suffix_punct': None,
        'possessive': None,
        'article': None,
        'article_cap': False,
        'article_space_prefix': False,
        'prep': None,
        'prep_cap': False,
        'prep_space_prefix': False,
        'space_prefix': False,
    }

    # Strip space prefix first
    if text.startswith(space_prefix):
        transformations['space_prefix'] = True
        text = text[len(space_prefix):]

    # Strip prefix punctuation
    if decompose_punctuation:
        punct, text = detect_prefix_punctuation(text)
        transformations['prefix_punct'] = punct

    # Strip preposition (must come before article since "the" can be in preposition phrase)
    if decompose_prepositions:
        prep, prep_cap, text = detect_preposition_prefix(text, prepositions)
        transformations['prep'] = prep
        transformations['prep_cap'] = prep_cap
        if track_article_prep_space_prefix and prep:
            transformations['prep_space_prefix'] = transformations['space_prefix'] and transformations['prefix_punct'] is None

    # Strip article
    if decompose_articles:
        article, article_cap, text = detect_article_prefix(
            text,
            include_possessive_determiners=decompose_possessive_determiners,
            include_demonstrative_determiners=decompose_demonstrative_determiners,
            include_quantifier_determiners=decompose_quantifier_determiners,
        )
        transformations['article'] = article
        transformations['article_cap'] = article_cap
        if track_article_prep_space_prefix and article:
            transformations['article_space_prefix'] = bool(transformations['prep']) or (
                transformations['space_prefix'] and transformations['prefix_punct'] is None
            )

    # Strip suffix punctuation
    if decompose_punctuation:
        punct = None
        if decompose_possessive_suffixes:
            # Support both "dog's." and "dogs'," by allowing one suffix punctuation
            # mark and one possessive suffix in either order.
            poss, text_after_poss = detect_possessive_suffix(text)
            if poss:
                transformations['possessive'] = poss
                text = text_after_poss
                punct, text = detect_suffix_punctuation(text, include_possessives=False)
            else:
                punct, text = detect_suffix_punctuation(text, include_possessives=False)
                poss, text_after_poss = detect_possessive_suffix(text)
                if poss:
                    transformations['possessive'] = poss
                    text = text_after_poss
        else:
            punct, text = detect_suffix_punctuation(text, include_possessives=True)
        transformations['suffix_punct'] = punct

    return text, transformations

__all__ = [
    'detect_prefix_punctuation',
    'detect_suffix_punctuation',
    'detect_possessive_suffix',
    'detect_article_prefix',
    'detect_preposition_prefix',
    'strip_all_affixes',
]
