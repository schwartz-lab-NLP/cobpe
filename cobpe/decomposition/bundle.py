import os
import json
import pickle
import hashlib
import shutil
import numpy as np
from collections import defaultdict
from transformers import AutoTokenizer
from cobpe.tokenization.fingerprints import fingerprint_named_files

from . import surface as _decomposition
from .prepositions import (
    DEFAULT_PREPOSITION_LIST,
    DEFAULT_PREPOSITION_PROFILE,
    classify_preposition_list,
    normalize_preposition_list,
    resolve_preposition_settings,
)
from .dual_stream import (
    identify_tokens_to_remove,
    PrunedTokenizerWrapper,
    build_sequence_map_from_decomposition,
    build_sequence_map_from_string_decomposition,
    DualStreamTokenizer,
)

get_variation_transformations = _decomposition.get_variation_transformations
strip_all_affixes = _decomposition.strip_all_affixes
get_label_maps_from_decomposition_map = _decomposition.get_label_maps_from_decomposition_map
UnifiedModifierArray = _decomposition.UnifiedModifierArray
UNIFIED_TRANSFORM_GROUPS = _decomposition.UNIFIED_TRANSFORM_GROUPS
DETERMINER_GROUP_NAME = _decomposition.DETERMINER_GROUP_NAME
ARTICLE_A = _decomposition.ARTICLE_A
ARTICLE_AN = _decomposition.ARTICLE_AN
ARTICLE_THE = _decomposition.ARTICLE_THE
ARTICLE_MY = _decomposition.ARTICLE_MY
ARTICLE_YOUR = _decomposition.ARTICLE_YOUR
ARTICLE_HIS = _decomposition.ARTICLE_HIS
ARTICLE_HER = _decomposition.ARTICLE_HER
ARTICLE_OUR = _decomposition.ARTICLE_OUR
ARTICLE_THEIR = _decomposition.ARTICLE_THEIR
ARTICLE_ITS = _decomposition.ARTICLE_ITS
ARTICLE_THIS = _decomposition.ARTICLE_THIS
ARTICLE_THAT = _decomposition.ARTICLE_THAT
ARTICLE_THESE = _decomposition.ARTICLE_THESE
ARTICLE_THOSE = _decomposition.ARTICLE_THOSE
ARTICLE_SOME = _decomposition.ARTICLE_SOME
ARTICLE_ANY = _decomposition.ARTICLE_ANY
ARTICLE_NO = _decomposition.ARTICLE_NO
ARTICLE_ALL = _decomposition.ARTICLE_ALL
ARTICLE_BOTH = _decomposition.ARTICLE_BOTH
ARTICLE_EACH = _decomposition.ARTICLE_EACH
ARTICLE_EVERY = _decomposition.ARTICLE_EVERY
ARTICLE_SEVERAL = _decomposition.ARTICLE_SEVERAL
ARTICLE_MANY = _decomposition.ARTICLE_MANY
ARTICLE_MUCH = _decomposition.ARTICLE_MUCH
ARTICLE_MORE = _decomposition.ARTICLE_MORE
ARTICLE_MOST = _decomposition.ARTICLE_MOST
ARTICLE_FEW = _decomposition.ARTICLE_FEW
ARTICLE_FEWER = _decomposition.ARTICLE_FEWER
ARTICLE_LITTLE = _decomposition.ARTICLE_LITTLE
ARTICLE_LESS = _decomposition.ARTICLE_LESS
ARTICLE_ANOTHER = _decomposition.ARTICLE_ANOTHER
PREFIX_PUNCT_BACKTICK = _decomposition.PREFIX_PUNCT_BACKTICK
PREFIX_PUNCT_CURLY = _decomposition.PREFIX_PUNCT_CURLY
PREFIX_PUNCT_DOUBLE_QUOTE = _decomposition.PREFIX_PUNCT_DOUBLE_QUOTE
PREFIX_PUNCT_HYPHEN = _decomposition.PREFIX_PUNCT_HYPHEN
PREFIX_PUNCT_PAREN = _decomposition.PREFIX_PUNCT_PAREN
PREFIX_PUNCT_SINGLE_QUOTE = _decomposition.PREFIX_PUNCT_SINGLE_QUOTE
PREFIX_PUNCT_SQUARE = _decomposition.PREFIX_PUNCT_SQUARE
SUFFIX_PUNCT_BACKTICK = _decomposition.SUFFIX_PUNCT_BACKTICK
SUFFIX_PUNCT_COLON = _decomposition.SUFFIX_PUNCT_COLON
SUFFIX_PUNCT_COMMA = _decomposition.SUFFIX_PUNCT_COMMA
SUFFIX_PUNCT_CURLY = _decomposition.SUFFIX_PUNCT_CURLY
SUFFIX_PUNCT_DOUBLE_QUOTE = _decomposition.SUFFIX_PUNCT_DOUBLE_QUOTE
SUFFIX_PUNCT_EXCLAIM = _decomposition.SUFFIX_PUNCT_EXCLAIM
SUFFIX_PUNCT_HYPHEN = _decomposition.SUFFIX_PUNCT_HYPHEN
SUFFIX_PUNCT_PAREN = _decomposition.SUFFIX_PUNCT_PAREN
SUFFIX_PUNCT_PERIOD = _decomposition.SUFFIX_PUNCT_PERIOD
SUFFIX_PUNCT_POSSESSIVE_PLURAL = _decomposition.SUFFIX_PUNCT_POSSESSIVE_PLURAL
SUFFIX_PUNCT_POSSESSIVE_S = _decomposition.SUFFIX_PUNCT_POSSESSIVE_S
POSSESSIVE_PLURAL = _decomposition.POSSESSIVE_PLURAL
POSSESSIVE_S = _decomposition.POSSESSIVE_S
SUFFIX_PUNCT_QUESTION = _decomposition.SUFFIX_PUNCT_QUESTION
SUFFIX_PUNCT_SEMICOLON = _decomposition.SUFFIX_PUNCT_SEMICOLON
SUFFIX_PUNCT_SINGLE_QUOTE = _decomposition.SUFFIX_PUNCT_SINGLE_QUOTE
SUFFIX_PUNCT_SQUARE = _decomposition.SUFFIX_PUNCT_SQUARE
NO_POSSESSIVE = _decomposition.NO_POSSESSIVE
NO_PREFIX_PUNCTUATION = _decomposition.NO_PREFIX_PUNCTUATION
NO_SUFFIX_PUNCTUATION = _decomposition.NO_SUFFIX_PUNCTUATION
NO_ARTICLE = _decomposition.NO_ARTICLE
NO_PREPOSITION = _decomposition.NO_PREPOSITION
NO_ARTICLE_CAPITALIZATION = _decomposition.NO_ARTICLE_CAPITALIZATION
ADD_ARTICLE_CAPITALIZATION = _decomposition.ADD_ARTICLE_CAPITALIZATION
NO_PREP_CAPITALIZATION = _decomposition.NO_PREP_CAPITALIZATION
ADD_PREP_CAPITALIZATION = _decomposition.ADD_PREP_CAPITALIZATION
NO_ARTICLE_SPACE_PREFIX = _decomposition.NO_ARTICLE_SPACE_PREFIX
ADD_ARTICLE_SPACE_PREFIX = _decomposition.ADD_ARTICLE_SPACE_PREFIX
NO_PREP_SPACE_PREFIX = _decomposition.NO_PREP_SPACE_PREFIX
ADD_PREP_SPACE_PREFIX = _decomposition.ADD_PREP_SPACE_PREFIX
NO_SPACE_PREFIX_TRANSFORM = _decomposition.NO_SPACE_PREFIX_TRANSFORM
WITH_SPACE_PREFIX_TRANSFORM = _decomposition.WITH_SPACE_PREFIX_TRANSFORM
REMOVE_SPACE_PREFIX_TRANSFORM = _decomposition.REMOVE_SPACE_PREFIX_TRANSFORM
NO_BASE_CAPITALIZATION_TRANSFORM = _decomposition.NO_BASE_CAPITALIZATION_TRANSFORM
ADD_BASE_CAPITALIZATION_TRANSFORM = _decomposition.ADD_BASE_CAPITALIZATION_TRANSFORM
ADD_ALL_CAPS_BASE_CAPITALIZATION_TRANSFORM = _decomposition.ADD_ALL_CAPS_BASE_CAPITALIZATION_TRANSFORM
REMOVE_BASE_CAPITALIZATION_TRANSFORM = _decomposition.REMOVE_BASE_CAPITALIZATION_TRANSFORM


class CompositionalTokenizerBundle:
    def __init__(self, tokenizer, model_init_data, tokenization_mode="dual_stream",
                 unified_modifier_array=None, dual_stream_tokenizer_config=None, sequence_map=None):
        """Initialize the tokenizer bundle.

        Args:
            tokenizer: The base tokenizer
            model_init_data: Dictionary containing model initialization data
            tokenization_mode: "dual_stream" (only supported mode in nanochat path)
            unified_modifier_array: UnifiedModifierArray instance (for 2D modifiers)
            dual_stream_tokenizer_config: Dict config for DualStreamTokenizer (optional)
            sequence_map: Pre-built SequenceMap instance (cached for performance)
        """
        self.tokenizer = tokenizer
        self.model_init_data = model_init_data
        self.tokenization_mode = tokenization_mode
        self.unified_modifier_array = unified_modifier_array
        self.dual_stream_tokenizer_config = dual_stream_tokenizer_config
        self.sequence_map = sequence_map

    def _build_manifest(self):
        base_vocab_size = len(self.model_init_data.get("non_inflection_indices", []))
        return {
            "format_version": 1,
            "base_tokenizer_size": self.model_init_data.get("base_tokenizer_size"),
            "base_vocab_size": base_vocab_size,
            "extended_vocab_size": len(self.tokenizer),
            "unified_modifier_dtype": self.model_init_data.get("unified_modifier_dtype"),
            "type_groups": self.model_init_data.get("type_groups", {}),
            "transformation_names_to_int": self.model_init_data.get("transformation_names_to_int", {}),
            "required_keys": [
                "base_tokenizer_size",
                "type_groups",
                "transformation_names_to_int",
                "token_id_to_base_id_mapping",
                "base_token_indices",
                "non_inflection_indices",
            ],
            "present_keys": sorted(self.model_init_data.keys()),
        }

    def save(self, save_path):
        os.makedirs(save_path, exist_ok=True)
        self.tokenizer.save_pretrained(os.path.join(save_path, "tokenizer"))

        # Save model_init_data with dual-stream metadata.
        bundle_data = {
            'format_version': 2,
            'model_init_data': self.model_init_data,
            'tokenization_mode': self.tokenization_mode,
            'unified_modifier_array': self.unified_modifier_array.to_dict() if self.unified_modifier_array else None,
            'dual_stream_tokenizer_config': self.dual_stream_tokenizer_config,
        }

        with open(os.path.join(save_path, "model_init_data.pkl"), "wb") as f:
            pickle.dump(bundle_data, f)
        with open(os.path.join(save_path, "model_init_manifest.json"), "w") as f:
            json.dump(self._build_manifest(), f, indent=2, sort_keys=True)

        # Save SequenceMap separately (can be large, so separate file)
        if self.sequence_map is not None:
            with open(os.path.join(save_path, "sequence_map.pkl"), "wb") as f:
                pickle.dump(self.sequence_map, f)
            print(f"Cached SequenceMap with {len(self.sequence_map)} sequences")

    @classmethod
    def load(cls, load_path, base_tokenizer_name=None):
        del base_tokenizer_name  # Fresh bundles persist complete tokenizer metadata.
        with open(os.path.join(load_path, "model_init_data.pkl"), "rb") as f:
            bundle_data = pickle.load(f)

        if not isinstance(bundle_data, dict) or 'model_init_data' not in bundle_data:
            raise ValueError(
                "Unsupported tokenizer bundle format. Rebuild the tokenizer bundle for fresh experiments."
            )

        tokenizer = AutoTokenizer.from_pretrained(os.path.join(load_path, "tokenizer"))
        print("Loaded tokenizer using AutoTokenizer (dual-stream mode)")

        model_init_data = bundle_data['model_init_data']
        tokenization_mode = bundle_data.get('tokenization_mode')
        if tokenization_mode != "dual_stream":
            raise ValueError(
                f"Unsupported tokenization_mode in bundle: {tokenization_mode!r}. "
                "CoBPE metadata construction supports modifier-bearing bundles only."
            )
        unified_modifier_array_dict = bundle_data.get('unified_modifier_array')
        unified_modifier_array = UnifiedModifierArray.from_dict(unified_modifier_array_dict) if unified_modifier_array_dict else None
        if unified_modifier_array is None:
            raise ValueError(
                "Unsupported dual_stream bundle without unified_modifier_array. "
                "Rebuild the tokenizer bundle for fresh experiments."
            )
        required_dtype = unified_modifier_array.recommended_modifier_dtype_name()
        stored_dtype = model_init_data.get('unified_modifier_dtype')
        if stored_dtype is None:
            model_init_data['unified_modifier_dtype'] = required_dtype
        else:
            try:
                stored_itemsize = np.dtype(stored_dtype).itemsize
                required_itemsize = np.dtype(required_dtype).itemsize
            except TypeError:
                stored_itemsize = 0
                required_itemsize = np.dtype(required_dtype).itemsize
            if stored_itemsize < required_itemsize:
                print(
                    "WARNING: bundle has undersized unified_modifier_dtype="
                    f"{stored_dtype!r}; upgrading to {required_dtype!r} to avoid modifier overflow."
                )
                model_init_data['unified_modifier_dtype'] = required_dtype

        dual_stream_tokenizer_config = bundle_data.get('dual_stream_tokenizer_config')
        if dual_stream_tokenizer_config is not None:
            cfg_dtype = dual_stream_tokenizer_config.get('modifier_dtype')
            final_dtype = model_init_data.get('unified_modifier_dtype')
            if cfg_dtype is None:
                dual_stream_tokenizer_config['modifier_dtype'] = final_dtype
            else:
                try:
                    cfg_itemsize = np.dtype(cfg_dtype).itemsize
                    final_itemsize = np.dtype(final_dtype).itemsize if final_dtype is not None else cfg_itemsize
                except TypeError:
                    cfg_itemsize = 0
                    final_itemsize = np.dtype(final_dtype).itemsize if final_dtype is not None else 0
                if final_dtype is not None and cfg_itemsize < final_itemsize:
                    dual_stream_tokenizer_config['modifier_dtype'] = final_dtype

        sequence_map = None
        sequence_map_path = os.path.join(load_path, "sequence_map.pkl")
        if os.path.exists(sequence_map_path):
            with open(sequence_map_path, "rb") as f:
                sequence_map = pickle.load(f)
            print(f"Loaded cached SequenceMap with {len(sequence_map)} sequences")

        print(f"Loaded bundle with tokenization_mode={tokenization_mode}")
        if unified_modifier_array:
            print(f"  UnifiedModifierArray: {unified_modifier_array}")

        return cls(tokenizer, model_init_data, tokenization_mode,
                   unified_modifier_array, dual_stream_tokenizer_config, sequence_map)


def _resolve_article_det_group_name(types_loss_indices_map):
    if DETERMINER_GROUP_NAME in types_loss_indices_map:
        return DETERMINER_GROUP_NAME
    if "article_det" in types_loss_indices_map:
        return "article_det"
    if "articles" in types_loss_indices_map:
        return "articles"
    return None


def _canonical_punct_surface(text: str) -> str:
    """Normalize punctuation surfaces, including common mojibake quote forms."""
    s = (text or "").strip()
    if not s:
        return ""
    replacements = {
        "âĢĻ": "’",
        "âĢĺ": "‘",
        "âĢľ": "“",
        "âĢĿ": "”",
        "âĢ¦": "…",
        "âĢ": "—",
        "âĢ": "–",
        "â€™": "’",
        "â€˜": "‘",
        "â€œ": "“",
        "â€": "”",
        "â€¦": "…",
        "â€”": "—",
        "â€“": "–",
    }
    for src, dst in replacements.items():
        s = s.replace(src, dst)
    return s


def _build_article_token_ids(
    tokenizer,
    include_possessive_determiners=False,
    include_demonstrative_determiners=False,
    include_quantifier_determiners=False,
):
    """Build set of token IDs that represent determiners in the determiner group."""
    article_words = {"the", "a", "an"}
    if include_possessive_determiners:
        article_words.update({"my", "your", "his", "her", "our", "their", "its"})
    if include_demonstrative_determiners:
        article_words.update({"this", "that", "these", "those"})
    if include_quantifier_determiners:
        article_words.update(
            {
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
            }
        )
    vocab = tokenizer.get_vocab()
    special_ids = set(getattr(tokenizer, "all_special_ids", []) or [])
    token_ids = set()
    for _token, token_id in vocab.items():
        token_id = int(token_id)
        if token_id in special_ids:
            continue
        decoded = tokenizer.decode(
            [token_id],
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        stripped = decoded.strip()
        if (
            stripped
            and stripped.isalpha()
            and stripped == stripped.lower()
            and stripped.lower() in article_words
        ):
            token_ids.add(token_id)
    return token_ids


def _build_prep_token_ids(tokenizer, preposition_list=None):
    """Build set of token IDs that represent prepositions."""
    if preposition_list is None:
        prepositions = list(DEFAULT_PREPOSITION_LIST)
    else:
        prepositions = normalize_preposition_list(preposition_list)

    prep_set = {p.lower() for p in prepositions}
    vocab = tokenizer.get_vocab()
    special_ids = set(getattr(tokenizer, "all_special_ids", []) or [])
    token_ids = set()
    for _token, token_id in vocab.items():
        token_id = int(token_id)
        if token_id in special_ids:
            continue
        decoded = tokenizer.decode(
            [token_id],
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        stripped = decoded.strip()
        if (
            stripped
            and stripped.isalpha()
            and stripped == stripped.lower()
            and stripped.lower() in prep_set
        ):
            token_ids.add(token_id)
    return token_ids


def _build_punct_token_ids(tokenizer, include_possessives=True):
    """Build sets of token IDs for prefix and suffix punctuation."""
    prefix_punct_base = ['"', "'", '`', '(', '[', '{', '-', '—', '\u2018', '\u2019', '\u201C', '\u201D']
    suffix_punct_base = ['"', "'", '`', ')', ']', '}', '.', '!', '?', ',', ';', ':', '-', '—', "\u2019"]
    suffix_punct_base.extend([
        ")'", ")\u2019", ")\"", ")\u201D",
        "]'", "]\u2019", "]\"", "]\u201D",
        ".'", ".\u2019", ".\"", ".\u201D",
        "'.", "\u2019.", "\".", "\u201D.",
        "!'", "!\u2019", "!\"", "!\u201D",
        "?'", "?\u2019", "?\"", "?\u201D",
        ",'", ",\u2019", ",\"", ",\u201D",
        "',", "\u2019,", "\",", "\u201D,",
        ".)", ").", "!)", "?)", ",)",
        ")-", "]-",
    ])
    if include_possessives:
        suffix_punct_base.extend(["'s", "s'", "\u2019s", "s\u2019"])
    prefix_punct_strings = set(prefix_punct_base)
    suffix_punct_strings = set(suffix_punct_base)

    vocab = tokenizer.get_vocab()
    special_ids = set(getattr(tokenizer, "all_special_ids", []) or [])
    prefix_ids = set()
    suffix_ids = set()

    for _token, token_id in vocab.items():
        token_id = int(token_id)
        if token_id in special_ids:
            continue
        decoded = tokenizer.decode(
            [token_id],
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        stripped = _canonical_punct_surface(decoded)
        if not stripped:
            continue
        if stripped in prefix_punct_strings:
            prefix_ids.add(token_id)
        if stripped in suffix_punct_strings:
            suffix_ids.add(token_id)

    return prefix_ids, suffix_ids


def _build_possessive_token_ids(tokenizer):
    """Build set of token IDs that represent English possessive suffixes."""
    possessive_strings = {"'s", "s'", "\u2019s", "s\u2019"}
    vocab = tokenizer.get_vocab()
    special_ids = set(getattr(tokenizer, "all_special_ids", []) or [])
    token_ids = set()
    for _token, token_id in vocab.items():
        token_id = int(token_id)
        if token_id in special_ids:
            continue
        decoded = tokenizer.decode(
            [token_id],
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        stripped = _canonical_punct_surface(decoded)
        if stripped in possessive_strings:
            token_ids.add(token_id)
    return token_ids


def _build_article_transform_map(tokenizer, article_token_ids, types_loss_indices_map, transformation_names_to_int):
    """Build mapping from determiner token_id to (group_name, relative_index)."""
    transform_map = {}

    group_name = _resolve_article_det_group_name(types_loss_indices_map)
    if group_name:
        start, end = types_loss_indices_map[group_name]
        determiner_to_transform = {
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

        for token_id in article_token_ids:
            decoded = tokenizer.decode(
                [int(token_id)],
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )
            clean_raw = decoded.strip()
            if clean_raw != clean_raw.lower():
                continue
            clean_str = clean_raw.lower()
            transform_name = determiner_to_transform.get(clean_str)
            if transform_name is None:
                continue
            idx = transformation_names_to_int.get(transform_name)
            if idx is None:
                continue
            rel_idx = idx - start
            if 0 <= rel_idx < (end - start):
                transform_map[token_id] = (group_name, rel_idx)

    return transform_map


def _build_preposition_transform_map(tokenizer, prep_token_ids, types_loss_indices_map, transformation_names_to_int, preposition_list=None):
    """Build mapping from preposition token_id to (group_name, relative_index)."""
    transform_map = {}

    prepositions = normalize_preposition_list(preposition_list) if preposition_list is not None else list(DEFAULT_PREPOSITION_LIST)
    preposition_set = {p.lower() for p in prepositions}

    if 'prepositions' in types_loss_indices_map:
        start, end = types_loss_indices_map['prepositions']
        for token_id in prep_token_ids:
            decoded = tokenizer.decode(
                [int(token_id)],
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )
            clean_raw = decoded.strip()
            if clean_raw != clean_raw.lower():
                continue
            clean_str = clean_raw.lower()
            if clean_str in preposition_set:
                transform_name = f"prep_{clean_str}"
                idx = transformation_names_to_int.get(transform_name)
                if idx is None:
                    continue
                rel_idx = idx - start
                if 0 <= rel_idx < (end - start):
                    transform_map[token_id] = ('prepositions', rel_idx)

    return transform_map


def _build_punct_transform_map(
    tokenizer,
    punct_token_ids,
    group_name,
    types_loss_indices_map,
    transformation_names_to_int,
    possessive_separate_group=False,
):
    """Build mapping from punctuation token_id to (group_name, relative_index)."""
    transform_map = {}

    if group_name in types_loss_indices_map:
        for token_id in punct_token_ids:
            clean_str = _canonical_punct_surface(tokenizer.decode(
                [int(token_id)],
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            ))
            target_group_name = group_name
            if group_name == 'prefix_punctuation':
                transform_name = {
                    '"': PREFIX_PUNCT_DOUBLE_QUOTE,
                    '\u201C': "punct_prefix_\u201c",
                    '\u201D': "punct_prefix_\u201d",
                    "'": PREFIX_PUNCT_SINGLE_QUOTE,
                    '\u2018': "punct_prefix_\u2018",
                    '\u2019': "punct_prefix_\u2019",
                    '`': PREFIX_PUNCT_BACKTICK,
                    '(': PREFIX_PUNCT_PAREN,
                    '[': PREFIX_PUNCT_SQUARE,
                    '{': PREFIX_PUNCT_CURLY,
                    '-': PREFIX_PUNCT_HYPHEN,
                    '—': "punct_prefix_—",
                    "'(": "punct_prefix_'(",
                    "\u2018(": "punct_prefix_\u2018(",
                    "\u2019(": "punct_prefix_\u2019(",
                    "\"(": "punct_prefix_\"(",
                    "\u201C(": "punct_prefix_\u201c(",
                    "\u201D(": "punct_prefix_\u201d(",
                    "'[": "punct_prefix_'[",
                    "\u2018[": "punct_prefix_\u2018[",
                    "\u2019[": "punct_prefix_\u2019[",
                    "\"[": "punct_prefix_\"[",
                    "\u201C[": "punct_prefix_\u201c[",
                    "\u201D[": "punct_prefix_\u201d[",
                    "('": "punct_prefix_('\"",
                    "(\u2018": "punct_prefix_(\u2018",
                    "(\u2019": "punct_prefix_(\u2019",
                    "-(": "punct_prefix_-(",
                    "-[": "punct_prefix_-[",
                }.get(clean_str)
            else:
                transform_name = {
                    '"': SUFFIX_PUNCT_DOUBLE_QUOTE,
                    "'": SUFFIX_PUNCT_SINGLE_QUOTE,
                    '\u2018': "punct_suffix_\u2018",
                    '\u2019': "punct_suffix_\u2019",
                    '\u201C': "punct_suffix_\u201c",
                    '\u201D': "punct_suffix_\u201d",
                    '`': SUFFIX_PUNCT_BACKTICK,
                    ')': SUFFIX_PUNCT_PAREN,
                    ']': SUFFIX_PUNCT_SQUARE,
                    '}': SUFFIX_PUNCT_CURLY,
                    '.': SUFFIX_PUNCT_PERIOD,
                    '!': SUFFIX_PUNCT_EXCLAIM,
                    '?': SUFFIX_PUNCT_QUESTION,
                    ',': SUFFIX_PUNCT_COMMA,
                    ';': SUFFIX_PUNCT_SEMICOLON,
                    ':': SUFFIX_PUNCT_COLON,
                    '-': SUFFIX_PUNCT_HYPHEN,
                    '—': "punct_suffix_—",
                    "'s": SUFFIX_PUNCT_POSSESSIVE_S,
                    "s'": SUFFIX_PUNCT_POSSESSIVE_PLURAL,
                    "\u2019s": "punct_suffix_\u2019s",
                    "s\u2019": "punct_suffix_s\u2019",
                    ")'": "punct_suffix_)'",
                    ")\u2019": "punct_suffix_)\u2019",
                    ")\"": "punct_suffix_)\"",
                    ")\u201D": "punct_suffix_)\u201d",
                    "]'": "punct_suffix_]'",
                    "]\u2019": "punct_suffix_]\u2019",
                    "]\"": "punct_suffix_]\"",
                    "]\u201D": "punct_suffix_]\u201d",
                    ".'": "punct_suffix_.'",
                    ".\u2019": "punct_suffix_.\u2019",
                    ".\"": "punct_suffix_.\"",
                    ".\u201D": "punct_suffix_.\u201d",
                    "!'": "punct_suffix_!'",
                    "!\u2019": "punct_suffix_!\u2019",
                    "!\"": "punct_suffix_!\"",
                    "!\u201D": "punct_suffix_!\u201d",
                    "?'": "punct_suffix_?'",
                    "?\u2019": "punct_suffix_?\u2019",
                    "?\"": "punct_suffix_?\"",
                    "?\u201D": "punct_suffix_?\u201d",
                    ",'": "punct_suffix_,'",
                    ",\u2019": "punct_suffix_,\u2019",
                    ",\"": "punct_suffix_,\"",
                    ",\u201D": "punct_suffix_,\u201d",
                    "',": "punct_suffix_',",
                    "\u2019,": "punct_suffix_\u2019,",
                    "\",": "punct_suffix_\",",
                    "\u201D,": "punct_suffix_\u201d,",
                    "'.": "punct_suffix_'.",
                    "\u2019.": "punct_suffix_\u2019.",
                    "\".": "punct_suffix_\".",
                    "\u201D.": "punct_suffix_\u201d.",
                    ".)": "punct_suffix_.)",
                    ").": "punct_suffix_).",
                    "!)": "punct_suffix_!)",
                    "?)": "punct_suffix_?)",
                    ",)": "punct_suffix_,)",
                    ")-": "punct_suffix_)-",
                    "]-": "punct_suffix_]-",
                }.get(clean_str)
                if possessive_separate_group and clean_str in {"'s", "s'", "\u2019s", "s\u2019"}:
                    target_group_name = "possessives"
                    transform_name = {
                        "'s": POSSESSIVE_S,
                        "\u2019s": "possessive_\u2019s",
                        "s'": POSSESSIVE_PLURAL,
                        "s\u2019": "possessive_s\u2019",
                    }.get(clean_str)
            if not transform_name:
                continue
            if target_group_name not in types_loss_indices_map:
                continue
            start, end = types_loss_indices_map[target_group_name]
            idx = transformation_names_to_int.get(transform_name)
            if idx is None:
                continue
            rel_idx = idx - start
            if 0 <= rel_idx < (end - start):
                transform_map[token_id] = (target_group_name, rel_idx)

    return transform_map


def _add_whitespace_bridge_sequences_from_vocab(
    *,
    tokenizer,
    sequence_map,
    transformation_names_to_int,
    types_loss_indices_map,
    max_tokens_per_word=None,
):
    """Add [space, token] -> token sequences for whitespace-isolating tokenizers.

    This preserves efficient space-prefix handling when the base tokenizer no longer
    has single tokens that decode to " <word>".
    """
    if "space_prefix" not in types_loss_indices_map:
        return {"enabled": False, "added": 0, "skipped_existing": 0, "reason": "no_space_group"}

    with_space_global_idx = transformation_names_to_int.get(WITH_SPACE_PREFIX_TRANSFORM)
    if with_space_global_idx is None:
        return {"enabled": False, "added": 0, "skipped_existing": 0, "reason": "no_with_space_transform"}

    start, end = types_loss_indices_map["space_prefix"]
    with_space_rel_idx = with_space_global_idx - start
    if with_space_rel_idx < 0 or with_space_rel_idx >= (end - start):
        return {"enabled": False, "added": 0, "skipped_existing": 0, "reason": "invalid_with_space_index"}

    # Detect whether whitespace is already fused into leading-space tokens.
    # If " dog" is single-token, we don't need bridging entries.
    probe = tuple(tokenizer.encode(" dog", add_special_tokens=False))
    if len(probe) <= 1:
        return {"enabled": False, "added": 0, "skipped_existing": 0, "reason": "leading_space_tokens_present"}

    space_ids = tuple(tokenizer.encode(" ", add_special_tokens=False))
    if len(space_ids) != 1:
        return {"enabled": False, "added": 0, "skipped_existing": 0, "reason": "no_single_space_token"}
    if max_tokens_per_word is not None and 2 > max_tokens_per_word:
        return {"enabled": False, "added": 0, "skipped_existing": 0, "reason": "max_tokens_per_word_lt_2"}
    space_id = int(space_ids[0])

    special_ids = set(getattr(tokenizer, "all_special_ids", []) or [])
    vocab_items = sorted(tokenizer.get_vocab().items(), key=lambda kv: kv[1])

    added = 0
    skipped_existing = 0
    for _token_str, token_id in vocab_items:
        token_id = int(token_id)
        if token_id in special_ids:
            continue
        decoded = tokenizer.decode(
            [token_id],
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        if not decoded or all(ch.isspace() for ch in decoded):
            continue

        variant_ids = (space_id, token_id)
        if sequence_map.get_exact_match(variant_ids) is not None:
            skipped_existing += 1
            continue

        modifier = sequence_map.modifier_manager.create_empty_modifier()
        sequence_map.modifier_manager.set_group_value(modifier, "space_prefix", with_space_rel_idx)
        sequence_map.add_sequence(variant_ids, (token_id,), modifier)
        added += 1

    return {
        "enabled": True,
        "added": added,
        "skipped_existing": skipped_existing,
        "reason": "ok",
    }


def _add_pure_surface_merge_sequences_from_vocab(
    *,
    tokenizer,
    sequence_map,
    max_tokens_per_word=None,
):
    """Add merge-only SequenceMap entries for pure whitespace/punctuation surfaces.

    For a token T with decoded surface S that is either:
      - whitespace-only, or
      - punctuation/symbol-only (no letters/digits/whitespace),
    if encode(S) produces multiple token IDs, we add:
      encode(S) -> (T) with an empty modifier.
    """

    special_ids = set(getattr(tokenizer, "all_special_ids", []) or [])
    vocab_items = sorted(tokenizer.get_vocab().items(), key=lambda kv: kv[1])

    added_whitespace = 0
    added_punct = 0
    skipped_existing = 0
    skipped_single = 0
    skipped_max_len = 0
    skipped_non_pure = 0

    for _token_str, token_id in vocab_items:
        token_id = int(token_id)
        if token_id in special_ids:
            continue

        decoded = tokenizer.decode(
            [token_id],
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        if not decoded:
            continue

        is_whitespace_only = all(ch.isspace() for ch in decoded)
        is_punct_only = all((not ch.isspace()) and (not ch.isalnum()) for ch in decoded)
        if not is_whitespace_only and not is_punct_only:
            skipped_non_pure += 1
            continue

        variant_ids = tuple(tokenizer.encode(decoded, add_special_tokens=False))
        if len(variant_ids) <= 1:
            skipped_single += 1
            continue
        if max_tokens_per_word is not None and len(variant_ids) > max_tokens_per_word:
            skipped_max_len += 1
            continue
        if sequence_map.get_exact_match(variant_ids) is not None:
            skipped_existing += 1
            continue

        modifier = sequence_map.modifier_manager.create_empty_modifier()
        sequence_map.add_sequence(variant_ids, (token_id,), modifier)
        if is_whitespace_only:
            added_whitespace += 1
        else:
            added_punct += 1

    return {
        "added_whitespace": added_whitespace,
        "added_punct": added_punct,
        "skipped_existing": skipped_existing,
        "skipped_single": skipped_single,
        "skipped_max_len": skipped_max_len,
        "skipped_non_pure": skipped_non_pure,
    }


def _add_single_char_capitalization_sequences_from_vocab(
    *,
    tokenizer,
    sequence_map,
    transformation_names_to_int,
    types_loss_indices_map,
    max_tokens_per_word=None,
):
    """Add direct single-letter uppercase aliases: U -> u + add_capitalization."""
    if "base_capitalization" not in types_loss_indices_map:
        return {
            "enabled": False,
            "added": 0,
            "skipped_existing": 0,
            "overwritten_existing": 0,
            "reason": "no_base_cap_group",
        }
    add_cap_global_idx = transformation_names_to_int.get(ADD_BASE_CAPITALIZATION_TRANSFORM)
    if add_cap_global_idx is None:
        return {
            "enabled": False,
            "added": 0,
            "skipped_existing": 0,
            "overwritten_existing": 0,
            "reason": "no_add_cap_transform",
        }

    start, end = types_loss_indices_map["base_capitalization"]
    add_cap_rel_idx = add_cap_global_idx - start
    if add_cap_rel_idx < 0 or add_cap_rel_idx >= (end - start):
        return {
            "enabled": False,
            "added": 0,
            "skipped_existing": 0,
            "overwritten_existing": 0,
            "reason": "invalid_add_cap_index",
        }

    special_ids = set(getattr(tokenizer, "all_special_ids", []) or [])
    vocab_items = sorted(tokenizer.get_vocab().items(), key=lambda kv: kv[1])
    base_cap_group_idx = sequence_map.modifier_manager.group_to_idx.get("base_capitalization", -1)
    added = 0
    skipped_existing = 0
    overwritten_existing = 0

    for _token_str, token_id in vocab_items:
        token_id = int(token_id)
        if token_id in special_ids:
            continue
        surface = tokenizer.decode(
            [token_id],
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        if not surface:
            continue
        alpha_positions = [idx for idx, ch in enumerate(surface) if ch.isalpha()]
        if len(alpha_positions) != 1:
            continue
        alpha_idx = alpha_positions[0]
        alpha_ch = surface[alpha_idx]
        if not alpha_ch.isupper():
            continue

        lowered_chars = list(surface)
        lowered_chars[alpha_idx] = alpha_ch.lower()
        lower_surface = "".join(lowered_chars)

        variant_ids = tuple(tokenizer.encode(surface, add_special_tokens=False))
        if len(variant_ids) == 0:
            continue
        if max_tokens_per_word is not None and len(variant_ids) > max_tokens_per_word:
            continue

        base_ids = tuple(tokenizer.encode(lower_surface, add_special_tokens=False))
        if len(base_ids) == 0:
            continue
        if max_tokens_per_word is not None and len(base_ids) > max_tokens_per_word:
            continue

        existing = sequence_map.get_exact_match(variant_ids)
        if existing is not None:
            existing_base_ids, existing_modifier = existing
            existing_cap_val = (
                int(existing_modifier[base_cap_group_idx])
                if 0 <= base_cap_group_idx < len(existing_modifier)
                else 0
            )
            if tuple(int(x) for x in existing_base_ids) == base_ids and existing_cap_val == int(add_cap_rel_idx):
                skipped_existing += 1
                continue
            overwritten_existing += 1

        modifier = sequence_map.modifier_manager.create_empty_modifier()
        sequence_map.modifier_manager.set_group_value(
            modifier,
            "base_capitalization",
            add_cap_rel_idx,
        )
        sequence_map.add_sequence(variant_ids, base_ids, modifier)
        added += 1

    return {
        "enabled": True,
        "added": added,
        "skipped_existing": skipped_existing,
        "overwritten_existing": overwritten_existing,
        "reason": "ok",
    }


def _build_sequence_map_for_bundle(tokenizer, model_init_data, unified_modifier_array, args):
    """Build SequenceMap during bundle creation.

    This is a one-time operation that happens when the bundle is first created.
    The SequenceMap is then cached and reused for all tokenization runs.
    """
    print("  Building SequenceMap for bundle (one-time cost)...")
    use_build_optimizations = bool(getattr(args, "enable_build_optimizations", False))
    max_tokens_per_word = getattr(args, "max_tokens_per_base_word", None)

    decompose_possessive_determiners = bool(
        getattr(args, "decompose_possessive_determiners", False) and getattr(args, "decompose_articles", False)
    )
    decompose_demonstrative_determiners = bool(
        getattr(args, "decompose_demonstrative_determiners", False) and getattr(args, "decompose_articles", False)
    )
    decompose_quantifier_determiners = bool(
        getattr(args, "decompose_quantifier_determiners", False) and getattr(args, "decompose_articles", False)
    )
    decompose_determiners = bool(args.decompose_articles)
    decompose_possessive_separate_group = bool(
        getattr(args, "decompose_punctuation", False) and getattr(args, "possessive_separate_group", False)
    )

    # Build token ID sets
    article_token_ids = _build_article_token_ids(
        tokenizer,
        include_possessive_determiners=decompose_possessive_determiners,
        include_demonstrative_determiners=decompose_demonstrative_determiners,
        include_quantifier_determiners=decompose_quantifier_determiners,
    ) if decompose_determiners else set()
    prep_token_ids = _build_prep_token_ids(tokenizer, args.preposition_list) if args.decompose_prepositions else set()
    if args.decompose_punctuation:
        prefix_punct_token_ids, suffix_punct_token_ids = _build_punct_token_ids(
            tokenizer,
            include_possessives=not decompose_possessive_separate_group,
        )
        if decompose_possessive_separate_group:
            suffix_punct_token_ids |= _build_possessive_token_ids(tokenizer)
    else:
        prefix_punct_token_ids, suffix_punct_token_ids = set(), set()

    # Get types_loss_indices_map - required for building SequenceMap
    types_loss_indices_map = model_init_data.get('types_loss_indices_map')
    if not types_loss_indices_map:
        raise ValueError(
            "types_loss_indices_map is missing from model_init_data! "
            "This is required for building SequenceMap with correct modifier indices. "
            "The bundle cache may be corrupted or from an old version. "
            "Try rebuilding with --overwrite_bundle_cache."
        )

    decomposition_map = model_init_data.get('decomposition_map', {})
    syntactic_decomposition_map = model_init_data.get('syntactic_decomposition_map', {})
    final_decomposition_map = model_init_data.get('final_decomposition_map', {})
    transformation_names_to_int = model_init_data.get('transformation_names_to_int', {})
    base_tokens = model_init_data.get('base_tokens')

    # Build transform mappings
    article_transforms = _build_article_transform_map(
        tokenizer,
        article_token_ids,
        types_loss_indices_map,
        transformation_names_to_int,
    ) if decompose_determiners and article_token_ids else {}
    preposition_transforms = _build_preposition_transform_map(
        tokenizer,
        prep_token_ids,
        types_loss_indices_map,
        transformation_names_to_int,
        preposition_list=args.preposition_list
    ) if args.decompose_prepositions and prep_token_ids else {}
    prefix_punct_transforms = _build_punct_transform_map(
        tokenizer,
        prefix_punct_token_ids,
        'prefix_punctuation',
        types_loss_indices_map,
        transformation_names_to_int,
        possessive_separate_group=decompose_possessive_separate_group,
    ) if args.decompose_punctuation and prefix_punct_token_ids else {}
    suffix_punct_transforms = _build_punct_transform_map(
        tokenizer,
        suffix_punct_token_ids,
        'suffix_punctuation',
        types_loss_indices_map,
        transformation_names_to_int,
        possessive_separate_group=decompose_possessive_separate_group,
    ) if args.decompose_punctuation and suffix_punct_token_ids else {}
    print(
        "  Prefix token coverage: "
        f"articles ids={len(article_token_ids):,} maps={len(article_transforms):,}, "
        f"prepositions ids={len(prep_token_ids):,} maps={len(preposition_transforms):,}, "
        f"prefix_punct ids={len(prefix_punct_token_ids):,} maps={len(prefix_punct_transforms):,}, "
        f"suffix_punct ids={len(suffix_punct_token_ids):,} maps={len(suffix_punct_transforms):,}"
        + (
            f", possessives maps={sum(1 for g, _ in suffix_punct_transforms.values() if g == 'possessives'):,}"
            if decompose_possessive_separate_group
            else ""
        )
    )

    # Build sequence map from token-id decomposition for reliable modifier indices
    sequence_map = build_sequence_map_from_decomposition(
        base_tokenizer=tokenizer,
        decomposition_map=final_decomposition_map,
        transformation_names_to_int=transformation_names_to_int,
        types_loss_indices_map=types_loss_indices_map,
        article_token_ids=article_token_ids if decompose_determiners else None,
        preposition_token_ids=prep_token_ids if args.decompose_prepositions else None,
        prefix_punct_token_ids=prefix_punct_token_ids if args.decompose_punctuation else None,
        suffix_punct_token_ids=suffix_punct_token_ids if args.decompose_punctuation else None,
        article_transforms=article_transforms,
        preposition_transforms=preposition_transforms,
        prefix_punct_transforms=prefix_punct_transforms,
        suffix_punct_transforms=suffix_punct_transforms,
        groups=unified_modifier_array.groups,
    )

    # Extend with string-based decomposition for multi-token bases/variants.
    # These should override any single-token mappings when the base is multi-token.
    build_sequence_map_from_string_decomposition(
        base_tokenizer=tokenizer,
        decomposition_map=decomposition_map,
        transformation_names_to_int=transformation_names_to_int,
        types_loss_indices_map=types_loss_indices_map,
        groups=unified_modifier_array.groups,
        space_prefix=" ",
        max_tokens_per_word=max_tokens_per_word,
        sequence_map=sequence_map,
        only_multi_token_base=True,
        base_tokens=base_tokens,
        enable_build_optimizations=use_build_optimizations,
    )
    if syntactic_decomposition_map:
        build_sequence_map_from_string_decomposition(
            base_tokenizer=tokenizer,
            decomposition_map=syntactic_decomposition_map,
            transformation_names_to_int=transformation_names_to_int,
            types_loss_indices_map=types_loss_indices_map,
            groups=unified_modifier_array.groups,
            space_prefix=" ",
            max_tokens_per_word=max_tokens_per_word,
            sequence_map=sequence_map,
            only_multi_token_base=False,
            base_tokens=base_tokens,
            overwrite_existing=False,
            enable_build_optimizations=use_build_optimizations,
        )
    decompose_spaces = not getattr(args, "dont_decompose_spaces", False)
    decompose_caps = not getattr(args, "dont_decompose_capitalized", False)
    if decompose_spaces or decompose_caps:
        syntactic_stats = add_space_cap_syntactic_sequences_from_vocab(
            base_tokenizer=tokenizer,
            sequence_map=sequence_map,
            decomposition_map=decomposition_map,
            base_tokens=base_tokens,
            transformation_names_to_int=transformation_names_to_int,
            types_loss_indices_map=types_loss_indices_map,
            groups=unified_modifier_array.groups,
            decompose_spaces=decompose_spaces,
            decompose_caps=decompose_caps,
            enable_all_caps_capitalization=bool(
                getattr(args, "enable_all_caps_capitalization", False)
            ),
            use_relative_space_cap_transforms=bool(
                getattr(args, "use_relative_space_cap_transforms", False)
            ),
            max_tokens_per_word=max_tokens_per_word,
            overwrite_existing=False,
            enable_build_optimizations=use_build_optimizations,
            promote_surface_alias_bases=bool(getattr(args, "promote_surface_alias_bases", False)),
        )
        print(
            "  Added global space/cap sequences directly to SequenceMap "
            f"(bases={syntactic_stats['bases']:,}, added={syntactic_stats['added']:,})"
        )
    modifier_generation_mode = _resolve_modifier_generation_mode(args).lower()
    if modifier_generation_mode == "full":
        print(
            "  NOTE: dual-stream modifier handling is SequenceMap-driven; "
            "using vocab-driven modifier candidates for punctuation/articles/prepositions."
        )
    modifier_stats = add_modifier_syntactic_sequences_from_vocab(
        base_tokenizer=tokenizer,
        sequence_map=sequence_map,
        decomposition_map=decomposition_map,
        transformation_names_to_int=transformation_names_to_int,
        types_loss_indices_map=types_loss_indices_map,
        groups=unified_modifier_array.groups,
        decompose_articles=decompose_determiners,
        decompose_possessive_determiners=decompose_possessive_determiners,
        decompose_demonstrative_determiners=decompose_demonstrative_determiners,
        decompose_quantifier_determiners=decompose_quantifier_determiners,
        decompose_prepositions=bool(args.decompose_prepositions),
        decompose_punctuation=bool(args.decompose_punctuation),
        decompose_possessive_separate_group=decompose_possessive_separate_group,
        preposition_list=args.preposition_list,
        decompose_article_prep_space_prefix=bool(
            getattr(args, "decompose_article_prep_space_prefix", False)
        ),
        max_tokens_per_word=max_tokens_per_word,
        overwrite_existing=False,
        enable_build_optimizations=use_build_optimizations,
    )
    if decompose_determiners or args.decompose_prepositions or args.decompose_punctuation:
        print(
            "  Added modifier candidates directly to SequenceMap "
            f"(added={modifier_stats['added']:,}, scanned={modifier_stats['scanned']:,}, "
            f"matched_bases={modifier_stats['matched_base']:,}, "
            f"skipped_existing={modifier_stats['skipped_existing']:,}, "
            f"ambiguous_base_variants={modifier_stats['ambiguous_base_variants']:,})"
        )

    whitespace_bridge_stats = _add_whitespace_bridge_sequences_from_vocab(
        tokenizer=tokenizer,
        sequence_map=sequence_map,
        transformation_names_to_int=transformation_names_to_int,
        types_loss_indices_map=types_loss_indices_map,
        max_tokens_per_word=max_tokens_per_word,
    )
    if whitespace_bridge_stats["enabled"]:
        print(
            "  Added whitespace bridge sequences "
            f"(added={whitespace_bridge_stats['added']:,}, "
            f"skipped_existing={whitespace_bridge_stats['skipped_existing']:,})"
        )
    else:
        print(f"  Skipped whitespace bridge sequences ({whitespace_bridge_stats['reason']})")

    pure_surface_merge_stats = _add_pure_surface_merge_sequences_from_vocab(
        tokenizer=tokenizer,
        sequence_map=sequence_map,
        max_tokens_per_word=max_tokens_per_word,
    )
    print(
        "  Added pure-surface merge sequences "
        f"(whitespace={pure_surface_merge_stats['added_whitespace']:,}, "
        f"punct={pure_surface_merge_stats['added_punct']:,}, "
        f"skipped_existing={pure_surface_merge_stats['skipped_existing']:,}, "
        f"skipped_single={pure_surface_merge_stats['skipped_single']:,}, "
        f"skipped_max_len={pure_surface_merge_stats['skipped_max_len']:,})"
    )

    # Add article/preposition/punctuation sequences
    if article_token_ids and article_transforms:
        for token_id, (group_name, rel_idx) in article_transforms.items():
            modifier = sequence_map.modifier_manager.create_empty_modifier()
            sequence_map.modifier_manager.set_group_value(modifier, group_name, rel_idx)
            sequence_map.add_sequence((token_id,), (token_id,), modifier)
    if prep_token_ids and preposition_transforms:
        for token_id, (group_name, rel_idx) in preposition_transforms.items():
            modifier = sequence_map.modifier_manager.create_empty_modifier()
            sequence_map.modifier_manager.set_group_value(modifier, group_name, rel_idx)
            sequence_map.add_sequence((token_id,), (token_id,), modifier)
    if prefix_punct_token_ids and prefix_punct_transforms:
        for token_id, (group_name, rel_idx) in prefix_punct_transforms.items():
            modifier = sequence_map.modifier_manager.create_empty_modifier()
            sequence_map.modifier_manager.set_group_value(modifier, group_name, rel_idx)
            sequence_map.add_sequence((token_id,), (token_id,), modifier)
    if suffix_punct_token_ids and suffix_punct_transforms:
        for token_id, (group_name, rel_idx) in suffix_punct_transforms.items():
            modifier = sequence_map.modifier_manager.create_empty_modifier()
            sequence_map.modifier_manager.set_group_value(modifier, group_name, rel_idx)
            sequence_map.add_sequence((token_id,), (token_id,), modifier)

    # Add single-letter uppercase aliases last so they win over generic identity
    # mappings (including prior article/preposition marker entries).
    single_char_cap_stats = _add_single_char_capitalization_sequences_from_vocab(
        tokenizer=tokenizer,
        sequence_map=sequence_map,
        transformation_names_to_int=transformation_names_to_int,
        types_loss_indices_map=types_loss_indices_map,
        max_tokens_per_word=max_tokens_per_word,
    )
    if single_char_cap_stats["enabled"]:
        print(
            "  Added single-char uppercase cap aliases "
            f"(added={single_char_cap_stats['added']:,}, "
            f"overwritten_existing={single_char_cap_stats['overwritten_existing']:,}, "
            f"skipped_existing={single_char_cap_stats['skipped_existing']:,})"
        )
    else:
        print(f"  Skipped single-char uppercase cap aliases ({single_char_cap_stats['reason']})")

    def _count_sequences_with_multitoken_base(seq_map):
        count = 0
        stack = [seq_map.root]
        while stack:
            node = stack.pop()
            if node.is_end and node.base_token_ids and len(node.base_token_ids) > 1:
                count += 1
            stack.extend(node.children.values())
        return count

    total_sequences = len(sequence_map)
    multi_base_sequences = _count_sequences_with_multitoken_base(sequence_map)
    skipped_disallowed = int(getattr(sequence_map, "skipped_disallowed_sequences", 0))
    disallowed_count = len(getattr(sequence_map, "disallowed_match_token_ids", set()) or set())
    if skipped_disallowed > 0:
        print(
            "    SequenceMap pruned disallowed whitespace-bearing sequences: "
            f"{skipped_disallowed:,} (token_ids with non-space whitespace={disallowed_count:,})"
        )
    print(f"    SequenceMap built with {total_sequences} total sequences")
    print(f"    SequenceMap sequences with len(base_ids)>1: {multi_base_sequences}")

    if bool(getattr(args, "verify_sequence_map_coverage", False)):
        # Coverage check: single-token variants whose base is multi-token should be mapped by SequenceMap.
        max_tokens_per_word = 5
        encoded_text_cache = {}

        def _encode_cached(text):
            if not use_build_optimizations:
                return tuple(tokenizer.encode(text, add_special_tokens=False))
            cached = encoded_text_cache.get(text)
            if cached is not None:
                return cached
            encoded = tuple(tokenizer.encode(text, add_special_tokens=False))
            encoded_text_cache[text] = encoded
            return encoded

        if decomposition_map:
            expected = []
            for base_word, variants in decomposition_map.items():
                base_ids = _encode_cached(base_word)
                if len(base_ids) <= 1 or len(base_ids) > max_tokens_per_word:
                    continue
                for variant_str, transforms in variants.items():
                    if variant_str == base_word:
                        continue
                    variant_ids = _encode_cached(variant_str)
                    if len(variant_ids) != 1 or len(variant_ids) > max_tokens_per_word:
                        continue
                    if base_tokens:
                        normalized_variant = variant_str.lstrip(" ")
                        candidate_key = f" {normalized_variant}" if normalized_variant else " "
                        candidate_base = base_tokens.get(candidate_key) or base_tokens.get(normalized_variant)
                        if isinstance(candidate_base, int):
                            continue
                    expected.append((base_word, variant_str, variant_ids[0], base_ids))

            if expected:
                covered = 0
                missing = []
                for base_word, variant_str, variant_id, base_ids in expected:
                    match_len, found_base_ids, _ = sequence_map.find_longest_match([variant_id], 0)
                    if match_len == 1 and found_base_ids == base_ids:
                        covered += 1
                    else:
                        if len(missing) < 5:
                            missing.append((base_word, variant_str, found_base_ids))
                if missing:
                    print(
                        f"SequenceMap coverage for single-token variants with multi-token bases: "
                        f"{covered}/{len(expected)} (missing {len(expected) - covered})"
                    )
                    print(f"  missing examples (base -> variant -> found_base_ids): {missing}")
                else:
                    print(
                        f"SequenceMap coverage for single-token variants with multi-token bases: "
                        f"{covered}/{len(expected)}"
                    )

    return sequence_map


def _resolve_modifier_generation_mode(args):
    mode = getattr(args, "modifier_generation_mode", None)
    if mode:
        mode = mode.lower()
        if mode != "auto":
            return mode
    tokenization_mode = getattr(args, "tokenization_mode", "dual_stream")
    if tokenization_mode == "dual_stream":
        return "vocab_only"
    return "full"


def _build_transform_to_group_idx(transformation_names_to_int, types_loss_indices_map, groups):
    int_to_name = {v: k for k, v in transformation_names_to_int.items()}
    transform_to_group_idx = {}
    for group_name in groups:
        if group_name not in types_loss_indices_map:
            continue
        start, end = types_loss_indices_map[group_name]
        for i in range(start, end):
            name = int_to_name.get(i)
            if name is not None:
                transform_to_group_idx[name] = (group_name, i - start)
    return transform_to_group_idx


def _space_cap_is_active_transform(transforms):
    return any(
        t in {
            WITH_SPACE_PREFIX_TRANSFORM,
            REMOVE_SPACE_PREFIX_TRANSFORM,
            ADD_BASE_CAPITALIZATION_TRANSFORM,
            ADD_ALL_CAPS_BASE_CAPITALIZATION_TRANSFORM,
            REMOVE_BASE_CAPITALIZATION_TRANSFORM,
        }
        for t in transforms
    )


def _canonical_space_cap_base(token_surface, decompose_caps):
    if token_surface is None:
        return None
    token_without_space = token_surface.lstrip(" ")
    if not token_without_space:
        return None

    if not decompose_caps:
        return token_without_space

    alphabetic_chars = [c for c in token_without_space if c.isalpha()]
    if alphabetic_chars and all(c.isupper() for c in alphabetic_chars):
        return token_without_space
    return token_without_space.lower()


def _build_space_cap_syntactic_decomposition_from_vocab(base_tokenizer, decomposition_map, args, base_tokens=None):
    """Build global space/capitalization variants separate from morphology map."""
    decompose_spaces = not getattr(args, "dont_decompose_spaces", False)
    decompose_caps = not getattr(args, "dont_decompose_capitalized", False)
    if not (decompose_spaces or decompose_caps):
        return {}

    use_relative = bool(getattr(args, "use_relative_space_cap_transforms", False))
    enable_all_caps_capitalization = bool(getattr(args, "enable_all_caps_capitalization", False))
    max_variant_tokens = getattr(args, "max_tokens_per_base_word", None)
    use_build_optimizations = bool(getattr(args, "enable_build_optimizations", False))
    promote_surface_alias_bases = bool(getattr(args, "promote_surface_alias_bases", False))

    syntactic_map = {}
    for (
        base_key,
        _base_ids,
        _base_rank,
        variant_str,
        transforms,
        _variant_ids,
    ) in _iter_space_cap_syntactic_candidates(
        base_tokenizer=base_tokenizer,
        decomposition_map=decomposition_map,
        decompose_spaces=decompose_spaces,
        decompose_caps=decompose_caps,
        use_relative=use_relative,
        enable_all_caps_capitalization=enable_all_caps_capitalization,
        base_tokens=base_tokens,
        max_variant_tokens=max_variant_tokens,
        enable_build_optimizations=use_build_optimizations,
        promote_surface_alias_bases=promote_surface_alias_bases,
    ):
        base_entry = syntactic_map.setdefault(base_key, {})
        base_entry[variant_str] = transforms
    return syntactic_map


def _build_vocab_only_decomposition_map(
    base_tokenizer,
    args,
    *,
    decompose_articles,
    decompose_possessive_determiners,
    decompose_demonstrative_determiners,
    decompose_quantifier_determiners,
    decompose_prepositions,
    decompose_punctuation,
    decompose_possessive_separate_group,
    preposition_list=None,
):
    """Build a minimal vocab-only decomposition map without UniMorph.

    This is used when drop_morphology_transforms is enabled: we still need
    canonical base entries so SequenceMap can add global space/cap and
    article/preposition/punctuation modifiers, but we do not need inflection or
    derivation discovery from external morphology resources.
    """
    decompose_caps = not getattr(args, "dont_decompose_capitalized", False)
    special_tokens = set(getattr(base_tokenizer, "all_special_tokens", []) or [])
    vocab_items = sorted(base_tokenizer.get_vocab().items(), key=lambda kv: kv[1])

    identity_transform_cache = {}
    base_tokens = {}
    decomposition_map = {}

    def _identity_transforms(base_key):
        cached = identity_transform_cache.get(base_key)
        if cached is not None:
            return list(cached)
        variants = get_variation_transformations(
            base_key,
            None,
            None,
            space_prefix=" ",
            decompose_spaces=False,
            decompose_capitalization=False,
            use_relative_space_cap_transforms=bool(getattr(args, "use_relative_space_cap_transforms", False)),
            article_space_prefix=None,
            prep_space_prefix=None,
        )
        transforms = list(variants.get(base_key, []))
        identity_transform_cache[base_key] = tuple(transforms)
        return list(transforms)

    base_count = 0
    for token, _token_id in vocab_items:
        if token in special_tokens:
            continue
        token_surface = base_tokenizer.convert_tokens_to_string([token])
        if not token_surface:
            continue

        base_text, _ = strip_all_affixes(
            token_surface,
            space_prefix=" ",
            decompose_punctuation=decompose_punctuation,
            decompose_articles=decompose_articles,
            decompose_prepositions=decompose_prepositions,
            prepositions=preposition_list,
            track_article_prep_space_prefix=False,
            decompose_possessive_determiners=decompose_possessive_determiners,
            decompose_demonstrative_determiners=decompose_demonstrative_determiners,
            decompose_quantifier_determiners=decompose_quantifier_determiners,
            decompose_possessive_suffixes=decompose_possessive_separate_group,
        )
        if not base_text:
            continue

        base_key = _canonical_space_cap_base(base_text, decompose_caps)
        if not base_key:
            continue

        if base_key not in base_tokens:
            base_ids = tuple(base_tokenizer.encode(base_key, add_special_tokens=False))
            if not base_ids:
                continue
            base_tokens[base_key] = base_ids[0] if len(base_ids) == 1 else base_ids
            base_count += 1

        base_entry = decomposition_map.setdefault(base_key, {})
        if base_key not in base_entry:
            base_entry[base_key] = _identity_transforms(base_key)

    print(
        "Built vocab-only decomposition map without UniMorph "
        f"({base_count:,} canonical bases)"
    )
    duplicates = {
        "duplicates": [],
        "transform_collisions": [],
        "derivation_decision_stats": {},
        "derivation_decision_samples": {},
        "invalid_derivation_rows_count": 0,
        "invalid_derivation_rows_samples": [],
    }
    return base_tokens, decomposition_map, {}, [], [], duplicates, {}


def _iter_space_cap_syntactic_candidates(
    base_tokenizer,
    decomposition_map,
    decompose_spaces,
    decompose_caps,
    use_relative,
    enable_all_caps_capitalization=False,
    base_tokens=None,
    max_variant_tokens=None,
    enable_build_optimizations=False,
    promote_surface_alias_bases=False,
):
    """Yield deduplicated global space/cap candidates from vocab + morph surfaces.

    Yields tuples:
      (base_key, base_ids, base_rank, variant_str, transforms, variant_ids)
    """
    if not (decompose_spaces or decompose_caps):
        return

    use_build_optimizations = bool(enable_build_optimizations)

    existing_morph_variants = set()
    morph_variant_owner = {}
    for base_word, variants in decomposition_map.items():
        existing_morph_variants.update(variants.keys())
        for variant_str, transforms in variants.items():
            if variant_str not in morph_variant_owner:
                morph_variant_owner[variant_str] = (base_word, list(transforms))

    special_tokens = set(getattr(base_tokenizer, "all_special_tokens", []) or [])
    vocab_items = sorted(base_tokenizer.get_vocab().items(), key=lambda kv: kv[1])
    encoded_cache = {}
    seen_base_keys = set()
    variant_to_owner = {}
    conflicts = 0
    yielded = 0

    def _encode_cached(text):
        if not use_build_optimizations:
            return tuple(base_tokenizer.encode(text, add_special_tokens=False))
        cached = encoded_cache.get(text)
        if cached is not None:
            return cached
        encoded = tuple(base_tokenizer.encode(text, add_special_tokens=False))
        encoded_cache[text] = encoded
        return encoded

    def _capitalize_first_alpha(text: str) -> str:
        chars = list(text)
        for idx, ch in enumerate(chars):
            if ch.isalpha():
                chars[idx] = ch.upper()
                break
        return "".join(chars)

    def _first_alpha_is_upper(text: str) -> bool:
        for ch in text:
            if ch.isalpha():
                return bool(ch.isupper())
        return False

    def _replace_transform(transforms, src_name: str, dst_name: str):
        rewritten = []
        for name in transforms:
            if name == src_name:
                rewritten.append(dst_name)
            else:
                rewritten.append(name)
        return list(dict.fromkeys(rewritten))

    def _prefer_single_token_alias_base(base_key: str, candidate_base_ids):
        """Prefer single-token aliases when canonical no-space base is multi-token.

        For whitespace-aware vocabularies, some lemmas only have an efficient single-token
        representation with a leading space (e.g., " dog"), while the no-space form
        tokenizes to multiple IDs. Similarly, some lemmas are only available as
        capitalized single tokens (e.g., " Germany"). In those cases, anchor the
        canonical base to the available single-token alias.
        """
        if not promote_surface_alias_bases:
            return candidate_base_ids, None
        if not isinstance(base_key, str) or not base_key:
            return candidate_base_ids, None
        if len(candidate_base_ids) <= 1:
            return candidate_base_ids, None

        def _resolve_ids_for_key(key: str):
            resolved = None
            if isinstance(base_tokens, dict):
                candidate = base_tokens.get(key)
                if isinstance(candidate, int):
                    resolved = (candidate,)
                elif isinstance(candidate, tuple):
                    resolved = candidate
                elif isinstance(candidate, list):
                    resolved = tuple(candidate)
            if resolved is None:
                resolved = _encode_cached(key)
            return resolved

        if decompose_spaces:
            spaced_key = f" {base_key}"
            spaced_ids = _resolve_ids_for_key(spaced_key)
            if len(spaced_ids) == 1:
                return spaced_ids, spaced_key

        if decompose_caps:
            capitalized_key = _capitalize_first_alpha(base_key)
            if capitalized_key != base_key:
                cap_ids = _resolve_ids_for_key(capitalized_key)
                if len(cap_ids) == 1:
                    return cap_ids, capitalized_key
                if decompose_spaces:
                    spaced_cap_key = f" {capitalized_key}"
                    spaced_cap_ids = _resolve_ids_for_key(spaced_cap_key)
                    if len(spaced_cap_ids) == 1:
                        return spaced_cap_ids, spaced_cap_key

        return candidate_base_ids, None

    def _adjust_surface_transforms_for_alias(base_key: str, base_alias_key, transforms):
        if not promote_surface_alias_bases:
            return transforms
        if not base_alias_key or not isinstance(base_alias_key, str):
            return transforms
        if not isinstance(base_key, str) or not base_key:
            return transforms

        adjusted = list(transforms)
        alias_has_space = base_alias_key.startswith(" ")
        base_has_space = base_key.startswith(" ")
        if alias_has_space and (not base_has_space):
            adjusted = _replace_transform(
                adjusted,
                NO_SPACE_PREFIX_TRANSFORM,
                REMOVE_SPACE_PREFIX_TRANSFORM,
            )

        alias_core = base_alias_key.lstrip(" ")
        base_core = base_key.lstrip(" ")
        if _first_alpha_is_upper(alias_core) and (not _first_alpha_is_upper(base_core)):
            adjusted = _replace_transform(
                adjusted,
                NO_BASE_CAPITALIZATION_TRANSFORM,
                REMOVE_BASE_CAPITALIZATION_TRANSFORM,
            )

        return adjusted

    def _process_candidate_surface(token_surface: str):
        nonlocal conflicts, yielded
        if not token_surface:
            return

        base_key = _canonical_space_cap_base(token_surface, decompose_caps)
        if not base_key or base_key in seen_base_keys:
            return
        seen_base_keys.add(base_key)

        morph_owner = morph_variant_owner.get(base_key)
        owner_base_word = morph_owner[0] if morph_owner is not None else base_key
        owner_transforms = morph_owner[1] if morph_owner is not None else []
        base_alias_key = None

        base_ids = None
        if isinstance(base_tokens, dict):
            candidate_base_keys = []
            for key in (
                owner_base_word,
                base_key,
                f" {base_key}" if base_key else None,
                base_key.lower() if isinstance(base_key, str) else None,
                f" {base_key.lower()}" if isinstance(base_key, str) and base_key else None,
            ):
                if key is None:
                    continue
                if key in candidate_base_keys:
                    continue
                candidate_base_keys.append(key)
            for key in candidate_base_keys:
                candidate = base_tokens.get(key)
                if isinstance(candidate, int):
                    base_ids = (candidate,)
                    break
                if isinstance(candidate, tuple):
                    base_ids = candidate
                    break
                if isinstance(candidate, list):
                    base_ids = tuple(candidate)
                    break
        if base_ids is None:
            base_ids = _encode_cached(owner_base_word)
        base_ids, base_alias_key = _prefer_single_token_alias_base(base_key, base_ids)
        if len(base_ids) == 0:
            return
        if max_variant_tokens is not None and len(base_ids) > max_variant_tokens:
            return
        base_rank = max(base_ids)

        generated = get_variation_transformations(
            base_key,
            None,
            None,
            space_prefix=" ",
            decompose_spaces=decompose_spaces,
            decompose_capitalization=decompose_caps,
            use_relative_space_cap_transforms=use_relative,
            include_all_caps_capitalization=enable_all_caps_capitalization,
            article_space_prefix=None,
            prep_space_prefix=None,
        )

        for variant_str, transforms in generated.items():
            if variant_str in existing_morph_variants and not (
                promote_surface_alias_bases and base_alias_key and variant_str == base_key
            ):
                continue
            combined_transforms = list(owner_transforms) + list(transforms)
            seen_transforms = set()
            deduped_transforms = []
            for transform_name in combined_transforms:
                if transform_name in seen_transforms:
                    continue
                seen_transforms.add(transform_name)
                deduped_transforms.append(transform_name)
            deduped_transforms = _adjust_surface_transforms_for_alias(
                base_key,
                base_alias_key,
                deduped_transforms,
            )
            if not _space_cap_is_active_transform(deduped_transforms):
                continue
            variant_ids = _encode_cached(variant_str)
            if len(variant_ids) == 0:
                continue
            if max_variant_tokens is not None and len(variant_ids) > max_variant_tokens:
                continue
            owner = variant_to_owner.get(variant_str)
            owner_key = owner_base_word if morph_owner is not None else base_key
            if owner is not None and owner != owner_key:
                conflicts += 1
                continue
            variant_to_owner[variant_str] = owner_key
            yielded += 1
            yield base_key, base_ids, base_rank, variant_str, deduped_transforms, variant_ids

    # First seed from tokenizer vocab (fast/common path).
    for token, _token_id in vocab_items:
        if token in special_tokens:
            continue
        token_surface = base_tokenizer.convert_tokens_to_string([token])
        for item in _process_candidate_surface(token_surface):
            yield item

    # Then seed from morphology-produced surfaces so space/cap closure is applied
    # even when a morphology form is not itself a vocab token (e.g., "whales").
    for variant_surface in sorted(existing_morph_variants):
        for item in _process_candidate_surface(variant_surface):
            yield item

    if conflicts:
        print(f"Skipped {conflicts:,} conflicting global space/cap candidates")
    if enable_build_optimizations:
        print(f"Global space/cap candidate generation: {yielded:,} candidates from {len(seen_base_keys):,} canonical bases")


def add_modifier_syntactic_sequences_from_vocab(
    base_tokenizer,
    sequence_map,
    decomposition_map,
    transformation_names_to_int,
    types_loss_indices_map,
    groups,
    decompose_articles,
    decompose_possessive_determiners,
    decompose_demonstrative_determiners,
    decompose_quantifier_determiners,
    decompose_prepositions,
    decompose_punctuation,
    decompose_possessive_separate_group=False,
    preposition_list=None,
    decompose_article_prep_space_prefix=False,
    max_tokens_per_word=None,
    overwrite_existing=False,
    enable_build_optimizations=False,
):
    """Add punctuation/article/preposition candidates directly into SequenceMap."""
    if not (
        decompose_articles
        or decompose_prepositions
        or decompose_punctuation
        or decompose_possessive_separate_group
    ):
        return {
            "added": 0,
            "scanned": 0,
            "matched_base": 0,
            "skipped_existing": 0,
            "ambiguous_base_variants": 0,
        }

    modifier_manager = sequence_map.modifier_manager
    transform_to_group_idx = _build_transform_to_group_idx(
        transformation_names_to_int=transformation_names_to_int,
        types_loss_indices_map=types_loss_indices_map,
        groups=groups,
    )
    use_build_optimizations = bool(enable_build_optimizations)
    encoded_cache = {}

    def _encode_cached(text):
        if not use_build_optimizations:
            return tuple(base_tokenizer.encode(text, add_special_tokens=False))
        cached = encoded_cache.get(text)
        if cached is not None:
            return cached
        encoded = tuple(base_tokenizer.encode(text, add_special_tokens=False))
        encoded_cache[text] = encoded
        return encoded

    variant_to_entry = {}
    ambiguous_base_variants = 0
    for base_key, variants in decomposition_map.items():
        for variant_str, transforms in variants.items():
            existing = variant_to_entry.get(variant_str)
            if existing is None:
                variant_to_entry[variant_str] = (base_key, list(transforms))
                continue
            existing_base, _ = existing
            if existing_base != base_key:
                ambiguous_base_variants += 1

    special_tokens = set(getattr(base_tokenizer, "all_special_tokens", []) or [])
    vocab_items = sorted(base_tokenizer.get_vocab().items(), key=lambda kv: kv[1])

    added = 0
    scanned = 0
    matched_base = 0
    fallback_base = 0
    skipped_existing = 0

    has_article_space_group = "article_space_prefix" in types_loss_indices_map
    has_prep_space_group = "prep_space_prefix" in types_loss_indices_map

    for token, _token_id in vocab_items:
        if token in special_tokens:
            continue
        token_surface = base_tokenizer.convert_tokens_to_string([token])
        if not token_surface:
            continue
        base_text, extracted = strip_all_affixes(
            token_surface,
            space_prefix=" ",
            decompose_punctuation=decompose_punctuation,
            decompose_articles=decompose_articles,
            decompose_prepositions=decompose_prepositions,
            prepositions=preposition_list,
            track_article_prep_space_prefix=decompose_article_prep_space_prefix,
            decompose_possessive_determiners=decompose_possessive_determiners,
            decompose_demonstrative_determiners=decompose_demonstrative_determiners,
            decompose_quantifier_determiners=decompose_quantifier_determiners,
            decompose_possessive_suffixes=decompose_possessive_separate_group,
        )
        if (
            extracted["prefix_punct"] is None
            and extracted["suffix_punct"] is None
            and extracted["possessive"] is None
            and extracted["article"] is None
            and extracted["prep"] is None
        ):
            continue

        scanned += 1
        base_variant = f" {base_text}" if extracted["space_prefix"] else base_text
        base_entry = variant_to_entry.get(base_variant)
        if base_entry is None:
            # Fallback: allow syntactic-only rewrites even when the base surface is
            # absent from the decomposition map (e.g., filtered proper nouns like "James").
            base_key, base_transforms = base_variant, []
            fallback_base += 1
        else:
            matched_base += 1
            base_key, base_transforms = base_entry

        variant_ids = _encode_cached(token_surface)
        if len(variant_ids) == 0:
            continue
        if max_tokens_per_word is not None and len(variant_ids) > max_tokens_per_word:
            continue
        if (not overwrite_existing) and sequence_map.get_exact_match(variant_ids) is not None:
            skipped_existing += 1
            continue

        base_ids = _encode_cached(base_key)
        if len(base_ids) == 0:
            continue
        if max_tokens_per_word is not None and len(base_ids) > max_tokens_per_word:
            continue

        modifier = modifier_manager.create_empty_modifier()
        for transform_name in base_transforms:
            group_rel = transform_to_group_idx.get(transform_name)
            if group_rel is None:
                continue
            group_name, rel_idx = group_rel
            modifier_manager.set_group_value(modifier, group_name, rel_idx)

        override_transforms = []
        if decompose_punctuation:
            override_transforms.append(extracted["prefix_punct"] or NO_PREFIX_PUNCTUATION)
            override_transforms.append(extracted["suffix_punct"] or NO_SUFFIX_PUNCTUATION)
        if decompose_possessive_separate_group:
            override_transforms.append(extracted["possessive"] or NO_POSSESSIVE)
        if decompose_articles:
            override_transforms.append(extracted["article"] or NO_ARTICLE)
            override_transforms.append(
                ADD_ARTICLE_CAPITALIZATION
                if extracted["article"] and extracted["article_cap"]
                else NO_ARTICLE_CAPITALIZATION
            )
            if decompose_article_prep_space_prefix and has_article_space_group:
                override_transforms.append(
                    ADD_ARTICLE_SPACE_PREFIX
                    if extracted["article"] and extracted["article_space_prefix"]
                    else NO_ARTICLE_SPACE_PREFIX
                )
        if decompose_prepositions:
            override_transforms.append(extracted["prep"] or NO_PREPOSITION)
            override_transforms.append(
                ADD_PREP_CAPITALIZATION
                if extracted["prep"] and extracted["prep_cap"]
                else NO_PREP_CAPITALIZATION
            )
            if decompose_article_prep_space_prefix and has_prep_space_group:
                override_transforms.append(
                    ADD_PREP_SPACE_PREFIX
                    if extracted["prep"] and extracted["prep_space_prefix"]
                    else NO_PREP_SPACE_PREFIX
                )

        for transform_name in override_transforms:
            group_rel = transform_to_group_idx.get(transform_name)
            if group_rel is None:
                continue
            group_name, rel_idx = group_rel
            modifier_manager.set_group_value(modifier, group_name, rel_idx)

        sequence_map.add_sequence(variant_ids, base_ids, modifier)
        added += 1

    return {
        "added": added,
        "scanned": scanned,
        "matched_base": matched_base,
        "fallback_base": fallback_base,
        "skipped_existing": skipped_existing,
        "ambiguous_base_variants": ambiguous_base_variants,
    }


def add_space_cap_syntactic_sequences_from_vocab(
    base_tokenizer,
    sequence_map,
    decomposition_map,
    base_tokens,
    transformation_names_to_int,
    types_loss_indices_map,
    groups,
    decompose_spaces,
    decompose_caps,
    enable_all_caps_capitalization=False,
    use_relative_space_cap_transforms=False,
    max_tokens_per_word=None,
    overwrite_existing=False,
    enable_build_optimizations=False,
    promote_surface_alias_bases=False,
):
    """Add global space/capitalization syntactic candidates directly into a SequenceMap."""
    if not (decompose_spaces or decompose_caps):
        return {"added": 0, "bases": 0, "variants": 0, "max_rank": -1}

    modifier_manager = sequence_map.modifier_manager
    transform_to_group_idx = _build_transform_to_group_idx(
        transformation_names_to_int=transformation_names_to_int,
        types_loss_indices_map=types_loss_indices_map,
        groups=groups,
    )

    added = 0
    bases_seen = set()
    max_rank = -1
    for (
        base_key,
        base_ids,
        base_rank,
        _variant_str,
        transforms,
        variant_ids,
    ) in _iter_space_cap_syntactic_candidates(
        base_tokenizer=base_tokenizer,
        decomposition_map=decomposition_map,
        decompose_spaces=decompose_spaces,
        decompose_caps=decompose_caps,
        use_relative=use_relative_space_cap_transforms,
        enable_all_caps_capitalization=enable_all_caps_capitalization,
        base_tokens=base_tokens,
        max_variant_tokens=max_tokens_per_word,
        enable_build_optimizations=enable_build_optimizations,
        promote_surface_alias_bases=promote_surface_alias_bases,
    ):
        bases_seen.add(base_key)
        if base_rank > max_rank:
            max_rank = base_rank
        modifier = modifier_manager.create_empty_modifier()
        for transform_name in transforms:
            if transform_name in transform_to_group_idx:
                group_name, rel_idx = transform_to_group_idx[transform_name]
                modifier_manager.set_group_value(modifier, group_name, rel_idx)
        if (not overwrite_existing) and sequence_map.get_exact_match(variant_ids) is not None:
            continue
        sequence_map.add_sequence(variant_ids, base_ids, modifier)
        added += 1

    return {
        "added": added,
        "bases": len(bases_seen),
        "max_rank": max_rank,
    }


def build_rank_to_space_cap_variants_from_vocab(
    tokenizer,
    decomposition_map,
    base_tokens,
    decompose_spaces,
    decompose_caps,
    enable_all_caps_capitalization=False,
    use_relative_space_cap_transforms=False,
    max_tokens_per_word=None,
    enable_build_optimizations=False,
    promote_surface_alias_bases=False,
):
    """Build rank->variants directly from vocab-wide global space/cap candidates."""
    rank_to_variants = defaultdict(set)
    used_bases = set()
    max_rank = -1
    for (
        base_key,
        _base_ids,
        base_rank,
        variant_str,
        _transforms,
        _variant_ids,
    ) in _iter_space_cap_syntactic_candidates(
        base_tokenizer=tokenizer,
        decomposition_map=decomposition_map,
        decompose_spaces=decompose_spaces,
        decompose_caps=decompose_caps,
        use_relative=use_relative_space_cap_transforms,
        enable_all_caps_capitalization=enable_all_caps_capitalization,
        base_tokens=base_tokens,
        max_variant_tokens=max_tokens_per_word,
        enable_build_optimizations=enable_build_optimizations,
        promote_surface_alias_bases=promote_surface_alias_bases,
    ):
        rank_to_variants[base_rank].add(variant_str)
        used_bases.add(base_key)
        if base_rank > max_rank:
            max_rank = base_rank
    return rank_to_variants, max_rank, len(used_bases)


def _normalize_language_tag_for_cache(language: str) -> str:
    if not language:
        return "en"
    primary = language.split(":", 1)[0].strip()
    if not primary:
        primary = language.strip()
    primary = primary.replace("-", "_")
    if primary.startswith("en"):
        return "en"
    return primary


def _tokenizer_has_space_prefixed_word_tokens(base_tokenizer, sample_limit=50000):
    """Heuristic: regular GPT-style tokenizers expose many leading-space word tokens."""
    vocab = base_tokenizer.get_vocab()
    if not vocab:
        return False
    special_tokens = set(getattr(base_tokenizer, "all_special_tokens", []) or [])
    scanned = 0
    for token, _token_id in sorted(vocab.items(), key=lambda kv: kv[1]):
        if token in special_tokens:
            continue
        if scanned >= sample_limit:
            break
        scanned += 1
        try:
            surface = base_tokenizer.convert_tokens_to_string([token])
        except Exception:
            continue
        if not isinstance(surface, str) or not surface.startswith(" "):
            continue
        stripped = surface.lstrip(" ")
        if stripped and any(ch.isalnum() for ch in stripped):
            return True
    return False


def resolve_promote_surface_alias_bases(args, base_tokenizer=None):
    """Resolve alias-base promotion mode.

    If user sets --promote_surface_alias_bases / --no-promote_surface_alias_bases,
    that explicit value wins. Otherwise, auto-enable for regular tokenizers and
    keep disabled for space-cap-normalized tokenizers.
    """
    resolved_origin = getattr(args, "_promote_surface_alias_bases_origin", None)
    current_value = getattr(args, "promote_surface_alias_bases", None)
    if resolved_origin in {"explicit", "auto_regular", "auto_space_cap"} and isinstance(current_value, bool):
        return current_value, resolved_origin

    if current_value is not None:
        resolved = bool(current_value)
        origin = "explicit"
    else:
        tokenizer = base_tokenizer if base_tokenizer is not None else AutoTokenizer.from_pretrained(args.base_tokenizer)
        regular_like = _tokenizer_has_space_prefixed_word_tokens(tokenizer)
        resolved = bool(regular_like)
        origin = "auto_regular" if regular_like else "auto_space_cap"

    args.promote_surface_alias_bases = resolved
    args._promote_surface_alias_bases_origin = origin
    return resolved, origin


def _compute_tokenizer_cache_descriptor(args, base_cache_dir=None):
    # Extract base name from tokenizer path (handles both HF Hub names and local paths)
    base_name = os.path.basename(args.base_tokenizer.rstrip(os.sep))
    if not base_name:  # Handle edge case of root path
        base_name = args.base_tokenizer.replace(os.sep, "_").replace("/", "_")

    features = []
    compact_features = []
    language = getattr(args, "language", "en")
    language_tag = _normalize_language_tag_for_cache(str(language))
    if language_tag != "en":
        features.append(f"lang_{language_tag}")
        compact_features.append(f"lang_{language_tag}")
    decompose_articles = bool(getattr(args, "decompose_articles", False))
    decompose_demo = bool(decompose_articles and getattr(args, "decompose_demonstrative_determiners", False))
    decompose_quant = bool(decompose_articles and getattr(args, "decompose_quantifier_determiners", False))
    decompose_prepositions = bool(getattr(args, "decompose_prepositions", False))
    preposition_profile = "off"
    preposition_list_for_key = None
    if decompose_prepositions:
        requested_profile = getattr(args, "preposition_profile", DEFAULT_PREPOSITION_PROFILE)
        requested_list = getattr(args, "preposition_list", None)
        preposition_list_for_key, preposition_profile = resolve_preposition_settings(
            requested_profile,
            requested_list,
        )
        preposition_list_for_key = list(preposition_list_for_key)
        preposition_profile = str(preposition_profile or "custom").strip().lower()
    if decompose_articles:
        compact_features.append("det_on")
    else:
        compact_features.append("det_off")
    if decompose_articles:
        if decompose_demo and decompose_quant:
            det_profile = "demo_quant"
            features.append("det_profile_demo_quant")
            compact_features.append("det_demo_quant")
        elif decompose_demo:
            det_profile = "demo"
            features.append("det_profile_demo")
            compact_features.append("det_demo")
        elif decompose_quant:
            det_profile = "quant"
            features.append("det_profile_quant")
            compact_features.append("det_quant")
        else:
            det_profile = "base"
            features.append("det_profile_base")
            compact_features.append("det_base")
    else:
        det_profile = "off"
        features.append("det_profile_off")
    if not args.dont_decompose_spaces:
        features.append("decomp_spaces")
        compact_features.append("spaces")
    if args.skip_if_no_space_prefix:
        features.append("filter_no_space_prefix")
    if not args.dont_decompose_capitalized:
        features.append("decomp_caps")
        compact_features.append("caps")
        if not bool(getattr(args, "enable_all_caps_capitalization", False)):
            features.append("no_all_caps_cap")
    if getattr(args, "use_relative_space_cap_transforms", False):
        features.append("rel_space_cap")
        compact_features.append("rel_space_cap")
    if getattr(args, "canonicalize_token_strings", True):
        features.append("canonical_tokens")
    if bool(getattr(args, "promote_surface_alias_bases", False)):
        features.append("promote_surface_alias_bases")
    if not args.filter_propernouns_and_aux:
        features.append("w_aux")
    if args.include_non_resource_latin_tokens:
        features.append("w_subwords")
    if args.include_non_latin_tokens:
        features.append("w_symbols")
    if not args.skip_stop_words:
        features.append("w_stopwords")
    if args.use_type_str_as_type:
        features.append("type_as_str")
    if args.merge_plural_and_present_singular:
        features.append("merged_plural_vp3s")
    if getattr(args, "prefer_popular_base_conflicts", True):
        features.append("prefer_popular_base_conflicts")
    else:
        features.append("no_prefer_popular_base_conflicts")
    if not args.skip_multi_token_words:
        features.append("multi_token")
    if args.skip_three_token_words and not args.skip_multi_token_words:
        features.append("upto_two_tokens")
    elif getattr(args, "skip_four_token_words", False) and not args.skip_multi_token_words:
        features.append("upto_three_tokens")
    if getattr(args, "skip_multi_token_bases", False):
        features.append("skip_multi_token_bases")

    # New transformation groups
    if args.decompose_punctuation:
        features.append("decomp_punct")
        compact_features.append("punct")
    if decompose_articles:
        features.append("decomp_articles")
    if decompose_articles and getattr(args, "decompose_possessive_determiners", False):
        features.append("decomp_possessive_determiners")
        compact_features.append("det_poss")
    if decompose_demo:
        features.append("decomp_demonstrative_determiners")
        features.append("det_demo")
    if decompose_quant:
        features.append("decomp_quantifier_determiners")
        features.append("det_quant")
    if decompose_prepositions:
        features.append("decomp_preps")
        compact_features.append("preps")
        features.append(f"preps_profile_{preposition_profile}")
        compact_features.append(f"preps_{preposition_profile}")
        if preposition_profile == "custom" and preposition_list_for_key:
            prep_str = "-".join(sorted(preposition_list_for_key))
            features.append(f"preps_{prep_str}")
            compact_features.append("preps_custom")
    if getattr(args, "decompose_article_prep_space_prefix", False):
        features.append("article_prep_space_prefix")
        compact_features.append("detprep_space")
    modifier_generation_mode = _resolve_modifier_generation_mode(args).lower()
    if modifier_generation_mode == "vocab_only":
        features.append("modifiers_vocab_only")
        compact_features.append("mod_vocab")
    features.append("dualstream_separate_space_cap")

    # Multi-token whitelist for new transformations
    if args.multitoken_allowed_groups and len(args.multitoken_allowed_groups) > 0:
        groups_str = "-".join(sorted(args.multitoken_allowed_groups))
        features.append(f"mtok_allowed_{groups_str}")
    if getattr(args, "max_tokens_per_base_word", None) is not None:
        features.append(f"max_tokens_per_base_{int(args.max_tokens_per_base_word)}")

    # Unified modifiers (2D format) - automatically enabled for dual_stream
    # (removed separate flag, now implied by dual_stream mode)
    if getattr(args, 'prune_punct_tokens', False):
        features.append("pruned_punct")
        compact_features.append("pruned_punct")
    if getattr(args, 'possessive_separate_group', False):
        features.append("poss_sep_group")
        compact_features.append("poss_sep")

    preposition_list_for_hash = None
    if decompose_prepositions:
        preposition_list_for_hash = list(preposition_list_for_key)
    else:
        raw_prep_list = getattr(args, "preposition_list", None)
        preposition_list_for_hash = list(raw_prep_list) if raw_prep_list is not None else None

    key_fields = {
        "base_name": base_name,
        "language_tag": language_tag,
        "det_profile": det_profile,
        "decompose_articles": decompose_articles,
        "decompose_possessive_determiners": bool(decompose_articles and getattr(args, "decompose_possessive_determiners", False)),
        "decompose_demonstrative_determiners": decompose_demo,
        "decompose_quantifier_determiners": decompose_quant,
        "decompose_prepositions": bool(decompose_prepositions),
        "decompose_punctuation": bool(getattr(args, "decompose_punctuation", False)),
        "preposition_list": preposition_list_for_hash,
        "decompose_article_prep_space_prefix": bool(getattr(args, "decompose_article_prep_space_prefix", False)),
    }
    if decompose_prepositions:
        key_fields["preposition_profile"] = preposition_profile
    key_payload = {
        "format_version": 3,
        "key_fields": key_fields,
        "full_features": list(features),
        "base_tokenizer_fingerprint": _fingerprint_local_tokenizer_source(args.base_tokenizer),
    }
    key_hash = hashlib.sha256(
        json.dumps(_json_safe(key_payload), ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:12]
    compact_feature_str = "_".join(compact_features) if compact_features else "base"
    cache_subdir = f"{base_name}__{compact_feature_str}__cfg_{key_hash}"

    # Avoid filesystem component length limits (commonly 255 bytes) while keeping stable uniqueness.
    max_component_len = 240
    if len(cache_subdir) > max_component_len:
        prefix_budget = max_component_len - len(key_hash) - 6
        safe_prefix = cache_subdir[:max(32, prefix_budget)]
        cache_subdir = f"{safe_prefix}__cfg_{key_hash}"

    cache_path = os.path.join(base_cache_dir, cache_subdir) if base_cache_dir else cache_subdir
    return {
        "cache_path": cache_path,
        "cache_subdir": cache_subdir,
        "cache_hash": key_hash,
        "compact_features": list(compact_features),
        "full_features": list(features),
        "key_payload": key_payload,
    }


def _fingerprint_local_tokenizer_source(tokenizer_source):
    """Fingerprint local HF tokenizer inputs used to build compositional metadata."""

    source = os.path.abspath(os.path.expanduser(str(tokenizer_source)))
    if not os.path.isdir(source):
        return None
    names = (
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "added_tokens.json",
    )
    fingerprint = fingerprint_named_files(source, names)
    return fingerprint["sha256"] if fingerprint["files"] else None


def get_tokenizer_cache_path(args, base_cache_dir=None):
    return _compute_tokenizer_cache_descriptor(args, base_cache_dir)["cache_path"]


def _json_safe(value):
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(v) for v in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return repr(value)


def _write_json(path, payload):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_json_safe(payload), f, ensure_ascii=False, indent=2, sort_keys=True)


def _build_decomposition_stage_cache_key(
    args,
    *,
    language,
    decompose_spaces,
    decompose_capitalized,
    map_decompose_punctuation,
    map_decompose_articles,
    map_decompose_possessive_determiners,
    map_decompose_demonstrative_determiners,
    map_decompose_quantifier_determiners,
    map_decompose_possessive_separate_group,
    map_decompose_prepositions,
    map_phrase_space_prefix,
    modifier_generation_mode,
):
    payload = {
        "version": 2,
        "base_tokenizer": getattr(args, "base_tokenizer", ""),
        "language": language,
        "decompose_spaces": bool(decompose_spaces),
        "decompose_capitalization": bool(decompose_capitalized),
        "enable_all_caps_capitalization": bool(getattr(args, "enable_all_caps_capitalization", False)),
        "skip_if_no_space_prefix": bool(getattr(args, "skip_if_no_space_prefix", False)),
        "use_relative_space_cap_transforms": bool(getattr(args, "use_relative_space_cap_transforms", False)),
        "promote_surface_alias_bases": bool(getattr(args, "promote_surface_alias_bases", False)),
        "canonicalize_token_strings": bool(getattr(args, "canonicalize_token_strings", True)),
        "decompose_punctuation": bool(map_decompose_punctuation),
        "decompose_articles": bool(map_decompose_articles),
        "decompose_possessive_determiners": bool(map_decompose_possessive_determiners),
        "decompose_demonstrative_determiners": bool(map_decompose_demonstrative_determiners),
        "decompose_quantifier_determiners": bool(map_decompose_quantifier_determiners),
        "decompose_possessive_separate_group": bool(map_decompose_possessive_separate_group),
        "decompose_prepositions": bool(map_decompose_prepositions),
        "decompose_article_prep_space_prefix": bool(map_phrase_space_prefix),
        "preposition_profile": getattr(args, "preposition_profile", "off"),
        "preposition_list": list(getattr(args, "preposition_list", []) or []),
        "modifier_generation_mode": modifier_generation_mode,
        "skip_multi_token_words": bool(getattr(args, "skip_multi_token_words", False)),
        "skip_three_token_words": bool(getattr(args, "skip_three_token_words", False)),
        "skip_four_token_words": bool(getattr(args, "skip_four_token_words", False)),
        "skip_multi_token_bases": bool(getattr(args, "skip_multi_token_bases", False)),
        "multitoken_allowed_groups": list(getattr(args, "multitoken_allowed_groups", []) or []),
        "max_vocab_items": getattr(args, "max_vocab_items", None),
        "log_second_pass_bases": bool(getattr(args, "log_second_pass_bases", False)),
        "enable_build_optimizations": bool(getattr(args, "enable_build_optimizations", False)),
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:20]


def create_compositional_tokenizer_bundle(args, bundle_cache_dir=None):
    base_tokenizer = AutoTokenizer.from_pretrained(args.base_tokenizer)
    promote_surface_alias_bases, promote_origin = resolve_promote_surface_alias_bases(
        args, base_tokenizer=base_tokenizer
    )
    if not bool(getattr(args, "_promote_surface_alias_bases_reported", False)):
        if promote_origin == "auto_regular":
            print(
                "Auto-enabled promote_surface_alias_bases for regular tokenizer "
                "(detected space-prefixed word tokens in vocab)."
            )
        elif promote_origin == "auto_space_cap":
            print(
                "Auto-disabled promote_surface_alias_bases for space-cap-style tokenizer "
                "(no space-prefixed word tokens detected)."
            )
        else:
            print(f"Using promote_surface_alias_bases={promote_surface_alias_bases} (explicit CLI setting).")
        args._promote_surface_alias_bases_reported = True
    language = getattr(args, "language", "en")
    tokenization_mode = getattr(args, 'tokenization_mode', 'dual_stream')
    skip_multi_token_bases = bool(getattr(args, "skip_multi_token_bases", False))
    if tokenization_mode != "dual_stream":
        raise ValueError("nanochat cobpe.decomposition supports dual_stream tokenization only.")

    # Language-specific handling of English-only transformations
    decompose_articles = args.decompose_articles
    decompose_possessive_determiners = bool(getattr(args, "decompose_possessive_determiners", False))
    decompose_demonstrative_determiners = bool(getattr(args, "decompose_demonstrative_determiners", False))
    decompose_quantifier_determiners = bool(getattr(args, "decompose_quantifier_determiners", False))
    if decompose_possessive_determiners and not decompose_articles:
        print("INFO: ignoring --decompose_possessive_determiners because --decompose_articles is disabled.")
        decompose_possessive_determiners = False
    if decompose_demonstrative_determiners and not decompose_articles:
        print("INFO: ignoring --decompose_demonstrative_determiners because --decompose_articles is disabled.")
        decompose_demonstrative_determiners = False
    if decompose_quantifier_determiners and not decompose_articles:
        print("INFO: ignoring --decompose_quantifier_determiners because --decompose_articles is disabled.")
        decompose_quantifier_determiners = False
    decompose_prepositions = args.decompose_prepositions
    preposition_list_explicit = bool(
        getattr(
            args,
            "_preposition_list_explicit",
            getattr(args, "preposition_list", None) is not None,
        )
    )
    requested_preposition_profile = getattr(args, "preposition_profile", DEFAULT_PREPOSITION_PROFILE)
    resolved_preposition_list = None
    resolved_preposition_profile = "off"
    if decompose_prepositions:
        resolved_preposition_list, resolved_preposition_profile = resolve_preposition_settings(
            requested_preposition_profile,
            getattr(args, "preposition_list", None),
        )
    decompose_possessive_separate_group = bool(getattr(args, "possessive_separate_group", False))
    if decompose_possessive_separate_group and not bool(args.decompose_punctuation):
        print("INFO: ignoring --possessive_separate_group because --decompose_punctuation is disabled.")
        decompose_possessive_separate_group = False
    modifier_generation_mode = _resolve_modifier_generation_mode(args)
    phrase_space_prefix = getattr(args, "decompose_article_prep_space_prefix", False)

    if language.lower() != "en":
        if decompose_possessive_separate_group:
            print(f"WARNING: Separate possessive suffix decomposition is English-only.")
            print(f"         Disabling for language '{language}'.")
            decompose_possessive_separate_group = False
        # Articles: English-only, no customization option -> always disable
        if (
            decompose_articles
            or decompose_possessive_determiners
            or decompose_demonstrative_determiners
            or decompose_quantifier_determiners
        ):
            print(f"WARNING: Determiner decomposition (articles/possessives) is English-only.")
            print(f"         Disabling for language '{language}'.")
            decompose_articles = False
            decompose_possessive_determiners = False
            decompose_demonstrative_determiners = False
            decompose_quantifier_determiners = False

        # Prepositions: Honor if user provided custom list, otherwise disable
        if decompose_prepositions:
            if not preposition_list_explicit:
                print(f"WARNING: Preposition decomposition defaults to English prepositions.")
                print(f"         Disabling for language '{language}'.")
                print(f"         To enable, provide --preposition_list with {language}-specific prepositions.")
                decompose_prepositions = False
                resolved_preposition_list = None
                resolved_preposition_profile = "off"
            else:
                resolved_preposition_profile = classify_preposition_list(resolved_preposition_list)
                print(f"INFO: Using custom preposition list for '{language}': {resolved_preposition_list}")

    decompose_determiners = bool(decompose_articles)

    # Keep downstream helpers consistent with effective language-gated choices.
    args.decompose_articles = bool(decompose_articles)
    args.decompose_possessive_determiners = bool(decompose_possessive_determiners)
    args.decompose_demonstrative_determiners = bool(decompose_demonstrative_determiners)
    args.decompose_quantifier_determiners = bool(decompose_quantifier_determiners)
    args.decompose_prepositions = bool(decompose_prepositions)
    args.preposition_list = list(resolved_preposition_list) if decompose_prepositions and resolved_preposition_list is not None else None
    args.preposition_profile = (
        str(resolved_preposition_profile).strip().lower()
        if decompose_prepositions
        else "off"
    )
    args.possessive_separate_group = bool(decompose_possessive_separate_group)

    if not (decompose_determiners or decompose_prepositions):
        phrase_space_prefix = False
    args.decompose_article_prep_space_prefix = bool(phrase_space_prefix)

    separate_global_space_cap = (tokenization_mode == "dual_stream")
    base_decompose_spaces = (not args.dont_decompose_spaces) and (not separate_global_space_cap)
    base_decompose_capitalized = (not args.dont_decompose_capitalized) and (not separate_global_space_cap)
    if separate_global_space_cap and (not args.dont_decompose_spaces or not args.dont_decompose_capitalized):
        print(
            "Dual-stream mode: space and capitalization are applied from the tokenizer vocabulary."
        )

    decomposition_map_controls_modifiers = (tokenization_mode == "dual_stream")
    map_decompose_punctuation = args.decompose_punctuation
    map_decompose_articles = decompose_determiners
    map_decompose_demonstrative_determiners = decompose_demonstrative_determiners
    map_decompose_quantifier_determiners = decompose_quantifier_determiners
    map_decompose_prepositions = decompose_prepositions
    map_decompose_possessive_separate_group = decompose_possessive_separate_group
    map_phrase_space_prefix = phrase_space_prefix
    if decomposition_map_controls_modifiers and (
        args.decompose_punctuation or decompose_determiners or decompose_prepositions
    ):
        map_decompose_punctuation = False
        map_decompose_articles = False
        map_decompose_demonstrative_determiners = False
        map_decompose_quantifier_determiners = False
        map_decompose_prepositions = False
        map_decompose_possessive_separate_group = False
        map_phrase_space_prefix = False
        print(
            "Dual-stream mode: punctuation/articles/prepositions are handled in SequenceMap "
            "(skipping decomposition-map modifier generation)."
        )

    stage_cache_key = _build_decomposition_stage_cache_key(
        args,
        language=language,
        decompose_spaces=base_decompose_spaces,
        decompose_capitalized=base_decompose_capitalized,
        map_decompose_punctuation=map_decompose_punctuation,
        map_decompose_articles=map_decompose_articles,
        map_decompose_possessive_determiners=decompose_possessive_determiners,
        map_decompose_demonstrative_determiners=map_decompose_demonstrative_determiners,
        map_decompose_quantifier_determiners=map_decompose_quantifier_determiners,
        map_decompose_possessive_separate_group=map_decompose_possessive_separate_group,
        map_decompose_prepositions=map_decompose_prepositions,
        map_phrase_space_prefix=map_phrase_space_prefix,
        modifier_generation_mode=modifier_generation_mode,
    )
    reuse_stage_cache_arg = getattr(args, "reuse_decomposition_stage_cache", None)
    if reuse_stage_cache_arg is None:
        use_stage_cache = True
    else:
        use_stage_cache = bool(reuse_stage_cache_arg)

    overwrite_stage_cache_arg = getattr(args, "overwrite_decomposition_stage_cache", None)
    if overwrite_stage_cache_arg is None:
        overwrite_stage_cache = bool(getattr(args, "overwrite_bundle_cache", False))
    else:
        overwrite_stage_cache = bool(overwrite_stage_cache_arg)
    stage_cache_dir = getattr(args, "decomposition_stage_cache_dir", None)
    if not stage_cache_dir:
        if bundle_cache_dir:
            stage_cache_dir = os.path.join(bundle_cache_dir, "_build_stage_cache")
        else:
            stage_cache_dir = os.path.join(getattr(args, "tokenizer_cache_dir", "tokenizer_cache"), "_build_stage_cache")
    stage_cache_path = os.path.join(stage_cache_dir, f"decomposition_stage_{stage_cache_key}.pkl")

    print("Building surface-marker decomposition without morphology resources.")
    base_tokens, decomposition_map, types_to_words, ambiguous, conflicts, duplicates, derivation_chains = (
        _build_vocab_only_decomposition_map(
            base_tokenizer,
            args,
            decompose_articles=decompose_determiners,
            decompose_possessive_determiners=decompose_possessive_determiners,
            decompose_demonstrative_determiners=decompose_demonstrative_determiners,
            decompose_quantifier_determiners=decompose_quantifier_determiners,
            decompose_prepositions=decompose_prepositions,
            decompose_punctuation=args.decompose_punctuation,
            decompose_possessive_separate_group=decompose_possessive_separate_group,
            preposition_list=args.preposition_list,
        )
    )
    if use_stage_cache:
        os.makedirs(stage_cache_dir, exist_ok=True)
        stage_payload = {
            "base_tokens": base_tokens,
            "decomposition_map": decomposition_map,
            "types_to_words": types_to_words,
            "ambiguous": ambiguous,
            "conflicts": conflicts,
            "duplicates": duplicates,
            "derivation_chains": derivation_chains,
        }
        with open(stage_cache_path, "wb") as f:
            pickle.dump(stage_payload, f)
        print(f"Saved decomposition stage cache to {stage_cache_path}")

    syntactic_decomposition_map = {}
    if tokenization_mode == "dual_stream":
        materialize_syntactic_map = bool(
            getattr(args, "materialize_syntactic_decomposition_map", False)
        )
        if materialize_syntactic_map:
            syntactic_decomposition_map = _build_space_cap_syntactic_decomposition_from_vocab(
                base_tokenizer,
                decomposition_map,
                args,
                base_tokens=base_tokens,
            )
            if syntactic_decomposition_map:
                syntactic_variant_count = sum(len(v) for v in syntactic_decomposition_map.values())
                print(
                    "Built global space/cap map "
                    f"({len(syntactic_decomposition_map):,} bases, {syntactic_variant_count:,} variants)"
                )
        else:
            print(
                "Dual-stream mode: global space/cap transformations will be streamed "
                "directly into SequenceMap (syntactic map not materialized)."
            )

    article_space_prefix_enabled = phrase_space_prefix and decompose_determiners
    prep_space_prefix_enabled = phrase_space_prefix and decompose_prepositions

    type_names_to_int, types_loss_indices_map = get_label_maps_from_decomposition_map(
        decomposition_map,
        force_articles=decompose_determiners,
        force_possessive_determiners=decompose_possessive_determiners,
        force_demonstrative_determiners=decompose_demonstrative_determiners,
        force_quantifier_determiners=decompose_quantifier_determiners,
        force_prepositions=decompose_prepositions,
        force_punctuation=args.decompose_punctuation,
        force_possessives=decompose_possessive_separate_group,
        force_article_space_prefix=article_space_prefix_enabled,
        force_prep_space_prefix=prep_space_prefix_enabled,
        preposition_list=args.preposition_list,
    )

    if (
        tokenization_mode == "dual_stream"
        and (not getattr(args, "dont_decompose_capitalized", False))
        and bool(getattr(args, "enable_all_caps_capitalization", False))
        and ADD_ALL_CAPS_BASE_CAPITALIZATION_TRANSFORM not in type_names_to_int
    ):
        raise ValueError(
            "Expected all-caps capitalization transform is missing from transformation names. "
            "This usually indicates an out-of-date checkout or stale module import path."
        )

    active_morph_groups = []

    tokenization_mode = getattr(args, 'tokenization_mode', 'dual_stream')
    if tokenization_mode != "dual_stream":
        raise ValueError("nanochat cobpe.decomposition supports dual_stream tokenization only.")

    # Dual-stream mode always uses unified modifiers (2D modifier arrays)
    # This is the modern, recommended approach for compositional tokenization
    use_unified_modifiers = True

    # For dual-stream mode, we DON'T extend the tokenizer vocabulary
    # Instead, we use the base tokenizer as-is and do post-processing via DualStreamTokenizer
    print(f"\n=== DUAL-STREAM MODE: Using base tokenizer without vocabulary extension ===")
    print(f"Base tokenizer vocab size: {len(base_tokenizer)}")

    # Use base tokenizer directly (no extension)
    tokenizer = base_tokenizer
    transformation_names_to_int = type_names_to_int

    # Apply vocabulary pruning if requested.
    # NOTE: dual-stream path should never fallback to tokenizer-extension flow.
    pruned_token_ids = set()
    if getattr(args, 'prune_punct_tokens', False):
        print(f"Pruning tokens with boundary punctuation...")
        tokens_to_remove = identify_tokens_to_remove(base_tokenizer)
        pruned_token_ids = set(tokens_to_remove)
        print(f"  Found {len(pruned_token_ids)} tokens to prune from vocabulary")
        # Note: We don't actually modify the tokenizer vocabulary, but we exclude these
        # tokens from the decomposition map and mark them as unavailable.

    # Build a minimal decomposition map - just single-token inflections/derivations from base vocab.
    # This maps token_id -> (base_id, type_ids) for tokens that ARE in the base vocab.
    final_decomposition_map = {}
    existing_inflection_ids = set()
    negative_type_ids = []
    na_type_ids = []
    use_build_optimizations = bool(getattr(args, "enable_build_optimizations", False))
    single_token_id_cache = {}

    # Process decomposition_map to find single-token transformations in base vocab
    def _single_token_id(text):
        if not use_build_optimizations:
            token_ids = base_tokenizer.encode(text, add_special_tokens=False)
            if len(token_ids) != 1:
                return None
            return token_ids[0]
        cached = single_token_id_cache.get(text)
        if cached is not None or text in single_token_id_cache:
            return cached
        token_ids = base_tokenizer.encode(text, add_special_tokens=False)
        if len(token_ids) != 1:
            single_token_id_cache[text] = None
            return None
        token_id = token_ids[0]
        single_token_id_cache[text] = token_id
        return token_id

    skipped_missing_base = 0
    skipped_missing_base_examples = []
    for base_token_key, variants_dict in decomposition_map.items():
        for variant_str, transform_types in variants_dict.items():
            # Check if variant exists as single token in base vocab
            variant_id = _single_token_id(variant_str)
            if variant_id is not None:
                base_id = _single_token_id(base_token_key)
                if base_id is None:
                    base_override = base_tokens.get(base_token_key)
                    if isinstance(base_override, int):
                        base_id = base_override
                if base_id is None:
                    # No single-token base ID; let SequenceMap handle multi-token base expansion.
                    skipped_missing_base += 1
                    if len(skipped_missing_base_examples) < 5:
                        skipped_missing_base_examples.append((base_token_key, variant_str))
                    continue

                # Skip pruned tokens
                if variant_id in pruned_token_ids or base_id in pruned_token_ids:
                    continue

                # Store type IDs (not one-hot) to match model expectations
                type_ids = []
                for t in transform_types:
                    idx = transformation_names_to_int.get(t)
                    if idx is not None:
                        type_ids.append(idx)
                # De-duplicate while preserving order
                type_ids = list(dict.fromkeys(type_ids))

                final_decomposition_map[variant_id] = (base_id, type_ids)
                if variant_id != base_id:
                    existing_inflection_ids.add(variant_id)

    # Also include single-token global space/capitalization variants in final_decomposition_map
    # so base-vocab pruning can remove representable surface forms in dual-stream mode.
    syntactic_single_token_added = 0
    syntactic_single_token_replaced_self = 0
    syntactic_single_token_conflicts = 0
    decompose_spaces = not getattr(args, "dont_decompose_spaces", False)
    decompose_caps = not getattr(args, "dont_decompose_capitalized", False)
    if tokenization_mode == "dual_stream" and (decompose_spaces or decompose_caps):
        for (
            _base_key,
            base_ids,
            _base_rank,
            _variant_str,
            transforms,
            variant_ids,
        ) in _iter_space_cap_syntactic_candidates(
            base_tokenizer=base_tokenizer,
            decomposition_map=decomposition_map,
            decompose_spaces=decompose_spaces,
            decompose_caps=decompose_caps,
            use_relative=bool(getattr(args, "use_relative_space_cap_transforms", False)),
            enable_all_caps_capitalization=bool(
                getattr(args, "enable_all_caps_capitalization", False)
            ),
            base_tokens=base_tokens,
            max_variant_tokens=getattr(args, "max_tokens_per_base_word", None),
            enable_build_optimizations=use_build_optimizations,
            promote_surface_alias_bases=bool(getattr(args, "promote_surface_alias_bases", False)),
        ):
            if len(variant_ids) != 1 or len(base_ids) != 1:
                continue
            variant_id = int(variant_ids[0])
            base_id = int(base_ids[0])
            if variant_id in pruned_token_ids or base_id in pruned_token_ids:
                continue

            type_ids = []
            for t in transforms:
                idx = transformation_names_to_int.get(t)
                if idx is not None:
                    type_ids.append(idx)
            type_ids = list(dict.fromkeys(type_ids))

            existing = final_decomposition_map.get(variant_id)
            if existing is None:
                final_decomposition_map[variant_id] = (base_id, type_ids)
                if variant_id != base_id:
                    existing_inflection_ids.add(variant_id)
                syntactic_single_token_added += 1
                continue

            existing_base_id, existing_type_ids = existing
            existing_type_ids = list(existing_type_ids) if existing_type_ids is not None else []
            if existing_base_id == variant_id and base_id != variant_id:
                # Replace self-mapping with canonical syntactic mapping.
                merged_type_ids = list(dict.fromkeys(existing_type_ids + type_ids))
                final_decomposition_map[variant_id] = (base_id, merged_type_ids)
                existing_inflection_ids.add(variant_id)
                syntactic_single_token_replaced_self += 1
            elif existing_base_id == base_id:
                # Same base: keep mapping, enrich transform labels.
                merged_type_ids = list(dict.fromkeys(existing_type_ids + type_ids))
                final_decomposition_map[variant_id] = (base_id, merged_type_ids)
            else:
                # Ambiguous base assignment; keep existing entry.
                syntactic_single_token_conflicts += 1

    print(f"Found {len(final_decomposition_map)} single-token decompositions in base vocab")
    print(f"  (inflections that exist as single tokens: {len(existing_inflection_ids)})")
    if syntactic_single_token_added or syntactic_single_token_replaced_self or syntactic_single_token_conflicts:
        print(
            "  (space/cap single-token additions: "
            f"added={syntactic_single_token_added}, "
            f"replaced_self={syntactic_single_token_replaced_self}, "
            f"conflicts={syntactic_single_token_conflicts})"
        )
    if pruned_token_ids:
        print(f"  (excluded {len(pruned_token_ids)} pruned tokens with boundary punctuation)")
    if skipped_missing_base:
        print(
            f"  (skipped {skipped_missing_base} single-token variants with multi-token bases; "
            f"handled by SequenceMap)"
        )
        if skipped_missing_base_examples:
            print(f"  examples: {skipped_missing_base_examples}")

    # For dual-stream mode, populate negative_type_ids and na_type_ids from transformation names
    # These are used by the model to identify NO/NA transformations
    for name, idx in transformation_names_to_int.items():
        if name.startswith('no_') or name.startswith('NO_'):
            negative_type_ids.append(idx)
        elif name.startswith('NA_') or name.startswith('na_'):
            na_type_ids.append(idx)

    all_inflections = existing_inflection_ids
    # Keep only single-token base IDs for model mappings; multi-token bases are handled via SequenceMap.
    base_token_indices = [v for v in base_tokens.values() if not isinstance(v, tuple)]
    untouched_indices = set(base_tokenizer.get_vocab().values()) - set(base_token_indices) - set(
        existing_inflection_ids)
    untouched_indices = sorted(list(untouched_indices))
    non_inflection_indices = set(base_token_indices) | set(untouched_indices)

    base_or_inflection_token_ids = list(set(base_token_indices) | all_inflections)
    # Ensure IDs are integers; multi-token bases are handled via SequenceMap (no single token ID).
    base_or_inflection_token_ids = [v for v in base_or_inflection_token_ids if not isinstance(v, tuple)]
    token_id_to_base_id_mapping = {token_id: base_token_id for token_id, (base_token_id, _) in
                                   final_decomposition_map.items()}
    for untouched_i in untouched_indices:
        # In dual-stream mode, some tokens may already be in mapping, skip them
        if untouched_i not in token_id_to_base_id_mapping:
            token_id_to_base_id_mapping[untouched_i] = untouched_i

    # In dual-stream mode, the model's base vocab indices must cover every base_id that appears in
    # token_id_to_base_id_mapping.values(). Some tokens can be both "base-like" and also appear as
    # decomposable variants of other lemmas; in that case they may have been excluded from
    # non_inflection_indices, but they still must exist in the base-vocab index space to avoid
    # out-of-bounds errors in set_token_mappings().
    unique_base_ids = sorted(set(token_id_to_base_id_mapping.values()) | set(base_token_indices))
    if len(non_inflection_indices) != len(unique_base_ids):
        # Debug mismatch: show which base IDs are missing/extra (with token strings).
        inv_vocab = {v: k for k, v in base_tokenizer.get_vocab().items()}
        missing_base_ids = sorted(set(unique_base_ids) - set(non_inflection_indices))
        extra_non_inflection_ids = sorted(set(non_inflection_indices) - set(unique_base_ids))

        if missing_base_ids:
            print("NOTE: base IDs present in mapping but missing from non_inflection_indices:")
            for base_id in missing_base_ids:
                base_tok = inv_vocab.get(base_id, None)
                base_tok_str = repr(base_tok) if base_tok is not None else "<unk_id>"
                # Show a few example token IDs that map to this base_id
                example_ext_ids = [ext_id for ext_id, b_id in token_id_to_base_id_mapping.items() if b_id == base_id][:3]
                example_ext_toks = [repr(inv_vocab.get(eid, "<unk_id>")) for eid in example_ext_ids]
                print(f"  base_id={base_id} token={base_tok_str} example_ext={list(zip(example_ext_ids, example_ext_toks))}")

        if extra_non_inflection_ids:
            print("NOTE: base IDs present in non_inflection_indices but absent from mapping:")
            for base_id in extra_non_inflection_ids:
                base_tok = inv_vocab.get(base_id, None)
                base_tok_str = repr(base_tok) if base_tok is not None else "<unk_id>"
                print(f"  base_id={base_id} token={base_tok_str}")

    # Base vocab indices are defined by the set of unique base IDs reachable via token_id_to_base_id_mapping.
    # Keep model_init_data['non_inflection_indices'] consistent with that set (it's used to size the base embedding table).
    prev_non_inflection_size = len(non_inflection_indices)
    non_inflection_indices = set(unique_base_ids)
    if len(non_inflection_indices) != prev_non_inflection_size:
        print(
            f"NOTE: Adjusting non_inflection_indices size {prev_non_inflection_size} → {len(non_inflection_indices)} "
            f"to match unique base IDs (dual-stream correctness)."
        )

    transformation_int_to_name = {v: k for k, v in transformation_names_to_int.items()}

    type_groups = {}
    transform_groups = []
    for group_name, group_indices in types_loss_indices_map.items():
        transforms = []
        for i in range(group_indices[0], group_indices[1]):
            if i not in transformation_int_to_name:
                raise ValueError(
                    "Inconsistent transformation mappings: "
                    f"missing transform name for id={i} in group={group_name}. "
                    "This usually means non-unique transformation names or stale cache artifacts. "
                    "Try rebuilding with --overwrite_bundle_cache --overwrite_decomposition_stage_cache."
                )
            transforms.append({"name": transformation_int_to_name[i], "id": i})
        transform_groups.append({"name": group_name, "transforms": transforms})
        type_groups[group_name] = len(transforms)



    # Calculate vocab stats
    inflections_single_token = set()
    tokenizer_vocab_ids = set(base_tokenizer.get_vocab().values())
    for inflection_token_id, (base_token_id, type_ids) in final_decomposition_map.items():
        if inflection_token_id == base_token_id:
            continue
        if inflection_token_id in tokenizer_vocab_ids:
            inflections_single_token.add(inflection_token_id)

    model_init_data = {
        'base_tokenizer_size': len(base_tokenizer),
        'base_token_indices': base_token_indices,
        'base_tokens': base_tokens,
        'final_decomposition_map': final_decomposition_map,
        'transform_groups': transform_groups,
        'untouched_indices': untouched_indices,
        'non_inflection_indices': sorted(list(non_inflection_indices)),
        'token_id_to_base_id_mapping': token_id_to_base_id_mapping,
        'base_or_inflection_token_ids': base_or_inflection_token_ids,
        'type_groups': type_groups,
        'transformation_names_to_int': transformation_names_to_int,
        'types_loss_indices_map': types_loss_indices_map,
        'decomposition_map': decomposition_map,
        'syntactic_decomposition_map': syntactic_decomposition_map,
        'negative_type_ids': negative_type_ids,
        'na_type_ids': na_type_ids,
        'all_inflections': list(all_inflections),
        'inflections_single_token': list(inflections_single_token),
        'active_morphological_groups': active_morph_groups,
        'decomposition_filtered': duplicates,
    }
    print(f"# base tokens: {len(base_token_indices)} - # untouched: {len(untouched_indices)} - # single token inflections: {len(inflections_single_token)} - # all inflections: {len(all_inflections)}")

    tokenization_mode = getattr(args, 'tokenization_mode', 'dual_stream')

    unified_modifier_array = None
    dual_stream_tokenizer_config = None

    print(f"Creating UnifiedModifierArray for 2D modifier format...")
    if tokenization_mode == 'dual_stream':
        def _is_default_or_na_transform(name: str) -> bool:
            lower = str(name).lower()
            return lower.startswith("no_") or lower.startswith("na_")

        def _group_has_non_default_values(group_name: str) -> bool:
            if group_name not in types_loss_indices_map:
                return False
            start, end = types_loss_indices_map[group_name]
            if end <= start:
                return False
            names_in_group = [transformation_int_to_name.get(i, "") for i in range(start, end)]
            return any(name and not _is_default_or_na_transform(name) for name in names_in_group)

        # Determine which groups are active based on args
        active_groups = []
        if not args.dont_decompose_spaces:
            active_groups.append('space_prefix')
        if not args.dont_decompose_capitalized:
            active_groups.append('base_capitalization')
        if decompose_determiners:
            active_groups.append(DETERMINER_GROUP_NAME)
            if article_space_prefix_enabled:
                active_groups.append('article_space_prefix')
            active_groups.append('article_capitalization')
        if args.decompose_prepositions:
            active_groups.append('prepositions')
            if prep_space_prefix_enabled:
                active_groups.append('prep_space_prefix')
            active_groups.append('prep_capitalization')
        if args.decompose_punctuation:
            active_groups.extend(['prefix_punctuation', 'suffix_punctuation'])
        if decompose_possessive_separate_group:
            active_groups.append('possessives')

        # Generic empty-group pruning: remove groups that have no learnable value
        # (only NO/NA defaults) or missing index ranges.
        pruned_active_groups = []
        for group in active_groups:
            if group not in types_loss_indices_map:
                print(f"  Skipping missing group: {group}")
                continue
            if not _group_has_non_default_values(group):
                start, end = types_loss_indices_map[group]
                names_in_group = [transformation_int_to_name.get(i, "") for i in range(start, end)]
                print(f"  Skipping empty group: {group} ({names_in_group})")
                continue
            pruned_active_groups.append(group)
        active_groups = pruned_active_groups

        print(f"  Active transformation groups: {active_groups}")

        # Filter types_loss_indices_map to only include active groups
        filtered_loss_indices_map = {
            group: types_loss_indices_map[group]
            for group in active_groups
            if group in types_loss_indices_map
        }

        # Create UnifiedModifierArray with filtered loss indices
        unified_modifier_array = UnifiedModifierArray(
            groups=active_groups,
            types_loss_indices_map=filtered_loss_indices_map
        )
        print(f"  Created: {unified_modifier_array}")

        # Update model_init_data with filtered loss indices to match UnifiedModifierArray
        model_init_data['types_loss_indices_map'] = filtered_loss_indices_map

        # Rebuild transformation_names_to_int from scratch with sequential indices
        # Build a list of (name, new_idx) tuples by iterating through filtered_loss_indices_map in order
        new_transformation_names = []

        # Get the original transformation_names list (in order)
        # We need to reconstruct this from transformation_names_to_int
        old_names_by_idx = {idx: name for name, idx in transformation_names_to_int.items()}

        # For each group in filtered_loss_indices_map (in the order they appear)
        for group in active_groups:
            if group in filtered_loss_indices_map:
                new_start, new_end = filtered_loss_indices_map[group]
                old_start, old_end = types_loss_indices_map[group]

                # Add transformation names for this group in order
                for offset in range(old_end - old_start):
                    old_idx = old_start + offset
                    if old_idx in old_names_by_idx:
                        name = old_names_by_idx[old_idx]
                        new_transformation_names.append(name)

        # Build new transformation_names_to_int with sequential indices
        new_transformation_names_to_int = {name: idx for idx, name in enumerate(new_transformation_names)}

        # Rebuild filtered_loss_indices_map with new sequential indices.
        # active_groups already excludes empty/no-op-only groups.
        new_filtered_loss_indices_map = {}
        current_idx = 0
        non_empty_groups = []
        for group in active_groups:
            if group in filtered_loss_indices_map:
                old_start, old_end = filtered_loss_indices_map[group]
                group_size = old_end - old_start
                if group_size > 0:
                    new_filtered_loss_indices_map[group] = (current_idx, current_idx + group_size)
                    current_idx += group_size
                    non_empty_groups.append(group)

        # Update active_groups to only include non-empty groups
        active_groups = non_empty_groups

        # Update model_init_data with filtered transformation_names_to_int and indices
        model_init_data['transformation_names_to_int'] = new_transformation_names_to_int
        model_init_data['types_loss_indices_map'] = new_filtered_loss_indices_map  # Use rebuilt version
        transformation_names_to_int = new_transformation_names_to_int  # Update local variable too
        filtered_loss_indices_map = new_filtered_loss_indices_map  # Update local variable too

        # Rebuild type_groups using new sequential indices (only for non-empty groups)
        new_type_groups = {}
        new_transform_groups = []
        for group_name in active_groups:
            if group_name in new_filtered_loss_indices_map:
                start, end = new_filtered_loss_indices_map[group_name]
                new_type_groups[group_name] = end - start
                # Build transforms list for this group
                transforms = []
                for i in range(start, end):
                    # Find the name for this index
                    for name, idx in new_transformation_names_to_int.items():
                        if idx == i:
                            transforms.append({"name": name, "id": i})
                            break
                new_transform_groups.append({"name": group_name, "transforms": transforms})

        model_init_data['type_groups'] = new_type_groups
        model_init_data['transform_groups'] = new_transform_groups
        type_groups = new_type_groups  # Update local variable too
        transform_groups = new_transform_groups  # Update local variable too

        # Store the active groups in model_init_data for model initialization
        model_init_data['unified_modifier_groups'] = active_groups

        # Recreate UnifiedModifierArray with new sequential indices
        unified_modifier_array = UnifiedModifierArray(
            groups=active_groups,
            types_loss_indices_map=new_filtered_loss_indices_map
        )
        print(f"  Recreated UnifiedModifierArray with sequential indices: {unified_modifier_array}")
        modifier_dtype_name = unified_modifier_array.recommended_modifier_dtype_name()
        model_init_data['unified_modifier_dtype'] = modifier_dtype_name
        print(f"  Modifier dtype: {modifier_dtype_name}")

        # Rebuild na_type_ids and negative_type_ids from the new transformation names
        # (simpler and more correct than trying to remap old indices)
        new_negative_type_ids = []
        new_na_type_ids = []
        for name, idx in new_transformation_names_to_int.items():
            if name.startswith('no_') or name.startswith('NO_'):
                new_negative_type_ids.append(idx)
            elif name.startswith('NA_') or name.startswith('na_'):
                new_na_type_ids.append(idx)

        model_init_data['negative_type_ids'] = new_negative_type_ids
        model_init_data['na_type_ids'] = new_na_type_ids

        # Remap final_decomposition_map type IDs to the new sequential indices
        if final_decomposition_map:
            old_total_types = len(old_names_by_idx)

            def _remap_type_ids(type_ids):
                if type_ids is None or isinstance(type_ids, str):
                    return []
                if hasattr(type_ids, "tolist"):
                    type_ids = type_ids.tolist()
                try:
                    size = len(type_ids)
                except TypeError:
                    return []
                if size == 0:
                    return []

                mapped = []
                is_one_hot = False
                if size == old_total_types:
                    is_one_hot = True
                    for val in type_ids:
                        if isinstance(val, bool):
                            continue
                        if isinstance(val, int):
                            if val not in (0, 1):
                                is_one_hot = False
                                break
                        elif isinstance(val, float):
                            if val < -1e-6 or val > 1.0 + 1e-6:
                                is_one_hot = False
                                break
                        else:
                            is_one_hot = False
                            break

                if is_one_hot:
                    indices = (i for i, val in enumerate(type_ids) if val > 0.5)
                else:
                    indices = (int(v.item() if hasattr(v, "item") else v) for v in type_ids)

                for idx in indices:
                    name = old_names_by_idx.get(idx)
                    if name is None:
                        continue
                    new_idx = new_transformation_names_to_int.get(name)
                    if new_idx is not None:
                        mapped.append(new_idx)

                return list(dict.fromkeys(mapped))

            remapped = {}
            for token_id, (base_id, type_ids) in final_decomposition_map.items():
                remapped[token_id] = (base_id, _remap_type_ids(type_ids))
            final_decomposition_map = remapped
            model_init_data['final_decomposition_map'] = final_decomposition_map

        # Create DualStreamTokenizer config for later instantiation
        # This contains the info needed to build the tokenizer wrapper
        dual_stream_tokenizer_config = {
            'active_groups': active_groups,
            'types_loss_indices_map': new_filtered_loss_indices_map,  # Use rebuilt version with sequential indices
            'transformation_names_to_int': transformation_names_to_int,
            'modifier_dtype': model_init_data.get('unified_modifier_dtype'),
            # Preserve tokenizer behavior knobs for metadata export and analysis.
            'decompose_articles': bool(decompose_determiners),
            'decompose_possessive_determiners': bool(decompose_possessive_determiners),
            'decompose_demonstrative_determiners': bool(decompose_demonstrative_determiners),
            'decompose_quantifier_determiners': bool(decompose_quantifier_determiners),
            'decompose_prepositions': bool(args.decompose_prepositions),
            'decompose_punctuation': bool(args.decompose_punctuation),
            'possessive_separate_group': bool(decompose_possessive_separate_group),
            'preposition_profile': str(getattr(args, "preposition_profile", "off")),
            'preposition_list': list(args.preposition_list) if args.preposition_list is not None else None,
            'word_boundary_safety': bool(args.word_boundary_safety),
            'multi_token_modifier_position': str(args.multi_token_modifier_position),
            'enable_all_caps_capitalization': bool(getattr(args, "enable_all_caps_capitalization", False)),
            # Article/preposition/punctuation token IDs are rebuilt at runtime.
        }

        print(f"  Stored dual_stream_tokenizer_config in bundle")

        # Build SequenceMap during bundle creation (one-time cost)
        # This avoids rebuilding it on every tokenization run
        sequence_map = _build_sequence_map_for_bundle(
            tokenizer=tokenizer,
            model_init_data=model_init_data,
            unified_modifier_array=unified_modifier_array,
            args=args
        )
    else:
        sequence_map = None

    return CompositionalTokenizerBundle(tokenizer, model_init_data, tokenization_mode,
                                        unified_modifier_array, dual_stream_tokenizer_config, sequence_map)


def get_or_create_tokenizer_bundle(args, cache_base_dir="tokenizer_cache"):
    cache_descriptor = _compute_tokenizer_cache_descriptor(args, cache_base_dir)
    cache_dir = cache_descriptor["cache_path"]
    cache_details_path = os.path.join(cache_dir, "cache_key_details.json")
    def _persist_cache_details():
        _write_json(
            cache_details_path,
            {
                "cache_subdir": cache_descriptor["cache_subdir"],
                "cache_hash": cache_descriptor["cache_hash"],
                "compact_features": cache_descriptor["compact_features"],
                "full_features": cache_descriptor["full_features"],
                "key_payload": cache_descriptor["key_payload"],
            },
        )

    if args.overwrite_bundle_cache and os.path.exists(cache_dir):
        print(f"Removing cached tokenizer bundle at {cache_dir}")
        if os.path.isdir(cache_dir):
            shutil.rmtree(cache_dir)
        else:
            os.remove(cache_dir)

    if not args.overwrite_bundle_cache and os.path.exists(cache_dir) and os.path.exists(os.path.join(cache_dir, "tokenizer")) and os.path.exists(
            os.path.join(cache_dir, "model_init_data.pkl")):
        print(f"Loading cached tokenizer bundle from {cache_dir}")
        cached_bundle = CompositionalTokenizerBundle.load(cache_dir, base_tokenizer_name=args.base_tokenizer)
        if not os.path.exists(cache_details_path):
            _persist_cache_details()
        if getattr(args, "tokenization_mode", "dual_stream") == "dual_stream" and cached_bundle.sequence_map is None:
            print("Cached bundle missing SequenceMap for dual_stream; rebuilding cache entry.")
            bundle = create_compositional_tokenizer_bundle(args, bundle_cache_dir=cache_dir)
            bundle.save(cache_dir)
            _persist_cache_details()
            return bundle
        if (
            getattr(args, "tokenization_mode", "dual_stream") == "dual_stream"
            and not getattr(args, "dont_decompose_capitalized", False)
            and bool(getattr(args, "enable_all_caps_capitalization", False))
        ):
            transformation_names = cached_bundle.model_init_data.get("transformation_names_to_int", {})
            if ADD_ALL_CAPS_BASE_CAPITALIZATION_TRANSFORM not in transformation_names:
                print("Cached bundle missing all-caps capitalization transform; rebuilding cache entry.")
                bundle = create_compositional_tokenizer_bundle(args, bundle_cache_dir=cache_dir)
                bundle.save(cache_dir)
                _persist_cache_details()
                return bundle
        return cached_bundle

    print(f"Creating and caching tokenizer bundle to {cache_dir}")
    bundle = create_compositional_tokenizer_bundle(args, bundle_cache_dir=cache_dir)
    bundle.save(cache_dir)
    _persist_cache_details()
    return bundle
