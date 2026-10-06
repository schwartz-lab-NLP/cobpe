"""
Dual-Stream Tokenizer for Compositional Tokenization.

This module provides a tokenizer wrapper that returns both token IDs and modifier arrays
for dual-stream compositional training. It handles:
- Multi-token sequence detection (e.g., "walk" + "ing" -> base="walk", modifier=GERUND)
- Article/preposition/punctuation attachment
- Efficient stream processing with trie-based lookups
"""

import os
import time
import numpy as np
from typing import Dict, List, Tuple, Optional, Set, Any
from collections import defaultdict
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast

from .surface import (
    UNIFIED_TRANSFORM_GROUPS,
    UnifiedModifierArray,
    NO_SPACE_PREFIX_TRANSFORM,
    WITH_SPACE_PREFIX_TRANSFORM,
    NO_BASE_CAPITALIZATION_TRANSFORM,
    ADD_BASE_CAPITALIZATION_TRANSFORM,
    NO_INFLECTION,
    NO_DERIVATION,
    NO_ARTICLE,
    NO_PREPOSITION,
    NO_PREP_CAPITALIZATION,
    NO_ARTICLE_CAPITALIZATION,
    NO_PREFIX_PUNCTUATION,
    NO_SUFFIX_PUNCTUATION,
    ARTICLE_THE,
    ARTICLE_A,
    ARTICLE_AN,
    ADD_ARTICLE_CAPITALIZATION,
    ADD_PREP_CAPITALIZATION,
)

_MODIFIER_DTYPE_BY_NAME = {
    "uint8": np.uint8,
    "uint16": np.uint16,
    "uint32": np.uint32,
    "uint64": np.uint64,
}


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


def _punct_surface_from_transform_name(name: str) -> Optional[str]:
    if not name:
        return None
    if name.startswith("punct_prefix_"):
        return name[len("punct_prefix_"):]
    if name.startswith("punct_suffix_"):
        return name[len("punct_suffix_"):]
    if name.startswith("possessive_"):
        return name[len("possessive_"):]
    return None


def _resolve_modifier_dtype(dtype_like: Any, fallback_name: str) -> Tuple[Any, str]:
    if dtype_like is None:
        name = fallback_name
    elif isinstance(dtype_like, str):
        name = dtype_like.strip().lower()
    elif isinstance(dtype_like, np.dtype):
        name = dtype_like.name
    else:
        name = np.dtype(dtype_like).name
    if name not in _MODIFIER_DTYPE_BY_NAME:
        raise ValueError(f"Unsupported modifier dtype: {dtype_like!r}")
    return _MODIFIER_DTYPE_BY_NAME[name], name


def _build_global_index_to_group_map(
    types_loss_indices_map: Dict[str, Tuple[int, int]],
    groups: List[str],
) -> List[Optional[Tuple[str, int]]]:
    if not types_loss_indices_map:
        return []
    max_end = max(end for start, end in types_loss_indices_map.values())
    mapping: List[Optional[Tuple[str, int]]] = [None] * max_end
    for group_name in groups:
        if group_name not in types_loss_indices_map:
            continue
        start, end = types_loss_indices_map[group_name]
        for global_idx in range(start, end):
            mapping[global_idx] = (group_name, global_idx - start)
    return mapping


def _iter_active_global_indices(type_ids: Any, total_types: int):
    if type_ids is None or isinstance(type_ids, str):
        return ()
    try:
        length = len(type_ids)
    except TypeError:
        return ()
    if length == 0:
        return ()

    if length == total_types:
        is_one_hot = True
        for val in type_ids:
            if hasattr(val, "item"):
                val = val.item()
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
            return (i for i, val in enumerate(type_ids) if (val.item() if hasattr(val, "item") else val) > 0.5)

    return (int(v.item() if hasattr(v, "item") else v) for v in type_ids)


class TrieNode:
    """Node in a trie structure for efficient sequence matching."""

    def __init__(self):
        self.children: Dict[int, 'TrieNode'] = {}
        self.is_end: bool = False
        # If this is the end of a sequence, store the mapping
        self.base_token_ids: Optional[List[int]] = None
        self.modifier: Optional[List[int]] = None


class SequenceMap:
    """Maps token sequences to their base forms and modifiers using a trie.

    This enables efficient O(n) processing of token streams by finding the longest
    matching sequence at each position.

    Example:
        sequence_map.add((" walk", "ing"), (" walk",), (space=ADD, cap=NO, infl=GERUND, ...))
        sequence_map.find_longest([" walk", "ing", "to", "the", "park"])
        -> (2, [" walk"], modifier_tuple)
    """

    def __init__(
        self,
        modifier_array_manager: UnifiedModifierArray,
        disallowed_match_token_ids: Optional[Set[int]] = None,
    ):
        """Initialize the sequence map.

        Args:
            modifier_array_manager: UnifiedModifierArray instance for managing modifiers.
        """
        self.root = TrieNode()
        self.modifier_manager = modifier_array_manager
        self.max_sequence_length = 0
        self.disallowed_match_token_ids: Set[int] = set(disallowed_match_token_ids or set())
        self.skipped_disallowed_sequences = 0

    def add_sequence(self, token_ids: Tuple[int, ...], base_ids: Tuple[int, ...],
                     modifier: List[int]) -> None:
        """Add a token sequence mapping.

        Args:
            token_ids: Tuple of token IDs that form the input sequence.
            base_ids: Tuple of base token IDs (before transformation).
            modifier: List of group-relative indices for each transformation group.
        """
        if len(token_ids) == 0:
            return
        if self.disallowed_match_token_ids and any(
            int(tid) in self.disallowed_match_token_ids for tid in token_ids
        ):
            self.skipped_disallowed_sequences += 1
            return

        node = self.root
        for tid in token_ids:
            if tid not in node.children:
                node.children[tid] = TrieNode()
            node = node.children[tid]

        node.is_end = True
        node.base_token_ids = list(base_ids)
        node.modifier = modifier

        self.max_sequence_length = max(self.max_sequence_length, len(token_ids))

    def find_longest_match(self, token_ids: List[int], start_idx: int) -> Tuple[int, Optional[List[int]], Optional[List[int]]]:
        """Find the longest matching sequence starting at start_idx.

        Args:
            token_ids: Full list of token IDs.
            start_idx: Starting position in the token list.

        Returns:
            Tuple of (match_length, base_token_ids, modifier).
            If no match, returns (0, None, None).
        """
        node = self.root
        best_match_length = 0
        best_base_ids = None
        best_modifier = None

        for i in range(start_idx, min(start_idx + self.max_sequence_length, len(token_ids))):
            tid = token_ids[i]
            if tid not in node.children:
                break

            node = node.children[tid]
            if node.is_end:
                best_match_length = i - start_idx + 1
                best_base_ids = node.base_token_ids
                best_modifier = node.modifier

        return best_match_length, best_base_ids, best_modifier

    def find_matches(self, token_ids: List[int], start_idx: int) -> List[Tuple[int, Optional[List[int]], Optional[List[int]]]]:
        """Find all matching sequences starting at start_idx, longest first."""
        node = self.root
        matches: List[Tuple[int, Optional[List[int]], Optional[List[int]]]] = []

        for i in range(start_idx, min(start_idx + self.max_sequence_length, len(token_ids))):
            tid = token_ids[i]
            if tid not in node.children:
                break

            node = node.children[tid]
            if node.is_end:
                match_len = i - start_idx + 1
                matches.append((match_len, node.base_token_ids, node.modifier))

        matches.sort(key=lambda x: x[0], reverse=True)
        return matches

    def get_exact_match(self, token_ids: Tuple[int, ...]) -> Optional[Tuple[List[int], List[int]]]:
        """Get an exact sequence mapping if present."""
        if len(token_ids) == 0:
            return None
        node = self.root
        for tid in token_ids:
            if tid not in node.children:
                return None
            node = node.children[tid]
        if not node.is_end or node.base_token_ids is None or node.modifier is None:
            return None
        return node.base_token_ids, node.modifier

    def __len__(self):
        """Return the number of sequences in the trie."""
        def count_sequences(node):
            count = 1 if node.is_end else 0
            for child in node.children.values():
                count += count_sequences(child)
            return count
        return count_sequences(self.root)

    def to_dict(self) -> Dict:
        """Serialize the sequence map to a dictionary."""
        sequences = []

        def traverse(node, path):
            if node.is_end:
                sequences.append({
                    'token_ids': path,
                    'base_ids': node.base_token_ids,
                    'modifier': node.modifier
                })
            for tid, child in node.children.items():
                traverse(child, path + [tid])

        traverse(self.root, [])
        return {
            'sequences': sequences,
            'max_sequence_length': self.max_sequence_length,
            'modifier_manager': self.modifier_manager.to_dict(),
            'disallowed_match_token_ids': sorted(int(x) for x in self.disallowed_match_token_ids),
            'skipped_disallowed_sequences': int(self.skipped_disallowed_sequences),
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'SequenceMap':
        """Deserialize from dictionary."""
        modifier_manager = UnifiedModifierArray.from_dict(data['modifier_manager'])
        seq_map = cls(
            modifier_manager,
            disallowed_match_token_ids=set(int(x) for x in data.get('disallowed_match_token_ids', [])),
        )
        seq_map.skipped_disallowed_sequences = int(data.get('skipped_disallowed_sequences', 0))

        for entry in data.get('sequences', []):
            seq_map.add_sequence(
                tuple(entry['token_ids']),
                tuple(entry['base_ids']),
                entry['modifier']
            )

        return seq_map


def _build_non_space_whitespace_token_ids(base_tokenizer: PreTrainedTokenizer) -> Set[int]:
    """Return token IDs with mixed text that include non-space whitespace.

    We intentionally keep *pure whitespace* tokens (e.g., "\n", "\n\n", "\t\t")
    eligible for SequenceMap matches so whitespace-run merges can be represented.
    """
    ids: Set[int] = set()
    vocab = base_tokenizer.get_vocab()
    special_ids = set(getattr(base_tokenizer, "all_special_ids", []) or [])
    for _tok, token_id in vocab.items():
        token_id = int(token_id)
        if token_id in special_ids:
            continue
        decoded = base_tokenizer.decode(
            [token_id],
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        if not decoded:
            continue
        has_non_space_ws = any(ch.isspace() and ch != " " for ch in decoded)
        if not has_non_space_ws:
            continue
        has_non_ws_char = any(not ch.isspace() for ch in decoded)
        if has_non_ws_char:
            ids.add(token_id)
    return ids


def build_sequence_map_from_decomposition(
    base_tokenizer: PreTrainedTokenizer,
    decomposition_map: Dict[int, Tuple[int, Any]],
    transformation_names_to_int: Dict[str, int],
    types_loss_indices_map: Dict[str, Tuple[int, int]],
    article_token_ids: Optional[Set[int]] = None,
    preposition_token_ids: Optional[Set[int]] = None,
    prefix_punct_token_ids: Optional[Set[int]] = None,
    suffix_punct_token_ids: Optional[Set[int]] = None,
    article_transforms: Optional[Dict[int, str]] = None,
    preposition_transforms: Optional[Dict[int, str]] = None,
    prefix_punct_transforms: Optional[Dict[int, str]] = None,
    suffix_punct_transforms: Optional[Dict[int, str]] = None,
    groups: Optional[List[str]] = None,
    prune_non_space_whitespace_sequences: bool = True,
) -> SequenceMap:
    """Build a sequence map from an existing decomposition map.

    This function converts the decomposition_map (which maps extended token IDs
    to base IDs and transformation IDs or one-hot vectors) into a sequence map
    that can detect multi-token sequences.

    Args:
        base_tokenizer: The base tokenizer for encoding/decoding.
        decomposition_map: Dict mapping token_id -> (base_id, type_ids).
        transformation_names_to_int: Mapping from transform names to global indices.
        types_loss_indices_map: Mapping from group names to (start, end) indices.
        article_token_ids: Set of token IDs that are articles.
        preposition_token_ids: Set of token IDs that are prepositions.
        prefix_punct_token_ids: Set of token IDs that are prefix punctuation.
        suffix_punct_token_ids: Set of token IDs that are suffix punctuation.
        article_transforms: Dict mapping article token_id to transform name.
        preposition_transforms: Dict mapping preposition token_id to transform name.
        prefix_punct_transforms: Dict mapping prefix punct token_id to transform name.
        suffix_punct_transforms: Dict mapping suffix punct token_id to transform name.
        groups: List of active transformation group names (defaults to UNIFIED_TRANSFORM_GROUPS).

    Returns:
        SequenceMap instance with all transformation sequences.
    """
    if groups is None:
        groups = UNIFIED_TRANSFORM_GROUPS

    global_idx_to_group = _build_global_index_to_group_map(types_loss_indices_map, groups)
    total_types = len(global_idx_to_group)

    modifier_manager = UnifiedModifierArray(
        groups=groups,
        types_loss_indices_map=types_loss_indices_map
    )

    disallowed_match_token_ids = (
        _build_non_space_whitespace_token_ids(base_tokenizer)
        if prune_non_space_whitespace_sequences
        else set()
    )
    sequence_map = SequenceMap(
        modifier_manager,
        disallowed_match_token_ids=disallowed_match_token_ids,
    )

    # Build inverse mapping from transform name to group-relative index
    transform_to_group_idx = {}
    for group_name in groups:
        if group_name in types_loss_indices_map:
            start, end = types_loss_indices_map[group_name]
            int_to_name = {v: k for k, v in transformation_names_to_int.items()}
            for i in range(start, end):
                if i in int_to_name:
                    transform_to_group_idx[int_to_name[i]] = (group_name, i - start)

    # Process the decomposition map - this contains single token mappings
    for extended_token_id, (base_token_id, type_ids) in decomposition_map.items():
        # Create modifier from type_ids
        modifier = modifier_manager.create_empty_modifier()

        # Extract transformations from type_ids (list of indices or one-hot vector)
        for global_idx in _iter_active_global_indices(type_ids, total_types):
            if 0 <= global_idx < total_types:
                group_info = global_idx_to_group[global_idx]
                if group_info is not None:
                    group_name, rel_idx = group_info
                    modifier_manager.set_group_value(modifier, group_name, rel_idx)

        # Add single token sequence
        sequence_map.add_sequence((extended_token_id,), (base_token_id,), modifier)

    # Multi-token sequences will be added by add_multitoken_sequences_from_linguistic_db()

    # Add article token mappings
    if article_token_ids and article_transforms:
        for token_id in article_token_ids:
            if token_id in article_transforms:
                transform_value = article_transforms[token_id]
                # Support both string (transform name) and tuple (group_name, rel_idx) formats
                if isinstance(transform_value, tuple):
                    group_name, rel_idx = transform_value
                    modifier = modifier_manager.create_empty_modifier()
                    modifier_manager.set_group_value(modifier, group_name, rel_idx)
                    sequence_map.add_sequence((token_id,), (token_id,), modifier)
                elif transform_value in transform_to_group_idx:
                    group_name, rel_idx = transform_to_group_idx[transform_value]
                    modifier = modifier_manager.create_empty_modifier()
                    modifier_manager.set_group_value(modifier, group_name, rel_idx)
                    # Article tokens map to themselves as markers
                    sequence_map.add_sequence((token_id,), (token_id,), modifier)

    # Add preposition token mappings
    if preposition_token_ids and preposition_transforms:
        for token_id in preposition_token_ids:
            if token_id in preposition_transforms:
                transform_value = preposition_transforms[token_id]
                if isinstance(transform_value, tuple):
                    group_name, rel_idx = transform_value
                    modifier = modifier_manager.create_empty_modifier()
                    modifier_manager.set_group_value(modifier, group_name, rel_idx)
                    sequence_map.add_sequence((token_id,), (token_id,), modifier)
                elif transform_value in transform_to_group_idx:
                    group_name, rel_idx = transform_to_group_idx[transform_value]
                    modifier = modifier_manager.create_empty_modifier()
                    modifier_manager.set_group_value(modifier, group_name, rel_idx)
                    sequence_map.add_sequence((token_id,), (token_id,), modifier)

    # Add punctuation token mappings
    if prefix_punct_token_ids and prefix_punct_transforms:
        for token_id in prefix_punct_token_ids:
            if token_id in prefix_punct_transforms:
                transform_value = prefix_punct_transforms[token_id]
                if isinstance(transform_value, tuple):
                    group_name, rel_idx = transform_value
                    modifier = modifier_manager.create_empty_modifier()
                    modifier_manager.set_group_value(modifier, group_name, rel_idx)
                    sequence_map.add_sequence((token_id,), (token_id,), modifier)
                elif transform_value in transform_to_group_idx:
                    group_name, rel_idx = transform_to_group_idx[transform_value]
                    modifier = modifier_manager.create_empty_modifier()
                    modifier_manager.set_group_value(modifier, group_name, rel_idx)
                    sequence_map.add_sequence((token_id,), (token_id,), modifier)

    if suffix_punct_token_ids and suffix_punct_transforms:
        for token_id in suffix_punct_token_ids:
            if token_id in suffix_punct_transforms:
                transform_value = suffix_punct_transforms[token_id]
                if isinstance(transform_value, tuple):
                    group_name, rel_idx = transform_value
                    modifier = modifier_manager.create_empty_modifier()
                    modifier_manager.set_group_value(modifier, group_name, rel_idx)
                    sequence_map.add_sequence((token_id,), (token_id,), modifier)
                elif transform_value in transform_to_group_idx:
                    group_name, rel_idx = transform_to_group_idx[transform_value]
                    modifier = modifier_manager.create_empty_modifier()
                    modifier_manager.set_group_value(modifier, group_name, rel_idx)
                    sequence_map.add_sequence((token_id,), (token_id,), modifier)

    return sequence_map


def build_sequence_map_from_string_decomposition(
    base_tokenizer: PreTrainedTokenizer,
    decomposition_map: Dict[str, Dict[str, List[str]]],
    transformation_names_to_int: Dict[str, int],
    types_loss_indices_map: Dict[str, Tuple[int, int]],
    groups: Optional[List[str]] = None,
    space_prefix: str = " ",
    max_tokens_per_word: Optional[int] = 3,
    sequence_map: Optional[SequenceMap] = None,
    only_multi_token_base: bool = False,
    base_tokens: Optional[Dict[str, Any]] = None,
    overwrite_existing: bool = True,
    enable_build_optimizations: bool = False,
    prune_non_space_whitespace_sequences: bool = True,
) -> SequenceMap:
    """Build or extend a sequence map from a string-based decomposition map.

    This uses text forms (with spaces/casing) and encodes them to token IDs,
    enabling multi-token base forms and variants.
    """
    if groups is None:
        groups = UNIFIED_TRANSFORM_GROUPS

    if sequence_map is None:
        modifier_manager = UnifiedModifierArray(
            groups=groups,
            types_loss_indices_map=types_loss_indices_map
        )
        disallowed_match_token_ids = (
            _build_non_space_whitespace_token_ids(base_tokenizer)
            if prune_non_space_whitespace_sequences
            else set()
        )
        sequence_map = SequenceMap(
            modifier_manager,
            disallowed_match_token_ids=disallowed_match_token_ids,
        )
    else:
        modifier_manager = sequence_map.modifier_manager

    # Map transform name -> (group_name, rel_idx)
    transform_to_group_idx = {}
    int_to_name = {v: k for k, v in transformation_names_to_int.items()}
    for group_name in groups:
        if group_name in types_loss_indices_map:
            start, end = types_loss_indices_map[group_name]
            for i in range(start, end):
                if i in int_to_name:
                    transform_to_group_idx[int_to_name[i]] = (group_name, i - start)

    use_build_optimizations = bool(enable_build_optimizations)
    encoded_text_cache: Dict[str, Tuple[int, ...]] = {}

    def _encode_cached(text: str) -> Tuple[int, ...]:
        if not use_build_optimizations:
            return tuple(base_tokenizer.encode(text, add_special_tokens=False))
        cached = encoded_text_cache.get(text)
        if cached is not None:
            return cached
        encoded = tuple(base_tokenizer.encode(text, add_special_tokens=False))
        encoded_text_cache[text] = encoded
        return encoded

    for base_word, variants in decomposition_map.items():
        if base_tokens and base_word in base_tokens:
            base_ids_value = base_tokens[base_word]
            if isinstance(base_ids_value, int):
                base_ids_tuple = (base_ids_value,)
            elif isinstance(base_ids_value, tuple):
                base_ids_tuple = base_ids_value
            else:
                base_ids_tuple = tuple(base_ids_value)
        else:
            base_ids_tuple = _encode_cached(base_word)
        if len(base_ids_tuple) == 0:
            continue
        if max_tokens_per_word is not None and len(base_ids_tuple) > max_tokens_per_word:
            continue

        for variant_str, transforms in variants.items():
            if variant_str == base_word:
                continue
            variant_ids_tuple = _encode_cached(variant_str)
            if len(variant_ids_tuple) == 0:
                continue
            if max_tokens_per_word is not None and len(variant_ids_tuple) > max_tokens_per_word:
                continue
            if only_multi_token_base and len(base_ids_tuple) <= 1 and len(variant_ids_tuple) <= 1:
                continue
            if base_tokens and len(base_ids_tuple) > 1 and len(variant_ids_tuple) == 1:
                normalized_variant = variant_str.lstrip(space_prefix)
                canonical_no_space = normalized_variant.lower()
                candidate_keys = [
                    f"{space_prefix}{normalized_variant}" if space_prefix else normalized_variant,
                    normalized_variant,
                    f"{space_prefix}{canonical_no_space}" if space_prefix else canonical_no_space,
                    canonical_no_space,
                    base_word,
                ]
                candidate_base = None
                for candidate_key in candidate_keys:
                    candidate_base = base_tokens.get(candidate_key)
                    if isinstance(candidate_base, int):
                        break
                if isinstance(candidate_base, int):
                    continue

            modifier = modifier_manager.create_empty_modifier()
            for transform_name in transforms:
                if transform_name in transform_to_group_idx:
                    group_name, rel_idx = transform_to_group_idx[transform_name]
                    modifier_manager.set_group_value(modifier, group_name, rel_idx)

            if (not overwrite_existing) and sequence_map.get_exact_match(variant_ids_tuple) is not None:
                continue
            sequence_map.add_sequence(variant_ids_tuple, base_ids_tuple, modifier)

    return sequence_map


class DualStreamTokenizer:
    """Tokenizer wrapper that returns both token IDs and modifier arrays.

    This tokenizer processes text through the base tokenizer, then transforms
    the token stream to extract base tokens and their modifiers (transformations).

    Features:
    - Article/preposition/punctuation attachment to adjacent tokens
    - Multi-token base word handling (modifiers attach to last token)
    - Optional word-boundary checks for sequence matches
    - Multi-token modifier attachment options (first/last/split)
    - Efficient trie-based sequence matching
    - 2D modifier array output (seq_len, num_groups)
    """

    def __init__(
        self,
        base_tokenizer: PreTrainedTokenizer,
        sequence_map: SequenceMap,
        modifier_manager: UnifiedModifierArray,
        decomposition_token_ids: Optional[Set[int]] = None,
        na_type_ids: Optional[Set[int]] = None,
        use_na_modifiers_for_non_decomposed: bool = True,
        word_boundary_safety: bool = False,
        multi_token_modifier_position: str = "last",
        article_token_ids: Optional[Set[int]] = None,
        preposition_token_ids: Optional[Set[int]] = None,
        prefix_punct_token_ids: Optional[Set[int]] = None,
        suffix_punct_token_ids: Optional[Set[int]] = None,
        article_transforms: Optional[Dict[int, Tuple[str, int]]] = None,
        preposition_transforms: Optional[Dict[int, Tuple[str, int]]] = None,
        prefix_punct_transforms: Optional[Dict[int, Tuple[str, int]]] = None,
        suffix_punct_transforms: Optional[Dict[int, Tuple[str, int]]] = None,
        modifier_dtype: Optional[Any] = None,
        enable_rust_stream: bool = True,
        rust_stream_verify: bool = False,
        rust_stream_verify_limit: int = 128,
        rust_stream_init_docs: int = 0,
        enable_rust_stream_fused: bool = True,
        rust_stream_fused_verify: bool = False,
        rust_stream_fused_verify_limit: int = 128,
        lowercase_cap_fallback: bool = True,
        enable_all_caps_capitalization: bool = False,
        transformation_names_to_int: Optional[Dict[str, int]] = None,
    ):
        """Initialize the dual-stream tokenizer.

        Args:
            base_tokenizer: The underlying tokenizer.
            sequence_map: SequenceMap for multi-token detection.
            modifier_manager: UnifiedModifierArray for managing modifiers.
            word_boundary_safety: Enforce word boundary checks for sequence matches.
            multi_token_modifier_position: Where to attach modifiers for multi-token bases.
            article_token_ids: Set of token IDs that are articles.
            preposition_token_ids: Set of token IDs that are prepositions.
            prefix_punct_token_ids: Set of token IDs that are prefix punctuation.
            suffix_punct_token_ids: Set of token IDs that are suffix punctuation.
            article_transforms: Dict mapping article token_id to (group_name, relative_idx).
            preposition_transforms: Dict mapping preposition token_id to (group_name, relative_idx).
            prefix_punct_transforms: Dict mapping prefix punct token_id to (group_name, relative_idx).
            suffix_punct_transforms: Dict mapping suffix punct token_id to (group_name, relative_idx).
        """
        self.base_tokenizer = base_tokenizer
        self.sequence_map = sequence_map
        self.modifier_manager = modifier_manager
        self.modifier_dtype, self.modifier_dtype_name = _resolve_modifier_dtype(
            modifier_dtype,
            self.modifier_manager.recommended_modifier_dtype_name(),
        )
        required_dtype_name = self.modifier_manager.recommended_modifier_dtype_name()
        if np.dtype(self.modifier_dtype).itemsize < np.dtype(required_dtype_name).itemsize:
            raise ValueError(
                f"modifier_dtype {self.modifier_dtype_name!r} is too small for active modifier groups; "
                f"minimum required dtype is {required_dtype_name!r}."
            )
        self.decomposition_token_ids = set(decomposition_token_ids) if decomposition_token_ids else None
        self.na_type_ids = set(int(i) for i in (na_type_ids or []))
        self.use_na_modifiers_for_non_decomposed = use_na_modifiers_for_non_decomposed
        self.na_modifier = self._build_na_modifier()
        self.word_boundary_safety = word_boundary_safety

        placement = str(multi_token_modifier_position)
        allowed_positions = {"last", "first", "split_by_role"}
        if placement not in allowed_positions:
            raise ValueError(f"Unknown multi_token_modifier_position: {multi_token_modifier_position}")
        self.multi_token_modifier_position = placement
        self._multi_token_first_groups = {
            "space_prefix",
            "base_capitalization",
            "determiners",
            "articles",
            "article_det",
            "article_space_prefix",
            "article_capitalization",
            "prepositions",
            "prep_space_prefix",
            "prep_capitalization",
            "prefix_punctuation",
        }

        # Special token sets
        self.article_token_ids = article_token_ids or set()
        self.preposition_token_ids = preposition_token_ids or set()
        self.prefix_punct_token_ids = prefix_punct_token_ids or set()
        self.suffix_punct_token_ids = suffix_punct_token_ids or set()
        self._token_text_cache: Dict[int, str] = {}
        self._token_decoded_cache: Dict[int, str] = {}
        self._token_has_space_prefix_cache: Dict[int, bool] = {}
        self._token_has_word_char_cache: Dict[int, bool] = {}
        self._special_token_id_set: Set[int] = set(
            int(tid) for tid in (getattr(self.base_tokenizer, "all_special_ids", []) or [])
        )
        # Guardrails for Python cap-fallback path: keep behavior but prevent
        # pathological work on very long uninterrupted word runs.
        self.cap_fallback_max_span_tokens = 128
        self.cap_fallback_max_surface_chars = 512

        # Transform mappings: token_id -> (group_name, relative_index)
        self.article_transforms = article_transforms or {}
        self.preposition_transforms = preposition_transforms or {}
        self.prefix_punct_transforms = prefix_punct_transforms or {}
        self.suffix_punct_transforms = suffix_punct_transforms or {}
        self.prefix_punct_text_map = self._build_punct_text_map(self.prefix_punct_transforms)
        self.suffix_punct_text_map = self._build_punct_text_map(self.suffix_punct_transforms)
        self.suffix_possessive_s = self.suffix_punct_text_map.get("'s") or self.suffix_punct_text_map.get("\u2019s")
        self.suffix_possessive_plural = self.suffix_punct_text_map.get("s'") or self.suffix_punct_text_map.get("s\u2019")
        self.suffix_single_quote = self.suffix_punct_text_map.get("'") or self.suffix_punct_text_map.get("\u2019")
        self.curly_apostrophe_ids = tuple(
            self.base_tokenizer.encode("\u2019", add_special_tokens=False)
        )
        self.curly_apostrophe_s_ids = tuple(
            self.base_tokenizer.encode("\u2019s", add_special_tokens=False)
        )
        self.apostrophe_like_chars = {"'", "`", "\u2019", "\u2018", "\u02BC", "\u05F3"}
        self.transformation_names_to_int = dict(transformation_names_to_int or {})
        self._group_rel_to_transform_name: Dict[str, Dict[int, str]] = {}
        if self.transformation_names_to_int and self.modifier_manager.types_loss_indices_map:
            int_to_name = {int(v): str(k) for k, v in self.transformation_names_to_int.items()}
            for group_name, bounds in self.modifier_manager.types_loss_indices_map.items():
                start, end = int(bounds[0]), int(bounds[1])
                rel_map: Dict[int, str] = {}
                for global_idx in range(start, end):
                    tname = int_to_name.get(global_idx)
                    if tname is not None:
                        rel_map[global_idx - start] = tname
                if rel_map:
                    self._group_rel_to_transform_name[group_name] = rel_map

        # Capitalization detection
        self.article_cap_group_idx = self.modifier_manager.group_to_idx.get('article_capitalization', -1)
        self.prep_cap_group_idx = self.modifier_manager.group_to_idx.get('prep_capitalization', -1)
        self.article_space_group_idx = self.modifier_manager.group_to_idx.get('article_space_prefix', -1)
        self.prep_space_group_idx = self.modifier_manager.group_to_idx.get('prep_space_prefix', -1)
        self.space_group_idx = self.modifier_manager.group_to_idx.get('space_prefix', -1)
        self.base_cap_group_idx = self.modifier_manager.group_to_idx.get('base_capitalization', -1)
        base_cap_group_size = int(self.modifier_manager.group_sizes.get('base_capitalization', 0))
        if enable_all_caps_capitalization:
            raise ValueError("All-caps capitalization transform is disabled and unsupported.")
        # Group-relative index convention for base_capitalization:
        # 0=no_cap, 1=add_capitalization.
        self.base_cap_add_rel_idx = 1 if base_cap_group_size > 1 else -1
        self.lowercase_cap_fallback_enabled = bool(lowercase_cap_fallback)
        if self.space_group_idx >= 0 and self.modifier_manager.group_sizes.get('space_prefix', 0) > 1:
            self.space_with_rel_idx = 1
        else:
            self.space_with_rel_idx = -1
        self.article_token_is_cap = self._build_capitalization_map(self.article_token_ids)
        self.preposition_token_is_cap = self._build_capitalization_map(self.preposition_token_ids)
        (
            self.whitespace_only_token_ids,
            self.single_space_token_ids,
            self.single_space_token_id,
            self._token_decoded_lower_endswith_s_cache,
            self.has_space_prefixed_word_tokens,
        ) = self._build_whitespace_token_ids()
        # Space-isolating tokenizers carry whitespace in detached " " tokens and
        # typically do not have leading-space word tokens in vocab.
        self.space_isolating_tokenizer = bool(self.single_space_token_ids) and (
            not bool(self.has_space_prefixed_word_tokens)
        )
        self.debug_profile_enabled = False
        self._last_debug_profile: Optional[Dict[str, float]] = None
        self.rust_stream_enabled = False
        self.rust_stream_verify = False
        self.rust_stream_fused_enabled = False
        self.rust_stream_fused_verify = False

    def _build_na_modifier(self) -> List[int]:
        """Build the fallback modifier for non-decomposed tokens.

        If NA labels exist, use them per group; otherwise fall back to the NO/default value
        (group-relative index 0). This keeps behavior valid when bundles collapse to a
        single negative class (NO only).
        """
        if not self.na_type_ids:
            return self.modifier_manager.create_empty_modifier()

        modifier = []
        indices_map = self.modifier_manager.types_loss_indices_map or {}
        for group in self.modifier_manager.groups:
            rel_idx = 0
            if group in indices_map:
                start, end = indices_map[group]
                for global_idx in range(start, end):
                    if global_idx in self.na_type_ids:
                        rel_idx = global_idx - start
                        break
            modifier.append(rel_idx)
        return modifier

    def _build_capitalization_map(self, token_ids: Set[int]) -> Dict[int, bool]:
        cap_map = {}
        if not token_ids:
            return cap_map
        for token_id in token_ids:
            token_text = self.base_tokenizer.decode(
                [int(token_id)],
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )
            token_text = token_text.lstrip("Ġ ").strip()
            cap_map[token_id] = bool(token_text) and token_text[0].isupper()
        return cap_map

    def _build_punct_text_map(self, punct_transforms: Dict[int, Tuple[str, int]]) -> Dict[str, Tuple[str, int]]:
        text_map = {}
        if not punct_transforms:
            return text_map
        for token_id, (group_name, rel_idx) in punct_transforms.items():
            token_text = _canonical_punct_surface(self._get_token_decoded_text(token_id))
            if token_text:
                text_map[token_text] = (group_name, rel_idx)
        return text_map

    def _build_whitespace_token_ids(self) -> Tuple[Set[int], Set[int], Optional[int], Dict[int, bool], bool]:
        """Collect whitespace-only token IDs and detect exact single-space token IDs."""
        whitespace_ids: Set[int] = set()
        single_space_ids: Set[int] = set()
        decoded_lower_endswith_s: Dict[int, bool] = {}
        has_space_prefixed_word_tokens = False
        vocab = self.base_tokenizer.get_vocab()
        special_ids = set(getattr(self.base_tokenizer, "all_special_ids", []) or [])
        for token_text, token_id in vocab.items():
            token_id = int(token_id)
            if token_id in special_ids:
                continue
            token_text = str(token_text)
            decoded = self.base_tokenizer.decode(
                [token_id],
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )
            if decoded and all(ch.isspace() for ch in decoded):
                whitespace_ids.add(token_id)
                if decoded == " ":
                    single_space_ids.add(token_id)
            elif token_text in {"Ġ", "▁"}:
                # Marker-only whitespace token when decoder preserves markers.
                whitespace_ids.add(token_id)
                single_space_ids.add(token_id)
            if decoded and decoded.startswith(" "):
                stripped = decoded.lstrip(" ")
                if stripped and any(ch.isalnum() for ch in stripped):
                    has_space_prefixed_word_tokens = True
            elif self._token_has_space_prefix(token_text):
                stripped = token_text.lstrip(" Ġ▁")
                if stripped and any(ch.isalnum() for ch in stripped):
                    has_space_prefixed_word_tokens = True
            decoded_lower_endswith_s[token_id] = decoded.strip().lower().endswith("s")
        single_space_id = min(single_space_ids) if single_space_ids else None
        return (
            whitespace_ids,
            single_space_ids,
            single_space_id,
            decoded_lower_endswith_s,
            has_space_prefixed_word_tokens,
        )

    @staticmethod
    def _match_token_sequence(raw_token_ids: List[int], start_idx: int, sequence: Tuple[int, ...]) -> bool:
        if not sequence:
            return False
        end_idx = start_idx + len(sequence)
        if end_idx > len(raw_token_ids):
            return False
        return tuple(raw_token_ids[start_idx:end_idx]) == sequence

    def _get_token_text(self, token_id: int) -> str:
        token_id = int(token_id)
        token_text = self._token_text_cache.get(token_id)
        if token_text is not None:
            return token_text
        token_text = self.base_tokenizer.convert_ids_to_tokens([token_id])[0]
        self._token_text_cache[token_id] = token_text
        return token_text

    def _get_token_decoded_text(self, token_id: int) -> str:
        token_id = int(token_id)
        decoded = self._token_decoded_cache.get(token_id)
        if decoded is not None:
            return decoded
        decoded = self.base_tokenizer.decode(
            [token_id],
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        self._token_decoded_cache[token_id] = decoded
        return decoded

    def _token_has_space_prefix_id(self, token_id: int) -> bool:
        token_id = int(token_id)
        cached = self._token_has_space_prefix_cache.get(token_id)
        if cached is not None:
            return cached
        decoded = self._get_token_decoded_text(token_id)
        has_space = bool(decoded.startswith(" "))
        if not has_space:
            token_text = self._token_text_cache.get(token_id)
            if token_text is None:
                token_text = self._get_token_text(token_id)
            has_space = self._token_has_space_prefix(token_text)
        self._token_has_space_prefix_cache[token_id] = bool(has_space)
        return bool(has_space)

    def _token_has_word_char_id(self, token_id: int) -> bool:
        token_id = int(token_id)
        cached = self._token_has_word_char_cache.get(token_id)
        if cached is not None:
            return cached
        if token_id in self._special_token_id_set:
            self._token_has_word_char_cache[token_id] = False
            return False
        decoded = self.base_tokenizer.decode(
            [token_id],
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        has_word = any(ch.isalnum() for ch in decoded)
        self._token_has_word_char_cache[token_id] = has_word
        return has_word

    @staticmethod
    def _is_capitalized_token_text(token_text: str) -> bool:
        token_text = token_text.lstrip("Ġ ▁").strip()
        return bool(token_text) and token_text[0].isupper()

    @staticmethod
    def _classify_segment_cap_rel(segment: str, add_cap_rel_idx: int) -> Optional[int]:
        if (not segment) or (not segment.isalpha()):
            return None
        first = segment[0]
        rest = segment[1:]
        if first.islower() and all(ch.islower() for ch in rest):
            return 0
        is_title = bool(first.isupper() and all(ch.islower() for ch in rest))
        if is_title and add_cap_rel_idx >= 0:
            return int(add_cap_rel_idx)
        return None

    @staticmethod
    def _split_camel_case_segments(surface: str) -> Optional[List[str]]:
        if not surface:
            return None
        boundaries = [0]
        has_internal_upper = False
        for idx in range(1, len(surface)):
            current = surface[idx]
            if not current.isupper():
                continue
            has_internal_upper = True
            prev = surface[idx - 1]
            next_is_lower = (idx + 1) < len(surface) and surface[idx + 1].islower()
            if prev.islower() or (prev.isupper() and next_is_lower):
                boundaries.append(idx)
        if not has_internal_upper:
            return None
        boundaries.append(len(surface))
        if len(boundaries) <= 2:
            return None
        segments: List[str] = []
        for left, right in zip(boundaries[:-1], boundaries[1:]):
            if left < right:
                segments.append(surface[left:right])
        return segments if len(segments) > 1 else None

    @staticmethod
    def _expand_caps_segments(segments: List[str]) -> List[str]:
        expanded: List[str] = []
        for segment in segments:
            if len(segment) > 1 and all(ch.isupper() for ch in segment):
                expanded.extend(list(segment))
            else:
                expanded.append(segment)
        return expanded

    def _modifier_has_only_surface_groups(self, modifier: List[int]) -> bool:
        for group_name, group_idx in self.modifier_manager.group_to_idx.items():
            if group_idx >= len(modifier):
                continue
            if group_name in {"space_prefix", "base_capitalization"}:
                continue
            if int(modifier[group_idx]) != 0:
                return False
        return True

    def _build_cap_run_metadata(
        self,
        raw_token_ids: List[int],
        token_has_space_prefix: List[bool],
        token_has_word_char: List[bool],
    ) -> Tuple[List[int], List[bool]]:
        """Precompute per-index word-run boundaries and downstream cap presence.

        For each raw index i:
        - run_end_by_idx[i] is the exclusive end of the contiguous in-word run.
        - has_cap_from_idx[i] is True when any token in [i, run_end_by_idx[i])
          starts with an uppercase character.
        """
        n_tokens = len(raw_token_ids)
        run_end_by_idx = [0] * n_tokens
        has_cap_from_idx = [False] * n_tokens
        idx = 0
        while idx < n_tokens:
            if not token_has_word_char[idx]:
                run_end_by_idx[idx] = idx + 1
                has_cap_from_idx[idx] = False
                idx += 1
                continue

            run_end = idx + 1
            while (
                run_end < n_tokens
                and token_has_word_char[run_end]
                and (not token_has_space_prefix[run_end])
            ):
                run_end += 1

            suffix_has_cap = False
            for probe_idx in range(run_end - 1, idx - 1, -1):
                if not suffix_has_cap:
                    probe_tok_id = int(raw_token_ids[probe_idx])
                    suffix_has_cap = self._is_capitalized_token_text(
                        self._get_token_decoded_text(probe_tok_id)
                    )
                run_end_by_idx[probe_idx] = run_end
                has_cap_from_idx[probe_idx] = suffix_has_cap

            idx = run_end

        return run_end_by_idx, has_cap_from_idx

    def _try_lowercase_cap_fallback(
        self,
        raw_token_ids: List[int],
        start_idx: int,
        span_end_idx: Optional[int] = None,
    ) -> Optional[Tuple[int, List[int], List[List[int]]]]:
        """Fallback: tokenize capped spans as lowercase base tokens + per-token cap modifiers."""
        if self.base_cap_group_idx < 0 or self.base_cap_add_rel_idx < 0:
            return None
        if start_idx >= len(raw_token_ids):
            return None

        first_tok_id = int(raw_token_ids[start_idx])
        if not self._token_has_word_char_id(first_tok_id):
            return None

        if span_end_idx is not None:
            end_idx = max(start_idx + 1, min(int(span_end_idx), len(raw_token_ids)))
        else:
            end_idx = start_idx + 1
            while end_idx < len(raw_token_ids):
                token_id = int(raw_token_ids[end_idx])
                if self._token_has_space_prefix_id(token_id) or (not self._token_has_word_char_id(token_id)):
                    break
                end_idx += 1
        span_len = end_idx - start_idx
        if span_len <= 0:
            return None
        if self.cap_fallback_max_span_tokens > 0 and span_len > int(self.cap_fallback_max_span_tokens):
            return None

        # If the span is immediately followed by an apostrophe token and then an
        # in-word continuation (e.g., I + ' + m), treat it as a contraction
        # boundary and skip cap fallback to avoid rewriting to i + ' + m.
        if end_idx < len(raw_token_ids):
            boundary_token_id = int(raw_token_ids[end_idx])
            boundary_surface = _canonical_punct_surface(self._get_token_decoded_text(boundary_token_id))
            if (
                boundary_surface in self.apostrophe_like_chars
                and (end_idx + 1) < len(raw_token_ids)
            ):
                next_token_id = int(raw_token_ids[end_idx + 1])
                if (
                    (not self._token_has_space_prefix_id(next_token_id))
                    and self._token_has_word_char_id(next_token_id)
                ):
                    return None

        surface = self.base_tokenizer.decode(
            raw_token_ids[start_idx:end_idx],
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        if not surface:
            return None
        surface = surface.strip()
        if self.cap_fallback_max_surface_chars > 0 and len(surface) > int(self.cap_fallback_max_surface_chars):
            return None
        if (not surface) or (not surface.isalpha()) or (not any(ch.isupper() for ch in surface)):
            return None

        split_segments = self._split_camel_case_segments(surface)
        is_title_surface = bool(surface[0].isupper() and (len(surface) == 1 or surface[1:].islower()))
        if split_segments is None and (not is_title_surface):
            # Do not normalize all-caps/non-title words at span level.
            return None
        segments = split_segments or [surface]
        segments = self._expand_caps_segments(segments)

        output_ids: List[int] = []
        output_mods: List[List[int]] = []
        for segment in segments:
            cap_rel_idx = self._classify_segment_cap_rel(segment, self.base_cap_add_rel_idx)
            if cap_rel_idx is None:
                return None
            lower_ids = self.base_tokenizer.encode(segment.lower(), add_special_tokens=False)
            if not lower_ids:
                return None
            cap_modifier = self.modifier_manager.create_empty_modifier()
            cap_modifier[self.base_cap_group_idx] = int(cap_rel_idx)
            per_token_mods = self._spread_multi_token_modifiers(cap_modifier, len(lower_ids))
            for tok_id, token_mod in zip(lower_ids, per_token_mods):
                output_ids.append(int(tok_id))
                output_mods.append(list(token_mod))

        if not output_ids or len(output_ids) != len(output_mods):
            return None
        return span_len, output_ids, output_mods

    def encode(self, text: str, add_special_tokens: bool = True) -> Tuple[List[int], np.ndarray]:
        """Encode text into token IDs and modifier array.

        Args:
            text: Input text to tokenize.
            add_special_tokens: Whether to add special tokens (BOS/EOS).

        Returns:
            Tuple of (token_ids, modifier_ids) where modifier_ids is (N, num_groups).
        """
        return self._encode_impl(text, add_special_tokens=add_special_tokens, return_metadata=False)

    def encode_with_metadata(self, text: str, add_special_tokens: bool = True) -> Tuple[List[int], np.ndarray, Dict[str, Any]]:
        """Encode text and return metadata for analysis."""
        return self._encode_impl(text, add_special_tokens=add_special_tokens, return_metadata=True)

    def set_debug_profile(self, enabled: bool) -> None:
        self.debug_profile_enabled = bool(enabled)
        if not self.debug_profile_enabled:
            self._last_debug_profile = None

    def pop_last_debug_profile(self) -> Optional[Dict[str, float]]:
        profile = self._last_debug_profile
        self._last_debug_profile = None
        return profile

    def _encode_impl(self, text: str, add_special_tokens: bool, return_metadata: bool):
        profile: Optional[Dict[str, float]] = None
        if self.debug_profile_enabled:
            profile = {
                "ds_text_chars": float(len(text)),
                "ds_pretokenize_probe_s": 0.0,
                "ds_base_encode_s": 0.0,
                "ds_stream_total_s": 0.0,
                "ds_stream_boundary_tokens_s": 0.0,
                "ds_stream_match_lookup_s": 0.0,
                "ds_stream_reconstruct_s": 0.0,
                "ds_stream_finalize_s": 0.0,
                "ds_raw_tokens": 0.0,
                "ds_output_tokens": 0.0,
            }
            self._run_pretokenize_probe(text, profile)

        if profile is None:
            fused_processed = self._process_text_fused(
                text,
                add_special_tokens=add_special_tokens,
                return_metadata=return_metadata,
            )
            if fused_processed is not None:
                self._last_debug_profile = None
                return fused_processed

        t_encode = time.perf_counter() if profile is not None else 0.0
        raw_ids = self.base_tokenizer.encode(text, add_special_tokens=add_special_tokens)
        if profile is not None:
            profile["ds_base_encode_s"] += time.perf_counter() - t_encode
            profile["ds_raw_tokens"] = float(len(raw_ids))

        t_stream = time.perf_counter() if profile is not None else 0.0
        processed = self._process_stream(raw_ids, return_metadata=return_metadata, profile=profile)
        if profile is not None:
            profile["ds_stream_total_s"] += time.perf_counter() - t_stream
            if return_metadata:
                output_ids = processed[0]
            else:
                output_ids = processed[0]
            profile["ds_output_tokens"] = float(len(output_ids))
            self._last_debug_profile = profile
        else:
            self._last_debug_profile = None
        return processed

    def _run_pretokenize_probe(self, text: str, profile: Dict[str, float]) -> None:
        backend = getattr(self.base_tokenizer, "backend_tokenizer", None)
        if backend is None:
            return
        pre_tokenizer = getattr(backend, "pre_tokenizer", None)
        if pre_tokenizer is None:
            return
        try:
            t0 = time.perf_counter()
            pre_tokenizer.pre_tokenize_str(text)
            profile["ds_pretokenize_probe_s"] += time.perf_counter() - t0
        except Exception:
            return

    def __call__(self, text: str, **kwargs) -> Dict[str, Any]:
        """HuggingFace-compatible call interface.

        Args:
            text: Input text to tokenize.
            **kwargs: Additional arguments (e.g., add_special_tokens).

        Returns:
            Dictionary with 'input_ids', 'modifier_ids', and 'attention_mask'.
        """
        add_special_tokens = kwargs.get('add_special_tokens', True)
        token_ids, modifier_ids = self.encode(text, add_special_tokens=add_special_tokens)

        return {
            'input_ids': token_ids,
            'modifier_ids': modifier_ids.tolist(),
            'attention_mask': [1] * len(token_ids)
        }

    def tokenize(self, text: str) -> Tuple[List[str], np.ndarray]:
        """Tokenize text into tokens (strings) and modifier array.

        Args:
            text: Input text to tokenize.

        Returns:
            Tuple of (tokens, modifier_ids).
        """
        token_ids, modifier_ids = self.encode(text, add_special_tokens=False)
        tokens = self.base_tokenizer.convert_ids_to_tokens(token_ids)
        return tokens, modifier_ids

    def encode_batch(self, texts: List[str], add_special_tokens: bool = True,
                     padding: bool = False, max_length: Optional[int] = None) -> Tuple[List[List[int]], List[np.ndarray]]:
        """Encode a batch of texts into token IDs and modifier arrays.

        Args:
            texts: List of input texts to tokenize.
            add_special_tokens: Whether to add special tokens (BOS/EOS).
            padding: Whether to pad sequences to the same length.
            max_length: Maximum sequence length (required if padding=True).

        Returns:
            Tuple of (batch_token_ids, batch_modifier_ids) where:
                - batch_token_ids: List of token ID lists
                - batch_modifier_ids: List of modifier arrays (each shape: [seq_len, num_groups])
        """
        batch_pairs = self.encode_many(texts, add_special_tokens=add_special_tokens, return_metadata=False)
        batch_token_ids = [token_ids for token_ids, _mods in batch_pairs]
        batch_modifier_ids = [mods for _token_ids, mods in batch_pairs]

        if padding:
            if max_length is None:
                max_length = max(len(ids) for ids in batch_token_ids)

            # Pad token IDs
            pad_id = self.base_tokenizer.pad_token_id or 0
            padded_token_ids = []
            padded_modifier_ids = []

            for token_ids, modifier_ids in zip(batch_token_ids, batch_modifier_ids):
                seq_len = len(token_ids)
                if seq_len < max_length:
                    # Pad tokens
                    padded_ids = token_ids + [pad_id] * (max_length - seq_len)
                    # Pad modifiers with zeros
                    padding_mods = np.zeros((max_length - seq_len, self.modifier_manager.num_groups), dtype=self.modifier_dtype)
                    padded_mods = np.vstack([modifier_ids, padding_mods])
                elif seq_len > max_length:
                    # Truncate
                    padded_ids = token_ids[:max_length]
                    padded_mods = modifier_ids[:max_length]
                else:
                    padded_ids = token_ids
                    padded_mods = modifier_ids

                padded_token_ids.append(padded_ids)
                padded_modifier_ids.append(padded_mods)

            batch_token_ids = padded_token_ids
            batch_modifier_ids = padded_modifier_ids

        return batch_token_ids, batch_modifier_ids

    def _batch_encode_raw_ids(self, texts: List[str], add_special_tokens: bool) -> List[List[int]]:
        if not texts:
            return []
        batch = self.base_tokenizer(
            texts,
            add_special_tokens=add_special_tokens,
            return_attention_mask=False,
            return_token_type_ids=False,
        )
        raw_ids_batch = batch.get("input_ids")
        if raw_ids_batch is None:
            raise ValueError("Batch encoding did not return input_ids.")
        return [list(map(int, ids)) for ids in raw_ids_batch]

    def _process_text_fused(
        self,
        text: str,
        *,
        add_special_tokens: bool,
        return_metadata: bool,
    ) -> Optional[Any]:
        return None

    def _process_text_batch_fused(
        self,
        texts: List[str],
        *,
        add_special_tokens: bool,
        return_metadata: bool,
    ) -> Optional[List[Any]]:
        return None

    def _process_stream_batch(
        self,
        raw_ids_batch: List[List[int]],
        return_metadata: bool = False,
    ) -> List[Any]:
        return [
            self._process_stream_python(raw_ids, return_metadata=return_metadata, profile=None)
            for raw_ids in raw_ids_batch
        ]

    def encode_many(
        self,
        texts: List[str],
        add_special_tokens: bool = True,
        return_metadata: bool = False,
    ) -> List[Any]:
        if not texts:
            return []
        fused_batch = self._process_text_batch_fused(
            texts,
            add_special_tokens=add_special_tokens,
            return_metadata=return_metadata,
        )
        if fused_batch is not None:
            return fused_batch
        raw_ids_batch = self._batch_encode_raw_ids(texts, add_special_tokens=add_special_tokens)
        return self._process_stream_batch(raw_ids_batch, return_metadata=return_metadata)

    def _process_stream_python(
        self,
        raw_token_ids: List[int],
        return_metadata: bool = False,
        profile: Optional[Dict[str, float]] = None,
    ) -> Tuple[List[int], np.ndarray]:
        """Process raw token stream to extract base tokens and modifiers.

        This implements the core dual-stream algorithm:
        1. Accumulate prefix modifiers (articles, prepositions, prefix punct)
        2. Find the next base token (possibly multi-token sequence)
        3. Attach accumulated modifiers to the base token
        4. Handle suffix punctuation by attaching to previous token

        Args:
            raw_token_ids: List of token IDs from base tokenizer.

        Returns:
            Tuple of (compressed_token_ids, modifier_array).
            If return_metadata=True, returns (compressed_token_ids, modifier_array, metadata).
        """
        output_ids: List[int] = []
        output_modifiers: List[List[int]] = []

        # Metadata counters (only used when return_metadata=True)
        sequence_match_count = 0
        multi_token_base_count = 0
        multi_token_base_token_count = 0
        prefix_modifier_count = 0
        suffix_modifier_count = 0

        # Pending prefix modifiers: list of (group_name, relative_idx, is_capitalized)
        pending_prefixes: List[Tuple[str, int, bool]] = []
        # Raw token ids consumed as prefixes; if no base token follows, flush them literally.
        pending_prefix_token_ids: List[int] = []
        # True when we consumed a leading whitespace token right before a detachable prefix.
        pending_space_prefix_from_whitespace = False

        def _build_literal_modifier(tok: int) -> List[int]:
            if (
                self.use_na_modifiers_for_non_decomposed
                and self.decomposition_token_ids is not None
                and tok not in self.decomposition_token_ids
            ):
                return list(self.na_modifier)
            return self.modifier_manager.create_empty_modifier()

        def _emit_literal_token(tok: int, pending: Optional[List[Tuple[str, int, bool]]] = None) -> None:
            if pending:
                base_modifier = _build_literal_modifier(tok)
                modifier = self._combine_modifiers(base_modifier, pending)
                output_ids.append(tok)
                output_modifiers.append(modifier)
                return
            output_ids.append(tok)
            output_modifiers.append(_build_literal_modifier(tok))

        def _pending_prefix_is_space_only() -> bool:
            if not pending_prefixes:
                return False
            if self.space_with_rel_idx < 0:
                return False
            if not pending_prefix_token_ids:
                return False
            if not all(tok in self.single_space_token_ids for tok in pending_prefix_token_ids):
                return False
            for group_name, rel_idx, _ in pending_prefixes:
                if group_name != 'space_prefix' or rel_idx != self.space_with_rel_idx:
                    return False
            return True

        def _consume_pending_space_into_literal(tok: int) -> bool:
            nonlocal pending_space_prefix_from_whitespace
            if not _pending_prefix_is_space_only():
                return False
            _emit_literal_token(tok, pending=list(pending_prefixes))
            pending_prefixes.clear()
            pending_prefix_token_ids.clear()
            pending_space_prefix_from_whitespace = False
            return True

        def _flush_pending_prefix_tokens_literal() -> None:
            nonlocal pending_space_prefix_from_whitespace
            if not pending_prefix_token_ids:
                pending_prefixes.clear()
                pending_space_prefix_from_whitespace = False
                return
            for pending_tok in pending_prefix_token_ids:
                _emit_literal_token(pending_tok)
            pending_prefixes.clear()
            pending_prefix_token_ids.clear()
            pending_space_prefix_from_whitespace = False

        def _expr_space_from_context(prev_is_single_space: bool) -> bool:
            # In space-isolating tokenizers, a preceding single-space token can
            # either represent expression-level leading space (no detached prefixes)
            # or separator space between detached prefix and base token.
            if pending_space_prefix_from_whitespace:
                return True
            return bool(prev_is_single_space and (not pending_prefixes))

        def _pending_has_prefix_punctuation() -> bool:
            return any(group_name == "prefix_punctuation" for group_name, _, _ in pending_prefixes)

        def _starts_with_detached_candidate(start_idx: int) -> bool:
            if start_idx < 0 or start_idx >= len(raw_token_ids):
                return False
            tok = int(raw_token_ids[start_idx])
            if tok in self.article_token_ids or tok in self.preposition_token_ids:
                return True
            return False

        def _raw_token_has_explicit_space_prefix(start_idx: int) -> bool:
            if start_idx < 0 or start_idx >= len(raw_token_ids):
                return False
            tok = int(raw_token_ids[start_idx])
            token_text = self._get_token_text(tok)
            return bool(self._token_has_space_prefix(token_text))

        def _can_attach_detached_modifier(start_idx: int, consumed_len: int = 1) -> bool:
            """Allow detached article/preposition only when whitespace-bounded.

            Rules:
            - modifier must be whitespace-separated from text before (or start-of-text),
              except when it is inside an already-detached prefix-punctuation wrapper.
            - modifier must be whitespace-separated from the following base token.
            - if prefix punctuation appears between modifier and next word, do not detach.
            """
            if consumed_len <= 0:
                return False
            if start_idx < 0 or start_idx >= len(raw_token_ids):
                return False
            span_end = min(start_idx + consumed_len, len(raw_token_ids))
            for mid in range(start_idx + 1, span_end):
                if raw_token_ids[mid] in self.prefix_punct_token_ids:
                    return False

            allow_left_from_prefix_wrap = _pending_has_prefix_punctuation()
            if start_idx == 0:
                left_ok = True
            else:
                has_space_here = _raw_token_has_explicit_space_prefix(start_idx)
                prev_tok = raw_token_ids[start_idx - 1]
                prev_is_whitespace = prev_tok in self.whitespace_only_token_ids
                # Detached modifier must be whitespace-bounded in the raw stream
                # (or start-of-text). Do not trust pending whitespace state here.
                left_ok = bool(
                    has_space_here
                    or prev_is_whitespace
                )
            if (not left_ok) and (not allow_left_from_prefix_wrap):
                return False

            j = start_idx + consumed_len
            if j >= len(raw_token_ids):
                return False
            saw_whitespace_between = False
            saw_non_space_whitespace = False
            while j < len(raw_token_ids):
                probe_tok = raw_token_ids[j]
                if probe_tok in self.whitespace_only_token_ids:
                    saw_whitespace_between = True
                    if probe_tok not in self.single_space_token_ids:
                        saw_non_space_whitespace = True
                    j += 1
                    continue
                break
            if j >= len(raw_token_ids):
                return False
            next_tok = raw_token_ids[j]
            if next_tok in self.prefix_punct_token_ids:
                return False
            if raw_token_has_word_char is not None:
                next_is_word = bool(raw_token_has_word_char[j])
            else:
                next_is_word = self._token_has_word_char_id(next_tok)
            if not next_is_word:
                # Keep detached determiners/prepositions literal when the next token is
                # punctuation-only or whitespace-only; attachment is word-target only.
                return False
            # Do not attach when the next lexical token is itself detachable,
            # except for preposition->article chains (e.g., "on the mat").
            if _starts_with_detached_candidate(j):
                current_tok = raw_token_ids[start_idx]
                next_is_article = next_tok in self.article_token_ids
                next_is_preposition = next_tok in self.preposition_token_ids
                allow_prep_article_chain = (
                    (current_tok in self.preposition_token_ids) and next_is_article
                )
                if next_is_preposition or (not allow_prep_article_chain):
                    return False
            if raw_token_has_space_prefix is not None:
                next_has_space_prefix = bool(raw_token_has_space_prefix[j])
            else:
                next_has_space_prefix = self._token_has_space_prefix_id(next_tok)
            if saw_non_space_whitespace:
                return False
            return bool(saw_whitespace_between or next_has_space_prefix)

        article_group_idx = self.modifier_manager.group_to_idx.get(
            "determiners",
            self.modifier_manager.group_to_idx.get(
                "article_det",
                self.modifier_manager.group_to_idx.get("articles", -1),
            ),
        )
        preposition_group_idx = self.modifier_manager.group_to_idx.get("prepositions", -1)

        def _strip_invalid_detached_modifier_groups(
            modifier_values: List[int],
            start_idx: int,
            consumed_len: int,
        ) -> List[int]:
            if _can_attach_detached_modifier(start_idx, consumed_len):
                return modifier_values
            cleaned = list(modifier_values)
            group_indices = (
                article_group_idx,
                preposition_group_idx,
                self.article_cap_group_idx,
                self.prep_cap_group_idx,
                self.article_space_group_idx,
                self.prep_space_group_idx,
            )
            for gidx in group_indices:
                if 0 <= gidx < len(cleaned):
                    cleaned[gidx] = 0
            return cleaned

        raw_token_has_space_prefix: Optional[List[bool]] = None
        raw_token_has_word_char: Optional[List[bool]] = None
        raw_space_prefix_prefix_sum: Optional[List[int]] = None
        if self.word_boundary_safety:
            t_boundary = time.perf_counter() if profile is not None else 0.0
            raw_token_has_space_prefix = [self._token_has_space_prefix_id(tok_id) for tok_id in raw_token_ids]
            raw_token_has_word_char = [self._token_has_word_char_id(tok_id) for tok_id in raw_token_ids]
            raw_space_prefix_prefix_sum = [0] * (len(raw_token_ids) + 1)
            running_space_prefix = 0
            for idx, has_space_prefix in enumerate(raw_token_has_space_prefix, start=1):
                if has_space_prefix:
                    running_space_prefix += 1
                raw_space_prefix_prefix_sum[idx] = running_space_prefix
            if profile is not None:
                profile["ds_stream_boundary_tokens_s"] += time.perf_counter() - t_boundary

        cap_run_end_by_idx: Optional[List[int]] = None
        cap_has_cap_from_idx: Optional[List[bool]] = None
        if (
            self.lowercase_cap_fallback_enabled
            and self.base_cap_group_idx >= 0
            and self.base_cap_add_rel_idx >= 0
            and raw_token_ids
        ):
            cap_token_has_space_prefix = (
                raw_token_has_space_prefix
                if raw_token_has_space_prefix is not None
                else [self._token_has_space_prefix_id(tok_id) for tok_id in raw_token_ids]
            )
            cap_token_has_word_char = (
                raw_token_has_word_char
                if raw_token_has_word_char is not None
                else [self._token_has_word_char_id(tok_id) for tok_id in raw_token_ids]
            )
            cap_run_end_by_idx, cap_has_cap_from_idx = self._build_cap_run_metadata(
                raw_token_ids,
                cap_token_has_space_prefix,
                cap_token_has_word_char,
            )

        i = 0
        while i < len(raw_token_ids):
            token_id = raw_token_ids[i]

            # Detached prefix transforms (article/preposition/prefix punct) must never
            # attach to pure whitespace rows like newline tokens.
            if (
                token_id in self.whitespace_only_token_ids
                and token_id not in self.single_space_token_ids
                and pending_prefixes
                and (not _pending_prefix_is_space_only())
            ):
                _flush_pending_prefix_tokens_literal()

            # In regular tokenizers, standalone single-space tokens that appear
            # between detached prefixes and the lexical base are separators.
            # Keep them pending so prefixes attach to the next word token.
            if (
                (not self.space_isolating_tokenizer)
                and token_id in self.single_space_token_ids
                and pending_prefixes
            ):
                pending_prefix_token_ids.append(token_id)
                pending_space_prefix_from_whitespace = False
                i += 1
                continue

            # If the tokenizer isolates single spaces, treat " " before any non-whitespace
            # token as a pending space-prefix modifier instead of emitting it literally.
            if (
                self.space_isolating_tokenizer
                and token_id in self.single_space_token_ids
                and (i + 1) < len(raw_token_ids)
            ):
                next_token_id = raw_token_ids[i + 1]
                if next_token_id not in self.whitespace_only_token_ids:
                    pending_prefix_token_ids.append(token_id)
                    # Separator space between detached prefixes (article/preposition/punct)
                    # and their base should not become expression-level space_prefix.
                    if (not pending_prefixes) and self.space_with_rel_idx >= 0:
                        pending_prefixes.append(('space_prefix', self.space_with_rel_idx, False))
                        pending_space_prefix_from_whitespace = True
                    else:
                        pending_space_prefix_from_whitespace = False
                    i += 1
                    continue

            if self.curly_apostrophe_s_ids and self._match_token_sequence(
                raw_token_ids, i, self.curly_apostrophe_s_ids
            ):
                if output_modifiers and self.suffix_possessive_s:
                    group_name, rel_idx = (
                        self.suffix_punct_text_map.get("\u2019s")
                        or self.suffix_possessive_s
                    )
                    self.modifier_manager.set_group_value(output_modifiers[-1], group_name, rel_idx)
                    suffix_modifier_count += 1
                    i += len(self.curly_apostrophe_s_ids)
                    continue

            if self.curly_apostrophe_ids and self._match_token_sequence(
                raw_token_ids, i, self.curly_apostrophe_ids
            ):
                if output_modifiers:
                    if self._attach_plural_possessive_if_contextual(
                        raw_token_ids,
                        None,
                        i,
                        output_ids,
                        output_modifiers,
                        preferred_entry=self.suffix_punct_text_map.get("s\u2019"),
                    ):
                        suffix_modifier_count += 1
                        i += len(self.curly_apostrophe_ids)
                        continue
                    if self.suffix_single_quote or self.suffix_punct_text_map.get("\u2019"):
                        group_name, rel_idx = (
                            self.suffix_punct_text_map.get("\u2019")
                            or self.suffix_single_quote
                        )
                        self.modifier_manager.set_group_value(output_modifiers[-1], group_name, rel_idx)
                        suffix_modifier_count += 1
                        i += len(self.curly_apostrophe_ids)
                        continue
                prefix_entry = self.prefix_punct_text_map.get("\u2019") or self.prefix_punct_text_map.get("'")
                if prefix_entry:
                    group_name, rel_idx = prefix_entry
                    pending_prefixes.append((group_name, rel_idx, False))
                    pending_prefix_token_ids.extend(raw_token_ids[i : i + len(self.curly_apostrophe_ids)])
                    prefix_modifier_count += 1
                    i += len(self.curly_apostrophe_ids)
                    continue

            # Check if this is a prefix modifier (article, preposition, prefix punct)
            is_prefix_punct = token_id in self.prefix_punct_token_ids
            is_suffix_punct = token_id in self.suffix_punct_token_ids
            if is_prefix_punct or is_suffix_punct:
                token_text = self._get_token_decoded_text(token_id)
                stripped_token = _canonical_punct_surface(token_text)
                if stripped_token in self.apostrophe_like_chars and self._attach_plural_possessive_if_contextual(
                    raw_token_ids,
                    None,
                    i,
                    output_ids,
                    output_modifiers,
                    preferred_entry=(
                        self.suffix_punct_text_map.get("s\u2019")
                        if stripped_token == "\u2019"
                        else self.suffix_punct_text_map.get("s'")
                    ),
                ):
                    suffix_modifier_count += 1
                    i += 1
                    continue
                if raw_token_has_space_prefix is not None:
                    has_space_prefix = raw_token_has_space_prefix[i]
                else:
                    has_space_prefix = self._token_has_space_prefix(token_text)
                if pending_space_prefix_from_whitespace:
                    has_space_prefix = True

                if is_prefix_punct and is_suffix_punct:
                    if not output_modifiers and not has_space_prefix:
                        treat_as_prefix = True
                    else:
                        treat_as_prefix = has_space_prefix
                else:
                    treat_as_prefix = is_prefix_punct

                if treat_as_prefix:
                    prefix_entry = self.prefix_punct_transforms.get(token_id)
                    if raw_token_has_word_char is not None:
                        next_is_word = (i + 1) < len(raw_token_ids) and raw_token_has_word_char[i + 1]
                    else:
                        next_is_word = (
                            (i + 1) < len(raw_token_ids)
                            and self._token_has_word_char_id(raw_token_ids[i + 1])
                        )
                    has_existing_prefix_group = (
                        prefix_entry is not None
                        and any(g == prefix_entry[0] for g, _, _ in pending_prefixes)
                    )
                    can_attach_prefix = (
                        prefix_entry is not None
                        and next_is_word
                        and not has_existing_prefix_group
                    )
                    if can_attach_prefix:
                        group_name, rel_idx = prefix_entry
                        pending_prefixes.append((group_name, rel_idx, False))
                        pending_prefix_token_ids.append(token_id)
                        prefix_modifier_count += 1
                        pending_space_prefix_from_whitespace = False
                    else:
                        if not _consume_pending_space_into_literal(token_id):
                            _flush_pending_prefix_tokens_literal()
                            _emit_literal_token(token_id)
                    i += 1
                    continue

                # Suffix punctuation - attach to previous token
                if output_modifiers and token_id in self.suffix_punct_transforms:
                    group_name, rel_idx = self.suffix_punct_transforms[token_id]
                    prev_is_word = bool(output_ids) and self._token_has_word_char_id(output_ids[-1])
                    prev_raw_is_whitespace = i > 0 and raw_token_ids[i - 1] in self.whitespace_only_token_ids
                    has_space_here = _raw_token_has_explicit_space_prefix(i)
                    group_idx = self.modifier_manager.group_to_idx.get(group_name, -1)
                    group_already_set = (
                        0 <= group_idx < len(output_modifiers[-1])
                        and int(output_modifiers[-1][group_idx]) != 0
                    )
                    if (
                        prev_is_word
                        and (not prev_raw_is_whitespace)
                        and (not has_space_here)
                        and (not group_already_set)
                    ):
                        self.modifier_manager.set_group_value(output_modifiers[-1], group_name, rel_idx)
                        suffix_modifier_count += 1
                    else:
                        if not _consume_pending_space_into_literal(token_id):
                            _flush_pending_prefix_tokens_literal()
                            _emit_literal_token(token_id)
                else:
                    if not _consume_pending_space_into_literal(token_id):
                        _flush_pending_prefix_tokens_literal()
                        _emit_literal_token(token_id)
                i += 1
                continue

            if (
                token_id in self.article_token_ids
                and token_id in self.article_transforms
                and _can_attach_detached_modifier(i, 1)
            ):
                # Article
                group_name, rel_idx = self.article_transforms[token_id]
                # Check if capitalized
                is_cap = self.article_token_is_cap.get(token_id, False)
                pending_prefixes.append((group_name, rel_idx, is_cap))
                if is_cap and self.article_cap_group_idx >= 0:
                    # Add capitalization marker
                    cap_idx = 1  # Assuming ADD_CAP is index 1
                    pending_prefixes.append(('article_capitalization', cap_idx, False))
                if self.article_space_group_idx >= 0:
                    if raw_token_has_space_prefix is not None:
                        has_space_prefix = raw_token_has_space_prefix[i]
                    else:
                        has_space_prefix = self._token_has_space_prefix_id(token_id)
                    if pending_space_prefix_from_whitespace:
                        has_space_prefix = True
                    if self.space_isolating_tokenizer:
                        has_space_prefix = pending_space_prefix_from_whitespace or (
                            i > 0 and raw_token_ids[i - 1] in self.single_space_token_ids
                        )
                    space_idx = 1 if has_space_prefix else 0
                    pending_prefixes.append(('article_space_prefix', space_idx, False))
                pending_prefix_token_ids.append(token_id)
                prefix_modifier_count += 1
                pending_space_prefix_from_whitespace = False
                i += 1
                continue

            if (
                token_id in self.preposition_token_ids
                and token_id in self.preposition_transforms
                and _can_attach_detached_modifier(i, 1)
            ):
                # Preposition
                group_name, rel_idx = self.preposition_transforms[token_id]
                # Check if capitalized
                is_cap = self.preposition_token_is_cap.get(token_id, False)
                pending_prefixes.append((group_name, rel_idx, is_cap))
                if is_cap and self.prep_cap_group_idx >= 0:
                    # Add capitalization marker
                    cap_idx = 1  # Assuming ADD_CAP is index 1
                    pending_prefixes.append(('prep_capitalization', cap_idx, False))
                if self.prep_space_group_idx >= 0:
                    if raw_token_has_space_prefix is not None:
                        has_space_prefix = raw_token_has_space_prefix[i]
                    else:
                        has_space_prefix = self._token_has_space_prefix_id(token_id)
                    if pending_space_prefix_from_whitespace:
                        has_space_prefix = True
                    if self.space_isolating_tokenizer:
                        has_space_prefix = pending_space_prefix_from_whitespace or (
                            i > 0 and raw_token_ids[i - 1] in self.single_space_token_ids
                        )
                    space_idx = 1 if has_space_prefix else 0
                    pending_prefixes.append(('prep_space_prefix', space_idx, False))
                pending_prefix_token_ids.append(token_id)
                prefix_modifier_count += 1
                pending_space_prefix_from_whitespace = False
                i += 1
                continue

            # Handle split prefixes (e.g., "T"+"he", "H"+"is") that sequence-map
            # into a single article/preposition base token with only surface (space/cap)
            # modifiers. These should remain detachable prefixes, not emitted bases.
            match_len = 0
            base_ids = None
            modifier = None
            t_match = time.perf_counter() if profile is not None else 0.0
            if self.word_boundary_safety:
                if (
                    raw_token_has_space_prefix is not None
                    and raw_token_has_word_char is not None
                    and raw_space_prefix_prefix_sum is not None
                ):
                    match_len, base_ids, modifier = self._find_longest_boundary_safe_match(
                        raw_token_ids,
                        i,
                        raw_token_has_space_prefix,
                        raw_token_has_word_char,
                        raw_space_prefix_prefix_sum,
                    )
            else:
                match_len, base_ids, modifier = self.sequence_map.find_longest_match(raw_token_ids, i)
            if profile is not None:
                profile["ds_stream_match_lookup_s"] += time.perf_counter() - t_match

            preferred_lowercase_fallback = None
            if (
                match_len > 0
                and base_ids
                and modifier
                and self._should_prefer_cap_fallback_over_match(
                    raw_token_ids,
                    i,
                    match_len,
                    modifier,
                    raw_token_has_space_prefix,
                    raw_token_has_word_char,
                )
            ):
                span_end = cap_run_end_by_idx[i] if cap_run_end_by_idx is not None else None
                preferred_lowercase_fallback = self._try_lowercase_cap_fallback(
                    raw_token_ids,
                    i,
                    span_end_idx=span_end,
                )
                if preferred_lowercase_fallback is not None:
                    match_len = 0
                    base_ids = None
                    modifier = None

            if (
                match_len > 1
                and base_ids
                and modifier
                and len(base_ids) == 1
            ):
                base_tok_id = int(base_ids[0])
                if self._modifier_has_only_surface_groups(modifier):
                    seq_surface = self.base_tokenizer.decode(
                        raw_token_ids[i:i + match_len],
                        skip_special_tokens=False,
                        clean_up_tokenization_spaces=False,
                    )
                    seq_surface_stripped = seq_surface.lstrip()
                    seq_is_cap = bool(seq_surface_stripped) and seq_surface_stripped[0].isupper()
                    if not seq_is_cap and self.base_cap_group_idx >= 0:
                        seq_is_cap = int(modifier[self.base_cap_group_idx]) > 0
                    prev_is_single_space = (
                        i > 0 and raw_token_ids[i - 1] in self.single_space_token_ids
                    )
                    if self.space_isolating_tokenizer:
                        seq_has_space_prefix = _expr_space_from_context(prev_is_single_space)
                    else:
                        seq_has_space_prefix = (
                            pending_space_prefix_from_whitespace
                            or bool(seq_surface.startswith(" "))
                            or prev_is_single_space
                        )

                    if (
                        base_tok_id in self.article_token_ids
                        and base_tok_id in self.article_transforms
                        and _can_attach_detached_modifier(i, match_len)
                    ):
                        group_name, rel_idx = self.article_transforms[base_tok_id]
                        pending_prefixes.append((group_name, rel_idx, seq_is_cap))
                        if seq_is_cap and self.article_cap_group_idx >= 0:
                            pending_prefixes.append(('article_capitalization', 1, False))
                        if self.article_space_group_idx >= 0:
                            pending_prefixes.append(('article_space_prefix', 1 if seq_has_space_prefix else 0, False))
                        pending_prefix_token_ids.extend(raw_token_ids[i:i + match_len])
                        prefix_modifier_count += 1
                        pending_space_prefix_from_whitespace = False
                        i += match_len
                        continue

                    if (
                        base_tok_id in self.preposition_token_ids
                        and base_tok_id in self.preposition_transforms
                        and _can_attach_detached_modifier(i, match_len)
                    ):
                        group_name, rel_idx = self.preposition_transforms[base_tok_id]
                        pending_prefixes.append((group_name, rel_idx, seq_is_cap))
                        if seq_is_cap and self.prep_cap_group_idx >= 0:
                            pending_prefixes.append(('prep_capitalization', 1, False))
                        if self.prep_space_group_idx >= 0:
                            pending_prefixes.append(('prep_space_prefix', 1 if seq_has_space_prefix else 0, False))
                        pending_prefix_token_ids.extend(raw_token_ids[i:i + match_len])
                        prefix_modifier_count += 1
                        pending_space_prefix_from_whitespace = False
                        i += match_len
                        continue

            if (
                match_len > 0
                and base_ids
                and modifier
                and self.space_isolating_tokenizer
                and self.space_group_idx >= 0
                and token_id not in self.whitespace_only_token_ids
            ):
                prev_is_single_space = i > 0 and raw_token_ids[i - 1] in self.single_space_token_ids
                seq_has_space_prefix = _expr_space_from_context(prev_is_single_space)
                modifier = list(modifier)
                modifier[self.space_group_idx] = (
                    self.space_with_rel_idx if (seq_has_space_prefix and self.space_with_rel_idx >= 0) else 0
                )
            if match_len > 0 and base_ids and modifier:
                modifier = _strip_invalid_detached_modifier_groups(list(modifier), i, match_len)

            if match_len > 0 and base_ids and modifier:
                # Found a sequence match
                sequence_match_count += 1
                if len(base_ids) > 1:
                    multi_token_base_count += 1
                    multi_token_base_token_count += len(base_ids)
                if (not pending_prefixes) and len(base_ids) == 1:
                    t_reconstruct = time.perf_counter() if profile is not None else 0.0
                    output_ids.append(base_ids[0])
                    output_modifiers.append(list(modifier))
                    pending_prefixes.clear()
                    pending_prefix_token_ids.clear()
                    pending_space_prefix_from_whitespace = False
                    if profile is not None:
                        profile["ds_stream_reconstruct_s"] += time.perf_counter() - t_reconstruct
                    i += match_len
                    continue
                t_reconstruct = time.perf_counter() if profile is not None else 0.0
                combined_modifier = self._combine_modifiers(modifier, pending_prefixes)
                base_len = len(base_ids)
                modifiers_per_token = self._spread_multi_token_modifiers(combined_modifier, base_len)
                for base_id, token_modifier in zip(base_ids, modifiers_per_token):
                    output_ids.append(base_id)
                    output_modifiers.append(token_modifier)
                if profile is not None:
                    profile["ds_stream_reconstruct_s"] += time.perf_counter() - t_reconstruct

                pending_prefixes.clear()
                pending_prefix_token_ids.clear()
                pending_space_prefix_from_whitespace = False
                i += match_len
                continue

            # ID-first policy: try sequence-map rewrites first; only then fallback.
            if self._token_has_word_char_id(token_id):
                if cap_has_cap_from_idx is not None:
                    has_cap_token = bool(cap_has_cap_from_idx[i])
                    cap_span_end = cap_run_end_by_idx[i] if cap_run_end_by_idx is not None else None
                else:
                    has_cap_token = self._is_capitalized_token_text(self._get_token_decoded_text(token_id))
                    probe_idx = i + 1
                    while (not has_cap_token) and probe_idx < len(raw_token_ids):
                        probe_tok = int(raw_token_ids[probe_idx])
                        if self._token_has_space_prefix_id(probe_tok) or (not self._token_has_word_char_id(probe_tok)):
                            break
                        has_cap_token = self._is_capitalized_token_text(self._get_token_decoded_text(probe_tok))
                        probe_idx += 1
                    cap_span_end = None

                if has_cap_token:
                    lowercase_fallback = preferred_lowercase_fallback
                    if lowercase_fallback is None:
                        lowercase_fallback = self._try_lowercase_cap_fallback(
                            raw_token_ids,
                            i,
                            span_end_idx=cap_span_end,
                        )
                    if lowercase_fallback is not None:
                        (
                            fallback_len,
                            fallback_base_ids,
                            fallback_modifiers,
                        ) = lowercase_fallback
                        if len(fallback_base_ids) == 1:
                            fallback_base_tok_id = int(fallback_base_ids[0])
                            fallback_is_cap = (
                                self.base_cap_group_idx >= 0
                                and len(fallback_modifiers) > 0
                                and int(fallback_modifiers[0][self.base_cap_group_idx]) > 0
                            )
                            prev_is_single_space = (
                                i > 0 and raw_token_ids[i - 1] in self.single_space_token_ids
                            )
                            if raw_token_has_space_prefix is not None:
                                has_space_prefix = raw_token_has_space_prefix[i]
                            else:
                                has_space_prefix = self._token_has_space_prefix_id(token_id)
                            fallback_has_space_prefix = _expr_space_from_context(prev_is_single_space)
                            # For whitespace-isolating tokenizers (space_cap), only literal
                            # single-space context should trigger space-prefix in cap fallback.
                            if not self.space_isolating_tokenizer:
                                fallback_has_space_prefix = fallback_has_space_prefix or has_space_prefix
                            if (
                                fallback_base_tok_id in self.article_token_ids
                                and fallback_base_tok_id in self.article_transforms
                                and _can_attach_detached_modifier(i, fallback_len)
                            ):
                                group_name, rel_idx = self.article_transforms[fallback_base_tok_id]
                                pending_prefixes.append((group_name, rel_idx, fallback_is_cap))
                                if fallback_is_cap and self.article_cap_group_idx >= 0:
                                    pending_prefixes.append(('article_capitalization', 1, False))
                                if self.article_space_group_idx >= 0:
                                    pending_prefixes.append(('article_space_prefix', 1 if fallback_has_space_prefix else 0, False))
                                pending_prefix_token_ids.extend(raw_token_ids[i:i + fallback_len])
                                prefix_modifier_count += 1
                                pending_space_prefix_from_whitespace = False
                                i += fallback_len
                                continue
                            if (
                                fallback_base_tok_id in self.preposition_token_ids
                                and fallback_base_tok_id in self.preposition_transforms
                                and _can_attach_detached_modifier(i, fallback_len)
                            ):
                                group_name, rel_idx = self.preposition_transforms[fallback_base_tok_id]
                                pending_prefixes.append((group_name, rel_idx, fallback_is_cap))
                                if fallback_is_cap and self.prep_cap_group_idx >= 0:
                                    pending_prefixes.append(('prep_capitalization', 1, False))
                                if self.prep_space_group_idx >= 0:
                                    pending_prefixes.append(('prep_space_prefix', 1 if fallback_has_space_prefix else 0, False))
                                pending_prefix_token_ids.extend(raw_token_ids[i:i + fallback_len])
                                prefix_modifier_count += 1
                                pending_space_prefix_from_whitespace = False
                                i += fallback_len
                                continue
                        if (not pending_prefixes) and len(fallback_base_ids) == 1:
                            t_reconstruct = time.perf_counter() if profile is not None else 0.0
                            output_ids.append(fallback_base_ids[0])
                            output_modifiers.append(list(fallback_modifiers[0]))
                            pending_prefixes.clear()
                            pending_prefix_token_ids.clear()
                            pending_space_prefix_from_whitespace = False
                            if profile is not None:
                                profile["ds_stream_reconstruct_s"] += time.perf_counter() - t_reconstruct
                            i += fallback_len
                            continue
                        t_reconstruct = time.perf_counter() if profile is not None else 0.0
                        fallback_modifiers = [list(row) for row in fallback_modifiers]
                        if pending_prefixes and fallback_modifiers:
                            fallback_modifiers[0] = self._combine_modifiers(
                                fallback_modifiers[0],
                                pending_prefixes,
                            )
                        for base_id, token_modifier in zip(fallback_base_ids, fallback_modifiers):
                            output_ids.append(base_id)
                            output_modifiers.append(token_modifier)
                        if profile is not None:
                            profile["ds_stream_reconstruct_s"] += time.perf_counter() - t_reconstruct
                        pending_prefixes.clear()
                        pending_prefix_token_ids.clear()
                        pending_space_prefix_from_whitespace = False
                        i += fallback_len
                        continue

            # Single token - no transformation, just attach pending prefixes
            t_reconstruct = time.perf_counter() if profile is not None else 0.0
            if (
                self.use_na_modifiers_for_non_decomposed
                and self.decomposition_token_ids is not None
                and token_id not in self.decomposition_token_ids
            ):
                base_modifier = self.na_modifier
            else:
                base_modifier = self.modifier_manager.create_empty_modifier()
            modifier = self._combine_modifiers(
                base_modifier,
                pending_prefixes
            )
            output_ids.append(token_id)
            output_modifiers.append(modifier)
            pending_prefixes.clear()
            pending_prefix_token_ids.clear()
            pending_space_prefix_from_whitespace = False
            if profile is not None:
                profile["ds_stream_reconstruct_s"] += time.perf_counter() - t_reconstruct
            i += 1

        # If the stream ended with standalone prefix tokens (e.g., trailing " A"),
        # keep them as literal tokens instead of dropping them.
        if pending_prefix_token_ids:
            t_reconstruct = time.perf_counter() if profile is not None else 0.0
            for tok in pending_prefix_token_ids:
                _emit_literal_token(tok)
            if profile is not None:
                profile["ds_stream_reconstruct_s"] += time.perf_counter() - t_reconstruct

        # Convert to numpy array
        t_finalize = time.perf_counter() if profile is not None else 0.0
        if output_modifiers:
            modifier_array = np.array(output_modifiers, dtype=self.modifier_dtype)
        else:
            modifier_array = np.zeros((0, self.modifier_manager.num_groups), dtype=self.modifier_dtype)
        if profile is not None:
            profile["ds_stream_finalize_s"] += time.perf_counter() - t_finalize

        if return_metadata:
            raw_len = len(raw_token_ids)
            compressed_len = len(output_ids)
            metadata = {
                "raw_len": raw_len,
                "compressed_len": compressed_len,
                "tokens_removed": raw_len - compressed_len,
                "sequence_match_count": sequence_match_count,
                "multi_token_base_count": multi_token_base_count,
                "multi_token_base_token_count": multi_token_base_token_count,
                "prefix_modifier_count": prefix_modifier_count,
                "suffix_modifier_count": suffix_modifier_count,
            }
            return output_ids, modifier_array, metadata

        return output_ids, modifier_array

    def _process_stream(
        self,
        raw_token_ids: List[int],
        return_metadata: bool = False,
        profile: Optional[Dict[str, float]] = None,
    ) -> Any:
        return self._process_stream_python(
            raw_token_ids,
            return_metadata=return_metadata,
            profile=profile,
        )

    def _combine_modifiers(self, base_modifier: List[int],
                           pending_prefixes: List[Tuple[str, int, bool]]) -> List[int]:
        """Combine base modifier with pending prefix modifiers.

        Args:
            base_modifier: Base modifier from sequence match.
            pending_prefixes: List of (group_name, relative_idx, is_cap) tuples.

        Returns:
            Combined modifier list.
        """
        if not pending_prefixes:
            # Defensive copy: output modifiers may be updated in-place later
            # (e.g., suffix punctuation attachment).
            return list(base_modifier)

        combined = list(base_modifier)

        # Apply pending prefixes (only closest one per group)
        applied_groups: Set[str] = set()

        # Process in reverse order (most recent = closest to the token)
        for group_name, rel_idx, _ in reversed(pending_prefixes):
            if group_name not in applied_groups:
                self.modifier_manager.set_group_value(combined, group_name, rel_idx)
                applied_groups.add(group_name)

        return combined

    def _spread_multi_token_modifiers(self, combined_modifier: List[int], base_len: int) -> List[List[int]]:
        """Distribute modifiers across a multi-token base according to the attachment mode."""
        if base_len <= 1:
            return [combined_modifier]

        empty_modifier = self.modifier_manager.create_empty_modifier()

        if self.multi_token_modifier_position == "last":
            return [empty_modifier for _ in range(base_len - 1)] + [combined_modifier]

        if self.multi_token_modifier_position == "first":
            return [combined_modifier] + [empty_modifier for _ in range(base_len - 1)]

        if self.multi_token_modifier_position == "split_by_role":
            first_modifier = self.modifier_manager.create_empty_modifier()
            last_modifier = self.modifier_manager.create_empty_modifier()
            for idx, group_name in enumerate(self.modifier_manager.groups):
                value = combined_modifier[idx]
                if value <= 0:
                    continue
                if group_name in self._multi_token_first_groups:
                    first_modifier[idx] = value
                else:
                    last_modifier[idx] = value
            if base_len == 2:
                return [first_modifier, last_modifier]
            return [first_modifier] + [empty_modifier for _ in range(base_len - 2)] + [last_modifier]

        raise ValueError(f"Unknown multi_token_modifier_position: {self.multi_token_modifier_position}")

    def _token_has_space_prefix(self, token: str) -> bool:
        return token.startswith("Ġ") or token.startswith("▁") or token.startswith(" ")

    def _token_has_word_char(self, token: str) -> bool:
        stripped = token.lstrip("Ġ ▁")
        for ch in stripped:
            if ch.isalnum():
                return True
        return False

    def _find_longest_boundary_safe_match(
        self,
        raw_token_ids: List[int],
        start_idx: int,
        token_has_space_prefix: List[bool],
        token_has_word_char: List[bool],
        space_prefix_prefix_sum: List[int],
    ) -> Tuple[int, Optional[List[int]], Optional[List[int]]]:
        """Find the longest trie match that respects token-level word boundaries."""
        n_tokens = len(raw_token_ids)
        if start_idx >= n_tokens:
            return 0, None, None

        start_inside_word = bool(
            start_idx > 0
            and token_has_word_char[start_idx]
            and (not token_has_space_prefix[start_idx])
            and token_has_word_char[start_idx - 1]
        )

        node = self.sequence_map.root
        max_end = min(start_idx + self.sequence_map.max_sequence_length, n_tokens)
        best_match_length = 0
        best_base_ids: Optional[List[int]] = None
        best_modifier: Optional[List[int]] = None

        for end_idx in range(start_idx, max_end):
            token_id = raw_token_ids[end_idx]
            node = node.children.get(token_id)
            if node is None:
                break

            if not node.is_end or node.base_token_ids is None or node.modifier is None:
                continue

            # Candidate span is [start_idx, end_idx], exclusive-end is span_end.
            span_end = end_idx + 1
            match_length = span_end - start_idx
            allow_intra_word_cap_alias = self._is_intra_word_cap_alias_match(match_length, node.modifier)

            if start_inside_word and (not allow_intra_word_cap_alias):
                continue

            # End boundary must be after the final token in the candidate span.
            if (
                span_end < n_tokens
                and token_has_word_char[end_idx]
                and (not token_has_space_prefix[span_end])
                and token_has_word_char[span_end]
            ):
                if allow_intra_word_cap_alias:
                    pass
                else:
                    continue

            # No internal token in the span may start with a space prefix.
            if match_length > 1:
                if all(token_has_word_char[j] for j in range(start_idx, span_end)):
                    if (space_prefix_prefix_sum[span_end] - space_prefix_prefix_sum[start_idx + 1]) > 0:
                        continue

            best_match_length = match_length
            best_base_ids = node.base_token_ids
            best_modifier = node.modifier

        return best_match_length, best_base_ids, best_modifier

    def _is_intra_word_cap_alias_match(self, match_length: int, modifier: Optional[List[int]]) -> bool:
        """Allow boundary-safe matching inside words for single-token cap aliases."""
        if match_length != 1 or modifier is None:
            return False
        if self.base_cap_group_idx < 0 or self.base_cap_add_rel_idx < 0:
            return False
        if not (0 <= self.base_cap_group_idx < len(modifier)):
            return False
        if int(modifier[self.base_cap_group_idx]) != int(self.base_cap_add_rel_idx):
            return False
        if not self._modifier_has_only_surface_groups(modifier):
            return False
        return True

    def _should_prefer_cap_fallback_over_match(
        self,
        raw_token_ids: List[int],
        start_idx: int,
        match_length: int,
        modifier: Optional[List[int]],
        token_has_space_prefix: Optional[List[bool]],
        token_has_word_char: Optional[List[bool]],
    ) -> bool:
        """Prefer whole-word cap fallback over a one-token cap alias inside a larger word."""
        if not self._is_intra_word_cap_alias_match(match_length, modifier):
            return False
        if token_has_space_prefix is None or token_has_word_char is None:
            return False
        if not (0 <= start_idx < len(raw_token_ids)):
            return False
        if not token_has_word_char[start_idx]:
            return False

        prev_continues_word = bool(
            start_idx > 0
            and token_has_word_char[start_idx - 1]
            and not token_has_space_prefix[start_idx]
        )
        next_idx = start_idx + match_length
        next_continues_word = bool(
            next_idx < len(raw_token_ids)
            and token_has_word_char[next_idx]
            and not token_has_space_prefix[next_idx]
        )
        return prev_continues_word or next_continues_word

    def _is_word_boundary_start(self, idx: int, raw_tokens: List[str]) -> bool:
        if idx == 0:
            return True
        token = raw_tokens[idx]
        if self._token_has_space_prefix(token):
            return True
        prev = raw_tokens[idx - 1]
        return not self._token_has_word_char(prev)

    def _is_word_boundary_end(self, end_idx: int, raw_tokens: List[str]) -> bool:
        if end_idx >= len(raw_tokens):
            return True
        token = raw_tokens[end_idx]
        if self._token_has_space_prefix(token):
            return True
        return not self._token_has_word_char(token)

    def _match_respects_word_boundaries(self, raw_tokens: List[str], start_idx: int, match_len: int) -> bool:
        if not self._is_word_boundary_start(start_idx, raw_tokens):
            return False
        end_idx = start_idx + match_len
        if not self._is_word_boundary_end(end_idx, raw_tokens):
            return False
        for j in range(start_idx + 1, end_idx):
            if self._token_has_space_prefix(raw_tokens[j]):
                return False
        return True

    def _infer_possessive_plural_rel_idx(self) -> int:
        """Best-effort fallback index for plural possessive in dedicated possessive group."""
        group_size = int(self.modifier_manager.group_sizes.get("possessives", 0))
        if group_size >= 3:
            return 2  # expected: no_possessive, possessive_'s, possessive_s'
        if group_size >= 2:
            return 1
        return -1

    def _attach_plural_possessive_if_contextual(
        self,
        raw_token_ids: List[int],
        raw_tokens: Optional[List[str]],
        i: int,
        output_ids: List[int],
        output_modifiers: List[List[int]],
        preferred_entry: Optional[Tuple[str, int]] = None,
    ) -> bool:
        """Attach apostrophe as plural possessive when context strongly indicates it."""
        if not output_ids or not output_modifiers:
            return False

        prev_text = self.base_tokenizer.decode(
            [int(output_ids[-1])],
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        ).strip().lower()
        if not prev_text.endswith("s"):
            return False

        # Look ahead to next non-whitespace token. Possessive apostrophe should
        # be followed by a word with an explicit boundary (space token/prefix).
        j = i + 1
        saw_whitespace = False
        while j < len(raw_token_ids):
            next_id = raw_token_ids[j]
            if next_id in self.whitespace_only_token_ids:
                saw_whitespace = True
                j += 1
                continue
            next_tok = (
                raw_tokens[j]
                if (raw_tokens is not None and j < len(raw_tokens))
                else self._get_token_text(next_id)
            )
            if not next_tok.strip():
                saw_whitespace = True
                j += 1
                continue
            if not self._token_has_word_char_id(next_id):
                return False
            if not (saw_whitespace or self._token_has_space_prefix(next_tok)):
                return False
            break
        if j >= len(raw_token_ids):
            return False

        if preferred_entry is not None:
            group_name, rel_idx = preferred_entry
            self.modifier_manager.set_group_value(output_modifiers[-1], group_name, rel_idx)
            return True

        if self.suffix_possessive_plural:
            group_name, rel_idx = self.suffix_possessive_plural
            self.modifier_manager.set_group_value(output_modifiers[-1], group_name, rel_idx)
            return True

        # Fallback when no dedicated s' token exists in vocab but possessives group is active.
        group_idx = self.modifier_manager.group_to_idx.get("possessives", -1)
        if group_idx >= 0:
            rel_idx = self._infer_possessive_plural_rel_idx()
            if rel_idx > 0:
                output_modifiers[-1][group_idx] = rel_idx
                return True

        return False

    def _normalize_modifier(self, modifier: List[int], ignore_group_indices: Set[int]) -> Tuple[int, ...]:
        if not ignore_group_indices:
            return tuple(int(v) for v in modifier)
        normalized = list(modifier)
        for idx in ignore_group_indices:
            if 0 <= idx < len(normalized):
                normalized[idx] = 0
        return tuple(int(v) for v in normalized)

    def _get_reverse_sequence_map(self, ignore_group_indices: Set[int]) -> Tuple[Dict[Tuple[Tuple[int, ...], Tuple[int, ...]], Tuple[int, ...]], int]:
        cache = getattr(self, "_reverse_sequence_map_cache", {})
        cache_key = tuple(sorted(ignore_group_indices))
        if cache_key in cache:
            return cache[cache_key]

        reverse_map: Dict[Tuple[Tuple[int, ...], Tuple[int, ...]], Tuple[int, ...]] = {}
        max_base_len = 1
        data = self.sequence_map.to_dict()
        for entry in data.get("sequences", []):
            base_ids = tuple(entry.get("base_ids", []))
            modifier = entry.get("modifier", [])
            if not base_ids:
                continue
            norm_modifier = self._normalize_modifier(modifier, ignore_group_indices)
            variant_ids = tuple(entry.get("token_ids", []))
            key = (base_ids, norm_modifier)
            existing = reverse_map.get(key)
            if existing is None or variant_ids < existing:
                reverse_map[key] = variant_ids
            if len(base_ids) > max_base_len:
                max_base_len = len(base_ids)

        cache[cache_key] = (reverse_map, max_base_len)
        self._reverse_sequence_map_cache = cache
        return reverse_map, max_base_len

    def decode(self, token_ids: List[int]) -> str:
        """Decode token IDs back to text.

        Note: This only decodes the token IDs, not the modifiers.

        Args:
            token_ids: List of token IDs.

        Returns:
            Decoded text string.
        """
        return self.base_tokenizer.decode(token_ids)

    def decode_with_modifiers(self, token_ids: List[int], modifier_array: np.ndarray) -> str:
        """Reconstruct original text from compressed tokens and modifiers.

        This function reverses the compression by reinserting articles, prepositions,
        and punctuation based on the modifier array.

        Args:
            token_ids: List of compressed token IDs.
            modifier_array: Modifier array of shape (seq_len, num_groups).

        Returns:
            Reconstructed text string.
        """
        if len(token_ids) != modifier_array.shape[0]:
            raise ValueError(f"Token count {len(token_ids)} doesn't match modifier rows {modifier_array.shape[0]}")

        ignore_groups = {
            'determiners',
            'articles',
            'article_det',
            'article_capitalization',
            'article_space_prefix',
            'prepositions',
            'prep_capitalization',
            'prep_space_prefix',
            'prefix_punctuation',
            'suffix_punctuation',
            'possessives',
        }
        ignore_group_indices = {
            idx for name, idx in self.modifier_manager.group_to_idx.items()
            if name in ignore_groups
        }
        reverse_map, max_base_len = self._get_reverse_sequence_map(ignore_group_indices)

        def _token_meta(token_id: int) -> Tuple[bool, bool]:
            decoded = self._get_token_decoded_text(int(token_id))
            stripped = decoded.lstrip(" Ġ▁")
            if not stripped:
                token = self._get_token_text(int(token_id))
                stripped = token.lstrip(" Ġ▁")
            is_cap = stripped[:1].isupper()
            has_space = self._token_has_space_prefix_id(int(token_id))
            return is_cap, has_space

        def _pick_token_id(
            candidates: List[int],
            prefer_cap: Optional[bool] = None,
            require_space: Optional[bool] = None,
        ) -> Optional[int]:
            if not candidates:
                return None
            filtered = candidates
            if require_space is not None:
                filtered = [tid for tid in candidates if _token_meta(tid)[1] == require_space]
                if not filtered:
                    filtered = candidates
            best = None
            best_score = -1
            for tid in filtered:
                is_cap, _ = _token_meta(tid)
                score = 0
                if prefer_cap is not None and is_cap == prefer_cap:
                    score += 1
                if score > best_score or (score == best_score and (best is None or tid < best)):
                    best_score = score
                    best = tid
            return best

        def _append_token_with_space(tokens_out: List[int], token_id: Optional[int], require_space: bool) -> None:
            if token_id is None:
                return
            if require_space:
                _, has_space = _token_meta(token_id)
                if (not has_space) and (self.single_space_token_id is not None):
                    tokens_out.append(int(self.single_space_token_id))
            tokens_out.append(int(token_id))

        def _append_text_with_space(tokens_out: List[int], text: str, require_space: bool) -> bool:
            text = text.strip()
            if not text:
                return False
            prefixed_text = f" {text}" if require_space else text
            ids = self.base_tokenizer.encode(prefixed_text, add_special_tokens=False)
            if not ids:
                return False
            tokens_out.extend(int(tid) for tid in ids)
            return True

        def _append_literal_text(tokens_out: List[int], text: str) -> bool:
            if not text:
                return False
            ids = self.base_tokenizer.encode(text, add_special_tokens=False)
            if not ids:
                return False
            tokens_out.extend(int(tid) for tid in ids)
            return True

        article_ids_by_rel: Dict[int, List[int]] = defaultdict(list)
        for token_id, (group_name, rel_idx) in self.article_transforms.items():
            if group_name in {'determiners', 'articles', 'article_det'}:
                article_ids_by_rel[rel_idx].append(token_id)

        prep_ids_by_rel: Dict[int, List[int]] = defaultdict(list)
        for token_id, (group_name, rel_idx) in self.preposition_transforms.items():
            if group_name == 'prepositions':
                prep_ids_by_rel[rel_idx].append(token_id)

        def _build_rel_text_map(ids_by_rel: Dict[int, List[int]]) -> Dict[int, str]:
            rel_text: Dict[int, str] = {}
            for rel_idx, candidates in ids_by_rel.items():
                best: Optional[str] = None
                for tid in candidates:
                    surface = self.base_tokenizer.decode(
                        [int(tid)],
                        skip_special_tokens=False,
                        clean_up_tokenization_spaces=False,
                    ).strip()
                    if not surface:
                        continue
                    normalized = surface.lower()
                    if (best is None) or (len(normalized) < len(best)) or (
                        len(normalized) == len(best) and normalized < best
                    ):
                        best = normalized
                if best:
                    rel_text[int(rel_idx)] = best
            return rel_text

        article_text_by_rel = _build_rel_text_map(article_ids_by_rel)
        prep_text_by_rel = _build_rel_text_map(prep_ids_by_rel)

        def _transform_name_for_rel(group_name: str, rel_idx: int) -> Optional[str]:
            rel_map = self._group_rel_to_transform_name.get(group_name)
            if not rel_map:
                return None
            return rel_map.get(int(rel_idx))

        def _punct_text_for_rel(group_name: str, rel_idx: int) -> Optional[str]:
            tname = _transform_name_for_rel(group_name, rel_idx)
            if not tname:
                return None
            return _punct_surface_from_transform_name(tname)

        prefix_punct_ids_by_rel: Dict[int, List[int]] = defaultdict(list)
        for token_id, (group_name, rel_idx) in self.prefix_punct_transforms.items():
            if group_name == 'prefix_punctuation':
                prefix_punct_ids_by_rel[rel_idx].append(token_id)

        suffix_punct_ids_by_rel: Dict[int, List[int]] = defaultdict(list)
        for token_id, (group_name, rel_idx) in self.suffix_punct_transforms.items():
            if group_name == 'suffix_punctuation':
                suffix_punct_ids_by_rel[rel_idx].append(token_id)
        possessive_ids_by_rel: Dict[int, List[int]] = defaultdict(list)
        for token_id, (group_name, rel_idx) in self.suffix_punct_transforms.items():
            if group_name == 'possessives':
                possessive_ids_by_rel[rel_idx].append(token_id)

        space_group_idx = self.modifier_manager.group_to_idx.get('space_prefix', -1)

        def _base_has_space(base_ids: Tuple[int, ...]) -> bool:
            if not base_ids:
                return False
            return self._token_has_space_prefix_id(int(base_ids[0]))

        def _expr_has_space(base_ids: Tuple[int, ...], modifiers: List[int]) -> bool:
            base_has_space = _base_has_space(base_ids)
            if not (0 <= space_group_idx < len(modifiers)):
                return base_has_space
            rel_idx = int(modifiers[space_group_idx])
            # space_prefix group order: NO, WITH, REMOVE, NA
            if rel_idx == 1:
                return True
            if rel_idx == 2:
                return False
            return base_has_space

        def _space_prefix_setting(modifiers: List[int], group_name: str, fallback: bool) -> bool:
            group_idx = self.modifier_manager.group_to_idx.get(group_name, -1)
            if not (0 <= group_idx < len(modifiers)):
                return fallback
            rel_idx = int(modifiers[group_idx])
            group_size = self.modifier_manager.group_sizes.get(group_name, 0)
            if group_size and rel_idx == group_size - 1:
                return fallback
            return rel_idx == 1

        def _apply_base_capitalization(surface: str, rel_idx: int) -> str:
            if rel_idx <= 0 or not surface:
                return surface
            cap_name = self._group_rel_to_transform_name.get("base_capitalization", {}).get(rel_idx)
            if rel_idx == 1:
                chars = list(surface)
                for idx, ch in enumerate(chars):
                    if ch.isalpha():
                        chars[idx] = ch.upper()
                        return "".join(chars)
                return surface
            if cap_name == "remove_capitalization" or rel_idx == 3:
                chars = list(surface)
                for idx, ch in enumerate(chars):
                    if ch.isalpha():
                        chars[idx] = ch.lower()
                        return "".join(chars)
                return surface
            return surface

        def _synthesize_surface_variant(
            base_ids: Tuple[int, ...],
            modifiers: List[int],
        ) -> Optional[Tuple[int, ...]]:
            base_cap_idx = self.modifier_manager.group_to_idx.get('base_capitalization', -1)
            for gidx, gval in enumerate(modifiers):
                if int(gval) == 0:
                    continue
                if gidx == space_group_idx:
                    continue
                if gidx == base_cap_idx:
                    continue
                return None

            surface = self.base_tokenizer.decode(
                list(base_ids),
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )
            if not surface:
                return None
            if 0 <= base_cap_idx < len(modifiers):
                surface = _apply_base_capitalization(surface, int(modifiers[base_cap_idx]))
            expected_space = _expr_has_space(base_ids, modifiers)
            if expected_space and not surface[:1].isspace():
                surface = " " + surface
            if (not expected_space) and surface[:1].isspace():
                surface = surface.lstrip()
            ids = self.base_tokenizer.encode(surface, add_special_tokens=False)
            if not ids:
                return None
            return tuple(int(x) for x in ids)

        def _decode_ids_text(ids: Tuple[int, ...]) -> str:
            return self.base_tokenizer.decode(
                list(ids),
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )

        def _first_alpha_is_upper(text: str) -> Optional[bool]:
            for ch in text:
                if ch.isalpha():
                    return bool(ch.isupper())
            return None

        def _prefer_synth_variant(
            variant_ids: Tuple[int, ...],
            synth_ids: Tuple[int, ...],
            base_ids: Tuple[int, ...],
            modifiers: List[int],
        ) -> bool:
            if variant_ids == synth_ids:
                return False
            variant_text = _decode_ids_text(variant_ids)
            synth_text = _decode_ids_text(synth_ids)

            # Preserve explicit space_prefix semantics.
            expected_space = _expr_has_space(base_ids, modifiers)
            if bool(variant_text[:1].isspace()) != expected_space and bool(synth_text[:1].isspace()) == expected_space:
                return True

            if 0 <= base_cap_idx < len(modifiers):
                cap_rel = int(modifiers[base_cap_idx])
                if cap_rel in (1, 3):
                    v_upper = _first_alpha_is_upper(variant_text)
                    s_upper = _first_alpha_is_upper(synth_text)
                    if cap_rel == 1 and v_upper is False and s_upper is True:
                        return True
                    if cap_rel == 3 and v_upper is True and s_upper is False:
                        return True
            return False

        def _enforce_base_capitalization_on_variant(
            variant_ids: Tuple[int, ...],
            modifiers: List[int],
        ) -> Tuple[int, ...]:
            if not variant_ids:
                return variant_ids
            if not (0 <= base_cap_idx < len(modifiers)):
                return variant_ids
            rel_idx = int(modifiers[base_cap_idx])
            if rel_idx <= 0:
                return variant_ids
            variant_text = _decode_ids_text(variant_ids)
            adjusted_text = _apply_base_capitalization(variant_text, rel_idx)
            if adjusted_text == variant_text:
                return variant_ids
            adjusted_ids = self.base_tokenizer.encode(adjusted_text, add_special_tokens=False)
            if not adjusted_ids:
                return variant_ids
            return tuple(int(x) for x in adjusted_ids)

        def _build_prefix_tokens(
            modifiers: List[int],
            base_ids: Tuple[int, ...],
        ) -> Tuple[List[int], bool]:
            prefix_tokens: List[int] = []
            expr_has_space = _expr_has_space(base_ids, modifiers)
            space_for_next = expr_has_space
            require_space_before_base = False

            # Prefix punctuation
            group_idx = self.modifier_manager.group_to_idx.get('prefix_punctuation', -1)
            if 0 <= group_idx < len(modifiers):
                rel_idx = int(modifiers[group_idx])
                if rel_idx > 0:
                    require_space = space_for_next
                    inserted = False
                    punct_text = _punct_text_for_rel("prefix_punctuation", rel_idx)
                    if punct_text:
                        inserted = _append_text_with_space(prefix_tokens, punct_text, require_space=require_space)
                    if not inserted:
                        tid = _pick_token_id(
                            prefix_punct_ids_by_rel.get(rel_idx, []),
                            require_space=require_space,
                        )
                        if tid is not None:
                            _append_token_with_space(prefix_tokens, tid, require_space=require_space)
                            inserted = True
                    # Prefix punctuation attaches to the following word.
                    if inserted:
                        space_for_next = False

            # Preposition (with capitalization)
            group_idx = self.modifier_manager.group_to_idx.get('prepositions', -1)
            if 0 <= group_idx < len(modifiers):
                rel_idx = int(modifiers[group_idx])
                if rel_idx > 0:
                    cap_idx = self.modifier_manager.group_to_idx.get('prep_capitalization', -1)
                    use_cap = bool(0 <= cap_idx < len(modifiers) and int(modifiers[cap_idx]) > 0)
                    require_space = _space_prefix_setting(modifiers, 'prep_space_prefix', space_for_next)
                    prep_text = prep_text_by_rel.get(rel_idx)
                    inserted = False
                    if prep_text:
                        surface = prep_text[:1].upper() + prep_text[1:] if use_cap else prep_text
                        inserted = _append_text_with_space(prefix_tokens, surface, require_space=require_space)
                    if not inserted:
                        tid = _pick_token_id(
                            prep_ids_by_rel.get(rel_idx, []),
                            prefer_cap=use_cap if use_cap else None,
                            require_space=require_space,
                        )
                        _append_token_with_space(prefix_tokens, tid, require_space=require_space)
                        inserted = tid is not None
                    if inserted:
                        require_space_before_base = True
                    space_for_next = True

            # Article (with capitalization)
            group_idx = self.modifier_manager.group_to_idx.get('determiners', -1)
            if group_idx < 0:
                group_idx = self.modifier_manager.group_to_idx.get('article_det', -1)
            if group_idx < 0:
                group_idx = self.modifier_manager.group_to_idx.get('articles', -1)
            if 0 <= group_idx < len(modifiers):
                rel_idx = int(modifiers[group_idx])
                if rel_idx > 0:
                    cap_idx = self.modifier_manager.group_to_idx.get('article_capitalization', -1)
                    use_cap = bool(0 <= cap_idx < len(modifiers) and int(modifiers[cap_idx]) > 0)
                    require_space = _space_prefix_setting(modifiers, 'article_space_prefix', space_for_next)
                    article_text = article_text_by_rel.get(rel_idx)
                    inserted = False
                    if article_text:
                        surface = article_text[:1].upper() + article_text[1:] if use_cap else article_text
                        inserted = _append_text_with_space(prefix_tokens, surface, require_space=require_space)
                    if not inserted:
                        tid = _pick_token_id(
                            article_ids_by_rel.get(rel_idx, []),
                            prefer_cap=use_cap if use_cap else None,
                            require_space=require_space,
                        )
                        _append_token_with_space(prefix_tokens, tid, require_space=require_space)
                        inserted = tid is not None
                    if inserted:
                        require_space_before_base = True
                    space_for_next = True

            return prefix_tokens, require_space_before_base

        def _has_prefix_punct(modifiers: List[int]) -> bool:
            group_idx = self.modifier_manager.group_to_idx.get('prefix_punctuation', -1)
            return 0 <= group_idx < len(modifiers) and int(modifiers[group_idx]) > 0

        def _space_only_modifier(modifiers: List[int]) -> bool:
            if not (0 <= space_group_idx < len(modifiers)):
                return False
            if int(modifiers[space_group_idx]) == 0:
                return False
            for gidx, gval in enumerate(modifiers):
                if gidx == space_group_idx:
                    continue
                if int(gval) != 0:
                    return False
            return True

        def _combine_modifier_span(start: int, length: int) -> List[int]:
            combined = [0] * self.modifier_manager.num_groups
            for group_idx in range(self.modifier_manager.num_groups):
                for offset in range(length):
                    val = int(modifier_array[start + offset][group_idx])
                    if val != 0:
                        combined[group_idx] = val
                        break
            return combined

        multi_token_first_group_indices = {
            idx
            for name, idx in self.modifier_manager.group_to_idx.items()
            if name in self._multi_token_first_groups
        }

        def _row_has_non_zero(row: np.ndarray) -> bool:
            for val in row:
                if int(val) != 0:
                    return True
            return False

        def _span_layout_valid(start: int, length: int) -> bool:
            """Check whether modifier placement over a span matches encode layout."""
            if length <= 1:
                return True

            if self.multi_token_modifier_position == "first":
                for off in range(1, length):
                    if _row_has_non_zero(modifier_array[start + off]):
                        return False
                return True

            if self.multi_token_modifier_position == "last":
                for off in range(0, length - 1):
                    if _row_has_non_zero(modifier_array[start + off]):
                        return False
                return True

            # split_by_role: first-token groups can only be on first row,
            # all other groups can only be on last row, interior rows must be empty.
            # In particular, do not merge spans that carry independent base-cap rows:
            # a span-level reverse-map hit can only reconstruct one base-cap decision.
            for off in range(length):
                row = modifier_array[start + off]
                if not _row_has_non_zero(row):
                    continue
                if off == 0:
                    for gidx, val in enumerate(row):
                        if int(val) == 0:
                            continue
                        if gidx not in multi_token_first_group_indices:
                            return False
                    continue
                if off == (length - 1):
                    for gidx, val in enumerate(row):
                        if int(val) == 0:
                            continue
                        if gidx in multi_token_first_group_indices:
                            return False
                    continue
                for gidx, val in enumerate(row):
                    if int(val) == 0:
                        continue
                    return False
                continue
            return True

        debug_decode = os.environ.get("VD_DECODE_DEBUG", "0") == "1"
        debug_decode_limit = 40
        debug_decode_count = 0
        debug_span = os.environ.get("VD_DECODE_SPAN_DEBUG", "0") == "1"
        debug_span_attempted_multi = 0
        debug_span_rejected_layout = 0
        debug_span_selected_multi = 0
        base_cap_idx = self.modifier_manager.group_to_idx.get('base_capitalization', -1)

        def _wrapper_space_only_modifier(modifiers: List[int]) -> bool:
            for gidx, gval in enumerate(modifiers):
                if int(gval) == 0:
                    continue
                if gidx == space_group_idx:
                    continue
                if gidx in ignore_group_indices:
                    continue
                return False
            return True

        reconstructed_ids: List[int] = []
        i = 0
        while i < len(token_ids):
            matched = False
            max_len = min(max_base_len, len(token_ids) - i)

            for length in range(max_len, 0, -1):
                if length > 1:
                    debug_span_attempted_multi += 1
                    if not _span_layout_valid(i, length):
                        debug_span_rejected_layout += 1
                        continue
                base_ids = tuple(token_ids[i:i + length])
                modifiers = _combine_modifier_span(i, length)
                needs_variant_lookup = False
                for gidx, gval in enumerate(modifiers):
                    if int(gval) == 0:
                        continue
                    if gidx in ignore_group_indices:
                        continue
                    if gidx == space_group_idx:
                        continue
                    needs_variant_lookup = True
                    break
                if needs_variant_lookup:
                    cap_space_only = True
                    for gidx, gval in enumerate(modifiers):
                        if int(gval) == 0:
                            continue
                        if gidx in ignore_group_indices:
                            continue
                        if gidx in {space_group_idx, base_cap_idx}:
                            continue
                        cap_space_only = False
                        break

                    modifier_for_variant = modifiers
                    if _has_prefix_punct(modifiers) and 0 <= space_group_idx < len(modifiers):
                        modifier_for_variant = list(modifiers)
                        modifier_for_variant[space_group_idx] = 0
                    variant_ids: Optional[Tuple[int, ...]] = None
                    synth_ids: Optional[Tuple[int, ...]] = None
                    norm_modifier = self._normalize_modifier(modifier_for_variant, ignore_group_indices)
                    variant_ids = reverse_map.get((base_ids, norm_modifier))
                    if variant_ids is None and modifier_for_variant is not modifiers:
                        norm_modifier = self._normalize_modifier(modifiers, ignore_group_indices)
                        variant_ids = reverse_map.get((base_ids, norm_modifier))
                    # Synthetic cap/space reconstruction is safe only for single-token spans.
                    # For multi-token spans, require an explicit reverse-map hit to avoid
                    # collapsing unrelated neighboring tokens.
                    if cap_space_only and length == 1:
                        synth_ids = _synthesize_surface_variant(base_ids, modifier_for_variant)
                        if variant_ids is None:
                            variant_ids = synth_ids
                        elif synth_ids is not None and _prefer_synth_variant(variant_ids, synth_ids, base_ids, modifiers):
                            variant_ids = synth_ids
                    if variant_ids is None:
                        if length == 1:
                            variant_ids = base_ids
                        else:
                            continue
                    variant_ids = _enforce_base_capitalization_on_variant(variant_ids, modifier_for_variant)
                else:
                    needs_surface_synthesis = False
                    if 0 <= space_group_idx < len(modifiers) and int(modifiers[space_group_idx]) != 0:
                        needs_surface_synthesis = True
                    if 0 <= base_cap_idx < len(modifiers) and int(modifiers[base_cap_idx]) != 0:
                        needs_surface_synthesis = True
                    if needs_surface_synthesis:
                        modifier_for_variant = modifiers
                        if _has_prefix_punct(modifiers) and 0 <= space_group_idx < len(modifiers):
                            modifier_for_variant = list(modifiers)
                            modifier_for_variant[space_group_idx] = 0
                        variant_ids: Optional[Tuple[int, ...]]
                        # Space-only rows should preserve lexical surface; only spacing
                        # should be materialized around the base token.
                        if length == 1 and _space_only_modifier(modifier_for_variant):
                            variant_ids = base_ids
                        elif length == 1 and _wrapper_space_only_modifier(modifier_for_variant):
                            synth_modifier = list(modifier_for_variant)
                            for group_idx in ignore_group_indices:
                                if 0 <= group_idx < len(synth_modifier):
                                    synth_modifier[group_idx] = 0
                            variant_ids = _synthesize_surface_variant(base_ids, synth_modifier)
                            if variant_ids is None:
                                variant_ids = base_ids
                        else:
                            norm_modifier = self._normalize_modifier(modifier_for_variant, ignore_group_indices)
                            variant_ids = reverse_map.get((base_ids, norm_modifier))
                            if length == 1:
                                synth_ids = _synthesize_surface_variant(base_ids, modifier_for_variant)
                                if variant_ids is None:
                                    variant_ids = synth_ids
                                elif synth_ids is not None and _prefer_synth_variant(variant_ids, synth_ids, base_ids, modifiers):
                                    variant_ids = synth_ids
                                if variant_ids is None:
                                    variant_ids = base_ids
                            elif variant_ids is None:
                                continue
                        variant_ids = _enforce_base_capitalization_on_variant(variant_ids, modifier_for_variant)
                    else:
                        variant_ids = base_ids

                suffix_tokens: List[int] = []
                prefix_tokens, require_space_before_base = _build_prefix_tokens(modifiers, base_ids)

                if debug_decode and debug_decode_count < debug_decode_limit and length > 1:
                    try:
                        base_text = self.base_tokenizer.decode(list(base_ids), skip_special_tokens=False, clean_up_tokenization_spaces=False)
                    except Exception:
                        base_text = "<base_decode_err>"
                    try:
                        variant_text = self.base_tokenizer.decode(list(variant_ids), skip_special_tokens=False, clean_up_tokenization_spaces=False)
                    except Exception:
                        variant_text = "<variant_decode_err>"
                    print(
                        "[decode-debug] "
                        f"span_start={i} span_len={length} "
                        f"base_text={base_text!r} variant_text={variant_text!r} "
                        f"require_space_before_base={require_space_before_base} mods={modifiers}",
                        flush=True,
                    )
                    debug_decode_count += 1

                # Suffix punctuation
                group_idx = self.modifier_manager.group_to_idx.get('possessives', -1)
                if 0 <= group_idx < len(modifiers):
                    rel_idx = int(modifiers[group_idx])
                    if rel_idx > 0:
                        tid = _pick_token_id(possessive_ids_by_rel.get(rel_idx, []))
                        if tid is not None:
                            suffix_tokens.append(tid)
                group_idx = self.modifier_manager.group_to_idx.get('suffix_punctuation', -1)
                if 0 <= group_idx < len(modifiers):
                    rel_idx = int(modifiers[group_idx])
                    if rel_idx > 0:
                        tid = _pick_token_id(suffix_punct_ids_by_rel.get(rel_idx, []))
                        if tid is not None:
                            suffix_tokens.append(tid)
                        else:
                            punct_text = _punct_text_for_rel("suffix_punctuation", rel_idx)
                            if punct_text:
                                _append_literal_text(suffix_tokens, punct_text)

                if (
                    prefix_tokens
                    and (self.single_space_token_id is not None)
                    and int(prefix_tokens[0]) == int(self.single_space_token_id)
                    and reconstructed_ids
                    and (int(reconstructed_ids[-1]) in self.whitespace_only_token_ids)
                ):
                    prefix_tokens = prefix_tokens[1:]
                reconstructed_ids.extend(prefix_tokens)
                if (not prefix_tokens) and (not require_space_before_base) and variant_ids:
                    needs_expr_space = _expr_has_space(base_ids, modifiers)
                    _, has_space = _token_meta(int(variant_ids[0]))
                    prev_is_ws = bool(reconstructed_ids) and (int(reconstructed_ids[-1]) in self.whitespace_only_token_ids)
                    if (
                        needs_expr_space
                        and (not has_space)
                        and (self.single_space_token_id is not None)
                        and (not prev_is_ws)
                    ):
                        reconstructed_ids.append(int(self.single_space_token_id))
                if require_space_before_base and variant_ids:
                    _, has_space = _token_meta(int(variant_ids[0]))
                    prev_is_ws = bool(reconstructed_ids) and (int(reconstructed_ids[-1]) in self.whitespace_only_token_ids)
                    if (not has_space) and (self.single_space_token_id is not None) and (not prev_is_ws):
                        reconstructed_ids.append(int(self.single_space_token_id))
                reconstructed_ids.extend(list(variant_ids))
                reconstructed_ids.extend(suffix_tokens)
                matched = True
                if length > 1:
                    debug_span_selected_multi += 1
                i += length
                break

            if not matched:
                modifiers = modifier_array[i]
                suffix_tokens: List[int] = []
                prefix_tokens, require_space_before_base = _build_prefix_tokens(modifiers, (token_ids[i],))

                group_idx = self.modifier_manager.group_to_idx.get('possessives', -1)
                if 0 <= group_idx < len(modifiers):
                    rel_idx = int(modifiers[group_idx])
                    if rel_idx > 0:
                        tid = _pick_token_id(possessive_ids_by_rel.get(rel_idx, []))
                        if tid is not None:
                            suffix_tokens.append(tid)
                group_idx = self.modifier_manager.group_to_idx.get('suffix_punctuation', -1)
                if 0 <= group_idx < len(modifiers):
                    rel_idx = int(modifiers[group_idx])
                    if rel_idx > 0:
                        tid = _pick_token_id(suffix_punct_ids_by_rel.get(rel_idx, []))
                        if tid is not None:
                            suffix_tokens.append(tid)
                        else:
                            punct_text = _punct_text_for_rel("suffix_punctuation", rel_idx)
                            if punct_text:
                                _append_literal_text(suffix_tokens, punct_text)

                if (
                    prefix_tokens
                    and (self.single_space_token_id is not None)
                    and int(prefix_tokens[0]) == int(self.single_space_token_id)
                    and reconstructed_ids
                    and (int(reconstructed_ids[-1]) in self.whitespace_only_token_ids)
                ):
                    prefix_tokens = prefix_tokens[1:]
                reconstructed_ids.extend(prefix_tokens)
                if require_space_before_base:
                    _, has_space = _token_meta(int(token_ids[i]))
                    prev_is_ws = bool(reconstructed_ids) and (int(reconstructed_ids[-1]) in self.whitespace_only_token_ids)
                    if (not has_space) and (self.single_space_token_id is not None) and (not prev_is_ws):
                        reconstructed_ids.append(int(self.single_space_token_id))
                reconstructed_ids.append(token_ids[i])
                reconstructed_ids.extend(suffix_tokens)
                i += 1

        if debug_span:
            print(
                "[decode-span-debug] "
                f"tokens={len(token_ids)} attempted_multi={debug_span_attempted_multi} "
                f"rejected_layout={debug_span_rejected_layout} selected_multi={debug_span_selected_multi}",
                flush=True,
            )

        return self.base_tokenizer.decode(
            reconstructed_ids,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )

    def get_reverse_collisions(
        self,
        max_entries: int = 50,
        max_variants_per_entry: int = 6,
        ignore_groups: Optional[Set[str]] = None,
    ) -> Dict[str, Any]:
        """Report collisions in the reverse map (base_ids + modifiers -> multiple variants)."""
        if ignore_groups is None:
            ignore_groups = {
                'determiners',
                'articles',
                'article_det',
                'article_capitalization',
                'article_space_prefix',
                'prepositions',
                'prep_capitalization',
                'prep_space_prefix',
                'prefix_punctuation',
                'suffix_punctuation',
                'possessives',
            }

        ignore_group_indices = {
            idx for name, idx in self.modifier_manager.group_to_idx.items()
            if name in ignore_groups
        }

        collisions: Dict[Tuple[Tuple[int, ...], Tuple[int, ...]], Set[Tuple[int, ...]]] = defaultdict(set)
        data = self.sequence_map.to_dict()
        for entry in data.get("sequences", []):
            base_ids = tuple(entry.get("base_ids", []))
            modifier = entry.get("modifier", [])
            token_ids = tuple(entry.get("token_ids", []))
            if not base_ids or not token_ids:
                continue
            norm_modifier = self._normalize_modifier(modifier, ignore_group_indices)
            collisions[(base_ids, norm_modifier)].add(token_ids)

        collision_items = []
        for (base_ids, norm_modifier), variants in collisions.items():
            if len(variants) <= 1:
                continue
            collision_items.append((base_ids, norm_modifier, sorted(variants)))

        collision_items.sort(key=lambda item: len(item[2]), reverse=True)

        results = []
        for base_ids, norm_modifier, variants in collision_items[:max_entries]:
            base_tokens = self.base_tokenizer.convert_ids_to_tokens(list(base_ids))
            base_text = self.base_tokenizer.decode(list(base_ids))
            variant_entries = []
            for variant_ids in variants[:max_variants_per_entry]:
                variant_tokens = self.base_tokenizer.convert_ids_to_tokens(list(variant_ids))
                variant_text = self.base_tokenizer.decode(list(variant_ids))
                variant_entries.append({
                    "variant_ids": list(variant_ids),
                    "variant_tokens": variant_tokens,
                    "variant_text": variant_text,
                })
            surface_texts = sorted({entry["variant_text"] for entry in variant_entries})
            results.append({
                "base_ids": list(base_ids),
                "base_tokens": base_tokens,
                "base_text": base_text,
                "modifier": list(norm_modifier),
                "variant_count": len(variants),
                "unique_surface_texts": surface_texts,
                "variants": variant_entries,
            })

        return {
            "collision_count": len(collision_items),
            "ignored_groups": sorted(ignore_groups),
            "max_entries": max_entries,
            "max_variants_per_entry": max_variants_per_entry,
            "collisions": results,
        }

    @property
    def vocab_size(self) -> int:
        """Get the vocabulary size of the base tokenizer."""
        return self.base_tokenizer.vocab_size

    @property
    def bos_token_id(self) -> Optional[int]:
        """Get the BOS token ID."""
        return self.base_tokenizer.bos_token_id

    @property
    def eos_token_id(self) -> Optional[int]:
        """Get the EOS token ID."""
        return self.base_tokenizer.eos_token_id

    @property
    def pad_token_id(self) -> Optional[int]:
        """Get the PAD token ID."""
        return self.base_tokenizer.pad_token_id

    def get_vocab(self) -> Dict[str, int]:
        """Get the vocabulary dictionary."""
        return self.base_tokenizer.get_vocab()

    def compute_compression_stats(self, text: str) -> Dict[str, Any]:
        """Compute compression statistics for a given text.

        Args:
            text: Input text to analyze.

        Returns:
            Dictionary containing compression statistics.
        """
        # Get raw tokenization
        raw_ids = self.base_tokenizer.encode(text, add_special_tokens=False)
        # Get compressed tokenization
        compressed_ids, modifier_array = self.encode(text, add_special_tokens=False)

        num_raw_tokens = len(raw_ids)
        num_compressed_tokens = len(compressed_ids)

        # Count modifiers by type
        modifier_usage = {group: 0 for group in self.modifier_manager.groups}
        non_zero_modifiers = 0
        multi_modifier_tokens = 0

        for i in range(len(compressed_ids)):
            token_mods = modifier_array[i]
            non_zero_count = sum(1 for val in token_mods if val > 0)
            if non_zero_count > 0:
                non_zero_modifiers += 1
            if non_zero_count > 1:
                multi_modifier_tokens += 1

            # Count each group usage
            for group_idx, group_name in enumerate(self.modifier_manager.groups):
                if group_idx < len(token_mods) and token_mods[group_idx] > 0:
                    modifier_usage[group_name] += 1

        compression_ratio = num_raw_tokens / num_compressed_tokens if num_compressed_tokens > 0 else 0
        tokens_removed = num_raw_tokens - num_compressed_tokens

        return {
            'raw_tokens': num_raw_tokens,
            'compressed_tokens': num_compressed_tokens,
            'tokens_removed': tokens_removed,
            'compression_ratio': compression_ratio,
            'tokens_with_modifiers': non_zero_modifiers,
            'tokens_with_multi_modifiers': multi_modifier_tokens,
            'modifier_usage_by_group': modifier_usage,
            'modifier_coverage_ratio': non_zero_modifiers / num_compressed_tokens if num_compressed_tokens > 0 else 0,
        }

    def get_tokenizer_stats(self) -> Dict[str, Any]:
        """Get general statistics about the tokenizer configuration.

        Returns:
            Dictionary containing tokenizer statistics.
        """
        return {
            'base_vocab_size': self.base_tokenizer.vocab_size,
            'num_transformation_groups': self.modifier_manager.num_groups,
            'transformation_groups': self.modifier_manager.groups,
            'num_sequences_in_map': len(self.sequence_map),
            'max_sequence_length': self.sequence_map.max_sequence_length,
            'num_article_tokens': len(self.article_token_ids),
            'num_preposition_tokens': len(self.preposition_token_ids),
            'num_prefix_punct_tokens': len(self.prefix_punct_token_ids),
            'num_suffix_punct_tokens': len(self.suffix_punct_token_ids),
        }

    def to_dict(self) -> Dict:
        """Serialize the dual-stream tokenizer configuration."""
        return {
            'sequence_map': self.sequence_map.to_dict(),
            'modifier_manager': self.modifier_manager.to_dict(),
            'article_token_ids': list(self.article_token_ids),
            'preposition_token_ids': list(self.preposition_token_ids),
            'prefix_punct_token_ids': list(self.prefix_punct_token_ids),
            'suffix_punct_token_ids': list(self.suffix_punct_token_ids),
            'article_transforms': {str(k): v for k, v in self.article_transforms.items()},
            'preposition_transforms': {str(k): v for k, v in self.preposition_transforms.items()},
            'prefix_punct_transforms': {str(k): v for k, v in self.prefix_punct_transforms.items()},
            'suffix_punct_transforms': {str(k): v for k, v in self.suffix_punct_transforms.items()},
            'transformation_names_to_int': dict(self.transformation_names_to_int),
        }

    @classmethod
    def from_dict(cls, data: Dict, base_tokenizer: PreTrainedTokenizer) -> 'DualStreamTokenizer':
        """Deserialize from dictionary.

        Args:
            data: Serialized configuration dictionary.
            base_tokenizer: The underlying tokenizer instance.

        Returns:
            DualStreamTokenizer instance.
        """
        sequence_map = SequenceMap.from_dict(data['sequence_map'])
        modifier_manager = UnifiedModifierArray.from_dict(data['modifier_manager'])

        return cls(
            base_tokenizer=base_tokenizer,
            sequence_map=sequence_map,
            modifier_manager=modifier_manager,
            article_token_ids=set(data.get('article_token_ids', [])),
            preposition_token_ids=set(data.get('preposition_token_ids', [])),
            prefix_punct_token_ids=set(data.get('prefix_punct_token_ids', [])),
            suffix_punct_token_ids=set(data.get('suffix_punct_token_ids', [])),
            article_transforms={int(k): tuple(v) for k, v in data.get('article_transforms', {}).items()},
            preposition_transforms={int(k): tuple(v) for k, v in data.get('preposition_transforms', {}).items()},
            prefix_punct_transforms={int(k): tuple(v) for k, v in data.get('prefix_punct_transforms', {}).items()},
            suffix_punct_transforms={int(k): tuple(v) for k, v in data.get('suffix_punct_transforms', {}).items()},
            transformation_names_to_int=data.get('transformation_names_to_int', {}),
        )


# Vocabulary Pruning Functions
# These functions help remove tokens with boundary punctuation from the tokenizer

# Boundary punctuation characters (excluding apostrophe for contractions)
BOUNDARY_PUNCT_CHARS = set('()[]{}"\'.!?,;:')
# Characters that indicate the start of a contraction (should not be removed)
CONTRACTION_PATTERNS = {"n't", "'s", "'ll", "'ve", "'re", "'d", "'m"}


def is_contraction_token(token: str) -> bool:
    """Check if a token is a contraction or part of one.

    Args:
        token: The token string to check.

    Returns:
        True if the token appears to be a contraction.
    """
    # Check if it contains an internal apostrophe (not at boundary)
    if "'" in token:
        # Find apostrophe position
        pos = token.find("'")
        if pos > 0 and pos < len(token) - 1:
            # Apostrophe is internal
            return True
        # Check for common contraction endings
        for pattern in CONTRACTION_PATTERNS:
            if token.endswith(pattern):
                return True
    return False


def has_boundary_punctuation(token: str) -> bool:
    """Check if a token has punctuation at its boundaries (start or end).

    Args:
        token: The token string to check.

    Returns:
        True if the token has boundary punctuation and is not a contraction.
    """
    if len(token) <= 1:
        return False  # Single character tokens are OK

    # Check if it's a contraction
    if is_contraction_token(token):
        return False

    # Strip space prefix if present (common in BPE tokenizers)
    check_token = token.lstrip(' ').lstrip('Ġ')  # Ġ is the space marker in some tokenizers

    if len(check_token) <= 1:
        return False

    # Check start and end
    has_start_punct = check_token[0] in BOUNDARY_PUNCT_CHARS
    has_end_punct = check_token[-1] in BOUNDARY_PUNCT_CHARS

    # Only flag if there are non-punct characters AND punct at boundary
    non_punct_chars = sum(1 for c in check_token if c not in BOUNDARY_PUNCT_CHARS and c not in ' Ġ')

    if non_punct_chars == 0:
        # Pure punctuation token - that's fine
        return False

    return has_start_punct or has_end_punct


def identify_tokens_to_remove(
    vocab: Dict[str, int],
    exclude_special_tokens: Optional[Set[str]] = None
) -> Set[str]:
    """Identify tokens that have boundary punctuation.

    Args:
        vocab: Vocabulary dictionary (token -> id).
        exclude_special_tokens: Set of special tokens to never remove.

    Returns:
        Set of token strings to remove.
    """
    exclude = exclude_special_tokens or set()
    tokens_to_remove = set()

    for token in vocab:
        if token in exclude:
            continue

        if has_boundary_punctuation(token):
            tokens_to_remove.add(token)

    return tokens_to_remove


def create_pruned_tokenizer_vocab(
    base_tokenizer: PreTrainedTokenizer,
    remove_boundary_punct: bool = True,
    exclude_special_tokens: Optional[Set[str]] = None
) -> Tuple[Dict[str, int], Set[str]]:
    """Create a pruned vocabulary by removing tokens with boundary punctuation.

    Args:
        base_tokenizer: The base tokenizer.
        remove_boundary_punct: Whether to remove boundary punctuation tokens.
        exclude_special_tokens: Set of special tokens to never remove.

    Returns:
        Tuple of (pruned_vocab, removed_tokens).
    """
    if not remove_boundary_punct:
        return base_tokenizer.get_vocab(), set()

    vocab = base_tokenizer.get_vocab()

    # Build exclude set
    exclude = exclude_special_tokens or set()

    # Add special tokens from the tokenizer
    for attr in ['bos_token', 'eos_token', 'pad_token', 'unk_token', 'sep_token', 'cls_token', 'mask_token']:
        token = getattr(base_tokenizer, attr, None)
        if token:
            exclude.add(token)

    # Identify tokens to remove
    tokens_to_remove = identify_tokens_to_remove(vocab, exclude)

    # Create new vocabulary without removed tokens
    # Note: This doesn't reindex - IDs stay the same but tokens are marked as invalid
    pruned_vocab = {token: idx for token, idx in vocab.items() if token not in tokens_to_remove}

    return pruned_vocab, tokens_to_remove


def filter_merges_for_removed_tokens(
    merges: List[str],
    removed_tokens: Set[str]
) -> List[str]:
    """Filter BPE merges to remove those that create removed tokens.

    Args:
        merges: List of BPE merge rules (e.g., "a b" meaning a+b -> ab).
        removed_tokens: Set of tokens that have been removed.

    Returns:
        Filtered list of merges.
    """
    filtered_merges = []

    for merge in merges:
        # Parse merge rule: "token1 token2"
        parts = merge.split(' ')
        if len(parts) != 2:
            continue

        # Check if the result of this merge is a removed token
        # The result is typically the concatenation of the two parts
        result = ''.join(parts)

        if result not in removed_tokens:
            filtered_merges.append(merge)

    return filtered_merges


class PrunedTokenizerWrapper:
    """Wrapper that prevents the use of tokens with boundary punctuation.

    This wrapper intercepts encoding requests and ensures that tokens with
    boundary punctuation (like "(dog" or "dog)") are not used, forcing the
    tokenizer to use more granular tokens instead.
    """

    def __init__(self, base_tokenizer: PreTrainedTokenizer, removed_tokens: Set[str]):
        """Initialize the pruned tokenizer wrapper.

        Args:
            base_tokenizer: The underlying tokenizer.
            removed_tokens: Set of token strings that should not be used.
        """
        self.base_tokenizer = base_tokenizer
        self.removed_tokens = removed_tokens
        self.removed_token_ids = set()

        # Build set of removed token IDs
        vocab = base_tokenizer.get_vocab()
        for token in removed_tokens:
            if token in vocab:
                self.removed_token_ids.add(vocab[token])

    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        """Encode text, avoiding removed tokens.

        This implementation is a simple fallback that re-encodes parts of the
        text that would use removed tokens. For a more robust solution, you
        would need to modify the BPE merges directly.

        Args:
            text: Input text to encode.
            add_special_tokens: Whether to add special tokens.

        Returns:
            List of token IDs.
        """
        # First, get the standard encoding
        token_ids = self.base_tokenizer.encode(text, add_special_tokens=add_special_tokens)

        # Check if any removed tokens are used
        if not any(tid in self.removed_token_ids for tid in token_ids):
            return token_ids

        # If removed tokens are present, we need to handle this
        # One approach: character-level re-encoding for affected segments
        # This is a simplified fallback - a full solution would modify BPE
        result_ids = []
        tokens = self.base_tokenizer.convert_ids_to_tokens(token_ids)

        for tid, token in zip(token_ids, tokens):
            if tid in self.removed_token_ids:
                # Re-encode this token character by character
                # This forces the tokenizer to use more basic tokens
                char_ids = []
                for char in token:
                    char_encoding = self.base_tokenizer.encode(char, add_special_tokens=False)
                    char_ids.extend(char_encoding)
                result_ids.extend(char_ids)
            else:
                result_ids.append(tid)

        return result_ids

    def __call__(self, text: str, **kwargs) -> Dict[str, Any]:
        """HuggingFace-compatible call interface."""
        token_ids = self.encode(text, add_special_tokens=kwargs.get('add_special_tokens', True))
        return {
            'input_ids': token_ids,
            'attention_mask': [1] * len(token_ids)
        }

    def decode(self, token_ids: List[int]) -> str:
        """Decode token IDs to text."""
        return self.base_tokenizer.decode(token_ids)

    def convert_ids_to_tokens(self, token_ids: List[int]) -> List[str]:
        """Convert token IDs to token strings."""
        return self.base_tokenizer.convert_ids_to_tokens(token_ids)

    def get_vocab(self) -> Dict[str, int]:
        """Get the vocabulary (excluding removed tokens conceptually)."""
        vocab = self.base_tokenizer.get_vocab()
        return {k: v for k, v in vocab.items() if k not in self.removed_tokens}

    @property
    def vocab_size(self) -> int:
        """Get vocabulary size (excluding removed tokens)."""
        return self.base_tokenizer.vocab_size - len(self.removed_token_ids)

    @property
    def bos_token_id(self) -> Optional[int]:
        return self.base_tokenizer.bos_token_id

    @property
    def eos_token_id(self) -> Optional[int]:
        return self.base_tokenizer.eos_token_id

    @property
    def pad_token_id(self) -> Optional[int]:
        return self.base_tokenizer.pad_token_id

    def __getattr__(self, name):
        """Forward attribute access to base tokenizer."""
        return getattr(self.base_tokenizer, name)
