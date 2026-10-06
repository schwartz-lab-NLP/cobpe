use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyList;
use rayon::prelude::*;
use rustc_hash::FxHashMap;
use serde::Deserialize;
use serde_json::json;
use std::collections::{HashMap, HashSet};
use tiktoken_rs::CoreBPE;

const SURFACE_GROUP_NAMES: [&str; 2] = ["space_prefix", "base_capitalization"];
const STRIP_NONLEXICAL_GROUPS: [&str; 6] = [
    "determiners",
    "pronouns",
    "prepositions",
    "postpositions",
    "article_capitalization",
    "prep_capitalization",
];
const STRIP_INVALID_DETACHED_GROUPS: [&str; 8] = [
    "determiners",
    "pronouns",
    "prepositions",
    "postpositions",
    "article_capitalization",
    "prep_capitalization",
    "article_space_prefix",
    "prep_space_prefix",
];

#[derive(Debug, Deserialize, Clone)]
struct Config {
    version: usize,
    num_modifier_groups: usize,
    default_modifier: Vec<u16>,
    #[serde(default)]
    group_names: Vec<String>,
    #[serde(default)]
    group_value_names: HashMap<String, Vec<String>>,
    entries: Vec<Entry>,
    #[serde(default)]
    reverse_entries: Vec<ReverseEntry>,
    #[serde(default)]
    token_meta: Vec<TokenMeta>,
    #[serde(default)]
    runtime: RuntimeConfig,
    base_bpe: BaseBpeConfig,
}

#[derive(Debug, Deserialize, Clone, Default)]
struct BaseBpeConfig {
    #[serde(default)]
    pattern: String,
    #[serde(default)]
    mergeable_ranks: Vec<BaseBpeRank>,
    #[serde(default)]
    special_tokens: HashMap<String, u32>,
}

#[derive(Debug, Deserialize, Clone, Default)]
struct BaseBpeRank {
    #[serde(default)]
    token: String,
    #[serde(default)]
    rank: u32,
}

#[derive(Debug, Deserialize, Clone)]
struct Entry {
    token_ids: Vec<u32>,
    base_ids: Vec<u32>,
    modifier_rows: Vec<Vec<u16>>,
}

#[derive(Debug, Deserialize, Clone)]
struct ReverseEntry {
    token_ids: Vec<u32>,
    base_ids: Vec<u32>,
    modifier_rows: Vec<Vec<u16>>,
    #[serde(default)]
    surface: Option<String>,
}

#[derive(Debug, Deserialize, Clone, Default)]
struct LiteralTransform {
    #[serde(default)]
    group_name: String,
    #[serde(default)]
    rel_idx: usize,
}

#[derive(Clone)]
struct PostpositionPattern {
    words: Vec<String>,
    transform: LiteralTransform,
}

struct PostpositionContext {
    transform_for_host: Vec<Option<LiteralTransform>>,
    host_for_raw_idx: Vec<Option<usize>>,
    skip_raw_idx: Vec<bool>,
}

#[derive(Debug, Deserialize, Clone, Default)]
struct TokenMeta {
    #[serde(default)]
    token_text: String,
    #[serde(default)]
    canonical_surface: String,
    #[serde(default)]
    has_space_prefix: bool,
    #[serde(default)]
    has_word_char: bool,
    #[serde(default)]
    is_whitespace_only: bool,
    #[serde(default)]
    is_single_ascii_space: bool,
    #[serde(default)]
    is_byte_fallback: bool,
    #[serde(default)]
    is_base_cap_representable: bool,
    #[serde(default)]
    determiner: Option<LiteralTransform>,
    #[serde(default)]
    pronoun: Option<LiteralTransform>,
    #[serde(default)]
    preposition: Option<LiteralTransform>,
    #[serde(default)]
    postposition: Option<LiteralTransform>,
    #[serde(default)]
    prefix_punctuation: Option<LiteralTransform>,
    #[serde(default)]
    suffix_punctuation: Option<LiteralTransform>,
}

#[derive(Debug, Deserialize, Clone)]
struct AttachmentLimits {
    #[serde(default = "default_attach_limit")]
    max_prefix_punctuation: usize,
    #[serde(default = "default_attach_limit")]
    max_suffix_punctuation: usize,
}

impl Default for AttachmentLimits {
    fn default() -> Self {
        Self {
            max_prefix_punctuation: default_attach_limit(),
            max_suffix_punctuation: default_attach_limit(),
        }
    }
}

#[derive(Debug, Deserialize, Clone, Default)]
struct RuntimeConfig {
    #[serde(default)]
    group_indices: HashMap<String, usize>,
    #[serde(default)]
    literal_maps: HashMap<String, HashMap<String, LiteralTransform>>,
    #[serde(default)]
    multi_token_first_group_indices: Vec<usize>,
    #[serde(default)]
    attachment_limits: AttachmentLimits,
}

#[derive(Clone)]
struct EntryValue {
    consumed_len: usize,
    base_ids: Vec<u32>,
    modifier_rows: Vec<Vec<u16>>,
}

#[derive(Clone)]
struct ReverseEntryValue {
    consumed_len: usize,
    base_ids: Vec<u32>,
    modifier_rows: Vec<Vec<u16>>,
    surface: Option<String>,
}

#[derive(Default)]
struct TrieNode {
    children: HashMap<u32, usize>,
    value: Option<EntryValue>,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct ReverseKey {
    token_id: u32,
    modifier_row: Vec<u16>,
}

#[derive(Default)]
struct ReverseTrieNode {
    children: HashMap<ReverseKey, usize>,
    value: Option<ReverseEntryValue>,
}

#[derive(Debug, Clone)]
struct PendingGroup {
    group_idx: usize,
    rel_idx: u16,
}

struct BaseBpeRuntime {
    core: CoreBPE,
    id_to_token_bytes: Vec<Option<Vec<u8>>>,
}

#[pyclass(name = "CompositionalProcessor")]
pub struct CompositionalTokenizer {
    base_bpe: BaseBpeRuntime,
    trie_nodes: Vec<TrieNode>,
    max_sequence_len: usize,
    reverse_trie_nodes: Vec<ReverseTrieNode>,
    max_reverse_sequence_len: usize,
    default_modifier: Vec<u16>,
    num_modifier_groups: usize,
    group_names: Vec<String>,
    group_value_names: HashMap<String, Vec<String>>,
    runtime: RuntimeConfig,
    token_meta: Vec<TokenMeta>,
    token_meta_is_explicit: bool,
    postpositions_by_first_word: HashMap<String, Vec<PostpositionPattern>>,
}

fn default_attach_limit() -> usize {
    1
}

fn bytes_to_latin1(bytes: &[u8]) -> String {
    bytes.iter().map(|byte| char::from(*byte)).collect()
}

fn latin1_to_bytes(text: &str) -> Option<Vec<u8>> {
    let mut out = Vec::with_capacity(text.len());
    for ch in text.chars() {
        let code = ch as u32;
        if code > 0xFF {
            return None;
        }
        out.push(code as u8);
    }
    Some(out)
}

fn decode_utf8_lossy(bytes: &[u8]) -> String {
    String::from_utf8_lossy(bytes).to_string()
}

fn first_alpha_is_upper(text: &str) -> Option<bool> {
    text.chars()
        .find(|ch| ch.is_alphabetic())
        .map(|ch| ch.is_uppercase())
}

fn capitalize_first_alpha(text: &str) -> String {
    let mut out = String::new();
    let mut changed = false;
    for ch in text.chars() {
        if !changed && ch.is_alphabetic() {
            for upper in ch.to_uppercase() {
                out.push(upper);
            }
            changed = true;
        } else {
            out.push(ch);
        }
    }
    out
}

fn lowercase_first_alpha(text: &str) -> String {
    let mut out = String::new();
    let mut changed = false;
    for ch in text.chars() {
        if !changed && ch.is_alphabetic() {
            for lower in ch.to_lowercase() {
                out.push(lower);
            }
            changed = true;
        } else {
            out.push(ch);
        }
    }
    out
}

fn is_capitalized_surface(text: &str) -> bool {
    let stripped = text.trim_start_matches(' ');
    stripped
        .chars()
        .find(|ch| ch.is_alphabetic())
        .map(|ch| ch.is_uppercase())
        .unwrap_or(false)
}

fn leading_ascii_spaces(text: &str) -> usize {
    text.bytes().take_while(|byte| *byte == b' ').count()
}

fn is_base_cap_representable_surface(text: &str) -> bool {
    let alpha_chars: Vec<char> = text.chars().filter(|ch| ch.is_alphabetic()).collect();
    if alpha_chars.is_empty() {
        return true;
    }
    if alpha_chars.iter().all(|ch| ch.is_lowercase()) {
        return true;
    }
    alpha_chars[0].is_uppercase() && alpha_chars.iter().skip(1).all(|ch| ch.is_lowercase())
}

fn split_camel_case_segments(surface: &str) -> Option<Vec<String>> {
    if surface.is_empty() {
        return None;
    }
    let chars: Vec<char> = surface.chars().collect();
    let mut boundaries = vec![0usize];
    let mut has_internal_upper = false;
    for idx in 1..chars.len() {
        let current = chars[idx];
        if !current.is_uppercase() {
            continue;
        }
        has_internal_upper = true;
        let prev = chars[idx - 1];
        let next_is_lower = (idx + 1) < chars.len() && chars[idx + 1].is_lowercase();
        if prev.is_lowercase() || (prev.is_uppercase() && next_is_lower) {
            boundaries.push(idx);
        }
    }
    if !has_internal_upper {
        return None;
    }
    boundaries.push(chars.len());
    if boundaries.len() <= 2 {
        return None;
    }
    let mut segments = Vec::new();
    for pair in boundaries.windows(2) {
        let left = pair[0];
        let right = pair[1];
        if left < right {
            segments.push(chars[left..right].iter().collect());
        }
    }
    if segments.len() > 1 {
        Some(segments)
    } else {
        None
    }
}

fn expand_caps_segments(segments: Vec<String>) -> Vec<String> {
    let mut expanded = Vec::new();
    for segment in segments {
        if segment.len() > 1 && segment.chars().all(|ch| ch.is_uppercase()) {
            expanded.extend(segment.chars().map(|ch| ch.to_string()));
        } else {
            expanded.push(segment);
        }
    }
    expanded
}

mod matching;
mod process;
mod python;
mod runtime;
mod surface;
