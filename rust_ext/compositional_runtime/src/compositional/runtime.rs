use super::*;

impl CompositionalTokenizer {
    pub(super) fn add_entry(&mut self, entry: &Entry) {
        let mut node_idx = 0usize;
        for token_id in &entry.token_ids {
            let child_idx = if let Some(idx) = self.trie_nodes[node_idx].children.get(token_id) {
                *idx
            } else {
                let idx = self.trie_nodes.len();
                self.trie_nodes.push(TrieNode::default());
                self.trie_nodes[node_idx].children.insert(*token_id, idx);
                idx
            };
            node_idx = child_idx;
        }
        self.trie_nodes[node_idx].value = Some(EntryValue {
            consumed_len: entry.token_ids.len(),
            base_ids: entry.base_ids.clone(),
            modifier_rows: entry.modifier_rows.clone(),
        });
        self.max_sequence_len = self.max_sequence_len.max(entry.token_ids.len());
    }

    pub(super) fn add_reverse_entry(&mut self, entry: &ReverseEntry) {
        let mut node_idx = 0usize;
        for (base_id, modifier_row) in entry.base_ids.iter().zip(entry.modifier_rows.iter()) {
            let key = ReverseKey {
                token_id: *base_id,
                modifier_row: modifier_row.clone(),
            };
            let child_idx = if let Some(idx) = self.reverse_trie_nodes[node_idx].children.get(&key)
            {
                *idx
            } else {
                let idx = self.reverse_trie_nodes.len();
                self.reverse_trie_nodes.push(ReverseTrieNode::default());
                self.reverse_trie_nodes[node_idx].children.insert(key, idx);
                idx
            };
            node_idx = child_idx;
        }
        self.reverse_trie_nodes[node_idx].value = Some(ReverseEntryValue {
            consumed_len: entry.base_ids.len(),
            base_ids: entry.base_ids.clone(),
            modifier_rows: entry.modifier_rows.clone(),
            surface: entry.surface.clone(),
        });
        self.max_reverse_sequence_len = self.max_reverse_sequence_len.max(entry.base_ids.len());
    }

    pub(super) fn group_idx(&self, group_name: &str) -> Option<usize> {
        self.runtime.group_indices.get(group_name).copied()
    }

    pub(super) fn empty_modifier(&self) -> Vec<u16> {
        self.default_modifier.clone()
    }

    pub(super) fn token_meta_ref(&self, token_id: u32) -> &TokenMeta {
        self.token_meta
            .get(token_id as usize)
            .unwrap_or(&self.token_meta[0])
    }

    pub(super) fn raw_span_has_byte_fallback(
        &self,
        raw_ids: &[u32],
        start_idx: usize,
        consumed_len: usize,
    ) -> bool {
        let end_idx = usize::min(start_idx + consumed_len, raw_ids.len());
        raw_ids[start_idx..end_idx]
            .iter()
            .any(|token_id| self.token_meta_ref(*token_id).is_byte_fallback)
    }

    pub(super) fn base_bpe_token_bytes(&self, token_id: u32) -> Option<&[u8]> {
        self.base_bpe
            .id_to_token_bytes
            .get(token_id as usize)
            .and_then(|opt| opt.as_deref())
    }

    pub(super) fn decode_token_bytes(&self, token_ids: &[u32]) -> Option<String> {
        let mut bytes = Vec::new();
        for token_id in token_ids {
            bytes.extend_from_slice(self.base_bpe_token_bytes(*token_id)?);
        }
        std::str::from_utf8(&bytes)
            .ok()
            .map(|text| text.to_string())
    }

    pub(super) fn byte_component_end(&self, token_ids: &[u32], start_idx: usize) -> usize {
        if start_idx >= token_ids.len() {
            return start_idx;
        }
        let Some(first_bytes) = self.base_bpe_token_bytes(token_ids[start_idx]) else {
            return start_idx + 1;
        };
        if std::str::from_utf8(first_bytes).is_ok() {
            return start_idx + 1;
        }
        let mut pending = Vec::new();
        for (end_idx, token_id) in token_ids.iter().enumerate().skip(start_idx) {
            let Some(token_bytes) = self.base_bpe_token_bytes(*token_id) else {
                break;
            };
            pending.extend_from_slice(token_bytes);
            if std::str::from_utf8(&pending).is_ok() {
                return end_idx + 1;
            }
        }
        start_idx + 1
    }

    pub(super) fn byte_component_has_word_char(&self, token_ids: &[u32], start_idx: usize) -> bool {
        let end_idx = self.byte_component_end(token_ids, start_idx);
        self.decode_token_bytes(&token_ids[start_idx..end_idx])
            .map(|text| {
                text.chars()
                    .any(|ch| ch.is_alphanumeric() || ch.is_alphabetic())
            })
            .unwrap_or(false)
    }

    pub(super) fn raw_position_has_word_char(&self, raw_ids: &[u32], idx: usize) -> bool {
        if idx >= raw_ids.len() {
            return false;
        }
        let meta = self.token_meta_ref(raw_ids[idx]);
        if meta.has_word_char {
            return true;
        }
        meta.is_byte_fallback && self.byte_component_has_word_char(raw_ids, idx)
    }

    pub(super) fn modifier_utf8_delta(&self, modifiers: &[u16]) -> Option<usize> {
        for (group_idx, value) in modifiers.iter().enumerate() {
            if *value == self.default_modifier[group_idx] {
                continue;
            }
            let group_name = self.group_names.get(group_idx)?;
            match group_name.as_str() {
                "space_prefix"
                | "base_capitalization"
                | "determiners"
                | "pronouns"
                | "article_det"
                | "articles"
                | "article_capitalization"
                | "article_space_prefix"
                | "prepositions"
                | "postpositions"
                | "prep_capitalization"
                | "prep_space_prefix"
                | "prefix_punctuation"
                | "suffix_punctuation" => {}
                _ => return None,
            }
        }

        let mut delta = 0usize;
        if self.space_setting(modifiers, "space_prefix") {
            delta += 1;
        }
        if let Some(value) = self.literal_from_group(modifiers, "prepositions", &["prep_"]) {
            delta += value.len() + 1;
        }
        let determiner = self
            .literal_from_group(modifiers, "determiners", &["det_", "article_"])
            .or_else(|| self.literal_from_group(modifiers, "article_det", &["det_", "article_"]))
            .or_else(|| self.literal_from_group(modifiers, "articles", &["article_", "det_"]));
        if let Some(value) = self.literal_from_group(modifiers, "pronouns", &["pron_"]) {
            delta += value.len() + 1;
        }
        if let Some(value) = determiner {
            delta += value.len() + 1;
        }
        if let Some(value) = self.literal_from_group(modifiers, "postpositions", &["postp_"]) {
            delta += value.len() + 1;
        }
        if let Some(value) =
            self.literal_from_group(modifiers, "prefix_punctuation", &["punct_prefix_"])
        {
            delta += value.len();
        }
        if let Some(value) =
            self.literal_from_group(modifiers, "suffix_punctuation", &["punct_suffix_"])
        {
            delta += value.len();
        }
        Some(delta)
    }

    pub(super) fn base_bpe_vocab_token(&self, token_id: u32) -> Option<String> {
        self.base_bpe_token_bytes(token_id).map(bytes_to_latin1)
    }

    pub(super) fn base_bpe_decoded_token(&self, token_id: u32) -> Option<String> {
        if let Ok(text) = self.base_bpe.core.decode(&[token_id]) {
            return Some(text);
        }
        self.base_bpe
            .core
            .decode_bytes(&[token_id])
            .ok()
            .map(|bytes| decode_utf8_lossy(&bytes))
    }

    pub(super) fn decode_ids(&self, ids: &[u32]) -> String {
        if let Ok(text) = self.base_bpe.core.decode(ids) {
            return text;
        }
        self.base_bpe
            .core
            .decode_bytes(ids)
            .map(|bytes| decode_utf8_lossy(&bytes))
            .unwrap_or_default()
    }

    pub(super) fn decode_single(&self, token_id: u32) -> String {
        self.decode_ids(&[token_id])
    }

    pub(super) fn encode_segment(&self, text: &str) -> Option<Vec<u32>> {
        Some(self.base_bpe.core.encode_ordinary(text))
    }

    pub(super) fn longest_reverse_match(
        &self,
        token_ids: &[u32],
        modifier_rows: &[Vec<u16>],
        start_idx: usize,
    ) -> Option<ReverseEntryValue> {
        let mut node_idx = 0usize;
        let mut best: Option<ReverseEntryValue> = None;
        let stop = usize::min(start_idx + self.max_reverse_sequence_len, token_ids.len());
        for pos in start_idx..stop {
            let key = ReverseKey {
                token_id: token_ids[pos],
                modifier_row: modifier_rows[pos].clone(),
            };
            let child_idx = match self.reverse_trie_nodes[node_idx].children.get(&key) {
                Some(idx) => *idx,
                None => break,
            };
            node_idx = child_idx;
            if let Some(value) = self.reverse_trie_nodes[node_idx].value.clone() {
                best = Some(value);
            }
        }
        best
    }

    pub(super) fn value_name(&self, group_name: &str, value: u16) -> Option<String> {
        self.group_value_names
            .get(group_name)
            .and_then(|names| names.get(value as usize))
            .cloned()
    }

    pub(super) fn suffix_literal_matches_raw(
        &self,
        token_id: u32,
        transform: &LiteralTransform,
    ) -> bool {
        let Some(value_name) = self.value_name(&transform.group_name, transform.rel_idx as u16)
        else {
            return false;
        };
        let Some(literal) = value_name.strip_prefix("punct_suffix_") else {
            return false;
        };
        self.decode_single(token_id)
            .trim_start_matches([' ', 'Ġ', '▁'])
            == literal
    }

    pub(super) fn literal_from_group(
        &self,
        modifiers: &[u16],
        group_name: &str,
        prefixes: &[&str],
    ) -> Option<String> {
        let group_idx = self.group_idx(group_name)?;
        let value = *modifiers.get(group_idx)?;
        if value == self.default_modifier[group_idx] {
            return None;
        }
        let value_name = self.value_name(group_name, value)?;
        for prefix in prefixes {
            if let Some(stripped) = value_name.strip_prefix(prefix) {
                return Some(stripped.to_string());
            }
        }
        None
    }

    pub(super) fn space_setting(&self, modifiers: &[u16], group_name: &str) -> bool {
        let Some(group_idx) = self.group_idx(group_name) else {
            return false;
        };
        let value = modifiers[group_idx];
        if value == self.default_modifier[group_idx] {
            return false;
        }
        let value_name = self
            .value_name(group_name, value)
            .unwrap_or_default()
            .to_lowercase();
        if value_name.starts_with("with_") || value_name.starts_with("add_") {
            return true;
        }
        if value_name.starts_with("remove_")
            || value_name.starts_with("lower_")
            || value_name.starts_with("no_")
            || value_name.starts_with("na_")
            || value_name.starts_with("none")
        {
            return false;
        }
        value == 1
    }

    pub(super) fn build_base_bpe_runtime(config: &BaseBpeConfig) -> PyResult<BaseBpeRuntime> {
        if config.pattern.is_empty() || config.mergeable_ranks.is_empty() {
            return Err(PyValueError::new_err(
                "base_bpe requires a pattern and mergeable_ranks",
            ));
        }
        let mut encoder: FxHashMap<Vec<u8>, u32> = FxHashMap::default();
        let mut special_tokens: FxHashMap<String, u32> = FxHashMap::default();
        let mut max_token_id = 0usize;
        let mut id_to_token_bytes: Vec<Option<Vec<u8>>> = Vec::new();

        for entry in &config.mergeable_ranks {
            let token_bytes = latin1_to_bytes(&entry.token).ok_or_else(|| {
                PyValueError::new_err("base_bpe mergeable_ranks contains non-latin1 token text")
            })?;
            let token_id = entry.rank as usize;
            if token_id >= id_to_token_bytes.len() {
                id_to_token_bytes.resize(token_id + 1, None);
            }
            id_to_token_bytes[token_id] = Some(token_bytes.clone());
            encoder.insert(token_bytes, entry.rank);
            max_token_id = max_token_id.max(token_id);
        }

        for (token, token_id) in &config.special_tokens {
            special_tokens.insert(token.clone(), *token_id);
            let token_id_usize = *token_id as usize;
            if token_id_usize >= id_to_token_bytes.len() {
                id_to_token_bytes.resize(token_id_usize + 1, None);
            }
            id_to_token_bytes[token_id_usize] = Some(token.as_bytes().to_vec());
            max_token_id = max_token_id.max(token_id_usize);
        }

        if id_to_token_bytes.len() <= max_token_id {
            id_to_token_bytes.resize(max_token_id + 1, None);
        }

        let core = CoreBPE::new(encoder, special_tokens, &config.pattern)
            .map_err(|e| PyValueError::new_err(format!("Failed to build base_bpe runtime: {e}")))?;
        Ok(BaseBpeRuntime {
            core,
            id_to_token_bytes,
        })
    }

    pub(super) fn build_token_meta_table_from_base_bpe(
        base_bpe: &BaseBpeRuntime,
        runtime: &RuntimeConfig,
        explicit: &[TokenMeta],
    ) -> Vec<TokenMeta> {
        if !explicit.is_empty() {
            return explicit.to_vec();
        }
        let det_map = runtime.literal_maps.get("determiners");
        let pronoun_map = runtime.literal_maps.get("pronouns");
        let prep_map = runtime.literal_maps.get("prepositions");
        let postp_map = runtime.literal_maps.get("postpositions");
        let prefix_map = runtime.literal_maps.get("prefix_punctuation");
        let suffix_map = runtime.literal_maps.get("suffix_punctuation");

        base_bpe
            .id_to_token_bytes
            .iter()
            .enumerate()
            .map(|(token_id, maybe_vocab_bytes)| {
                let vocab_bytes = maybe_vocab_bytes.clone().unwrap_or_default();
                let is_byte_fallback =
                    !vocab_bytes.is_empty() && std::str::from_utf8(&vocab_bytes).is_err();
                let vocab_token = bytes_to_latin1(&vocab_bytes);
                let decoded_token = if let Ok(text) = base_bpe.core.decode(&[token_id as u32]) {
                    text
                } else if let Ok(bytes) = base_bpe.core.decode_bytes(&[token_id as u32]) {
                    decode_utf8_lossy(&bytes)
                } else {
                    String::new()
                };
                let surface_text = if decoded_token.is_empty() {
                    vocab_token.as_str()
                } else {
                    decoded_token.as_str()
                };
                let canonical_surface = if is_byte_fallback {
                    String::new()
                } else {
                    surface_text
                        .trim_start_matches([' ', 'Ġ', '▁'])
                        .to_lowercase()
                };
                let stripped = surface_text.trim_start_matches([' ', 'Ġ', '▁']);
                let has_space_prefix =
                    vocab_bytes.first().copied() == Some(b' ') || surface_text.starts_with(' ');
                TokenMeta {
                    token_text: surface_text.to_string(),
                    canonical_surface: canonical_surface.clone(),
                    has_space_prefix,
                    has_word_char: !is_byte_fallback
                        && stripped.chars().any(|ch| ch.is_alphanumeric()),
                    is_whitespace_only: surface_text.trim().is_empty(),
                    is_single_ascii_space: vocab_bytes == [b' '] || surface_text == " ",
                    is_byte_fallback,
                    is_base_cap_representable: !is_byte_fallback
                        && is_base_cap_representable_surface(surface_text),
                    determiner: if is_byte_fallback {
                        None
                    } else {
                        det_map.and_then(|m| m.get(&canonical_surface)).cloned()
                    },
                    pronoun: if is_byte_fallback {
                        None
                    } else {
                        pronoun_map.and_then(|m| m.get(&canonical_surface)).cloned()
                    },
                    preposition: if is_byte_fallback {
                        None
                    } else {
                        prep_map.and_then(|m| m.get(&canonical_surface)).cloned()
                    },
                    postposition: if is_byte_fallback {
                        None
                    } else {
                        postp_map.and_then(|m| m.get(&canonical_surface)).cloned()
                    },
                    prefix_punctuation: if is_byte_fallback {
                        None
                    } else {
                        prefix_map.and_then(|m| m.get(&canonical_surface)).cloned()
                    },
                    suffix_punctuation: if is_byte_fallback {
                        None
                    } else {
                        suffix_map.and_then(|m| m.get(&canonical_surface)).cloned()
                    },
                }
            })
            .collect()
    }
}
