use super::*;

impl CompositionalTokenizer {
    fn encode_text_impl(&self, text: &str) -> PyResult<Vec<u32>> {
        Ok(self.base_bpe.core.encode_ordinary(text))
    }

    fn build_result_object(
        &self,
        py: Python<'_>,
        output_ids: Vec<u32>,
        modifier_rows: Vec<Vec<u16>>,
    ) -> PyResult<Py<PyAny>> {
        Ok((output_ids, modifier_rows)
            .into_pyobject(py)?
            .unbind()
            .into_any())
    }

    fn group_value_name(&self, group_name: &str, rel_idx: usize) -> Option<String> {
        self.group_value_names
            .get(group_name)
            .and_then(|values| values.get(rel_idx))
            .cloned()
    }

    fn literal_debug_value(&self, literal: &Option<LiteralTransform>) -> serde_json::Value {
        let Some(transform) = literal else {
            return serde_json::Value::Null;
        };
        json!({
            "group_name": transform.group_name,
            "rel_idx": transform.rel_idx,
            "value_name": self.group_value_name(&transform.group_name, transform.rel_idx),
        })
    }

    fn token_debug_value(&self, token_id: u32) -> serde_json::Value {
        let meta = self.token_meta_ref(token_id);
        let vocab_token = self.base_bpe_vocab_token(token_id);
        let decoded_token = self.base_bpe_decoded_token(token_id);
        json!({
            "id": token_id,
            "vocab_token": vocab_token,
            "decoded_token": decoded_token,
            "token_text": meta.token_text,
            "canonical_surface": meta.canonical_surface,
            "has_space_prefix": meta.has_space_prefix,
            "has_word_char": meta.has_word_char,
            "is_whitespace_only": meta.is_whitespace_only,
            "is_single_ascii_space": meta.is_single_ascii_space,
            "is_byte_fallback": meta.is_byte_fallback,
            "is_base_cap_representable": meta.is_base_cap_representable,
            "determiner": self.literal_debug_value(&meta.determiner),
            "pronoun": self.literal_debug_value(&meta.pronoun),
            "preposition": self.literal_debug_value(&meta.preposition),
            "postposition": self.literal_debug_value(&meta.postposition),
            "prefix_punctuation": self.literal_debug_value(&meta.prefix_punctuation),
            "suffix_punctuation": self.literal_debug_value(&meta.suffix_punctuation),
        })
    }
}

#[pymethods]
impl CompositionalTokenizer {
    #[new]
    fn new(config_json: &str) -> PyResult<Self> {
        let cfg: Config = serde_json::from_str(config_json).map_err(|e| {
            PyValueError::new_err(format!("Failed to parse compositional runtime config: {e}"))
        })?;
        if cfg.version != 1 {
            return Err(PyValueError::new_err(format!(
                "Unsupported compositional runtime config version: {}",
                cfg.version
            )));
        }
        if cfg.default_modifier.len() != cfg.num_modifier_groups {
            return Err(PyValueError::new_err(
                "default_modifier width must match num_modifier_groups",
            ));
        }
        let base_bpe = Self::build_base_bpe_runtime(&cfg.base_bpe)?;
        let token_meta_is_explicit = !cfg.token_meta.is_empty();
        let mut token_meta =
            Self::build_token_meta_table_from_base_bpe(&base_bpe, &cfg.runtime, &cfg.token_meta);
        if token_meta.is_empty() {
            token_meta.push(TokenMeta::default());
        }

        // Compile compound postposition literals once when the tokenizer is
        // constructed.  The processing hot path only needs to look up
        // candidates by their first word; it should not split every literal
        // at every possible raw-token position.
        let mut postpositions_by_first_word: HashMap<String, Vec<PostpositionPattern>> =
            HashMap::new();
        if let Some(postposition_map) = cfg.runtime.literal_maps.get("postpositions") {
            for (literal, transform) in postposition_map {
                let words: Vec<String> = literal
                    .split(' ')
                    .filter(|word| !word.is_empty())
                    .map(str::to_owned)
                    .collect();
                if let Some(first_word) = words.first() {
                    postpositions_by_first_word
                        .entry(first_word.clone())
                        .or_default()
                        .push(PostpositionPattern {
                            words,
                            transform: transform.clone(),
                        });
                }
            }
        }

        let mut tokenizer = Self {
            base_bpe,
            trie_nodes: vec![TrieNode::default()],
            max_sequence_len: 1,
            reverse_trie_nodes: vec![ReverseTrieNode::default()],
            max_reverse_sequence_len: 1,
            default_modifier: cfg.default_modifier.clone(),
            num_modifier_groups: cfg.num_modifier_groups,
            group_names: cfg.group_names.clone(),
            group_value_names: cfg.group_value_names.clone(),
            runtime: cfg.runtime.clone(),
            token_meta,
            token_meta_is_explicit,
            postpositions_by_first_word,
        };

        for entry in &cfg.entries {
            if entry.modifier_rows.len() != entry.base_ids.len() {
                return Err(PyValueError::new_err(
                    "modifier_rows length must match base_ids length",
                ));
            }
            for row in &entry.modifier_rows {
                if row.len() != tokenizer.num_modifier_groups {
                    return Err(PyValueError::new_err("modifier row width mismatch"));
                }
            }
            tokenizer.add_entry(entry);
        }
        for entry in &cfg.reverse_entries {
            if entry.modifier_rows.len() != entry.base_ids.len() {
                return Err(PyValueError::new_err(
                    "reverse entry modifier_rows length must match base_ids length",
                ));
            }
            for row in &entry.modifier_rows {
                if row.len() != tokenizer.num_modifier_groups {
                    return Err(PyValueError::new_err(
                        "reverse entry modifier row width mismatch",
                    ));
                }
            }
            if entry.token_ids.is_empty() || entry.base_ids.is_empty() {
                return Err(PyValueError::new_err(
                    "reverse entries must provide non-empty token_ids and base_ids",
                ));
            }
            tokenizer.add_reverse_entry(entry);
        }
        Ok(tokenizer)
    }

    fn process_ids(&self, raw_ids: Vec<u32>, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let (output_ids, modifier_rows) = self.process_ids_impl(&raw_ids);
        self.build_result_object(py, output_ids, modifier_rows)
    }

    fn process_ids_batch(
        &self,
        raw_ids_batch: Vec<Vec<u32>>,
        py: Python<'_>,
    ) -> PyResult<Py<PyAny>> {
        let out = PyList::empty(py);
        for raw_ids in raw_ids_batch {
            let (output_ids, modifier_rows) = self.process_ids_impl(&raw_ids);
            out.append(self.build_result_object(py, output_ids, modifier_rows)?)?;
        }
        Ok(out.unbind().into_any())
    }

    fn process_text(&self, text: String, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let raw_ids = self.encode_text_impl(text.as_str())?;
        let (output_ids, modifier_rows) = self.process_ids_impl(&raw_ids);
        self.build_result_object(py, output_ids, modifier_rows)
    }

    fn encode_text(&self, text: String) -> PyResult<Vec<u32>> {
        self.encode_text_impl(&text)
    }

    fn process_text_batch(&self, texts: Vec<String>, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let out = PyList::empty(py);
        let raw_ids_batch: Vec<Vec<u32>> = texts
            .iter()
            .map(|text| self.base_bpe.core.encode_ordinary(text))
            .collect();
        let processed: Vec<(Vec<u32>, Vec<Vec<u16>>)> = py.detach(|| {
            raw_ids_batch
                .par_iter()
                .map(|raw_ids| self.process_ids_impl(raw_ids))
                .collect()
        });
        for (output_ids, modifier_rows) in processed {
            out.append(self.build_result_object(py, output_ids, modifier_rows)?)?;
        }
        Ok(out.unbind().into_any())
    }

    fn decode_with_modifiers(
        &self,
        token_ids: Vec<u32>,
        modifier_rows: Vec<Vec<u16>>,
    ) -> PyResult<String> {
        self.reconstruct_surface_impl(&token_ids, &modifier_rows)
    }

    fn utf8_len_with_modifiers_batch(
        &self,
        token_ids: Vec<u32>,
        modifier_rows: Vec<Vec<u16>>,
    ) -> PyResult<Vec<u32>> {
        if token_ids.len() != modifier_rows.len() {
            return Err(PyValueError::new_err(format!(
                "token_ids and modifier_rows length mismatch: {} != {}",
                token_ids.len(),
                modifier_rows.len()
            )));
        }
        let mut out = Vec::with_capacity(token_ids.len());
        for (token_id, modifier_row) in token_ids.into_iter().zip(modifier_rows.into_iter()) {
            if modifier_row.len() != self.num_modifier_groups {
                return Err(PyValueError::new_err(format!(
                    "modifier row width mismatch: expected {}, got {}",
                    self.num_modifier_groups,
                    modifier_row.len()
                )));
            }
            if self.token_meta_ref(token_id).is_byte_fallback {
                if let (Some(token_bytes), Some(delta)) = (
                    self.base_bpe_token_bytes(token_id),
                    self.modifier_utf8_delta(&modifier_row),
                ) {
                    out.push((token_bytes.len() + delta) as u32);
                    continue;
                }
            }
            let surface = self.reconstruct_surface_impl(&[token_id], &[modifier_row])?;
            out.push(surface.len() as u32);
        }
        Ok(out)
    }

    fn debug_tokenize_text_json(&self, text: String) -> PyResult<String> {
        let raw_ids = self.encode_text_impl(&text)?;
        let raw_tokens: Vec<serde_json::Value> = raw_ids
            .iter()
            .map(|token_id| self.token_debug_value(*token_id))
            .collect();
        Ok(json!({
            "text": text,
            "raw_ids": raw_ids,
            "raw_tokens": raw_tokens,
        })
        .to_string())
    }

    fn debug_process_text_json(&self, text: String) -> PyResult<String> {
        let raw_ids = self.encode_text_impl(&text)?;
        let raw_tokens: Vec<serde_json::Value> = raw_ids
            .iter()
            .map(|token_id| self.token_debug_value(*token_id))
            .collect();
        let (output_ids, modifier_rows) = self.process_ids_impl(&raw_ids);
        let roundtrip_text = self.reconstruct_surface_impl(&output_ids, &modifier_rows)?;
        Ok(json!({
            "text": text,
            "raw_ids": raw_ids,
            "raw_tokens": raw_tokens,
            "output_ids": output_ids,
            "modifier_rows": modifier_rows,
            "roundtrip_text": roundtrip_text,
        })
        .to_string())
    }
}
