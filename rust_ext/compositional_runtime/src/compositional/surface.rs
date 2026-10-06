use super::*;

impl CompositionalTokenizer {
    pub(super) fn apply_base_capitalization(&self, surface: &str, modifiers: &[u16]) -> String {
        let Some(group_idx) = self.group_idx("base_capitalization") else {
            return surface.to_string();
        };
        let value = modifiers[group_idx];
        if value == self.default_modifier[group_idx] {
            return surface.to_string();
        }
        let value_name = self
            .value_name("base_capitalization", value)
            .unwrap_or_default()
            .to_lowercase();
        if value_name.starts_with("add_") || value_name.starts_with("with_") || value == 1 {
            return capitalize_first_alpha(surface);
        }
        if value_name.starts_with("remove_") || value_name.starts_with("lower_") {
            return lowercase_first_alpha(surface);
        }
        surface.to_string()
    }

    pub(super) fn synthesize_surface(&self, lexical_surface: &str, modifiers: &[u16]) -> String {
        let mut surface = if lexical_surface.is_empty() {
            String::new()
        } else if lexical_surface.trim().is_empty() {
            lexical_surface.to_string()
        } else {
            lexical_surface.trim_start_matches(' ').to_string()
        };
        surface = self.apply_base_capitalization(&surface, modifiers);

        let prefix_punct =
            self.literal_from_group(modifiers, "prefix_punctuation", &["punct_prefix_"]);
        let mut preposition = self.literal_from_group(modifiers, "prepositions", &["prep_"]);
        let postposition = self.literal_from_group(modifiers, "postpositions", &["postp_"]);
        let pronoun = self.literal_from_group(modifiers, "pronouns", &["pron_"]);
        let mut determiner = self
            .literal_from_group(modifiers, "determiners", &["det_", "article_"])
            .or_else(|| self.literal_from_group(modifiers, "article_det", &["det_", "article_"]))
            .or_else(|| self.literal_from_group(modifiers, "articles", &["article_", "det_"]));
        let suffix_punct =
            self.literal_from_group(modifiers, "suffix_punctuation", &["punct_suffix_"]);

        if let Some(text) = preposition.as_ref() {
            if self.space_setting(modifiers, "prep_capitalization") {
                preposition = Some(capitalize_first_alpha(text));
            }
        }
        if let Some(text) = determiner.as_ref() {
            if self.space_setting(modifiers, "article_capitalization") {
                determiner = Some(capitalize_first_alpha(text));
            }
        }

        let mut pieces = Vec::new();
        if let Some(value) = preposition {
            if !value.is_empty() {
                pieces.push(value);
            }
        }
        if let Some(value) = pronoun {
            if !value.is_empty() {
                pieces.push(value);
            }
        }
        if let Some(value) = determiner {
            if !value.is_empty() {
                pieces.push(value);
            }
        }
        if !surface.is_empty() {
            pieces.push(surface);
        }
        if let Some(value) = postposition {
            if !value.is_empty() {
                pieces.push(value);
            }
        }

        let mut expr = pieces.join(" ");
        if let Some(value) = prefix_punct {
            expr = format!("{value}{expr}");
        }
        if let Some(value) = suffix_punct {
            expr = format!("{expr}{value}");
        }
        if self.space_setting(modifiers, "space_prefix")
            && !expr.is_empty()
            && !expr.starts_with(char::is_whitespace)
        {
            expr.insert(0, ' ');
        }
        expr
    }

    pub(super) fn combine_modifier_rows(&self, modifier_rows: &[Vec<u16>]) -> Vec<u16> {
        let mut combined = self.default_modifier.clone();
        for row in modifier_rows {
            for (group_idx, value) in row.iter().enumerate() {
                if *value != self.default_modifier[group_idx] {
                    combined[group_idx] = *value;
                }
            }
        }
        combined
    }

    pub(super) fn combine_pending(
        &self,
        base_modifier: &[u16],
        pending_groups: &[PendingGroup],
    ) -> Vec<u16> {
        if pending_groups.is_empty() {
            return base_modifier.to_vec();
        }
        let mut combined = base_modifier.to_vec();
        let mut applied = HashSet::new();
        for group in pending_groups.iter().rev() {
            if applied.insert(group.group_idx) {
                combined[group.group_idx] = group.rel_idx;
            }
        }
        combined
    }

    pub(super) fn spread_multi_token_modifiers(
        &self,
        combined_modifier: &[u16],
        base_len: usize,
    ) -> Vec<Vec<u16>> {
        if base_len <= 1 {
            return vec![combined_modifier.to_vec()];
        }
        let first_groups: HashSet<usize> = self
            .runtime
            .multi_token_first_group_indices
            .iter()
            .copied()
            .collect();
        let mut first_modifier = self.empty_modifier();
        let mut last_modifier = self.empty_modifier();
        for (idx, value) in combined_modifier.iter().enumerate() {
            if *value == self.default_modifier[idx] {
                continue;
            }
            if first_groups.contains(&idx) {
                first_modifier[idx] = *value;
            } else {
                last_modifier[idx] = *value;
            }
        }
        if base_len == 2 {
            return vec![first_modifier, last_modifier];
        }
        let mut out = vec![first_modifier];
        for _ in 0..(base_len - 2) {
            out.push(self.empty_modifier());
        }
        out.push(last_modifier);
        out
    }

    pub(super) fn modifier_has_active_group(&self, modifier: &[u16], group_name: &str) -> bool {
        let Some(group_idx) = self.group_idx(group_name) else {
            return false;
        };
        modifier[group_idx] != self.default_modifier[group_idx]
    }

    pub(super) fn modifier_has_only_surface_groups(&self, modifier: &[u16]) -> bool {
        let mut allowed = HashSet::new();
        for group_name in SURFACE_GROUP_NAMES {
            if let Some(group_idx) = self.group_idx(group_name) {
                allowed.insert(group_idx);
            }
        }
        for (idx, value) in modifier.iter().enumerate() {
            if *value == self.default_modifier[idx] {
                continue;
            }
            if !allowed.contains(&idx) {
                return false;
            }
        }
        true
    }

    pub(super) fn is_intra_word_cap_alias_match(
        &self,
        match_length: usize,
        modifier: &[u16],
    ) -> bool {
        if match_length != 1 {
            return false;
        }
        let Some(group_idx) = self.group_idx("base_capitalization") else {
            return false;
        };
        modifier[group_idx] == 1 && self.modifier_has_only_surface_groups(modifier)
    }

    pub(super) fn strip_groups(&self, modifier_values: &[u16], group_names: &[&str]) -> Vec<u16> {
        let mut cleaned = modifier_values.to_vec();
        for group_name in group_names {
            if let Some(group_idx) = self.group_idx(group_name) {
                cleaned[group_idx] = self.default_modifier[group_idx];
            }
        }
        cleaned
    }

    pub(super) fn strip_invalid_detached_modifier_groups(
        &self,
        modifier_values: &[u16],
        raw_ids: &[u32],
        start_idx: usize,
        consumed_len: usize,
        pending_has_prefix_punct: bool,
    ) -> Vec<u16> {
        if self.can_attach_detached_modifier(
            raw_ids,
            start_idx,
            consumed_len,
            pending_has_prefix_punct,
        ) {
            return modifier_values.to_vec();
        }
        self.strip_groups(modifier_values, &STRIP_INVALID_DETACHED_GROUPS)
    }

    pub(super) fn strip_nonlexical_surface_groups(
        &self,
        modifier_values: &[u16],
        base_ids: &[u32],
        raw_ids: &[u32],
        start_idx: usize,
        consumed_len: usize,
    ) -> Vec<u16> {
        if base_ids
            .iter()
            .any(|base_id| self.token_meta_ref(*base_id).has_word_char)
        {
            return modifier_values.to_vec();
        }
        let mut cleaned = self.strip_groups(modifier_values, &STRIP_NONLEXICAL_GROUPS);
        let punctuation_groups = ["prefix_punctuation", "suffix_punctuation"];
        let has_punctuation = punctuation_groups.iter().any(|group_name| {
            self.group_idx(group_name)
                .map(|idx| cleaned[idx] != self.default_modifier[idx])
                .unwrap_or(false)
        });
        if has_punctuation {
            let end_idx = usize::min(start_idx + consumed_len, raw_ids.len());
            let raw_surface = self.decode_ids(&raw_ids[start_idx..end_idx]);
            let base_surface = self.decode_ids(base_ids);
            if self.synthesize_surface(&base_surface, &cleaned) != raw_surface {
                cleaned = self.strip_groups(&cleaned, &punctuation_groups);
            }
        }
        cleaned
    }

    pub(super) fn raw_expr_has_leading_space(&self, raw_ids: &[u32], start_idx: usize) -> bool {
        if start_idx == 0 {
            return false;
        }
        let prev_token_id = raw_ids[start_idx - 1];
        if self.token_meta_ref(prev_token_id).is_whitespace_only {
            return false;
        }
        self.token_meta_ref(raw_ids[start_idx]).has_space_prefix
    }

    pub(super) fn next_non_whitespace_idx(
        &self,
        raw_ids: &[u32],
        start_idx: usize,
    ) -> Option<usize> {
        let mut idx = start_idx;
        while idx < raw_ids.len() {
            if !self.token_meta_ref(raw_ids[idx]).is_whitespace_only {
                return Some(idx);
            }
            idx += 1;
        }
        None
    }

    pub(super) fn token_can_host_expr_space(&self, raw_ids: &[u32], start_idx: usize) -> bool {
        let Some(next_idx) = self.next_non_whitespace_idx(raw_ids, start_idx) else {
            return false;
        };
        if next_idx != start_idx {
            return false;
        }
        let next_meta = self.token_meta_ref(raw_ids[next_idx]);
        if next_meta.has_space_prefix {
            return false;
        }
        if self.raw_position_has_word_char(raw_ids, next_idx) {
            return true;
        }
        if next_meta.determiner.is_some()
            || next_meta.pronoun.is_some()
            || next_meta.preposition.is_some()
        {
            return true;
        }
        if next_meta.prefix_punctuation.is_some() {
            let lookahead_idx = self.next_non_whitespace_idx(raw_ids, next_idx + 1);
            return lookahead_idx
                .map(|idx| self.raw_position_has_word_char(raw_ids, idx))
                .unwrap_or(false);
        }
        false
    }

    pub(super) fn following_boundary_space_count(
        &self,
        raw_ids: &[u32],
        start_idx: usize,
    ) -> Option<(usize, usize)> {
        let mut count = 0usize;
        let mut idx = start_idx;
        while idx < raw_ids.len() && self.token_meta_ref(raw_ids[idx]).is_whitespace_only {
            if !self.token_meta_ref(raw_ids[idx]).is_single_ascii_space {
                return None;
            }
            count += 1;
            idx += 1;
        }
        if idx >= raw_ids.len() {
            return None;
        }
        count += leading_ascii_spaces(&self.token_meta_ref(raw_ids[idx]).token_text);
        Some((count, idx))
    }

    pub(super) fn apply_contextual_space_prefix(
        &self,
        modifier_values: &[u16],
        raw_ids: &[u32],
        start_idx: usize,
        use_pending_space: bool,
        pending_leading_space: bool,
    ) -> Vec<u16> {
        let Some(space_idx) = self.group_idx("space_prefix") else {
            return modifier_values.to_vec();
        };
        let mut normalized = modifier_values.to_vec();
        let expr_has_space = if use_pending_space {
            pending_leading_space
        } else {
            self.raw_expr_has_leading_space(raw_ids, start_idx)
        };
        normalized[space_idx] = if expr_has_space {
            1
        } else {
            self.default_modifier[space_idx]
        };
        normalized
    }

    pub(super) fn apply_contextual_base_cap(
        &self,
        modifier_values: &[u16],
        raw_ids: &[u32],
        start_idx: usize,
        consumed_len: usize,
    ) -> Vec<u16> {
        let Some(cap_idx) = self.group_idx("base_capitalization") else {
            return modifier_values.to_vec();
        };
        let mut normalized = modifier_values.to_vec();
        if normalized[cap_idx] == self.default_modifier[cap_idx] {
            return normalized;
        }
        if self.raw_span_has_byte_fallback(raw_ids, start_idx, consumed_len) {
            normalized[cap_idx] = self.default_modifier[cap_idx];
            return normalized;
        }
        let raw_surface = self
            .decode_ids(&raw_ids[start_idx..usize::min(start_idx + consumed_len, raw_ids.len())]);
        if !is_capitalized_surface(&raw_surface) || !is_base_cap_representable_surface(&raw_surface)
        {
            normalized[cap_idx] = self.default_modifier[cap_idx];
        }
        normalized
    }

    pub(super) fn entry_has_case_mismatch(
        &self,
        entry: &EntryValue,
        modifier_values: &[u16],
        raw_ids: &[u32],
        start_idx: usize,
        consumed_len: usize,
    ) -> bool {
        let Some(cap_idx) = self.group_idx("base_capitalization") else {
            return false;
        };
        if modifier_values[cap_idx] != self.default_modifier[cap_idx] {
            return false;
        }
        let raw_surface = self
            .decode_ids(&raw_ids[start_idx..usize::min(start_idx + consumed_len, raw_ids.len())]);
        let base_surface = self.decode_ids(&entry.base_ids);
        if raw_surface == base_surface {
            return false;
        }
        let raw_letters: String = raw_surface
            .chars()
            .filter(|ch| ch.is_alphabetic())
            .flat_map(|ch| ch.to_lowercase())
            .collect();
        let base_letters: String = base_surface
            .chars()
            .filter(|ch| ch.is_alphabetic())
            .flat_map(|ch| ch.to_lowercase())
            .collect();
        if raw_letters.is_empty() || raw_letters != base_letters {
            return false;
        }
        if !is_base_cap_representable_surface(&raw_surface) {
            return true;
        }
        let raw_is_upper = first_alpha_is_upper(&raw_surface);
        let base_is_upper = first_alpha_is_upper(&base_surface);
        matches!((raw_is_upper, base_is_upper), (Some(left), Some(right)) if left != right)
    }

    pub(super) fn lexical_surface_for_reverse_entry(&self, entry: &ReverseEntryValue) -> String {
        self.decode_ids(&entry.base_ids)
    }

    pub(super) fn reverse_entry_surface(&self, entry: &ReverseEntryValue) -> String {
        let combined = self.combine_modifier_rows(&entry.modifier_rows);
        let base_surface = self.lexical_surface_for_reverse_entry(entry);
        let Some(surface) = entry.surface.as_ref() else {
            return self.synthesize_surface(&base_surface, &combined);
        };
        let synthesized_base = self.synthesize_surface(&base_surface, &combined);
        if *surface == synthesized_base {
            return surface.clone();
        }
        if surface.to_lowercase() == synthesized_base.to_lowercase() {
            return synthesized_base;
        }
        // Some hand-authored/legacy entries store the lexical surface before
        // applying its modifier rows. Apply the rows once in that case.
        self.synthesize_surface(surface, &combined)
    }

    pub(super) fn reconstruct_surface_impl(
        &self,
        token_ids: &[u32],
        modifier_rows: &[Vec<u16>],
    ) -> PyResult<String> {
        if token_ids.len() != modifier_rows.len() {
            return Err(PyValueError::new_err(format!(
                "token_ids and modifier_rows length mismatch: {} != {}",
                token_ids.len(),
                modifier_rows.len()
            )));
        }
        let mut chunks = Vec::new();
        let mut idx = 0usize;
        while idx < token_ids.len() {
            if self.token_meta_ref(token_ids[idx]).is_byte_fallback {
                let component_end = self.byte_component_end(token_ids, idx);
                if let Some(lexical_surface) =
                    self.decode_token_bytes(&token_ids[idx..component_end])
                {
                    let combined = self.combine_modifier_rows(&modifier_rows[idx..component_end]);
                    if combined == self.default_modifier {
                        chunks.push(lexical_surface);
                    } else {
                        chunks.push(self.synthesize_surface(&lexical_surface, &combined));
                    }
                    idx = component_end;
                    continue;
                }
            }
            if let Some(entry) = self.longest_reverse_match(token_ids, modifier_rows, idx) {
                chunks.push(self.reverse_entry_surface(&entry));
                idx += entry.consumed_len;
                continue;
            }
            if modifier_rows[idx] == self.default_modifier {
                let literal_start = idx;
                idx += 1;
                while idx < token_ids.len() {
                    if modifier_rows[idx] != self.default_modifier {
                        break;
                    }
                    if self.token_meta_ref(token_ids[idx]).is_byte_fallback {
                        break;
                    }
                    if self
                        .longest_reverse_match(token_ids, modifier_rows, idx)
                        .is_some()
                    {
                        break;
                    }
                    idx += 1;
                }
                chunks.push(self.decode_ids(&token_ids[literal_start..idx]));
                continue;
            }
            let lexical_surface = self.decode_single(token_ids[idx]);
            chunks.push(self.synthesize_surface(&lexical_surface, &modifier_rows[idx]));
            idx += 1;
        }
        Ok(chunks.concat())
    }
}
